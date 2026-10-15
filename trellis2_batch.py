"""
Batch image -> GLB with TRELLIS.2, tuned for a 12 GB GPU (RTX 4070 Ti) and game-engine import.

Modified from TRELLIS.2's example.py (MIT, Microsoft):
  * quality presets (draft / game / high / ultra) instead of many flags
  * two phases: 1) generate every image and save its mesh, 2) unload the model and export.
    The export gets the whole GPU and most of the RAM, and a failed export can be
    redone with --export-only without generating again.
  * GPU-memory fallbacks for the export on 12 GB cards
  * exports a game-ready GLB (PNG textures instead of WebP)
  * optional real-world scale (TRELLIS outputs fit a 1 m cube)
  * open background-removal model instead of the gated RMBG-2.0

Usage (from the repo root, inside the trellis2 conda env):
  python trellis2_batch.py                       # photos/ -> glb/draft/
  python trellis2_batch.py --quality high        # photos/ -> glb/high/
  python trellis2_batch.py -q high --export-only # redo the export of saved meshes only
Any explicit flag (--res, --steps, --faces, --tex, --remesh) overrides the preset.
"""
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import gc
import time
from pathlib import Path
from types import SimpleNamespace

import torch

EXTS = {".png", ".jpg", ".jpeg", ".webp"}

# res   = voxel resolution of the generation (the biggest quality lever)
# steps = sampling steps per stage (12 is the model default)
# faces = triangle count of the exported mesh
# tex   = baked texture size in pixels
PRESETS = {
    "draft": dict(res="512",          steps=12, faces=50_000,    tex=2048),  # ~1 min, quick look
    "game":  dict(res="1024_cascade", steps=12, faces=100_000,   tex=2048),  # good detail, light for CARLA
    "high":  dict(res="1024_cascade", steps=25, faces=500_000,   tex=4096),  # close-up assets, use Nanite
    "ultra": dict(res="1536_cascade", steps=25, faces=1_000_000, tex=4096),  # max; likely too big for 12 GB VRAM
}

# The raw mesh is reduced to this many faces before it is saved (the 1024 raw mesh has ~12M).
SAVE_FACES = 3_000_000


# ----------------------------------------------------------------------------- helpers

def is_oom(e: BaseException) -> bool:
    # PyTorch raises OutOfMemoryError; CuMesh/o-voxel raise RuntimeError("... out of memory")
    return isinstance(e, torch.cuda.OutOfMemoryError) or "out of memory" in str(e).lower()


def free_gpu():
    # Return PyTorch's cached GPU blocks to the driver. CuMesh allocates with plain
    # cudaMalloc, so memory PyTorch keeps cached is invisible to it.
    gc.collect()
    torch.cuda.empty_cache()


def gpu_free_gb() -> float:
    free, _total = torch.cuda.mem_get_info()
    return free / 1e9


def simplify(vertices, faces, target):
    import cumesh
    m = cumesh.CuMesh()
    m.init(vertices.cuda(), faces.cuda())
    m.simplify(target)
    v, f = m.read()
    del m
    return v, f


def use_rembg_model(name: str):
    """Swap the background-removal model that pipeline.json names.

    The default there, briaai/RMBG-2.0, is gated and non-commercial (CC BY-NC).
    ZhengPeng7/BiRefNet is the open MIT model RMBG-2.0 is built on; it uses the
    same loader and preprocessing in trellis2/pipelines/rembg/BiRefNet.py.
    """
    import trellis2.pipelines.rembg as rembg_mod
    base = rembg_mod.BiRefNet

    class _BiRefNetOverride(base):
        def __init__(self, *args, **kwargs):
            super().__init__(model_name=name)

    rembg_mod.BiRefNet = _BiRefNetOverride


# ----------------------------------------------------------------------------- phase 1

def generate_all(images, mesh_dir, a):
    from PIL import Image
    from trellis2.pipelines import Trellis2ImageTo3DPipeline

    if a.rembg:
        use_rembg_model(a.rembg)
    pipe = Trellis2ImageTo3DPipeline.from_pretrained(a.model)
    pipe.cuda()
    steps = {"steps": a.steps}  # merged with pipeline.json defaults

    for img_path in images:
        mesh_file = mesh_dir / f"{img_path.stem}.pt"
        t0 = time.time()
        mesh = None
        try:
            # RGBA input with a clean alpha skips the automatic background removal
            mesh = pipe.run(Image.open(img_path), seed=a.seed, pipeline_type=a.res,
                            max_num_tokens=a.max_tokens,
                            sparse_structure_sampler_params=steps,
                            shape_slat_sampler_params=steps,
                            tex_slat_sampler_params=steps)[0]
            raw = mesh.faces.shape[0]
            v, f = mesh.vertices, mesh.faces
            if raw > SAVE_FACES:
                free_gpu()
                v, f = simplify(v, f, SAVE_FACES)
            torch.save(dict(vertices=v.cpu(), faces=f.cpu(), attrs=mesh.attrs.cpu(),
                            coords=mesh.coords.cpu(), layout=mesh.layout,
                            voxel_size=mesh.voxel_size, res=a.res), mesh_file)
            print(f"gen  {img_path.name}  {time.time() - t0:.0f}s  ({raw:,} raw faces -> {f.shape[0]:,} saved)")
        except Exception as e:
            if not is_oom(e):
                raise
            print(f"OOM  {img_path.name} during generation: try --max-tokens 32768 or a lower --quality")
        finally:
            del mesh
            v = f = None
            free_gpu()

    # unload the model so the export gets the GPU and the RAM
    del pipe
    free_gpu()


# ----------------------------------------------------------------------------- phase 2

def export_attempts(a, res):
    remesh_first = {"on": True, "off": False, "auto": res == "512"}[a.remesh]
    if remesh_first:
        # dual-contouring remesh at the generation resolution: cleanest topology, most memory
        return [dict(pre=3_000_000, remesh=True), dict(pre=1_000_000, remesh=True),
                dict(pre=1_000_000, remesh=False), dict(pre=600_000, remesh=False)]
    # at 1024+ the remesh grid alone does not fit in 12 GB, so simplify the generated surface
    return [dict(pre=2_000_000, remesh=False), dict(pre=1_000_000, remesh=False),
            dict(pre=600_000, remesh=False)]


def export_one(mesh_file, out, a):
    import o_voxel

    d = torch.load(mesh_file, weights_only=False)
    m = SimpleNamespace(vertices=d["vertices"].cuda(), faces=d["faces"].cuda(),
                        attrs=d["attrs"].cuda(), coords=d["coords"].cuda(),
                        layout=d["layout"], voxel_size=d["voxel_size"])
    res = d.get("res", a.res)
    del d
    last_msg = ""
    for i, att in enumerate(export_attempts(a, res), 1):
        ok = False
        try:
            free_gpu()
            if m.faces.shape[0] > att["pre"]:
                m.vertices, m.faces = simplify(m.vertices, m.faces, att["pre"])
                free_gpu()
            target = min(a.faces, m.faces.shape[0])
            print(f"     export attempt {i}: {m.faces.shape[0]:,} -> {target:,} faces, remesh={att['remesh']},"
                  f" {gpu_free_gb():.1f} GB GPU free")
            glb = o_voxel.postprocess.to_glb(
                vertices=m.vertices, faces=m.faces, attr_volume=m.attrs, coords=m.coords,
                attr_layout=m.layout, voxel_size=m.voxel_size,
                aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
                decimation_target=target, texture_size=a.tex,
                remesh=att["remesh"], remesh_band=1, remesh_project=0,
            )
            ok = True
        except Exception as e:
            if not is_oom(e):
                raise
            last_msg = str(e).strip().splitlines()[-1]
            # do not keep `e`: its traceback would pin the failed attempt's GPU buffers
        if ok:
            if a.size > 0:
                glb.apply_scale(a.size / float(max(glb.extents)))
            glb.export(str(out), extension_webp=False)  # PNG textures import everywhere
            del glb, m
            free_gpu()
            return True
        print(f"     attempt {i} ran out of GPU memory")
    del m
    free_gpu()
    print(f"     last error: {last_msg}")
    return False


# ----------------------------------------------------------------------------- main

def main():
    p = argparse.ArgumentParser(description="TRELLIS.2 image -> GLB")
    p.add_argument("--quality", "-q", default="draft", choices=list(PRESETS),
                   help="preset: draft, game, high, ultra")
    p.add_argument("--in", dest="inp", default="photos", help="image file or folder (default: photos)")
    p.add_argument("--out", default=None, help="output folder (default: glb/<quality>)")
    p.add_argument("--size", type=float, default=0.0,
                   help="scale so the longest side is this many metres (0 = keep 1 m cube)")
    p.add_argument("--overwrite", action="store_true", help="redo images that already have a GLB")
    p.add_argument("--export-only", action="store_true",
                   help="skip generation, export the meshes saved in <out>/_meshes")
    p.add_argument("--keep-mesh", action="store_true",
                   help="keep <out>/_meshes/*.pt after a successful export (for re-exporting)")
    # fine-tuning; each overrides the preset
    p.add_argument("--res", default=None, choices=["512", "1024", "1024_cascade", "1536_cascade"])
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--faces", type=int, default=None)
    p.add_argument("--tex", type=int, default=None)
    p.add_argument("--remesh", default="auto", choices=["auto", "on", "off"],
                   help="auto: remesh at 512 only (1024 remesh needs more than 12 GB)")
    p.add_argument("--max-tokens", type=int, default=49152,
                   help="detail budget for the cascades; lower it (e.g. 32768) on out-of-memory")
    p.add_argument("--seed", type=int, default=42, help="change it to get a different variation")
    p.add_argument("--model", default="microsoft/TRELLIS.2-4B")
    p.add_argument("--rembg", default="ZhengPeng7/BiRefNet",
                   help="background-removal model; 'briaai/RMBG-2.0' once you have access")
    a = p.parse_args()

    for k, v in PRESETS[a.quality].items():
        if getattr(a, k) is None:
            setattr(a, k, v)
    out_dir = Path(a.out or f"glb/{a.quality}")
    mesh_dir = out_dir / "_meshes"
    mesh_dir.mkdir(parents=True, exist_ok=True)

    if a.export_only:
        todo = sorted(mesh_dir.glob("*.pt"))
        if not todo:
            raise SystemExit(f"no saved meshes in '{mesh_dir}'")
    else:
        src = Path(a.inp)
        if not src.exists():
            raise SystemExit(f"'{src}' not found - put your images in a '{src}' folder next to this script")
        images = [src] if src.is_file() else sorted(f for f in src.iterdir() if f.suffix.lower() in EXTS)
        if not images:
            raise SystemExit(f"no .png/.jpg/.webp images in '{src}'")
        pending = []
        for img in images:
            if (out_dir / f"{img.stem}.glb").exists() and not a.overwrite:
                print(f"skip {img.stem}.glb (exists, use --overwrite to redo)")
            elif (mesh_dir / f"{img.stem}.pt").exists() and not a.overwrite:
                print(f"skip generating {img.name} (mesh already saved, exporting it)")
            else:
                pending.append(img)
        print(f"quality={a.quality}: res={a.res} steps={a.steps} faces={a.faces} tex={a.tex}"
              f" | {len(pending)} image(s) to generate -> {out_dir}")
        if pending:
            generate_all(pending, mesh_dir, a)
        todo = [mesh_dir / f"{img.stem}.pt" for img in images
                if (mesh_dir / f"{img.stem}.pt").exists()
                and (a.overwrite or not (out_dir / f"{img.stem}.glb").exists())]

    for mesh_file in todo:
        out = out_dir / f"{mesh_file.stem}.glb"
        t0 = time.time()
        print(f"export {mesh_file.stem}")
        if export_one(mesh_file, out, a):
            print(f"ok   {out.name}  {time.time() - t0:.0f}s  {out.stat().st_size / 1e6:.1f} MB")
            if not a.keep_mesh:
                mesh_file.unlink()
        else:
            print(f"FAIL {mesh_file.stem}: mesh kept in {mesh_dir}; close GPU-heavy apps and run again "
                  f"with --export-only (add e.g. --faces 200000 --tex 2048 for a lighter export)")


if __name__ == "__main__":
    main()
