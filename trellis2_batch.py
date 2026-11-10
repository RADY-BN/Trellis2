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
  python trellis2_batch.py                       # photos_clean/ -> glb/draft/
  python trellis2_batch.py --quality high        # photos_clean/ -> glb/high/
  python trellis2_batch.py -q high --export-only # redo the export of saved meshes only
Any explicit flag (--res, --steps, --faces, --tex, --remesh) overrides the preset.
Input: by default each photo's building is cut out at full resolution and scaled to fill a 1024 px
square before TRELLIS sees it (TRELLIS would shrink the whole photo to 1024 px first). --no-fit: off.
Low-poly: blender_lowpoly.py on the exported GLB.
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
    "draft":    dict(res="512",          steps=12, faces=50_000,    tex=2048),  # ~1 min, quick look
    "game":     dict(res="1024_cascade", steps=12, faces=100_000,   tex=2048),  # good detail, light
    "high":     dict(res="1024_cascade", steps=25, faces=500_000,   tex=4096),  # close-up, use Nanite
    "ultra":    dict(res="1536_cascade", steps=25, faces=1_000_000, tex=4096),  # likely > 12 GB VRAM
}

# The raw mesh is reduced to this many faces before it is saved (the 1024 raw mesh has ~12M).
SAVE_FACES = 3_000_000


# ----------------------------------------------------------------------------- helpers

def is_oom(e: BaseException) -> bool:
    # PyTorch raises OutOfMemoryError; CuMesh/o-voxel raise RuntimeError("... out of memory")
    return isinstance(e, torch.cuda.OutOfMemoryError) or "out of memory" in str(e).lower()


def clear_cuda_error():
    """CUDA extensions (CuMesh) leave a failed cudaMalloc in the runtime's 'last error'. PyTorch then
    reports that stale "out of memory" at its next kernel launch, somewhere unrelated, so the next
    attempt fails too. Reading the error with cudaGetLastError resets it."""
    import ctypes
    for name in ("libcudart.so.12", "libcudart.so"):
        try:
            ctypes.CDLL(name).cudaGetLastError()
            return
        except (OSError, AttributeError):
            continue


def patch_fill_holes():
    """TRELLIS fills small holes in the raw decoded mesh (often >12M triangles) with CuMesh, which can
    need more GPU memory than generating did. Skip it when it does not fit: the export phase works on
    a smaller mesh and closes holes there."""
    from trellis2.representations.mesh import base
    if getattr(base.Mesh.fill_holes, "_safe", False):
        return
    original = base.Mesh.fill_holes

    def fill_holes(self, *args, **kwargs):
        try:
            return original(self, *args, **kwargs)
        except Exception as e:
            if not is_oom(e):
                raise
        clear_cuda_error()
        free_gpu()
        print(f"     skipped hole filling on the raw mesh ({self.faces.shape[0]:,} triangles, not enough GPU memory)")

    fill_holes._safe = True
    base.Mesh.fill_holes = fill_holes


def free_gpu():
    # Return PyTorch's cached GPU blocks to the driver. CuMesh allocates with plain
    # cudaMalloc, so memory PyTorch keeps cached is invisible to it.
    gc.collect()
    torch.cuda.empty_cache()


ERROR_LOG = None  # set in main(): <out>/errors.log


def where_failed(e: BaseException) -> str:
    """Last TRELLIS.2/script frame plus the error text; the full traceback goes to errors.log."""
    import traceback
    frames = traceback.extract_tb(e.__traceback__)
    ours = [fr for fr in frames if "site-packages" not in fr.filename] or frames
    fr = ours[-1]
    msg = str(e).strip().splitlines()[-1][:160] if str(e).strip() else type(e).__name__
    if ERROR_LOG is not None:
        with open(ERROR_LOG, "a") as fh:
            fh.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            fh.write("".join(traceback.format_exception(type(e), e, e.__traceback__)))
    return f"{Path(fr.filename).name}:{fr.lineno} {fr.name}: {msg}"


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
            # transformers 5 loads the checkpoint's fp16 weights as fp16, but the wrapper
            # feeds float32 images ("Input type (float) and bias type (c10::Half)")
            self.model.float()

    rembg_mod.BiRefNet = _BiRefNetOverride


FIT_SIZE = 1024  # TRELLIS.2 shrinks every input to 1024 px (longest side) before it crops to the object


def fit_to_frame(pipe, image, fill):
    """Cut the building out at full resolution and scale it so it fills a 1024 x 1024 transparent image.

    TRELLIS.2 itself shrinks the WHOLE photo to 1024 px first and only then crops to the object: a
    building covering 60 % of a 1550 px photo reaches the model at ~650 px. Cut out and scaled here,
    it reaches the model at ~1000 px. The transparent background also skips TRELLIS' own cut-out.
    Returns (image, building size in the photo, building size given to TRELLIS)."""
    import numpy as np
    from PIL import Image

    img = image
    alpha = np.asarray(img)[:, :, 3] if img.mode == "RGBA" else None
    if alpha is None or (alpha == 255).all():
        img = img.convert("RGB")
        if getattr(pipe, "low_vram", False):
            pipe.rembg_model.to(pipe.device)
        img = pipe.rembg_model(img)               # BiRefNet at full resolution -> RGBA
        if getattr(pipe, "low_vram", False):
            pipe.rembg_model.cpu()
        alpha = np.asarray(img)[:, :, 3]
    ys, xs = np.nonzero(alpha > 0.8 * 255)        # the same threshold TRELLIS uses
    if len(xs) == 0:
        return image, 0, 0
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    side = max(x1 - x0, y1 - y0)
    pad = max(2, int(0.02 * side))                # keep the soft edge of the cut-out
    crop = img.crop((max(0, x0 - pad), max(0, y0 - pad), min(img.width, x1 + pad), min(img.height, y1 + pad)))
    target = int(round(fill * FIT_SIZE))
    s = target / (side + 2 * pad)
    size = (max(1, round(crop.width * s)), max(1, round(crop.height * s)))
    crop = crop.convert("RGBa").resize(size, Image.Resampling.LANCZOS).convert("RGBA")  # no colour fringes
    canvas = Image.new("RGBA", (FIT_SIZE, FIT_SIZE), (0, 0, 0, 0))
    canvas.paste(crop, ((FIT_SIZE - crop.width) // 2, (FIT_SIZE - crop.height) // 2))
    return canvas, int(side), int(round(side * s))


# ----------------------------------------------------------------------------- phase 1

def pipeline_to_cpu(pipe):
    """After an OOM the stage that failed is still on the GPU (low_vram moves it back only on success)."""
    for mdl in getattr(pipe, "models", {}).values():
        if hasattr(mdl, "cpu"):
            mdl.cpu()
    for name in ("image_cond_model", "rembg_model"):
        obj = getattr(pipe, name, None)
        if obj is not None and hasattr(obj, "cpu"):
            obj.cpu()


def generate_one(pipe, image, res, a):
    """Returns (vertices, faces, attrs, coords, layout, voxel_size, raw_faces) on the CPU."""
    steps = {"steps": a.steps}  # merged with pipeline.json defaults
    mesh = pipe.run(image, seed=a.seed, pipeline_type=res, max_num_tokens=a.max_tokens,
                    sparse_structure_sampler_params=steps,
                    shape_slat_sampler_params=steps,
                    tex_slat_sampler_params=steps)[0]
    raw = mesh.faces.shape[0]
    v, f = mesh.vertices, mesh.faces
    if raw > SAVE_FACES:
        free_gpu()
        try:
            v, f = simplify(v, f, SAVE_FACES)
            ok = True
        except Exception as e:
            if not is_oom(e):
                raise
            ok = False
        if not ok:
            clear_cuda_error()
            # the export phase simplifies it later, with the model unloaded
            print(f"     not enough GPU memory to simplify now ({gpu_free_gb():.1f} GB free), saving the full mesh")
            v, f = mesh.vertices, mesh.faces
    out = (v.cpu(), f.cpu(), mesh.attrs.cpu(), mesh.coords.cpu(), mesh.layout, mesh.voxel_size, raw)
    del mesh, v, f
    return out


def save_inputs_only(images, mesh_dir, a):
    """Only cut out and fit the photos (BiRefNet, seconds), save them to <out>/_inputs/ - to check what
    TRELLIS would get, without generating."""
    from PIL import Image
    import trellis2.pipelines.rembg as rembg_mod

    if a.rembg:
        use_rembg_model(a.rembg)
    rembg = rembg_mod.BiRefNet()
    rembg.cuda()
    pipe = SimpleNamespace(rembg_model=rembg, low_vram=False, device="cuda")
    out = mesh_dir.parent / "_inputs"
    out.mkdir(exist_ok=True)
    for img_path in images:
        image, before, after = fit_to_frame(pipe, Image.open(img_path), a.fill)
        image.save(out / f"{img_path.stem}.png")
        print(f"input {img_path.name}: building {before} px -> {after} px, saved to {out / (img_path.stem + '.png')}")


# Saved meshes are written into WSL's own disk first and then copied next to the output: torch.save straight
# onto the Windows drive (/mnt/c) can fail ("PytorchStreamWriter failed writing file data/0") and leave a
# broken .pt behind. A mesh that cannot be copied over stays in LOCAL_MESHES and is exported from there.
LOCAL_MESHES = Path.home() / ".cache" / "trellis2_meshes"


def local_dir(mesh_dir):
    import hashlib
    return LOCAL_MESHES / hashlib.md5(str(mesh_dir.resolve()).encode()).hexdigest()[:10]


def mem_info():
    try:
        kv = {k: int(v.split()[0]) / 1e6 for k, v in (ln.split(":", 1) for ln in open("/proc/meminfo"))}
        return f"{kv['MemAvailable']:.1f} GB RAM free, {kv['SwapFree']:.1f} GB swap free"
    except (OSError, KeyError, ValueError):
        return "RAM unknown"


def disk_info(path):
    import shutil
    try:
        return f"{shutil.disk_usage(path).free / 1e9:.0f} GB free on {path}"
    except OSError:
        return f"disk of {path} unknown"


def mesh_ok(path):
    """a saved mesh is a complete zip archive (torch.save); a failed save leaves a broken one"""
    import zipfile
    try:
        return path.is_file() and path.stat().st_size > 1024 and zipfile.is_zipfile(path)
    except OSError:
        return False


def find_mesh(mesh_dir, stem):
    """the saved mesh of <stem>: next to the output, else in WSL's own disk; broken files are deleted"""
    for path in (mesh_dir / f"{stem}.pt", local_dir(mesh_dir) / f"{stem}.pt"):
        if mesh_ok(path):
            return path
        if path.exists():
            print(f"     {path} is broken (its save did not finish): deleted, {stem} is generated again")
            path.unlink(missing_ok=True)
    return None


def save_mesh(data, mesh_file):
    """Save in WSL, copy over in pieces, rename at the end. Returns the path it is in, or None."""
    import shutil
    local = local_dir(mesh_file.parent) / mesh_file.name
    local.parent.mkdir(parents=True, exist_ok=True)
    sizes = sum(t.numel() * t.element_size() for t in data.values() if isinstance(t, torch.Tensor)) / 1e6
    part = local.with_name(local.name + ".part")
    try:
        torch.save(data, part)
        os.replace(part, local)
    except Exception as e:
        part.unlink(missing_ok=True)
        print(f"     saving the mesh ({sizes:,.0f} MB) in WSL failed: {str(e).strip().splitlines()[-1][:160]}"
              f" ({mem_info()}, {disk_info(local.parent)}); trying {mesh_file.parent}")
        local = None
    tmp = mesh_file.with_name(mesh_file.name + ".part")
    try:
        if local is not None:
            with open(local, "rb") as src, open(tmp, "wb") as dst:
                shutil.copyfileobj(src, dst, 16 * 1024 * 1024)      # 16 MB pieces
        else:
            torch.save(data, tmp)
        os.replace(tmp, mesh_file)
    except Exception as e:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        why = f"{str(e).strip().splitlines()[-1][:160]} ({mem_info()}, {disk_info(mesh_file.parent)})"
        if local is None:
            print(f"     could not save the mesh anywhere: {why}")
            return None
        try:
            mesh_file.unlink(missing_ok=True)      # an older mesh there must not win over this one
        except OSError:
            pass
        print(f"     could not copy the mesh to {mesh_file}: {why}\n     it stays in {local} and is exported from there")
        return local
    if local is not None:
        local.unlink(missing_ok=True)
    return mesh_file


def generate_all(images, mesh_dir, a):
    from PIL import Image
    from trellis2.pipelines import Trellis2ImageTo3DPipeline

    if a.rembg:
        use_rembg_model(a.rembg)
    pipe = Trellis2ImageTo3DPipeline.from_pretrained(a.model)
    pipe.cuda()
    patch_fill_holes()
    # if the GPU runs out of memory at 1024/1536, try the same image again at 512
    resolutions = [a.res] + (["512"] if a.res != "512" and not a.no_fallback else [])

    for img_path in images:
        mesh_file = mesh_dir / f"{img_path.stem}.pt"
        image = Image.open(img_path)
        if a.fit:
            image, before, after = fit_to_frame(pipe, image, a.fill)
            if after:
                print(f"fit  {img_path.name}: building {before} px in the photo -> {after} px for TRELLIS"
                      f" (without --fit TRELLIS would see ~{int(before * min(1, 1024 / max(Image.open(img_path).size)))} px)")
            if a.save_input:
                (mesh_dir.parent / "_inputs").mkdir(exist_ok=True)
                image.save(mesh_dir.parent / "_inputs" / f"{img_path.stem}.png")
        for res in resolutions:
            t0 = time.time()
            where = None
            try:
                # RGBA input with a clean alpha skips the automatic background removal
                v, f, attrs, coords, layout, voxel_size, raw = generate_one(pipe, image, res, a)
            except Exception as e:
                if not is_oom(e):
                    raise
                where = where_failed(e)
                del e
            if where is None:
                saved = save_mesh(dict(vertices=v, faces=f, attrs=attrs, coords=coords, layout=layout,
                                       voxel_size=voxel_size, res=res), mesh_file)
                if saved is not None:
                    print(f"gen  {img_path.name}  {time.time() - t0:.0f}s  res={res}"
                          f"  ({raw:,} raw faces -> {f.shape[0]:,} saved)")
                else:
                    print(f"FAIL {img_path.name}: generated but not saved (see above), going on with the next photo")
                del v, f, attrs, coords
                free_gpu()
                break
            clear_cuda_error()
            pipeline_to_cpu(pipe)
            free_gpu()
            print(f"OOM  {img_path.name} at res={res} in {where} ({gpu_free_gb():.1f} GB GPU free now)")
            if res == resolutions[-1]:
                print("     giving up on this image: close GPU-heavy apps (check Task Manager > GPU memory) "
                      "or use a lower --quality")
            else:
                print("     retrying at res=512 (use --no-fallback to skip)")

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
            clear_cuda_error()
            # do not keep `e`: its traceback would pin the failed attempt's GPU buffers
        if ok:
            if a.height > 0:
                glb.apply_scale(a.height / float(glb.extents[1]))  # glTF is Y-up
            elif a.size > 0:
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
    p.add_argument("--in", dest="inp", default="photos_clean", help="image file or folder (default: photos_clean)")
    p.add_argument("--out", default=None, help="output folder (default: glb/<quality>)")
    p.add_argument("--size", type=float, default=0.0,
                   help="scale so the longest side is this many metres (0 = keep 1 m cube)")
    p.add_argument("--height", type=float, default=0.0,
                   help="scale so the height is this many metres (overrides --size)")
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
                   help="1536 only: token budget before it drops towards 1024 (no effect at 1024)")
    p.add_argument("--no-fallback", action="store_true",
                   help="do not retry at 512 when the GPU runs out of memory at 1024/1536")
    p.add_argument("--seed", type=int, default=42, help="change it to get a different variation")
    p.add_argument("--no-fit", dest="fit", action="store_false",
                   help="give TRELLIS the photo as it is (by default the building is cut out at full "
                        "resolution and scaled to fill a 1024 px square: ~50 %% more detail)")
    p.add_argument("--fill", type=float, default=0.96,
                   help="share of the 1024 px square the building fills (default 0.96)")
    p.add_argument("--save-input", action="store_true",
                   help="save the image TRELLIS gets to <out>/_inputs/ (to check the cut-out)")
    p.add_argument("--inputs-only", action="store_true",
                   help="only cut out + fit the photos and save them to <out>/_inputs/ (seconds, no generation)")
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
    global ERROR_LOG
    ERROR_LOG = out_dir / "errors.log"

    if a.export_only:
        stems = sorted({p.stem for d in (mesh_dir, local_dir(mesh_dir)) for p in d.glob("*.pt")})
        todo = [m for m in (find_mesh(mesh_dir, s) for s in stems) if m is not None]
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
            elif not a.overwrite and find_mesh(mesh_dir, img.stem) is not None:
                print(f"skip generating {img.name} (mesh already saved, exporting it)")
            else:
                pending.append(img)
        if a.inputs_only:
            save_inputs_only(images, mesh_dir, a)
            return
        print(f"quality={a.quality}: res={a.res} steps={a.steps} faces={a.faces} tex={a.tex}"
              f" | {len(pending)} image(s) to generate -> {out_dir}")
        if pending:
            generate_all(pending, mesh_dir, a)
        todo = [m for m in (find_mesh(mesh_dir, img.stem) for img in images
                            if a.overwrite or not (out_dir / f"{img.stem}.glb").exists()) if m is not None]

    for mesh_file in todo:
        out = out_dir / f"{mesh_file.stem}.glb"
        t0 = time.time()
        print(f"export {mesh_file.stem}")
        if export_one(mesh_file, out, a):
            print(f"ok   {out.name}  {time.time() - t0:.0f}s  {out.stat().st_size / 1e6:.1f} MB")
            if not a.keep_mesh:
                mesh_file.unlink()
        else:
            print(f"FAIL {mesh_file.stem}: mesh kept in {mesh_file.parent}; close GPU-heavy apps and run again "
                  f"with --export-only (add e.g. --faces 200000 --tex 2048 for a lighter export)")


if __name__ == "__main__":
    main()
