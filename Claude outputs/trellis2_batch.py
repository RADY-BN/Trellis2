"""
Batch image -> GLB with TRELLIS.2, tuned for a 16 GB GPU and game-engine import.

Modified from TRELLIS.2's example.py (MIT, Microsoft):
  * quality presets (draft / game / high / ultra) instead of many flags
  * loops over a folder of images, skips ones already done
  * no preview video (saves time and VRAM)
  * exports a game-ready GLB (PNG textures instead of WebP)
  * optional real-world scale (TRELLIS outputs fit a 1 m cube)
  * open background-removal model instead of the gated RMBG-2.0

Usage (from the repo root, inside the trellis2 conda env):
  python trellis2_batch.py                       # photos/ -> glb/draft/
  python trellis2_batch.py --quality high        # photos/ -> glb/high/
  python trellis2_batch.py --quality high --size 12 --in photos/church.jpg
Any explicit flag (--res, --steps, --faces, --tex) overrides the preset.
"""
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import gc
import time
from pathlib import Path

import torch
from PIL import Image
from trellis2.pipelines import Trellis2ImageTo3DPipeline
import o_voxel

EXTS = {".png", ".jpg", ".jpeg", ".webp"}

# res   = voxel resolution of the generation (the biggest quality lever)
# steps = sampling steps per stage (12 is the model default)
# faces = triangle count of the exported mesh
# tex   = baked texture size in pixels
PRESETS = {
    "draft": dict(res="512",          steps=12, faces=50_000,    tex=2048),  # ~1 min, quick look
    "game":  dict(res="1024_cascade", steps=12, faces=100_000,   tex=2048),  # good detail, light for CARLA
    "high":  dict(res="1024_cascade", steps=25, faces=500_000,   tex=4096),  # close-up assets, use Nanite
    "ultra": dict(res="1536_cascade", steps=25, faces=1_000_000, tex=4096),  # max; may exceed 16 GB VRAM
}


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


def main():
    p = argparse.ArgumentParser(description="TRELLIS.2 image -> GLB")
    p.add_argument("--quality", "-q", default="draft", choices=list(PRESETS),
                   help="preset: draft, game, high, ultra")
    p.add_argument("--in", dest="inp", default="photos", help="image file or folder (default: photos)")
    p.add_argument("--out", default=None, help="output folder (default: glb/<quality>)")
    p.add_argument("--size", type=float, default=0.0,
                   help="scale so the longest side is this many metres (0 = keep 1 m cube)")
    p.add_argument("--overwrite", action="store_true", help="redo images that already have a GLB")
    # fine-tuning; each overrides the preset
    p.add_argument("--res", default=None, choices=["512", "1024", "1024_cascade", "1536_cascade"])
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--faces", type=int, default=None)
    p.add_argument("--tex", type=int, default=None)
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

    if a.rembg:
        use_rembg_model(a.rembg)

    src = Path(a.inp)
    if not src.exists():
        raise SystemExit(f"'{src}' not found - put your images in a '{src}' folder next to this script")
    images = [src] if src.is_file() else sorted(f for f in src.iterdir() if f.suffix.lower() in EXTS)
    if not images:
        raise SystemExit(f"no .png/.jpg/.webp images in '{src}'")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"quality={a.quality}: res={a.res} steps={a.steps} faces={a.faces} tex={a.tex}"
          f" | {len(images)} image(s) -> {out_dir}")

    pipe = Trellis2ImageTo3DPipeline.from_pretrained(a.model)
    pipe.cuda()
    steps = {"steps": a.steps}  # merged with pipeline.json defaults

    for img_path in images:
        out = out_dir / f"{img_path.stem}.glb"
        if out.exists() and not a.overwrite:
            print(f"skip {out.name} (exists, use --overwrite to redo)")
            continue
        t0 = time.time()
        mesh = glb = None
        try:
            # RGBA input with a clean alpha skips the automatic background removal
            image = Image.open(img_path)
            mesh = pipe.run(image, seed=a.seed, pipeline_type=a.res,
                            max_num_tokens=a.max_tokens,
                            sparse_structure_sampler_params=steps,
                            shape_slat_sampler_params=steps,
                            tex_slat_sampler_params=steps)[0]
            mesh.simplify(16777216)  # nvdiffrast limit, as in example.py

            glb = o_voxel.postprocess.to_glb(
                vertices=mesh.vertices,
                faces=mesh.faces,
                attr_volume=mesh.attrs,
                coords=mesh.coords,
                attr_layout=mesh.layout,
                voxel_size=mesh.voxel_size,
                aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
                decimation_target=a.faces,
                texture_size=a.tex,
                remesh=True,
                remesh_band=1,
                remesh_project=0,
            )
            if a.size > 0:
                glb.apply_scale(a.size / float(max(glb.extents)))
            glb.export(str(out), extension_webp=False)  # PNG textures import everywhere
            print(f"ok   {out.name}  {time.time() - t0:.0f}s  {out.stat().st_size / 1e6:.1f} MB")
        except torch.cuda.OutOfMemoryError:
            print(f"OOM  {img_path.name}: try a lower --quality or add --max-tokens 32768")
        finally:
            del mesh, glb
            gc.collect()
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
