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

  * low-poly mode (preset "building"): few large triangles, detail baked into the texture

Usage (from the repo root, inside the trellis2 conda env):
  python trellis2_batch.py                       # photos/ -> glb/draft/
  python trellis2_batch.py --quality high        # photos/ -> glb/high/
  python trellis2_batch.py -q building --height 12          # low-poly, 12 m tall, base at origin
  python trellis2_batch.py -q building --export-only --faces 800   # redo with fewer triangles
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
# lowpoly = merge flat areas into big triangles and bake the detail into the texture
PRESETS = {
    "draft":    dict(res="512",          steps=12, faces=50_000,    tex=2048, lowpoly=False),  # ~1 min, quick look
    "game":     dict(res="1024_cascade", steps=12, faces=100_000,   tex=2048, lowpoly=False),  # good detail, light
    "high":     dict(res="1024_cascade", steps=25, faces=500_000,   tex=4096, lowpoly=False),  # close-up, use Nanite
    "ultra":    dict(res="1536_cascade", steps=25, faces=1_000_000, tex=4096, lowpoly=False),  # likely > 12 GB VRAM
    # low-poly buildings: flat areas become few large triangles, detail goes into the texture.
    # --faces is the intermediate mesh there; the final count follows the shape.
    "building":     dict(res="1024_cascade", steps=12, faces=80_000, tex=4096, lowpoly=True,  # ~10-30k tris,
                         proxy_res=512, close=2, plane_tol=0.002, min_plane=2e-5),             # recesses kept
    "building-low": dict(res="1024_cascade", steps=12, faces=3_000,  tex=2048, lowpoly=True,  # a few hundred
                         proxy_res=128, close=2, plane_tol=0.008, min_plane=0.003),           # tris, far away
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
        for res in resolutions:
            t0 = time.time()
            where = None
            try:
                # RGBA input with a clean alpha skips the automatic background removal
                v, f, attrs, coords, layout, voxel_size, raw = generate_one(pipe, Image.open(img_path), res, a)
            except Exception as e:
                if not is_oom(e):
                    raise
                where = where_failed(e)
                del e
            if where is None:
                torch.save(dict(vertices=v, faces=f, attrs=attrs, coords=coords, layout=layout,
                                voxel_size=voxel_size, res=res), mesh_file)
                print(f"gen  {img_path.name}  {time.time() - t0:.0f}s  res={res}"
                      f"  ({raw:,} raw faces -> {f.shape[0]:,} saved)")
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


# ----------------------------------------------------------------------------- low-poly export

def place(vertices, a):
    """Y-up vertices -> base centred on the origin, then scaled by --height or --size (metres)."""
    lo, hi = vertices.min(axis=0), vertices.max(axis=0)
    vertices = vertices - [(lo[0] + hi[0]) / 2, lo[1], (lo[2] + hi[2]) / 2]
    ext = hi - lo
    if a.height > 0:
        vertices = vertices * (a.height / ext[1])
    elif a.size > 0:
        vertices = vertices * (a.size / ext.max())
    return vertices


def export_lowpoly(mesh_file, out, a):
    """Few large triangles; colour/roughness/metal baked from the generated model.

    1. lowpoly_building.building_lowpoly (CPU): closed solid proxy -> ~--faces triangles ->
       planes (walls, roof faces) -> each plane re-triangulated from its outline.
    2. UV unwrap with charts split at --hard-angle, so walls get hard edges.
    3. Bake: for every texel, cast a ray along the low-poly normal onto the detailed
       surface (closest point as fallback) and sample TRELLIS' PBR voxel volume there.
    """
    import math
    import cv2
    import numpy as np
    import trimesh
    import trimesh.visual
    from PIL import Image
    import cumesh
    import nvdiffrast.torch as dr
    from flex_gemm.ops.grid_sample import grid_sample_3d
    from lowpoly_building import building_lowpoly

    d = torch.load(mesh_file, weights_only=False)
    attrs = d["attrs"].cuda()
    coords = d["coords"].cuda()
    layout = d["layout"]
    vs = torch.as_tensor(d["voxel_size"], dtype=torch.float32, device="cuda").reshape(-1)
    vs = vs.repeat(3) if vs.numel() == 1 else vs
    dense_v, dense_f = d["vertices"].float().numpy(), d["faces"].long().numpy()

    # 1a. low-poly building on the CPU
    t0 = time.time()
    lv, lf, n_planes = building_lowpoly(dense_v, dense_f, proxy_res=a.proxy_res, close=a.close,
                                        mid_faces=a.faces, angle=a.plane_angle, tol=a.plane_tol,
                                        min_plane=a.min_plane)
    lo_v = torch.from_numpy(np.ascontiguousarray(lv, dtype=np.float32)).cuda()
    lo_f = torch.from_numpy(np.ascontiguousarray(lf, dtype=np.int32)).cuda()
    # make the normals point outwards (positive signed volume)
    tri = lo_v[lo_f.long()]
    if torch.einsum("ij,ij->i", tri[:, 0], torch.cross(tri[:, 1], tri[:, 2], dim=-1)).sum() < 0:
        lo_f = lo_f[:, [0, 2, 1]].contiguous()

    # 1b. detailed reference surface for the bake (GPU)
    m = cumesh.CuMesh()
    m.init(d["vertices"].cuda().float().contiguous(), d["faces"].cuda().int().contiguous())
    del d
    free_gpu()
    m.fill_holes(max_hole_perimeter=3e-2)
    m.simplify(1_000_000)
    hi_v, hi_f = m.read()
    print(f"     low-poly: {lo_f.shape[0]:,} triangles, {n_planes} planes ({time.time() - t0:.0f}s; "
          f"bake reference {hi_f.shape[0]:,})")

    # 2. UVs; charts split where the surface bends more than --hard-angle
    u = cumesh.CuMesh()
    u.init(lo_v, lo_f)
    out_v, out_f, out_uv, _ = u.uv_unwrap(
        compute_charts_kwargs={"threshold_cone_half_angle_rad": math.radians(a.hard_angle),
                               "refine_iterations": 0, "global_iterations": 1, "smooth_strength": 1},
        return_vmaps=True)
    del u, m
    out_v = out_v.cuda().float()
    out_f = out_f.cuda().int().contiguous()
    out_uv = out_uv.cuda().float()
    # vertices are split at chart seams, so these normals are smooth inside a wall, hard at corners
    normals_np = np.asarray(trimesh.Trimesh(out_v.cpu().numpy(), out_f.cpu().numpy(), process=False)
                            .vertex_normals, dtype=np.float32)
    nrm = torch.from_numpy(normals_np).cuda()

    # 3. bake
    T = a.tex
    ctx = dr.RasterizeCudaContext()
    uv_clip = torch.cat([out_uv * 2 - 1, torch.zeros_like(out_uv[:, :1]), torch.ones_like(out_uv[:, :1])],
                        dim=-1).unsqueeze(0).contiguous()
    rast, _ = dr.rasterize(ctx, uv_clip, out_f, resolution=[T, T])
    mask = rast[0, ..., 3] > 0
    pos = dr.interpolate(out_v.unsqueeze(0).contiguous(), rast, out_f)[0][0][mask]
    n = dr.interpolate(nrm.unsqueeze(0).contiguous(), rast, out_f)[0][0][mask]
    n = torch.nn.functional.normalize(n, dim=-1)

    bvh = cumesh.cuBVH(hi_v, hi_f)
    cage = a.cage
    hit, fid, depth = bvh.ray_trace(pos + n * cage, -n)
    ok = (fid >= 0) & (depth >= 0) & (depth <= 2 * cage)
    sample = torch.where(ok[:, None], hit, pos)
    if (~ok).any():
        _, fid2, uvw = bvh.unsigned_distance(pos[~ok], return_uvw=True)
        tri_hi = hi_v[hi_f[fid2.long()].long()]
        sample[~ok] = (tri_hi * uvw.unsqueeze(-1)).sum(dim=1)
    print(f"     bake: {mask.sum().item():,} texels, {ok.float().mean().item() * 100:.0f}% by ray")

    grid_size = (1.0 / vs).round().int()
    tex_attrs = torch.zeros(T, T, attrs.shape[1], device="cuda")
    tex_attrs[mask] = grid_sample_3d(
        attrs,
        torch.cat([torch.zeros_like(coords[:, :1]), coords], dim=-1),
        shape=torch.Size([1, attrs.shape[1], *grid_size.tolist()]),
        grid=((sample + 0.5) / vs).reshape(1, -1, 3),
        mode="trilinear",
    )

    mask_np = mask.cpu().numpy()
    def channel(name):
        return np.clip(tex_attrs[..., layout[name]].cpu().numpy() * 255, 0, 255).astype(np.uint8)
    base_color, metallic, roughness, alpha = (channel(k) for k in ("base_color", "metallic", "roughness", "alpha"))
    holes = (~mask_np).astype(np.uint8)  # dilate into the gaps so UV seams do not show
    base_color = cv2.inpaint(base_color, holes, 3, cv2.INPAINT_TELEA)
    metallic = cv2.inpaint(metallic, holes, 1, cv2.INPAINT_TELEA)[..., None]
    roughness = cv2.inpaint(roughness, holes, 1, cv2.INPAINT_TELEA)[..., None]
    alpha = cv2.inpaint(alpha, holes, 1, cv2.INPAINT_TELEA)[..., None]
    material = trimesh.visual.material.PBRMaterial(
        baseColorTexture=Image.fromarray(np.concatenate([base_color, alpha], axis=-1)),
        baseColorFactor=np.array([255, 255, 255, 255], dtype=np.uint8),
        metallicRoughnessTexture=Image.fromarray(np.concatenate([np.zeros_like(metallic), roughness, metallic], axis=-1)),
        metallicFactor=1.0,
        roughnessFactor=1.0,
        alphaMode="OPAQUE",
        doubleSided=True,
    )

    # TRELLIS frame (Z up) -> glTF (Y up), as in o_voxel.postprocess.to_glb
    v_np = out_v.cpu().numpy()
    uv_np = out_uv.cpu().numpy()
    v_np[:, 1], v_np[:, 2] = v_np[:, 2], -v_np[:, 1].copy()
    normals_np[:, 1], normals_np[:, 2] = normals_np[:, 2], -normals_np[:, 1].copy()
    uv_np[:, 1] = 1 - uv_np[:, 1]
    v_np = place(v_np, a)
    trimesh.Trimesh(vertices=v_np, faces=out_f.cpu().numpy(), vertex_normals=normals_np, process=False,
                    visual=trimesh.visual.TextureVisuals(uv=uv_np, material=material)
                    ).export(str(out), extension_webp=False)
    del bvh, hi_v, hi_f, lo_v, lo_f, attrs, coords, rast, pos, n, sample, tex_attrs
    free_gpu()
    return True


# ----------------------------------------------------------------------------- main

def main():
    p = argparse.ArgumentParser(description="TRELLIS.2 image -> GLB")
    p.add_argument("--quality", "-q", default="draft", choices=list(PRESETS),
                   help="preset: draft, game, high, ultra")
    p.add_argument("--in", dest="inp", default="photos", help="image file or folder (default: photos)")
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
    p.add_argument("--lowpoly", action="store_const", const=True, default=None,
                   help="low-poly export: --faces is the triangle budget, detail goes into the texture")
    p.add_argument("--hard-angle", type=float, default=30.0,
                   help="low-poly: surfaces bending more than this (degrees) get a hard edge")
    p.add_argument("--cage", type=float, default=0.02,
                   help="low-poly: bake search distance as a fraction of the object size")
    p.add_argument("--min-part", type=float, default=1e-4, help=argparse.SUPPRESS)  # no longer used
    p.add_argument("--plane-tol", type=float, default=None,
                   help="low-poly: bumps smaller than this (fraction of the object size) are flattened into the plane; "
                        "bigger = fewer triangles, smaller = more detail kept")
    p.add_argument("--plane-angle", type=float, default=20.0,
                   help="low-poly: surfaces within this angle (degrees) can be one plane")
    p.add_argument("--min-plane", type=float, default=None,
                   help="low-poly: planes smaller than this share of the surface join a similar neighbour")
    p.add_argument("--proxy-res", type=int, default=None,
                   help="low-poly: voxel grid used to make the building one closed solid")
    p.add_argument("--close", type=int, default=None,
                   help="low-poly: openings up to about 2x this many voxels (windows, doors) are sealed")
    p.add_argument("--max-tokens", type=int, default=49152,
                   help="1536 only: token budget before it drops towards 1024 (no effect at 1024)")
    p.add_argument("--no-fallback", action="store_true",
                   help="do not retry at 512 when the GPU runs out of memory at 1024/1536")
    p.add_argument("--seed", type=int, default=42, help="change it to get a different variation")
    p.add_argument("--model", default="microsoft/TRELLIS.2-4B")
    p.add_argument("--rembg", default="ZhengPeng7/BiRefNet",
                   help="background-removal model; 'briaai/RMBG-2.0' once you have access")
    a = p.parse_args()

    for k, v in PRESETS[a.quality].items():
        if getattr(a, k) is None:
            setattr(a, k, v)
    for k, v in PRESETS["building"].items():  # --lowpoly with a non-building preset
        if getattr(a, k, None) is None:
            setattr(a, k, v)
    out_dir = Path(a.out or f"glb/{a.quality}")
    mesh_dir = out_dir / "_meshes"
    mesh_dir.mkdir(parents=True, exist_ok=True)
    global ERROR_LOG
    ERROR_LOG = out_dir / "errors.log"

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
              f"{' low-poly' if a.lowpoly else ''}"
              f" | {len(pending)} image(s) to generate -> {out_dir}")
        if pending:
            generate_all(pending, mesh_dir, a)
        todo = [mesh_dir / f"{img.stem}.pt" for img in images
                if (mesh_dir / f"{img.stem}.pt").exists()
                and (a.overwrite or not (out_dir / f"{img.stem}.glb").exists())]

    for mesh_file in todo:
        out = out_dir / f"{mesh_file.stem}.glb"
        t0 = time.time()
        print(f"export {mesh_file.stem}" + (" (low-poly)" if a.lowpoly else ""))
        if (export_lowpoly if a.lowpoly else export_one)(mesh_file, out, a):
            print(f"ok   {out.name}  {time.time() - t0:.0f}s  {out.stat().st_size / 1e6:.1f} MB")
            if not a.keep_mesh:
                mesh_file.unlink()
        else:
            print(f"FAIL {mesh_file.stem}: mesh kept in {mesh_dir}; close GPU-heavy apps and run again "
                  f"with --export-only (add e.g. --faces 200000 --tex 2048 for a lighter export)")


if __name__ == "__main__":
    main()
