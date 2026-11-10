r"""
trellis_lod2.py - a detailed building from ONE photo with TRELLIS.2, the LoD2 model as a guide for its rough shape.

TRELLIS.2 works in stages:
  1. rough shape: which cells of a 64 x 64 x 64 grid the building fills (~1.7 m cells on the Dom)
  2. detail:      the real surface inside those cells (windows, buttresses, curves) - from the photo
  3. texture:     colour, roughness, metal - from the photo
This script changes only stage 1: instead of starting from pure noise it starts from the LoD2 model with some noise
added, so the result keeps the LoD2's size and layout but may round it off, add curves and small changes (--follow).
With --follow 0 the shape is plain TRELLIS.2 from the photo and the LoD2 only gives the real size and place.
Stages 2 and 3 are TRELLIS.2's own, untouched. The result is put back into the LoD2 model's frame (metres, Y up,
ground at 0, front +Z), so <lod2>.json still says where it stands in the world.

WSL, trellis2 env, in the TRELLIS.2 folder (next to trellis2_batch.py, whose helpers it uses):

    H=/mnt/c/Users/mbalooch/Desktop/Myfiles/Hunyuan3D/Hunyuan3D-2.1
    python trellis_lod2.py --check
    python trellis_lod2.py $H/out/buildings/dom_lod2.glb $H/in/dom_corner/dom_az045.png --follow 0
        -> glb/lod2/dom_trellis_f000_az045_s42.glb (+ .json: where it stands in the world)
    LoD2 as a shape guide: --preview --follow 0.7 0.5 (stage 1 only, seconds each, *_stage1.png) to look first,
    then the same without --preview.

--follow F (several allowed, default 0.7 0.6 0.5): how closely the rough shape follows the LoD2 model
(= the share of TRELLIS' rough-shape steps taken from the LoD2 instead of generated)
    1.0   exact: the LoD2 cells themselves (grown by --margin cells, so relief can stick out); nothing can be added
    0.8   practically the LoD2
    0.7   close: roof lines, small parts change
    0.6   medium: layout and footprint kept, roofs / towers / corners may change
    0.5   loose: footprint mostly kept, the upper parts follow the photo (spires, roofs)
    0.3   only a hint
    0.0   no LoD2 shape: plain TRELLIS.2 from the photo, only placed onto the LoD2 afterwards (the usual choice)
    The stage-1 pictures (*_stage1.png, also written for full runs) show it from above, front and side:
    grey = LoD2 and result agree, red = LoD2 cell left out, blue = cell added.

Turn (--yaw): TRELLIS.2 has no camera - it decides itself how the building stands in its grid. So the script first
lets TRELLIS.2 build the rough shape from the photo alone (first seed) and turns the LoD2 model until it matches that
(written as *_fit.png: grey = both, red = LoD2 only, blue = TRELLIS only). If the fit picture looks wrong (front and
back swapped ...), give the turn yourself, e.g. --yaw 180 (degrees about up, as printed in the log).

Other options:
    --seeds 42 7       one result per seed (different versions)
    -q high            preset from trellis2_batch.py: draft (512) / game / high (default) / ultra
    --headroom 0.2     room above the LoD2 roof in the grid (share of its height) so a taller spire fits
    --fit icp          after generating: small fit (scale, turn about up, shift) onto the LoD2, worst 20 % ignored
    --overwrite        redo results that exist (else skipped: a stopped job goes on where it was)
    --export-only      only export the meshes kept in <out>/_meshes (after a failed export)
    --restarts 3       after a CUDA driver error / GPU reset ("device not ready", "illegal memory access" ...)
                       the process can't use the GPU any more, so the script starts itself again in a fresh
                       process (up to 3 times) and goes on with the jobs not done yet; a job that breaks the
                       GPU twice is skipped. 0 = stop at the first such error.

GPU memory: the script uses PyTorch's plain allocator instead of "expandable_segments" (which TRELLIS.2 and
trellis2_batch.py turn on): under WSL2 its cuMemMap calls can fail with "CUDA driver error: device not ready".
To get it back anyway: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python trellis_lod2.py ...
Then as for every model: blender_lowpoly.py on <result>.glb for the light version.

Needs the stage-1 ENCODER of TRELLIS (microsoft/TRELLIS-image-large, MIT, 119 MB, downloaded on the first run);
TRELLIS.2's own stage-1 decoder already comes from that repo.
"""
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:512,garbage_collection_threshold:0.8")

import argparse
import json
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

HERE = Path(__file__).resolve().parent
EXTS = {".png", ".jpg", ".jpeg", ".webp"}
ENCODER = "microsoft/TRELLIS-image-large/ckpts/ss_enc_conv3d_16l8_fp16"
GRID = 64              # stage-1 grid (decoder output)
FILL = 0.95            # longest side of the building in TRELLIS' unit cube (training: 0.99999; a little room
                       # at the sides so the generator can add relief there too)


RESTART_CODE = 75     # child -> supervisor: "this process' CUDA context is broken, start a new one"
MAX_GPU_TRIES = 2     # a job that breaks the GPU this often in one run is skipped
CURRENT = {"stem": None}


def log(*a):
    print(*a, flush=True)


def is_cuda_fatal(e):
    """a CUDA error that is not "out of memory": the process' CUDA context is broken, only a new process helps"""
    s = str(e)
    return isinstance(e, RuntimeError) and "out of memory" not in s.lower() and any(
        k in s for k in ("CUDA driver error", "CUDA error", "CUBLAS_STATUS", "cuDNN error", "CUDNN_STATUS"))


def gpu_fails():
    f = os.environ.get("TRELLIS_LOD2_FAILS")
    try:
        return json.loads(Path(f).read_text()) if f else {}
    except (OSError, ValueError):
        return {}


def gpu_fail_add(stem):
    f = os.environ.get("TRELLIS_LOD2_FAILS")
    if f and stem:
        d = gpu_fails()
        d[stem] = d.get(stem, 0) + 1
        Path(f).write_text(json.dumps(d))


# ----------------------------------------------------------------------------- photo

def angle_tag(stem):
    m = re.search(r"az(?:im)?_?(-?\d+(?:\.\d+)?)", stem, re.I)
    return f"az{int(round(float(m.group(1)))) % 360:03d}" if m else re.sub(r"[^A-Za-z0-9]+", "_", stem)[:24]


def white_key(img):
    """building cut out of a PLAIN WHITE background; None if the border is not white"""
    from PIL import Image
    from scipy.ndimage import binary_dilation, binary_propagation, label
    a = np.asarray(img.convert("RGB")).astype(np.int16)
    white = (a.min(-1) > 232) & ((a.max(-1) - a.min(-1)) < 20)
    edge = np.r_[white[0], white[-1], white[:, 0], white[:, -1]]
    if edge.mean() < 0.9:
        return None
    lab, _ = label(white)
    border = np.unique(np.r_[lab[0], lab[-1], lab[:, 0], lab[:, -1]])
    bg = np.isin(lab, border[border > 0])
    ys = np.nonzero(~bg)[0]
    if len(ys):                                             # light bluish ground shadow at the foot: background
        r, g, b = a[..., 0], a[..., 1], a[..., 2]
        cool = (b > g + 8) & (((a.mean(-1) > 185) & (b - r > 4)) | ((a.mean(-1) > 145) & (b - r > 25)))
        cool[: ys.min() + int(0.6 * (ys.max() - ys.min()))] = False
        bg = binary_propagation(bg, mask=bg | cool)
    fg = ~bg
    parts, n = label(fg)
    if n > 1:                                               # drop loose specks (< 0.2 % of the building)
        size = np.bincount(parts.ravel())
        size[0] = 0
        fg = np.isin(parts, np.flatnonzero(size >= 0.002 * size.sum()))
    rim = fg & binary_dilation(~fg)
    alpha = np.where(fg, 255, 0).astype(np.uint8)
    alpha[rim] = 128
    return Image.fromarray(np.dstack([a.astype(np.uint8), alpha]), "RGBA")


def cut_photo(path):
    """RGBA with a real alpha -> as it is; plain white background -> cut by colour; else TRELLIS' own BiRefNet"""
    from PIL import Image
    im = Image.open(path)
    if im.mode in ("RGBA", "LA") and np.asarray(im.convert("RGBA"))[..., 3].min() < 250:
        return im.convert("RGBA"), "alpha"
    k = white_key(im)
    if k is not None:
        return k, "white background"
    return im.convert("RGB"), "BiRefNet"


# ----------------------------------------------------------------------------- geometry (numpy only)

def load_mesh(path):
    import trimesh
    m = trimesh.load(str(path), force="mesh", process=False)
    m = trimesh.Trimesh(np.asarray(m.vertices, np.float64), np.asarray(m.faces, np.int64), process=True)
    return np.asarray(m.vertices), np.asarray(m.faces)


def rot_y(deg):
    """turn about up (Y); +90 turns the +Z side to +X"""
    t = np.radians(deg)
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def placement(V, yaw, headroom):
    """LoD2 (metres) -> TRELLIS' unit cube, glTF axes: Vc = k * (R @ v - c). Centred like TRELLIS' training data
    (bbox centre, longest side = FILL), but with `headroom` x height of free room above the roof."""
    R = rot_y(yaw)
    W = V @ R.T
    lo, hi = W.min(0), W.max(0)
    h = hi[1] - lo[1]
    H = h * (1 + headroom)
    ext = np.array([hi[0] - lo[0], H, hi[2] - lo[2]])
    k = FILL / ext.max()
    c = np.array([(lo[0] + hi[0]) / 2, lo[1] + H / 2, (lo[2] + hi[2]) / 2])
    return dict(yaw=float(yaw), R=R, k=float(k), c=c, headroom=float(headroom))


def to_cube(P, pl):
    return pl["k"] * (P @ pl["R"].T - pl["c"])


def back_matrix(pl):
    """4x4: unit cube (glTF axes, as TRELLIS exports) -> LoD2 frame (metres)"""
    M = np.eye(4)
    M[:3, :3] = pl["R"].T / pl["k"]
    M[:3, 3] = pl["R"].T @ pl["c"]
    return M


def gl_to_trellis(P):
    """glTF axes (Y up) -> TRELLIS' grid axes (Z up); the inverse of what o_voxel's to_glb does"""
    return np.stack([P[:, 0], -P[:, 2], P[:, 1]], 1)


def trellis_to_gl(P):
    return np.stack([P[:, 0], P[:, 2], -P[:, 1]], 1)


def sample_surface(V, F, n, rng, skip_bottom=False):
    """n random points on the surface (area weighted); skip_bottom: not on downward faces (LoD2 has no floor)"""
    tri = V[F]
    cr = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    area = 0.5 * np.linalg.norm(cr, axis=1)
    if skip_bottom:
        area = np.where(cr[:, 1] / (2 * area + 1e-20) < -0.7, 0.0, area)
    fi = rng.choice(len(F), n, p=area / area.sum())
    r1, r2 = rng.random(n), rng.random(n)
    s = np.sqrt(r1)
    w = np.stack([1 - s, s * (1 - r2), s * r2], 1)
    return (tri[fi] * w[:, :, None]).sum(1)


def voxelize(Vc, F, res=GRID):
    """surface cells of the mesh (unit cube, glTF axes) on TRELLIS' res^3 grid, indexed [x, y, z] in TRELLIS axes -
    the same as TRELLIS.2's training data (cells the surface passes through)"""
    Pt = gl_to_trellis(Vc)
    tri = Pt[F]
    area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1).sum()
    n = int(np.clip(area / (0.25 / res) ** 2 * 3, 200_000, 4_000_000))      # ~12 points per cell face
    P = np.concatenate([sample_surface(Pt, F, n, np.random.default_rng(0)), Pt])
    idx = np.clip(np.floor((P + 0.5) * res).astype(np.int64), 0, res - 1)
    occ = np.zeros((res, res, res), bool)
    occ[idx[:, 0], idx[:, 1], idx[:, 2]] = True
    return occ


def cells_to_points(occ):
    """cell centres of an occupancy grid (TRELLIS axes) -> unit cube, glTF axes"""
    res = occ.shape[0]
    return trellis_to_gl((np.argwhere(occ) + 0.5) / res - 0.5)


def follow_stats(occ_lod2, occ):
    """share of LoD2 cells the result kept (within 1 cell) and share of result cells that are new (> 1 cell away)"""
    from scipy.ndimage import binary_dilation
    st = np.ones((3, 3, 3), bool)
    kept = (occ_lod2 & binary_dilation(occ, st)).sum() / max(occ_lod2.sum(), 1)
    new = (occ & ~binary_dilation(occ_lod2, st)).sum() / max(occ.sum(), 1)
    return float(kept), float(new)


def _score(src, tree_f, f_pts, keep=0.8, iters=8):
    """align src to the free sample with scale + shift only (no turn), return trimmed symmetric distance"""
    from scipy.spatial import cKDTree
    cur = src
    for _ in range(iters):
        d, j = tree_f.query(cur)
        ok = d <= np.quantile(d, keep)
        a, b = cur[ok], f_pts[j[ok]]
        ma, mb = a.mean(0), b.mean(0)
        sc = np.sqrt(((b - mb) ** 2).sum() / max(((a - ma) ** 2).sum(), 1e-12))
        sc = float(np.clip(sc, 0.7, 1.4))
        cur = (cur - ma) * sc + mb
    d1 = tree_f.query(cur)[0]
    d2 = cKDTree(cur).query(f_pts)[0]
    t1 = np.sort(d1)[: int(keep * len(d1))].mean()
    t2 = np.sort(d2)[: int(keep * len(d2))].mean()
    return float(t1 + t2)


def find_yaw(V, F, free_occ, headroom):
    """turn of the LoD2 model about up that best matches TRELLIS' own rough shape (free_occ). Returns
    (best yaw, score, list of (yaw, score) of the best distinct candidates)"""
    from scipy.spatial import cKDTree
    f_pts = cells_to_points(free_occ)
    tree_f = cKDTree(f_pts)
    S = sample_surface(V, F, 6000, np.random.default_rng(3))

    def score(y):
        return _score(to_cube(S, placement(V, y, headroom)), tree_f, f_pts)

    coarse = {float(y): score(y) for y in range(0, 360, 5)}
    best = min(coarse, key=coarse.get)
    fine = {float(y % 360): score(y) for y in np.arange(best - 4, best + 5, 1.0)}
    best = min(fine, key=fine.get)
    allc = {**coarse, **fine}
    ranked, out = sorted(allc.items(), key=lambda t: t[1]), []
    for y, s in ranked:                                   # distinct candidates (> 20 deg apart)
        if all(min(abs(y - y2), 360 - abs(y - y2)) > 20 for y2, _ in out):
            out.append((y, s))
        if len(out) == 3:
            break
    return best, fine[best], out


def fit_to(src, tgt_tree, tgt, iters=40, keep=0.8):
    """similarity src -> LoD2 surface: uniform scale, turn about Y (up), shift; the worst (1-keep) of the point
    pairs are ignored each round (parts TRELLIS added, e.g. a longer spire). Returns 4x4, mean dist before/after."""
    M = np.eye(4)
    cur = src.copy()
    d0 = tgt_tree.query(src)[0]
    before = float(np.mean(np.sort(d0)[: int(keep * len(d0))]))
    for _ in range(iters):
        d, j = tgt_tree.query(cur)
        ok = d <= np.quantile(d, keep)
        a, b = cur[ok], tgt[j[ok]]
        ma, mb = a.mean(0), b.mean(0)
        xa, xb = a - ma, b - mb
        th = np.arctan2((xa[:, 2] * xb[:, 0] - xa[:, 0] * xb[:, 2]).sum(), (xa[:, 0] * xb[:, 0] + xa[:, 2] * xb[:, 2]).sum())
        c, s = np.cos(th), np.sin(th)
        R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
        sc = (np.einsum("ij,ij->", xa @ R.T, xb)) / (xa ** 2).sum()
        T = np.eye(4)
        T[:3, :3] = sc * R
        T[:3, 3] = mb - sc * R @ ma
        cur = cur @ T[:3, :3].T + T[:3, 3]
        M = T @ M
        if abs(th) < 1e-5 and abs(sc - 1) < 1e-5 and np.linalg.norm(T[:3, 3]) < 1e-4:
            break
    d1 = tgt_tree.query(cur)[0]
    after = float(np.mean(np.sort(d1)[: int(keep * len(d1))]))
    return M, before, after


def picture(path, occ_lod2, occ, title):
    """top / front / side views: grey = both, red = LoD2 only, blue = result only"""
    from PIL import Image, ImageDraw
    res, px = occ.shape[0], 4
    views = []
    # grid axes (TRELLIS): x right, y towards the back (= glTF -z), z up. Every view: rows = the 2nd remaining
    # axis flipped (up / back at the top), columns = the 1st (x, or y = back to the right for the side view)
    for name, ax in (("from above (front at the bottom)", 2), ("front (glTF +z)", 1), ("side (from +x)", 0)):
        a, b = occ_lod2.any(ax), occ.any(ax)
        img = np.full((res, res, 3), 255, np.uint8)
        img[a & b] = (120, 120, 120)
        img[a & ~b] = (220, 40, 40)
        img[~a & b] = (40, 90, 230)
        img = img.transpose(1, 0, 2)[::-1]
        views.append((name, Image.fromarray(np.ascontiguousarray(img)).resize((res * px, res * px), Image.NEAREST)))
    W = res * px
    canvas = Image.new("RGB", (3 * W + 40, W + 46), (255, 255, 255))
    d = ImageDraw.Draw(canvas)
    d.text((10, 4), title, fill=(0, 0, 0))
    for i, (name, im) in enumerate(views):
        x = 10 + i * (W + 10)
        canvas.paste(im, (x, 36))
        d.rectangle([x - 1, 35, x + W, 36 + W], outline=(200, 200, 200))
        d.text((x, 22), name, fill=(80, 80, 80))
    canvas.save(path)


# ----------------------------------------------------------------------------- TRELLIS.2 stage 1

def noise_level(follow, r=5.0):
    """--follow -> where in TRELLIS' rough-shape schedule we start: follow = the share of its (evenly spaced) steps
    that are skipped and taken from the LoD2 instead. Its schedule is warped (rescale_t = 5: most steps at high
    noise), so this spreads the useful range: 0.8 -> noise 0.56 (LoD2 kept), 0.7 -> 0.68 (small changes),
    0.6 -> 0.77, 0.5 -> 0.83 (upper parts free), 0.3 -> 0.92. Returns (s0, t0)."""
    s0 = 1.0 - float(follow)
    return s0, r * s0 / (1 + (r - 1) * s0)


def stage1(pipe, cond, seed, steps, z0=None, follow=0.0):
    """rough-shape latent. follow 0: TRELLIS' own (pure noise, same noise as its run() with this seed);
    0 < follow < 1: start from the LoD2 latent z0, noised to the level noise_level(follow), and run the rest of
    TRELLIS' schedule from there (with the photo as condition)"""
    import torch
    flow = pipe.models["sparse_structure_flow_model"]
    sampler = pipe.sparse_structure_sampler
    prm = {**pipe.sparse_structure_sampler_params, "steps": steps}
    r = float(prm.get("rescale_t", 1.0))
    torch.manual_seed(seed)
    noise = torch.randn(1, flow.in_channels, flow.resolution, flow.resolution, flow.resolution).to(pipe.device)
    if z0 is None or follow <= 0:
        x, s0 = noise, 1.0
    else:
        s0, t0 = noise_level(follow, r)
        sig = sampler.sigma_min + (1 - sampler.sigma_min) * t0
        x = (1 - t0) * z0.to(noise.dtype) + sig * noise
    s = np.linspace(s0, 0, steps + 1)
    t = (r * s / (1 + (r - 1) * s)).tolist()
    if pipe.low_vram:
        flow.to(pipe.device)
    with torch.no_grad():
        for t_cur, t_prev in zip(t[:-1], t[1:]):
            x = sampler.sample_once(flow, x, t_cur, t_prev, cond["cond"], neg_cond=cond["neg_cond"],
                                    guidance_strength=prm["guidance_strength"],
                                    guidance_interval=prm["guidance_interval"],
                                    guidance_rescale=prm.get("guidance_rescale", 0.0)).pred_x_prev
    if pipe.low_vram:
        flow.cpu()
    return x


def decode_ss(pipe, z):
    import torch
    dec = pipe.models["sparse_structure_decoder"]
    if pipe.low_vram:
        dec.to(pipe.device)
    with torch.no_grad():
        occ = (dec(z) > 0)[0, 0].cpu().numpy()
    if pipe.low_vram:
        dec.cpu()
    return occ


def encode_ss(enc, occ, device):
    import torch
    enc.to(device)
    with torch.no_grad():
        z = enc(torch.from_numpy(occ).float()[None, None].to(device), sample_posterior=False)
    enc.cpu()
    return z


def coords_for(occ, res_name):
    """occupancy (64^3) -> the coords TRELLIS' stage 2 takes (32^3 for 512/cascade, 64^3 for 1024)"""
    import torch
    ss_res = {"512": 32, "1024": 64, "1024_cascade": 32, "1536_cascade": 32}[res_name]
    o = torch.from_numpy(occ)[None, None]
    if ss_res != o.shape[2]:
        ratio = o.shape[2] // ss_res
        o = torch.nn.functional.max_pool3d(o.float(), ratio, ratio, 0) > 0.5
    return torch.argwhere(o)[:, [0, 2, 3, 4]].int()


def stages23(pipe, coords, cond512, cond1024, res, a):
    """TRELLIS.2's own detail + texture stages on the given rough shape (as in Trellis2ImageTo3DPipeline.run)"""
    import torch
    sp = {"steps": a.steps}
    coords = coords.to(pipe.device)
    M = pipe.models
    if res == "512":
        shape = pipe.sample_shape_slat(cond512, M["shape_slat_flow_model_512"], coords, sp)
        tex = pipe.sample_tex_slat(cond512, M["tex_slat_flow_model_512"], shape, sp)
        r = 512
    elif res == "1024":
        shape = pipe.sample_shape_slat(cond1024, M["shape_slat_flow_model_1024"], coords, sp)
        tex = pipe.sample_tex_slat(cond1024, M["tex_slat_flow_model_1024"], shape, sp)
        r = 1024
    else:
        shape, r = pipe.sample_shape_slat_cascade(cond512, cond1024, M["shape_slat_flow_model_512"],
                                                  M["shape_slat_flow_model_1024"], 512,
                                                  1536 if res == "1536_cascade" else 1024,
                                                  coords, sp, a.max_tokens)
        tex = pipe.sample_tex_slat(cond1024, M["tex_slat_flow_model_1024"], shape, sp)
    torch.cuda.empty_cache()
    return pipe.decode_latent(shape, tex, r)[0]


def save_mesh(tb, mesh, path, res, extra):
    """like trellis2_batch.generate_one: reduce the raw mesh to SAVE_FACES (if it fits) and save it on the CPU"""
    import torch
    raw = mesh.faces.shape[0]
    v, f = mesh.vertices, mesh.faces
    if raw > tb.SAVE_FACES:
        tb.free_gpu()
        try:
            v, f = tb.simplify(v, f, tb.SAVE_FACES)
        except Exception as e:
            if not tb.is_oom(e):
                raise
            tb.clear_cuda_error()
            log("     not enough GPU memory to simplify now, saving the full mesh")
            v, f = mesh.vertices, mesh.faces
    torch.save(dict(vertices=v.cpu(), faces=f.cpu(), attrs=mesh.attrs.cpu(), coords=mesh.coords.cpu(),
                    layout=mesh.layout, voxel_size=mesh.voxel_size, res=res, **extra), path)
    return raw, f.shape[0]


# ----------------------------------------------------------------------------- export (phase 2)

def export_placed(tb, pt, out, a, tgt, tree, lod2_size, meta_base):
    import torch
    import o_voxel
    d = torch.load(pt, weights_only=False)
    pl = d["place"]
    pl = dict(pl, R=np.asarray(pl["R"]), c=np.asarray(pl["c"]))
    info = d.get("info", {})
    m = SimpleNamespace(vertices=d["vertices"].cuda(), faces=d["faces"].cuda(), attrs=d["attrs"].cuda(),
                        coords=d["coords"].cuda(), layout=d["layout"], voxel_size=d["voxel_size"])
    res = d.get("res", a.res)
    del d
    glb = None
    for i, att in enumerate(tb.export_attempts(a, res), 1):
        try:
            tb.free_gpu()
            if m.faces.shape[0] > att["pre"]:
                m.vertices, m.faces = tb.simplify(m.vertices, m.faces, att["pre"])
                tb.free_gpu()
            target = min(a.faces, m.faces.shape[0])
            log(f"     export attempt {i}: {m.faces.shape[0]:,} -> {target:,} faces, remesh={att['remesh']}, "
                f"{tb.gpu_free_gb():.1f} GB GPU free")
            glb = o_voxel.postprocess.to_glb(
                vertices=m.vertices, faces=m.faces, attr_volume=m.attrs, coords=m.coords,
                attr_layout=m.layout, voxel_size=m.voxel_size, aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
                decimation_target=target, texture_size=a.tex,
                remesh=att["remesh"], remesh_band=1, remesh_project=0)
            break
        except Exception as e:
            if not tb.is_oom(e):
                raise
            tb.clear_cuda_error()
            log(f"     attempt {i} ran out of GPU memory")
    del m
    tb.free_gpu()
    if glb is None:
        return False

    glb.apply_transform(back_matrix(pl))                  # unit cube -> LoD2 frame (metres)
    fit = {"mode": a.fit}
    if a.fit == "icp":
        V, F = np.asarray(glb.vertices), np.asarray(glb.faces)
        src = sample_surface(V, F, 30_000, np.random.default_rng(1), skip_bottom=True)
        M, d0, d1 = fit_to(src, tree, tgt)
        sc = float(np.cbrt(np.linalg.det(M[:3, :3])))
        yaw = float(np.degrees(np.arctan2(M[0, 2], M[0, 0])))
        fit.update(scale=round(sc, 4), turn_deg=round(yaw, 2), shift_m=np.round(M[:3, 3], 2).tolist(),
                   mean_dist_m_before=round(d0, 3), mean_dist_m_after=round(d1, 3))
        if d1 < d0 and 0.8 < sc < 1.25 and abs(yaw) < 15:
            glb.apply_transform(M)
            log(f"     fit onto LoD2: scale {sc:.3f}, turn {yaw:+.1f} deg, shift {np.round(M[:3, 3], 2)} m; "
                f"mean distance {d0:.2f} -> {d1:.2f} m")
        else:
            fit["used"] = False
            log(f"     fit rejected (scale {sc:.3f}, turn {yaw:+.1f} deg, {d0:.2f} -> {d1:.2f} m): kept the placement")
    glb.apply_translation([0.0, -float(glb.vertices[:, 1].min()), 0.0])     # stands on the ground (y = 0)
    glb.export(str(out), extension_webp=False)
    sz = glb.extents
    meta = dict(meta_base, name=out.stem, **info, fit=fit, triangles=int(len(glb.faces)),
                size_m={"x": round(float(sz[0]), 2), "height": round(float(sz[1]), 2), "z": round(float(sz[2]), 2)},
                lod2_size_m={"x": round(float(lod2_size[0]), 2), "height": round(float(lod2_size[1]), 2),
                             "z": round(float(lod2_size[2]), 2)})
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    log(f"     {sz[0]:.1f} x {sz[1]:.1f} x {sz[2]:.1f} m (LoD2 {lod2_size[0]:.1f} x {lod2_size[1]:.1f} x "
        f"{lod2_size[2]:.1f})")
    del glb
    tb.free_gpu()
    return True


# ----------------------------------------------------------------------------- main

def check():
    import torch
    log(f"python {sys.version.split()[0]}, torch {torch.__version__}, cuda {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        log(f"GPU {p.name}, {p.total_memory / 2**30:.1f} GB, {torch.cuda.mem_get_info()[0] / 1e9:.1f} GB free")
    import trellis2_batch  # noqa: F401
    from trellis2.pipelines import Trellis2ImageTo3DPipeline  # noqa: F401
    import o_voxel  # noqa: F401
    for mod in ["trimesh", "scipy", "PIL", "cumesh"]:
        try:
            log(f"  {mod} {getattr(__import__(mod), '__version__', 'ok')}")
        except Exception as e:
            log(f"  {mod} MISSING ({e})")
    from trellis2 import models as tmodels
    enc = tmodels.from_pretrained(ENCODER)
    log(f"stage-1 encoder OK ({sum(p.numel() for p in enc.parameters()) / 1e6:.0f} M parameters)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("lod2", nargs="?", help="LoD2 GLB from lod2_to_glb.py (metres, Y up, front +Z)")
    ap.add_argument("photos", nargs="*", help="photo(s) of the building; white or transparent background is best")
    ap.add_argument("--follow", nargs="+", type=float, default=[0.7, 0.6, 0.5])
    ap.add_argument("--seeds", nargs="+", type=int, default=[42])
    ap.add_argument("-q", "--quality", default="high", choices=["draft", "game", "high", "ultra"])
    ap.add_argument("--yaw", default="auto", help="'auto' or the LoD2 turn about up in degrees")
    ap.add_argument("--headroom", type=float, default=0.2, help="free room above the roof (share of the height)")
    ap.add_argument("--margin", type=int, default=1, help="--follow 1: LoD2 cells grown by this many cells")
    ap.add_argument("--fit", choices=["icp", "none"], default="icp")
    ap.add_argument("--preview", action="store_true", help="stage 1 only: the *_stage1.png pictures, no models")
    ap.add_argument("--res", default=None, choices=["512", "1024", "1024_cascade", "1536_cascade"])
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--faces", type=int, default=None)
    ap.add_argument("--tex", type=int, default=None)
    ap.add_argument("--remesh", default="auto", choices=["auto", "on", "off"])
    ap.add_argument("--max-tokens", type=int, default=49152)
    ap.add_argument("--restarts", type=int, default=3,
                    help="fresh processes after a CUDA driver error / GPU reset (0 = off)")
    ap.add_argument("--no-fallback", action="store_true", help="no retry at 512 when the GPU runs out of memory")
    ap.add_argument("--name", help="name of the results (default: the LoD2 file's name without _lod2)")
    ap.add_argument("--out", default=str(HERE / "glb" / "lod2"))
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--export-only", action="store_true")
    ap.add_argument("--keep-mesh", action="store_true", help="keep <out>/_meshes/*.pt after the export")
    ap.add_argument("--model", default="microsoft/TRELLIS.2-4B")
    ap.add_argument("--rembg", default="ZhengPeng7/BiRefNet")
    ap.add_argument("--check", action="store_true", help="only check the setup")
    a = ap.parse_args()
    sys.path.insert(0, str(HERE))
    if a.check:
        return check()
    if not a.lod2 or (not a.photos and not a.export_only):
        ap.error("give the LoD2 GLB and a photo (or --check)")
    import trellis2_batch as tb
    for k, v in tb.PRESETS[a.quality].items():
        if getattr(a, k, None) is None:
            setattr(a, k, v)
    follows = [float(np.clip(f, 0, 1)) for f in a.follow]
    t0 = float(os.environ.get("TRELLIS_LOD2_T0", "inf"))      # start of this run (set by supervise())

    def done(f):         # result there - and with --overwrite: made in this run (by an earlier, restarted process)
        return f.exists() and (not a.overwrite or f.stat().st_mtime >= t0)

    t_all = time.time()
    lod2 = Path(a.lod2).resolve()
    V, F = load_mesh(lod2)
    size = V.max(0) - V.min(0)
    name = a.name or re.sub(r"_lod2$", "", lod2.stem)
    out = Path(a.out).resolve()
    mesh_dir = out / "_meshes"
    (out if a.preview else mesh_dir).mkdir(parents=True, exist_ok=True)
    tb.ERROR_LOG = out / "errors.log"
    meta_src = lod2.with_suffix(".json")
    place_w = json.loads(meta_src.read_text()) if meta_src.exists() else {}
    log(f"{lod2.name}: {len(F):,} triangles, {size[0]:.1f} x {size[1]:.1f} x {size[2]:.1f} m (x, height, z)")
    log(f"quality={a.quality}: res={a.res} steps={a.steps} faces={a.faces} tex={a.tex}")

    # ---- jobs
    jobs = []                                        # (photo, tag, follow, seed, stem)
    for p in [Path(s).resolve() for s in a.photos]:
        if not p.is_file():
            sys.exit(f"not found: {p}")
        tag = angle_tag(p.stem)
        for fo in follows:
            for seed in a.seeds:
                stem = f"{name}_trellis_f{int(round(fo * 100)):03d}_{tag}_s{seed}"
                if a.preview:
                    jobs.append((p, tag, fo, seed, stem))
                elif done(out / f"{stem}.glb"):
                    log(f"  {stem}.glb: already there, skipped (--overwrite to redo)")
                elif done(mesh_dir / f"{stem}.pt"):
                    log(f"  {stem}: mesh already generated, only exporting it")
                elif gpu_fails().get(stem, 0) >= MAX_GPU_TRIES:
                    log(f"  {stem}: broke the GPU {MAX_GPU_TRIES}x in this run, skipped "
                        f"(try it alone later, e.g. with --res 512)")
                else:
                    jobs.append((p, tag, fo, seed, stem))
    if a.export_only:
        jobs = []

    # ---- phase 1: generate
    if jobs:
        import torch
        from trellis2.pipelines import Trellis2ImageTo3DPipeline
        from trellis2 import models as tmodels
        if a.rembg:
            tb.use_rembg_model(a.rembg)
        log("loading TRELLIS.2 ...")
        pipe = Trellis2ImageTo3DPipeline.from_pretrained(a.model)
        pipe.cuda()
        tb.patch_fill_holes()
        enc = tmodels.from_pretrained(ENCODER).eval()
        resolutions = [a.res] + (["512"] if a.res != "512" and not a.no_fallback else [])

        by_photo = {}
        for j in jobs:
            by_photo.setdefault(j[0], []).append(j)
        for photo, pjobs in by_photo.items():
            tag = pjobs[0][1]
            img, how = cut_photo(photo)
            log(f"{photo.name}: cut by {how}")
            img = pipe.preprocess_image(img)
            cond512 = pipe.get_cond([img], 512)
            cond1024 = pipe.get_cond([img], 1024) if a.res != "512" else None

            # turn: match the LoD2 to TRELLIS' own rough shape from this photo
            if a.yaw == "auto":
                t1 = time.time()
                free = decode_ss(pipe, stage1(pipe, cond512, a.seeds[0], a.steps))
                if free.sum() < 50:
                    sys.exit(f"TRELLIS made (almost) nothing from {photo.name} - check the photo, or give --yaw")
                yaw, sc, cands = find_yaw(V, F, free, a.headroom)
                pl = placement(V, yaw, a.headroom)
                log(f"  turn: {yaw:.0f} deg (fit {sc:.4f}; other candidates: "
                    + ", ".join(f"{y:.0f} deg {s:.4f}" for y, s in cands[1:]) + f"; {time.time() - t1:.0f} s)")
                if len(cands) > 1 and cands[1][1] < 1.1 * cands[0][1]:
                    log(f"  NOTE: {cands[1][0]:.0f} deg fits almost as well - check {name}_trellis_{tag}_fit.png; "
                        f"if it looks wrong, run again with --yaw {cands[1][0]:.0f}")
                occ_l = voxelize(to_cube(V, pl), F)
                picture(out / f"{name}_trellis_{tag}_fit.png", occ_l, free,
                        f"{photo.name}: LoD2 turned {yaw:.0f} deg vs TRELLIS' own rough shape (seed {a.seeds[0]})")
            else:
                yaw = float(a.yaw)
                pl = placement(V, yaw, a.headroom)
                occ_l = voxelize(to_cube(V, pl), F)
                log(f"  turn: {yaw:.0f} deg (given)")
            z0 = encode_ss(enc, occ_l, pipe.device)
            back = decode_ss(pipe, z0)
            kept, new = follow_stats(occ_l, back)
            log(f"  LoD2 -> {int(occ_l.sum()):,} cells; encode/decode check: {kept * 100:.0f}% kept, "
                f"{new * 100:.0f}% extra (should be ~95+ / <10)")

            for _, _, fo, seed, stem in pjobs:
                t1 = time.time()
                CURRENT["stem"] = stem
                lvl = "exact LoD2 cells" if fo >= 1 else ("no LoD2" if fo <= 0 else
                                                          f"start at noise {noise_level(fo)[1]:.2f}")
                log(f"[{stem}] follow {fo:g} ({lvl}), seed {seed}  (started {time.strftime('%H:%M')})")
                if fo >= 1.0:
                    from scipy.ndimage import binary_dilation
                    occ = binary_dilation(occ_l, np.ones((3, 3, 3), bool), iterations=a.margin) if a.margin else occ_l
                    occ[:, :, : np.argwhere(occ_l)[:, 2].min()] = False      # not below the ground
                    torch.manual_seed(seed)
                else:
                    occ = decode_ss(pipe, stage1(pipe, cond512, seed, a.steps, z0, fo))
                kept, new = follow_stats(occ_l, occ)
                log(f"  rough shape: {int(occ.sum()):,} cells, {kept * 100:.0f}% of the LoD2 kept, "
                    f"{new * 100:.0f}% new")
                picture(out / f"{stem}_stage1.png", occ_l, occ,
                        f"{stem}: follow {fo:g} - kept {kept * 100:.0f}% of LoD2, {new * 100:.0f}% new")
                if a.preview:
                    continue
                info = {"made_by": "trellis_lod2.py", "lod2": str(lod2), "photo": str(photo), "follow": fo,
                        "seed": seed, "quality": a.quality, "steps": a.steps, "turn_deg": pl["yaw"],
                        "headroom": a.headroom, "stage1_lod2_kept": round(kept, 3), "stage1_new": round(new, 3)}
                place_s = {"yaw": pl["yaw"], "R": pl["R"].tolist(), "k": pl["k"], "c": pl["c"].tolist()}
                for res in resolutions:
                    ok = False
                    try:
                        mesh = stages23(pipe, coords_for(occ, res), cond512, cond1024, res, a)
                        raw, kept_f = save_mesh(tb, mesh, mesh_dir / f"{stem}.pt", res,
                                                {"place": place_s, "info": dict(info, res=res)})
                        del mesh
                        ok = True
                    except Exception as e:
                        if not tb.is_oom(e):
                            raise
                        where = tb.where_failed(e)
                        del e
                    if ok:
                        log(f"  generated at {res}: {raw:,} raw faces -> {kept_f:,} saved ({time.time() - t1:.0f} s)")
                        tb.free_gpu()
                        break
                    tb.clear_cuda_error()
                    tb.pipeline_to_cpu(pipe)
                    tb.free_gpu()
                    log(f"  OOM at {res} in {where} ({tb.gpu_free_gb():.1f} GB GPU free)"
                        + ("; retrying at 512" if res != resolutions[-1] else "; giving up on this one"))
                CURRENT["stem"] = None
        del pipe, enc
        tb.free_gpu()
    if a.preview:
        log(f"preview done -> {out}")
        return

    # ---- phase 2: export, back into the LoD2 frame
    todo = sorted(mesh_dir.glob(f"{name}_trellis_*.pt"))
    if not a.export_only:
        todo = [p for p in todo if a.overwrite or not (out / f"{p.stem}.glb").exists()]
    if not todo:
        log("nothing to export")
        return
    from scipy.spatial import cKDTree
    tgt = sample_surface(V, F, 200_000, np.random.default_rng(0))
    tree = cKDTree(tgt)
    meta_base = {"placement": {k: place_w[k] for k in ["origin_utm32", "origin_latlon", "ground_height_dhhn2016",
                                                       "axes", "facing", "turn_deg_about_up", "place_in_world"]
                               if k in place_w}}
    for pt in todo:
        t1 = time.time()
        res_glb = out / f"{pt.stem}.glb"
        log(f"export {pt.stem}")
        if export_placed(tb, pt, res_glb, a, tgt, tree, size, meta_base):
            log(f"ok   {res_glb.name}  {time.time() - t1:.0f} s  {res_glb.stat().st_size / 1e6:.1f} MB")
            if not a.keep_mesh:
                pt.unlink()
        else:
            log(f"FAILED {pt.stem}: kept {pt.name}; try --export-only --faces 200000 --tex 2048")
    log(f"done in {(time.time() - t_all) / 60:.1f} min -> {out}")


def run_child():
    """main(); a fatal CUDA error ends this process with RESTART_CODE so that supervise() starts a fresh one"""
    try:
        main()
        return 0
    except Exception as e:
        tb = sys.modules.get("trellis2_batch")
        if not (os.environ.get("TRELLIS_LOD2_CHILD") and is_cuda_fatal(e)) or "--preview" in sys.argv:
            raise
        where = tb.where_failed(e) if tb else f"{type(e).__name__}: {str(e)[:160]}"   # full traceback -> errors.log
        stem = CURRENT["stem"]
        gpu_fail_add(stem)
        log(f"  GPU ERROR{' in ' + stem if stem else ''}: {where}")
        log("  this process can't use the GPU any more (CUDA context broken)")
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(RESTART_CODE)        # no normal shutdown: freeing CUDA memory can hang on a broken context


def supervise():
    """run main() in a child process; after a fatal CUDA error (driver error, GPU reset) start a fresh one,
    which goes on with the jobs not done yet. The supervisor itself never touches the GPU."""
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--restarts", type=int, default=3)
    restarts = max(0, pre.parse_known_args()[0].restarts)
    fd, fails = tempfile.mkstemp(prefix="trellis_lod2_", suffix=".json")
    os.close(fd)
    env = dict(os.environ, TRELLIS_LOD2_CHILD="1", TRELLIS_LOD2_T0=repr(time.time()), TRELLIS_LOD2_FAILS=fails)
    cmd = [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]]
    try:
        for n in range(restarts + 1):
            code = subprocess.call(cmd, env=env)
            if code != RESTART_CODE:
                return code
            if n < restarts:
                log(f"===== starting again in a fresh process in 15 s (restart {n + 1}/{restarts}) =====")
                time.sleep(15)            # give the driver a moment after a GPU reset
        log(f"stopped: still GPU errors after {restarts} restarts - see errors.log")
        return 1
    except KeyboardInterrupt:
        return 130
    finally:
        Path(fails).unlink(missing_ok=True)


if __name__ == "__main__":
    sys.exit(run_child() if os.environ.get("TRELLIS_LOD2_CHILD") else supervise())
