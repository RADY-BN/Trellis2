"""
lowpoly.py - make a dense generated model (Pixal3D, TRELLIS.2, Hunyuan, Meshy, any GLB) lighter without
losing what you can see: hidden inner skins removed, flat surfaces merged into a handful of triangles,
every visible detail (windows, frames, bolts, columns, steps, signs, tracery) kept within a few
millimetres of the detailed model, colour + roughness + a normal map baked on. For buildings, churches,
stations, kiosks, art, props. CPU only (no GPU needed).

  python lowpoly.py glb/high/house.glb                # -> glb/high/lowpoly_visual/house.glb (default "visual")
  python lowpoly.py glb/full/church.glb --height 45 -q optimize   # -> glb/full/optimized/church.glb
  python lowpoly.py glb/full -q medium               # every GLB in the folder, lighter low-poly version
  python lowpoly.py glb/full/church.glb --error-mm 8  # even closer to the detailed model, more triangles
  python check.py glb/full glb/full/optimized        # how well do they match? (check.py: Pixal3D folder)
  python lowpoly.py out/church_mv.glb --height 45 --fill-openings   # close holes you can look through
  python lowpoly.py glb/high/house.glb -q planar      # flat walls/roofs as a handful of triangles each

Presets
  visual    (default) looks the same as the detailed model: repaired first (see 3. below), then reduced only
            where the eye cannot tell: shape within half a grid step of the generator everywhere it can be
            seen, nothing flattened or straightened, smooth shading, colour + normal map baked from the
            original. TRELLIS.2 tests (~480k triangles each): apartments 58.7k, house 147k, building 170k;
            0 gaps, 0 non-manifold edges, 0 triangles facing inward. In Unreal 5 (CARLA 0.10) the repaired
            full-detail mesh with Nanite is the other way to get the same look.
  planar    simplified look, fewest triangles (flat walls straightened) - for far-away / background use: every flat area (wall, roof, floor, sign) is rebuilt from its outline with the
            fewest triangles - a plain wall is 2 triangles, a wall with a few corners 10-20 - instead of
            following the generator's ripple. Flat areas are found with a tolerance of ~4 grid steps (the
            ripple TRELLIS/Pixal3D leave is 1-2 steps), shallow relief up to ~6 steps (frames, ledges, small
            recesses) is pressed into the wall and kept in the normal map + texture, edges between two walls
            become exactly straight. Curved parts (domes, spires, ornaments) are reduced as usual. The model
            stays closed. Test building (TRELLIS.2, 490k triangles): planar 25.6k, medium 118k (33k with
            --height 20); Paderborn Dom (480k): planar 67k - mostly the curved spire, towers, buttresses.
            Fine relief (window reveals) goes into the normal map: for a landmark lower it, e.g.
            --relief 0.003 --plane-tol 0.003, or use optimize / high.
  optimize  (default) every detail the generator really made stays: the shape error allowed is ~0.4 of the
            generator's grid step (1024 steps over the model: kiosk 5 mm, office block 7 mm, 40 m church
            17 mm). On the test models: 1-30% of the triangles, 99.9% of the surface within that error.
  high / medium / low   lighter low-poly versions (12 / 25 / 50 mm); shallow relief (1.5 / 3 / 8 cm) is
            flattened into the walls and kept in the normal map only.

The detailed model is the reference, and the error is a limit *everywhere* that can be seen, not an
average: a column, a step or a sign cannot be given up to save triangles on a wall.

Detail follows the real size: with the height known (--height, a sizes.txt next to the models with
'<name> <height in m>' lines, a name like 'church_45m.glb', or a model already in metres) the error is
set in millimetres (METRIC); otherwise relative to the model (PRESETS). The texture size follows the
surface area.

What it does
  1. merge    weld split vertices, drop loose crumbs
  2. level    stand the model upright: the "up" (within --max-tilt of +Y) that makes the most surface
              exactly vertical (walls) or level (floors, roofs, table tops)
  3. repair   (default, --solid on) every triangle turned to the side it is seen from (generated models have
              5-75% wound the wrong way - invisible in Unreal), hidden inner skins dropped, one closed shell
              built on a --voxel grid from them (gaps closed, hollow inside and the space under the open base
              filled - flat bottom), then put back onto the original surface so nothing moves.
              --solid auto / off: the old behaviour below.
     surface  keep only the triangles that can be seen from outside (rays from both sides of every
              triangle): generated models are thick-walled shells whose inner skin is often more than half
              of the triangles, and the tunnels to it (hundreds of tiny handles) stop simplification. The
              openings this leaves are capped where nobody can see the cap (a cap that would lie across a
              window is left out), every triangle faces the side it is seen from. A broken shell (open,
              many loose edges, e.g. TRELLIS) is rebuilt on a voxel grid.
  4. flat     find the flat areas (walls, roofs, floors, signs...) and make them exactly flat: walls exactly
              vertical, floors exactly level, so they merge into a few triangles. Only noise (and with
              --relief, shallow relief) is flattened; the hand-over to the rest spreads over two rings.
              -q planar instead rebuilds each flat area from its outline: outline points onto the plane
              (onto the line where two planes meet, the corner where three meet), every stretch of outline
              between two corners simplified once for both sides (no cracks), the area filled by a
              constrained Delaunay triangulation without inner points. Outlines that would cross themselves
              are repaired first; nothing is filled from a broken outline.
  5. reduce   quadric edge collapse in steps, stopping when more than bad_frac of the visible surface is
              farther than --error from the reference (measured both ways; surfaces sandwiched inside thin
              slabs do not count). Thin parts (signs, panes, railings, antennas) are weighted up from the
              start; after a pass, the places that are still off are weighted up and it goes on, so small
              details are the last to go. --faces caps the count.
  6. bake     new UVs (xatlas, in parallel pieces); colour, roughness/metal and a tangent-space normal map
              baked from the original model in chunks (~2 GB RAM for 4096). Flat areas shade perfectly
              flat, edges sharper than --hard-angle shade hard, curves smooth.
  7. place    base centred on the origin, optional real size (--height / --size), Y-up GLB with tangents
              (in Unreal: import normals AND tangents, so the normal map matches)

Needs (pip): numpy scipy scikit-image trimesh pymeshlab libigl fast-simplification xatlas embreex
             opencv-python pillow   (+ triangle shapely for -q planar)
"""
import argparse
import math
import time
from collections import deque
from pathlib import Path

import numpy as np
import trimesh

# error     = how far the reduced shape may be from the reference anywhere (fraction of the size; a
#             share bad_frac of the surface may be farther)
# faces     = most triangles allowed; min_faces = fewest
# mid       = working triangle count before reducing
# tol       = noise smaller than this (fraction of the size) still counts as flat
# angle     = surfaces bending less than this (degrees) can be one flat area
# min_plane = flat areas smaller than this share of the surface are treated as curved
# min_part  = loose pieces smaller than this share of the surface are dropped as crumbs
# relief    = details on a flat area shallower than this (fraction of the size) are flattened into it
# smooth    = de-noise passes on the non-flat parts (0: they are kept exactly as generated)
# tex       = texture size; voxel = grid when the surface has to be rebuilt
PRESETS = {
    "optimize": dict(error=0.0004, faces=600_000, min_faces=500, mid=800_000, tol=0.0008, angle=10.0,
                     min_plane=0.00005, relief=0.0, smooth=0, tex=4096, voxel=640, bad_frac=0.0005,
                     min_part=0.0001),
    "low":      dict(error=0.003,  faces=60_000,  min_faces=200, mid=300_000, tol=0.004, angle=15.0,
                     min_plane=0.0002, relief=0.004, smooth=0, tex=2048, voxel=384, bad_frac=0.002),
    "medium":   dict(error=0.0012, faces=250_000, min_faces=300, mid=400_000, tol=0.003, angle=15.0,
                     min_plane=0.0001, relief=0.0015, smooth=0, tex=4096, voxel=512, bad_frac=0.002),
    "high":     dict(error=0.0006, faces=500_000, min_faces=500, mid=700_000, tol=0.002, angle=12.0,
                     min_plane=0.00005, relief=0.0008, smooth=0, tex=4096, voxel=640, bad_frac=0.001),
    # visual (default): looks the same as the detailed model - shape within half a grid step of the generator
    # (1/1024 of the size) everywhere it can be seen, nothing flattened or straightened, shading from a baked
    # normal map; only what the eye cannot tell apart is reduced
    "visual":   dict(error=0.0005, faces=2_000_000, min_faces=500, mid=1_200_000, tol=0.0, angle=10.0,
                     min_plane=0.0001, relief=0.0, smooth=0, tex=4096, voxel=512, bad_frac=0.0005,
                     min_part=0.0001, flat=False),
    # planar: tol / relief / error in grid steps of the generator (1/1024 of the size), whatever the real
    # size - the ripple to look through is the generator's, not the building's
    "planar":   dict(error=0.0015, faces=600_000, min_faces=100, mid=400_000, tol=0.004, angle=20.0,
                     min_plane=0.0001, relief=0.006, smooth=3, tex=4096, voxel=512, bad_frac=0.003,
                     min_part=0.0001, planar=True),
}
# When the real size is known (--height / --size, a sizes.txt, or a model already in metres), the
# detail is set in metres instead: the same shape error everywhere, but never more than rel_error and
# never less than min_rel of the size (Pixal3D/TRELLIS build the model on a 1024 grid: below ~1/3 of a
# grid step there is only noise, and a small kiosk must not be made coarse). A building with lots of
# detail keeps more triangles than a plain one of the same size.
# error   = largest shape error allowed (metres); tol = noise that still counts as flat (m)
# relief  = details on flat surfaces shallower than this are flattened into them (m; 0 = none)
# texel   = texture resolution aimed at (metres per texture pixel), within min_tex..max_tex pixels
METRIC = {
    "optimize": dict(error=0.005, rel_error=0.01, min_rel=0.0004, tol=0.008, tol_rel=0.0006, relief=0.0,
                     texel=0.006, min_tex=2048, max_tex=4096, faces=600_000, min_faces=500),
    "low":      dict(error=0.05,  rel_error=0.012, min_rel=0.0006, tol=0.04, tol_rel=0.001, relief=0.08,
                     texel=0.03, min_tex=512, max_tex=2048, faces=60_000, min_faces=200),
    "medium":   dict(error=0.025, rel_error=0.005, min_rel=0.0004, tol=0.02, tol_rel=0.001, relief=0.03,
                     texel=0.012, min_tex=1024, max_tex=4096, faces=250_000, min_faces=300),
    "high":     dict(error=0.012, rel_error=0.0025, min_rel=0.0003, tol=0.01, tol_rel=0.001, relief=0.015,
                     texel=0.006, min_tex=2048, max_tex=4096, faces=500_000, min_faces=500),
    "planar":   dict(texel=0.01, min_tex=1024, max_tex=4096, faces=600_000, min_faces=100),   # shape: see PRESETS
    "visual":   dict(texel=0.006, min_tex=2048, max_tex=4096, faces=2_000_000, min_faces=500),  # shape: see PRESETS
}
BAD_FRAC = 0.002     # share of the surface that may be farther than --error from the reference
MESH_EXTS = {".glb", ".gltf", ".obj", ".ply", ".stl", ".off"}


def read_sizes(path):
    """{name: height in metres} from a sizes.txt: one '<name> <height>' per line (name = file name
    without extension, may contain spaces; '#' starts a comment). Missing file -> {}."""
    import re
    out = {}
    path = Path(path)
    if not path.is_file():
        return out
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.split("#")[0].strip()
        m = re.match(r'^"?(.*?)"?[\s,;:=]+(\d+(?:\.\d+)?)\s*m?\s*$', line)
        if m and float(m.group(2)) > 0:
            out[Path(m.group(1).strip()).stem if Path(m.group(1).strip()).suffix.lower() in
                {".png", ".jpg", ".jpeg", ".webp", ".glb", ".gltf", ".obj", ".ply"} else m.group(1).strip()] = \
                float(m.group(2))
    return out


def height_from_name(stem):
    """'church_45m' / 'kiosk 3.2m' -> 45.0 / 3.2; None when the name carries no height."""
    import re
    m = re.search(r"(?:^|[_\s\-])(\d+(?:\.\d+)?)\s?m$", stem)
    return float(m.group(1)) if m else None


def log(msg):
    print(f"     {msg}", flush=True)

# ============================================================================ input

def load_parts(src):
    """File path, trimesh.Scene or trimesh.Trimesh -> list of world-space Trimesh (visuals kept)."""
    obj = trimesh.load(str(src), process=False) if isinstance(src, (str, Path)) else src
    if isinstance(obj, trimesh.Trimesh):
        return [obj]
    parts = []
    for node in obj.graph.nodes_geometry:
        T, gname = obj.graph[node]
        g = obj.geometry[gname]
        if not isinstance(g, trimesh.Trimesh) or len(g.faces) == 0:
            continue
        g = g.copy()
        g.apply_transform(T)
        parts.append(g)
    if not parts:
        raise ValueError(f"no triangle mesh in {src}")
    return parts


def combine(parts):
    V, F, P, off = [], [], [], 0
    for i, p in enumerate(parts):
        V.append(np.asarray(p.vertices, np.float64))
        F.append(np.asarray(p.faces, np.int64) + off)
        P.append(np.full(len(p.faces), i, np.int32))
        off += len(p.vertices)
    return np.concatenate(V), np.concatenate(F), np.concatenate(P)

# ============================================================================ clean

def weld(V, F, eps):
    """Merge vertices closer than eps, drop degenerate and duplicate triangles and unused vertices."""
    lo = V.min(0)
    q = np.floor((V - lo) / eps + 0.5).astype(np.int64)
    if q.max() < (1 << 20):
        key = (q[:, 0] << 42) | (q[:, 1] << 21) | q[:, 2]
        _, idx, inv = np.unique(key, return_index=True, return_inverse=True)
    else:
        _, idx, inv = np.unique(q, axis=0, return_index=True, return_inverse=True)
    inv = inv.reshape(-1)
    V2, F2 = V[idx], inv[F]
    ok = (F2[:, 0] != F2[:, 1]) & (F2[:, 1] != F2[:, 2]) & (F2[:, 0] != F2[:, 2])
    F2 = F2[ok]
    s = np.sort(F2, axis=1)
    n = len(V2)
    if n < 2_000_000:
        _, first = np.unique((s[:, 0] * n + s[:, 1]) * n + s[:, 2], return_index=True)
    else:
        _, first = np.unique(s, axis=0, return_index=True)
    F2 = F2[np.sort(first)]
    used, F3 = np.unique(F2, return_inverse=True)
    return V2[used], F3.reshape(-1, 3)


def face_components(F, nv):
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    a = np.concatenate([F[:, 0], F[:, 1]])
    b = np.concatenate([F[:, 1], F[:, 2]])
    g = coo_matrix((np.ones(len(a), np.int8), (a, b)), shape=(nv, nv))
    _, lab = connected_components(g, directed=False)
    return lab[F[:, 0]]


def face_normals_areas(V, F):
    tri = V[F]
    cr = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    a = np.linalg.norm(cr, axis=1)
    n = cr / np.maximum(a, 1e-30)[:, None]
    return n, a / 2


def edge_stats(F):
    """(open edges, non-manifold edges, all edges)."""
    e = np.sort(np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), axis=1)
    n = int(F.max()) + 1
    _, c = np.unique(e[:, 0] * n + e[:, 1], return_counts=True)
    return int((c == 1).sum()), int((c > 2).sum()), len(c)


def clean_surface(V, F, min_part=0.0005, repair=True):
    """Keep the model's own surface: drop tiny loose bits (less than min_part of the area),
    repair non-manifold edges/vertices and close small holes."""
    comp = face_components(F, len(V))
    area = face_normals_areas(V, F)[1]
    ca = np.bincount(comp, area)
    keep = ca[comp] >= min_part * ca.sum()
    F = F[keep]
    used, inv = np.unique(F, return_inverse=True)
    V, F = V[used], inv.reshape(-1, 3)
    pml = _pymeshlab()
    if pml is not None and repair:
        ms = pml.MeshSet()
        ms.add_mesh(pml.Mesh(np.asarray(V, np.float64), np.asarray(F, np.int32)))
        ms.meshing_repair_non_manifold_edges(method="Remove Faces")
        ms.meshing_repair_non_manifold_vertices()
        try:
            ms.meshing_close_holes(maxholesize=60, selfintersection=False)
        except Exception:
            pass
        m = ms.current_mesh()
        V, F = m.vertex_matrix().astype(np.float64), m.face_matrix().astype(np.int64)
    return V, F

# ============================================================================ level

def _unit(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def rot_between(a, b):
    """Smallest rotation taking unit vector a onto unit vector b."""
    a, b = _unit(np.asarray(a, float)), _unit(np.asarray(b, float))
    v, c = np.cross(a, b), float(a @ b)
    if c > 1 - 1e-12:
        return np.eye(3)
    if c < -1 + 1e-12:
        p = _unit(np.cross(a, [1, 0, 0] if abs(a[0]) < 0.9 else [0, 1, 0]))
        return 2 * np.outer(p, p) - np.eye(3)
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K / (1 + c)


def estimate_up(V, F, prior=(0, 1, 0), max_tilt=40.0, delta=2.5):
    """Up direction near `prior` that makes the most surface area exactly vertical or horizontal.
    Returns (up, tilt_degrees, share_of_area_explained, share_at_prior)."""
    n, a = face_normals_areas(V, F)
    keep = a > 0
    n, a = n[keep], a[keep]
    n[n[:, 1] < 0] *= -1                      # the score ignores the sign of a normal
    q = np.floor(n * 120 + 0.5).astype(np.int64) + 128
    key = (q[:, 0] << 16) | (q[:, 1] << 8) | q[:, 2]
    uk, inv = np.unique(key, return_inverse=True)
    w = np.bincount(inv, a)
    nd = np.zeros((len(uk), 3))
    np.add.at(nd, inv, n * a[:, None])
    nd = _unit(nd)
    total = w.sum()
    sd, d2 = math.sin(math.radians(delta)), math.radians(delta) ** 2

    def score(U):
        out = []
        for i in range(0, len(U), 64):
            d = nd @ U[i:i + 64].T
            k = np.exp(-(d / sd) ** 2) + np.exp(-2 * (1 - np.abs(d)) / d2)
            out.append(w @ k)
        return np.concatenate(out) / total

    prior = _unit(np.asarray(prior, float))
    e1 = _unit(np.cross(prior, [1, 0, 0] if abs(prior[0]) < 0.9 else [0, 0, 1]))
    e2 = np.cross(prior, e1)

    def cands(center, half_deg, step_deg):
        c1 = _unit(np.cross(center, e2))
        c2 = np.cross(center, c1)
        t = np.tan(np.radians(np.arange(-half_deg, half_deg + 1e-9, step_deg)))
        X, Z = np.meshgrid(t, t)
        U = _unit(center + X.reshape(-1, 1) * c1 + Z.reshape(-1, 1) * c2)
        return U[np.degrees(np.arccos(np.clip(U @ prior, -1, 1))) <= max_tilt + 1e-6]

    U = cands(prior, max_tilt, 2.0)
    s = score(U)
    best = U[np.argmax(s)]
    U = cands(best, 3.0, 0.25)
    s = score(U)
    i = int(np.argmax(s))
    best, best_s = U[i], float(s[i])
    prior_s = float(score(prior[None])[0])
    tilt = math.degrees(math.acos(min(1.0, float(best @ prior))))
    return best, tilt, best_s, prior_s


def wall_yaw(V, F, up=(0, 1, 0)):
    """Angle (radians) around up that turns the main wall directions onto the X/Z axes."""
    n, a = face_normals_areas(V, F)
    sel = np.abs(n[:, 1]) < 0.1
    if a[sel].sum() < 0.05 * a.sum():
        return 0.0
    phi = np.arctan2(n[sel, 2], n[sel, 0])
    c = (a[sel] * np.exp(4j * phi)).sum()
    return float(np.angle(c) / 4)

# ============================================================================ rebuilt surface

def sample_surface(V, F, spacing, seed=0):
    """Points on the triangles (about 2 per spacing^2 of area) with their triangle's normal."""
    pts, nrm = [], []
    rng = np.random.default_rng(seed)
    for i in range(0, len(F), 500_000):
        tri = V[F[i:i + 500_000]]
        cr = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        area2 = np.linalg.norm(cr, axis=1)
        cnt = rng.poisson(area2 / 2 / spacing ** 2 * 2) + 1
        fi = np.repeat(np.arange(len(tri)), cnt)
        r = rng.random((len(fi), 2))
        flip = r.sum(1) > 1
        r[flip] = 1 - r[flip]
        t = tri[fi]
        pts.append((t[:, 0] + r[:, :1] * (t[:, 1] - t[:, 0]) + r[:, 1:] * (t[:, 2] - t[:, 0])).astype(np.float32))
        nrm.append((cr / np.maximum(area2, 1e-30)[:, None])[fi].astype(np.float32))
    return np.concatenate(pts), np.concatenate(nrm)


def orient_outward(V, F):
    """Consistent triangle winding per part, normals pointing out (pymeshlab when available)."""
    pml = _pymeshlab()
    if pml is not None:
        try:
            ms = pml.MeshSet()
            ms.add_mesh(pml.Mesh(np.asarray(V, np.float64), np.asarray(F, np.int32)))
            ms.meshing_re_orient_faces_coherently()
            m = ms.current_mesh()
            if m.face_number() == len(F):
                F = m.face_matrix().astype(np.int64)
                V = m.vertex_matrix().astype(np.float64)
        except Exception:
            pass
    # each loose part: normals should point away from the part's centre on average
    comp = face_components(F, len(V))
    n, a = face_normals_areas(V, F)
    fc = V[F].mean(1)
    w = np.bincount(comp, a)[:, None]
    cen = np.stack([np.bincount(comp, a * fc[:, k]) for k in range(3)], 1) / np.maximum(w, 1e-30)
    out = np.bincount(comp, a * np.einsum("ij,ij->i", fc - cen[comp], n))
    flip = out[comp] < 0
    F = F.copy()
    F[flip] = F[flip][:, ::-1]
    return V, F


def outer_surface(V, F, res=512, close=1, cap=True):
    """Remove hidden parts and repair the rest. Generated models often carry separate pieces inside
    (floors, inner skins, leftovers) which poke through the walls once the model is simplified. The space
    reachable from outside (through gaps wider than ~2*close voxels of a res^3 grid; an open bottom is
    capped) is air; a triangle is visible when air is right in front of it. Separate pieces that are
    (nearly) all invisible are removed. The kept triangles are the original ones: nothing gets blurred."""
    from scipy import ndimage

    lo, hi = V.min(0), V.max(0)
    vox = float((hi - lo).max()) / res
    margin = close + 5
    lo = lo - margin * vox
    dims = np.ceil((hi - lo) / vox).astype(int) + margin + 1
    pts, _ = sample_surface(V, F, vox / 2)
    ii = np.clip(np.floor((pts - lo) / vox + 0.5).astype(np.int64), 0, dims - 1)
    del pts
    near = np.zeros(dims, bool)
    near[ii[:, 0], ii[:, 1], ii[:, 2]] = True
    del ii
    free = ndimage.distance_transform_cdt(~near, metric="chessboard") > close
    if cap:
        ys = np.where(near.any(axis=(0, 2)))[0]
        if len(ys):
            for y in range(ys[0], min(ys[0] + close + 2, ys[-1] + 1)):
                blk = ndimage.binary_fill_holes(ndimage.binary_dilation(near[:, y, :], iterations=close))
                free[:, y, :] &= ~blk
    del near
    lab, _n = ndimage.label(free)
    del free
    border = np.unique(np.concatenate([lab[0].ravel(), lab[-1].ravel(), lab[:, 0].ravel(), lab[:, -1].ravel(),
                                       lab[:, :, 0].ravel(), lab[:, :, -1].ravel()]))
    air = np.isin(lab, border[border > 0])
    del lab
    air = ndimage.distance_transform_cdt(~air, metric="chessboard") <= close + 1
    fn = face_normals_areas(V, F)[0]
    c = V[F].mean(1)
    seen = np.zeros(len(F), bool)
    for o in (0.5, 1.0, 1.5, 2.5):
        q = np.floor((c + fn * o * vox - lo) / vox + 0.5).astype(np.int64)
        ok = np.all((q >= 0) & (q < dims), axis=1)
        val = np.ones(len(F), bool)
        q = np.clip(q, 0, dims - 1)
        val[ok] = air[q[ok, 0], q[ok, 1], q[ok, 2]]
        seen |= val
    del air
    # remove the separate parts that are (nearly) all hidden; parts of the visible shell stay whole, so
    # no hole is cut anywhere
    fa = face_normals_areas(V, F)[1]
    comp = face_components(F, len(V))
    hidden = np.bincount(comp, fa * ~seen) / np.maximum(np.bincount(comp, fa), 1e-30)
    drop = hidden[comp] >= 0.8
    n_hidden = int(len(np.unique(comp[drop])))
    Vk, Fk = clean_surface(V, F[~drop])
    return Vk, Fk, n_hidden


def remove_hidden(V, F, n_dirs=48, max_hole=500):
    """Delete every triangle that cannot be seen from outside, then cap the openings that leaves.
    Generated models are usually thick-walled shells: next to the outer skin there is an inner skin
    facing into the walls, plus leftovers inside - often more than half of all triangles. They cost
    triangles, and the tunnels connecting them to the outside (hundreds of tiny handles) stop any
    simplification below a few thousand triangles. A triangle is visible when at least one of n_dirs
    rays from it (to either side) escapes the model."""
    try:
        from trimesh.ray.ray_pyembree import RayMeshIntersector
        rmi = RayMeshIntersector(trimesh.Trimesh(V, F, process=False))
    except Exception:
        log("embreex is missing (pip install embreex): hidden triangles are kept")
        return V, F, 0.0
    fn, fa = face_normals_areas(V, F)
    c = V[F].mean(1)
    size = float((V.max(0) - V.min(0)).max())
    k = np.arange(n_dirs) + 0.5                       # evenly spread directions (Fibonacci sphere)
    phi = np.arccos(1 - 2 * k / n_dirs)
    th = math.pi * (1 + 5 ** 0.5) * k
    dirs = np.stack([np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)], 1)
    # both sides of every triangle are tested (generated models are not reliably oriented); the side a
    # ray escaped from is the outside, so it also tells which way the triangle has to face
    out_votes = np.zeros(len(F))
    for d in dirs:
        side = fn @ d
        todo = np.flatnonzero(np.abs(side) > 0.05)
        for i in range(0, len(todo), 1 << 20):
            t = todo[i:i + (1 << 20)]
            o = c[t] + fn[t] * (np.sign(side[t]) * size * 1e-4)[:, None]
            hit = rmi.intersects_any(o, np.tile(d, (len(t), 1)))
            np.add.at(out_votes, t[~hit], np.sign(side[t[~hit]]) * np.abs(side[t[~hit]]))
    vis = out_votes != 0
    hidden_share = float(fa[~vis].sum() / fa.sum())
    F = F.copy()
    flip = out_votes < 0                      # seen from its back: turn it around
    F[flip] = F[flip][:, ::-1]
    Fk = F[vis]
    used, inv = np.unique(Fk, return_inverse=True)
    Vk, Fk = V[used], inv.reshape(-1, 3)
    pml = _pymeshlab()
    if pml is not None:
        ms = pml.MeshSet()
        ms.add_mesh(pml.Mesh(np.asarray(Vk, np.float64), np.asarray(Fk, np.int32)))
        ms.meshing_remove_connected_component_by_face_number(mincomponentsize=30)
        for _ in range(2):
            ms.meshing_repair_non_manifold_edges(method="Remove Faces")
            ms.meshing_repair_non_manifold_vertices()
        new = None
        try:
            ms.meshing_close_holes(maxholesize=max_hole, selfintersection=False, newfaceselected=True)
            new = ms.current_mesh().face_selection_array().astype(bool)
        except Exception:
            pass
        m = ms.current_mesh()
        Vk, Fk = m.vertex_matrix().astype(np.float64), m.face_matrix().astype(np.int64)
        if new is not None and new.any() and len(new) == len(Fk):
            Fk = _drop_visible_caps(Vk, Fk, new, rmi_dirs=dirs, size=size)
            used, inv = np.unique(Fk, return_inverse=True)
            Vk, Fk = Vk[used], inv.reshape(-1, 3)
    return Vk, Fk, hidden_share


def _drop_visible_caps(V, F, new, rmi_dirs, size):
    """A hole cap is only right where nobody can see it (deep in a tunnel to the removed inner skin).
    A cap that can be seen from outside spans an opening in plain view - e.g. a lid across a rose
    window whose back was removed - and would cover real detail: such caps are taken out again."""
    try:
        from trimesh.ray.ray_pyembree import RayMeshIntersector
    except Exception:
        return F
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    ni = np.flatnonzero(new)
    rmi = RayMeshIntersector(trimesh.Trimesh(V, F, process=False))
    fn = face_normals_areas(V, F[ni])[0]
    c = V[F[ni]].mean(1)
    seen = np.zeros(len(ni), bool)
    for d in rmi_dirs:
        side = fn @ d
        t = np.flatnonzero((np.abs(side) > 0.05) & ~seen)
        if not len(t):
            continue
        o = c[t] + fn[t] * (np.sign(side[t]) * size * 1e-4)[:, None]
        seen[t[~rmi.intersects_any(o, np.tile(d, (len(t), 1)))]] = True
    if not seen.any():
        return F
    # whole caps: connected groups of new triangles (sharing an edge)
    sub = F[ni]
    e = np.concatenate([sub[:, [0, 1]], sub[:, [1, 2]], sub[:, [2, 0]]])
    e = np.sort(e, axis=1)
    key = e[:, 0] * len(V) + e[:, 1]
    fid = np.tile(np.arange(len(ni)), 3)
    o = np.argsort(key, kind="stable")
    key, fid = key[o], fid[o]
    same = np.flatnonzero(key[1:] == key[:-1])
    g = coo_matrix((np.ones(len(same)), (fid[same], fid[same + 1])), shape=(len(ni), len(ni)))
    _, lab = connected_components(g, directed=False)
    bad_caps = np.unique(lab[seen])
    drop = np.zeros(len(F), bool)
    drop[ni[np.isin(lab, bad_caps)]] = True
    return F[~drop]


def fill_openings(occ, min_share=0.02):
    """Close holes you can look through and fill a hollow inside (voxels in, voxels out).
    Generated buildings are often hollow boxes with thick walls; a gap in a wall then opens the hollow
    inside, which shows as a hole (common on the sides no view showed). Every slice of the grid along X,
    Y and Z is filled where it is enclosed (a hole in a wall is enclosed by the wall in the slices running
    through the wall), then whatever can no longer be reached from outside is filled. Of that, only big
    connected pieces are kept - the hollow inside together with the holes into it; window and door
    recesses are small pieces of their own and stay as they are."""
    from scipy import ndimage
    out = occ.copy()
    for ax in range(3):
        src = np.moveaxis(occ, ax, 0)
        dst = np.moveaxis(out, ax, 0)
        for i in range(src.shape[0]):
            if src[i].any():
                dst[i] |= ndimage.binary_fill_holes(src[i])
    lab, _n = ndimage.label(~out)
    border = np.unique(np.concatenate([lab[0].ravel(), lab[-1].ravel(), lab[:, 0].ravel(), lab[:, -1].ravel(),
                                       lab[:, :, 0].ravel(), lab[:, :, -1].ravel()]))
    out |= ~np.isin(lab, border[border > 0])
    del lab
    add = out & ~occ
    lab, n = ndimage.label(add)
    if not n:
        return occ
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    keep = sizes >= min_share * max(int(occ.sum()), 1)
    return occ | keep[lab]


def project_back(Vs, Fs, Vref, Fref, vox, reach=1.5):
    """Put the rebuilt surface back onto the original one: every vertex within `reach` voxels of the original
    surface moves onto its closest point there (the voxel grid alone would round edges and shift walls by up to
    a voxel). Vertices on thin parts (a slab thinner than ~2.5 voxels made around a single sheet) are left
    alone - both sides would land on the same sheet - and so is any move that would turn a triangle over."""
    import igl
    n = smooth_vertex_normals(Vs, Fs, vox * 1e-3)
    d2, _, cp = igl.point_mesh_squared_distance(Vs, np.asarray(Vref, np.float64), np.asarray(Fref, np.int64))
    disp = cp - Vs
    dist = np.sqrt(np.maximum(d2, 0))
    along = np.einsum("ij,ij->i", disp, n)
    ok = (dist <= reach * vox) & (np.abs(along) >= 0.6 * dist)
    try:
        from trimesh.ray.ray_pyembree import RayMeshIntersector
        rmi = RayMeshIntersector(trimesh.Trimesh(Vs, Fs, process=False))
        idx = np.flatnonzero(ok & (along < 0))
        start = 0.75 * vox                             # past the grid's own staircase next to the vertex
        loc, ri, _ = rmi.intersects_location(Vs[idx] - n[idx] * start, -n[idx], multiple_hits=False)
        thick = np.full(len(idx), np.inf)
        thick[ri] = start + np.linalg.norm(loc - (Vs[idx[ri]] - n[idx[ri]] * start), axis=1)
        ok[idx[thick < 2.5 * vox]] = False
    except Exception:
        pass
    # along the vertex normal, by the distance to the original surface measured that way; points that could
    # not be measured cleanly take the movement of their neighbours (so the surface moves as a whole - a
    # mix of moved and unmoved points would leave small dents and bumps), points with no original surface
    # nearby (patches over gaps, the bottom) stay
    from scipy.sparse import csr_matrix
    disp = np.zeros_like(Vs)
    disp[ok] = n[ok] * along[ok, None]
    free = ~ok & (dist <= 2.5 * vox)
    if free.any():
        e = np.concatenate([Fs[:, [0, 1]], Fs[:, [1, 2]], Fs[:, [2, 0]]])
        e = np.concatenate([e, e[:, ::-1]])
        A = csr_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(len(Vs), len(Vs)))
        A.data[:] = 1.0
        deg = np.maximum(np.asarray(A.sum(1)).ravel(), 1)[:, None]
        for _ in range(40):
            disp[free] = (A @ disp)[free] / deg[free]
    Vp = Vs + disp
    fn0 = np.cross(Vs[Fs[:, 1]] - Vs[Fs[:, 0]], Vs[Fs[:, 2]] - Vs[Fs[:, 0]])
    for _ in range(4):                                 # undo moves that turn a triangle over
        fn1 = np.cross(Vp[Fs[:, 1]] - Vp[Fs[:, 0]], Vp[Fs[:, 2]] - Vp[Fs[:, 0]])
        bad = np.einsum("ij,ij->i", fn0, fn1) <= 0.05 * np.linalg.norm(fn0, axis=1) * np.linalg.norm(fn1, axis=1)
        if not bad.any():
            break
        vb = np.unique(Fs[bad])
        Vp[vb] = Vs[vb]
    moved = np.linalg.norm(Vp - Vs, axis=1) > 1e-3 * vox
    return Vp, float(moved.mean())


def orient_by_sight(V, F, n_dirs=96, down=-0.15, min_seen=0.08):
    """Generated models have 5-75% of their triangles wound the wrong way and carry hidden inner skins. Every
    triangle is turned to face the side it is seen from (rays to both sides, sideways and upwards only - rays
    going down would look into the hollow inside through the open base), and triangles nobody can see from
    outside (inner skins, insides of hollow buildings) are dropped - also those seen only through a window
    or a gap (less than `min_seen` of the open directions), which belong to the inside. Returns (V, F, share
    of the surface turned, share dropped)."""
    try:
        from trimesh.ray.ray_pyembree import RayMeshIntersector
        rmi = RayMeshIntersector(trimesh.Trimesh(V, F, process=False))
    except Exception:
        log("embreex is missing (pip install embreex): triangles kept as they are")
        return V, F, 0.0, 0.0
    fn, fa = face_normals_areas(V, F)
    c = V[F].mean(1)
    size = float((V.max(0) - V.min(0)).max())
    k = np.arange(n_dirs) + 0.5
    phi = np.arccos(1 - 2 * k / n_dirs)
    th = math.pi * (1 + 5 ** 0.5) * k
    dirs = np.stack([np.cos(th) * np.sin(phi), np.cos(phi), np.sin(th) * np.sin(phi)], 1)
    dirs = dirs[dirs[:, 1] > down]
    front, back = np.zeros(len(F)), np.zeros(len(F))
    full = np.zeros(len(F))
    for d in dirs:
        s_ = fn @ d
        full += np.abs(s_) * (np.abs(s_) > 0.05) * 0.5
        for sign, acc in ((1.0, front), (-1.0, back)):
            t = np.flatnonzero(sign * s_ > 0.05)
            for i in range(0, len(t), 1 << 20):
                tt = t[i:i + (1 << 20)]
                hit = rmi.intersects_any(c[tt] + fn[tt] * (sign * size * 1e-5), np.tile(d, (len(tt), 1)))
                acc[tt[~hit]] += np.abs(s_[tt[~hit]])
    seen = np.maximum(front, back) >= min_seen * np.maximum(full, 1e-9)
    flip = back > front
    F = F.copy()
    F[flip] = F[flip][:, ::-1]
    A = fa.sum()
    return V, F[seen], float(fa[flip & seen].sum() / A), float(fa[~seen].sum() / A)


def fill_covered(field, band, vox, close=2):
    """Air that is closed in on all sides (in its horizontal slice, gaps up to ~2*close cells bridged) AND
    has the model above it is a hidden inside: the hollow of a generated building and the space under its
    open base. Made solid, so the bottom is flat and no inner walls are left. Courtyards (open to the sky),
    arcades and passages (open at the side) stay open."""
    from scipy import ndimage
    occ = field < 0
    above = np.flip(np.logical_or.accumulate(np.flip(occ, axis=1), axis=1), axis=1)
    n_add = 0
    for y in range(occ.shape[1] - 1):
        sl = occ[:, y, :]
        cov = above[:, y + 1, :]
        if not cov.any():
            continue
        cl = sl | ndimage.binary_closing(sl, iterations=close) if sl.any() else sl
        add = ndimage.binary_fill_holes(cl) & ~sl & cov
        if add.any():
            field[:, y, :][add] = -band * vox
            n_add += int(add.sum())
    # what the outside cannot reach through gaps wider than ~2*close cells is hidden inside too (bubbles and
    # thin layers of air left between filled slices); small dead-end pockets next to the outside (window and
    # door recesses) stay open
    solid = field < 0
    far = ndimage.distance_transform_cdt(~solid, metric="chessboard") > close
    lab, _ = ndimage.label(far)
    del far
    border = np.unique(np.concatenate([lab[0].ravel(), lab[-1].ravel(), lab[:, 0].ravel(), lab[:, -1].ravel(),
                                       lab[:, :, 0].ravel(), lab[:, :, -1].ravel()]))
    reach = np.isin(lab, border[border > 0])
    del lab
    reach = ndimage.binary_dilation(reach, iterations=close + 1, mask=~solid)
    un = ~solid & ~reach
    lab, n = ndimage.label(un)
    if n:
        sizes = np.bincount(lab.ravel())
        sizes[0] = 0
        touch = np.zeros(n + 1, bool)
        touch[np.unique(lab[ndimage.binary_dilation(reach) & un])] = True
        keep = touch & (sizes < 0.002 * max(int(solid.sum()), 1))
        keep[0] = True
        fill = ~keep[lab]
        field[fill] = -band * vox
        n_add += int(fill.sum())
    return n_add


def solid_surface(V, F, res=512, band=3, thick=0.5, fill=False, covered=False, close=2, orient=True):
    """Rebuild the model as one clean closed surface (Y up) on a res^3 grid.

    Near the surface the field is the exact distance to the original triangles; inside/outside
    comes from the generalized winding number (libigl), which stays right for meshes with holes,
    cracks, open bottoms, flipped or doubled triangles and hidden insides. Flat walls stay flat,
    edges stay sharp, thin parts (leaves, signs, railings) become slabs `thick` voxels thick.
    fill=True: also close openings into the model (see fill_openings) and fill it solid inside."""
    import igl
    from scipy import ndimage
    from skimage import measure

    if orient:
        V, F = orient_outward(V, F)
    margin = band + 3
    lo, hi = V.min(0), V.max(0)
    vox = float((hi - lo).max()) / res
    lo = lo - margin * vox
    dims = np.ceil((hi - lo) / vox).astype(int) + margin + 1
    pts, _ = sample_surface(V, F, vox / 2)
    ii = np.clip(np.floor((pts - lo) / vox + 0.5).astype(np.int64), 0, dims - 1)
    del pts
    near = np.zeros(dims, bool)
    near[ii[:, 0], ii[:, 1], ii[:, 2]] = True
    del ii
    near = ndimage.binary_dilation(near, iterations=band)
    # away from the surface: every connected air/solid pocket is either inside or outside;
    # ask the winding number at the point of each pocket that is farthest from the surface
    lab, n = ndimage.label(~near)
    depth = ndimage.distance_transform_cdt(~near, metric="taxicab")
    reps = np.array(ndimage.maximum_position(depth, lab, np.arange(1, n + 1)), np.float64).reshape(-1, 3)
    del depth
    w = igl.fast_winding_number(np.asarray(V, np.float64), np.asarray(F, np.int64), lo + reps * vox) if n else np.zeros(0)
    pocket_inside = np.concatenate([[False], w > 0.5])
    field = np.where(pocket_inside[lab], -band * vox, band * vox).astype(np.float32)
    del lab
    # near the surface: exact signed distance, pushed out by `thick`/2 so thin sheets get a thickness
    bi = np.argwhere(near)
    del near
    r = 0.5 * thick * vox if thick > 0 else 0.0
    for i in range(0, len(bi), 4_000_000):
        b = bi[i:i + 4_000_000]
        S = igl.signed_distance(lo + b * vox, np.asarray(V, np.float64), np.asarray(F, np.int64),
                                sign_type=igl.SIGNED_DISTANCE_TYPE_FAST_WINDING_NUMBER)[0]
        field[b[:, 0], b[:, 1], b[:, 2]] = np.clip(S - r, -band * vox, band * vox)
    del bi
    if fill:
        occ = field < 0
        add = fill_openings(occ) & ~occ
        field[add] = -band * vox
        log(f"fill: {int(add.sum()):,} voxels of openings and hollow inside filled")
        del occ, add
    elif covered:
        n_add = fill_covered(field, band, vox, close=close)
        if n_add:
            log(f"hollow inside and the space under the open base filled ({n_add:,} grid cells): flat bottom, "
                f"no hidden inner walls")
    mv, mf, _, _ = measure.marching_cubes(field, 0.0, allow_degenerate=True)
    del field
    mv = mv.astype(np.float64) * vox + lo
    mv, mf = weld(mv, mf.astype(np.int64), vox * 1e-3)
    if trimesh.Trimesh(mv, mf, process=False).volume < 0:
        mf = mf[:, ::-1]
    # drop loose crumbs (less than 0.05% of the surface)
    comp = face_components(mf, len(mv))
    ca = np.bincount(comp, face_normals_areas(mv, mf)[1])
    mf = mf[ca[comp] >= 0.0005 * ca.sum()]
    used, inv = np.unique(mf, return_inverse=True)
    return mv[used], inv.reshape(-1, 3), vox

# ============================================================================ decimation

def _pymeshlab():
    try:
        import pymeshlab
        return pymeshlab
    except Exception:
        return None


def decimate_fast(V, F, target):
    import fast_simplification
    if len(F) <= target:
        return V, F
    v, f = fast_simplification.simplify(np.asarray(V, np.float32), np.asarray(F, np.int32),
                                        target_reduction=1 - target / len(F), agg=7)
    return v.astype(np.float64), f.astype(np.int64)


def decimate_qem(V, F, target, quality=0.3):
    """Quadric edge collapse (pymeshlab) to `target` triangles, topology and boundaries kept."""
    if len(F) <= target:
        return V, F
    pml = _pymeshlab()
    if pml is None:
        return decimate_fast(V, F, target)
    ms = pml.MeshSet()
    ms.add_mesh(pml.Mesh(np.asarray(V, np.float64), np.asarray(F, np.int32)))
    ms.meshing_decimation_quadric_edge_collapse(
        targetfacenum=int(target), qualitythr=quality, preserveboundary=True, boundaryweight=2.0,
        preservenormal=True, preservetopology=True, optimalplacement=True, planarquadric=False,
        autoclean=True)
    m = ms.current_mesh()
    return m.vertex_matrix().astype(np.float64), m.face_matrix().astype(np.int64)


def area_samples(V, F, n, seed=0):
    """About n points spread evenly over the surface: (points, triangle of each point, area per point)."""
    fa = face_normals_areas(V, F)[1]
    A = float(fa.sum())
    rng = np.random.default_rng(seed)
    fi = np.repeat(np.arange(len(F)), rng.poisson(fa / A * n))
    r = rng.random((len(fi), 2))
    flip = r.sum(1) > 1
    r[flip] = 1 - r[flip]
    t = V[F[fi]]
    P = t[:, 0] + r[:, :1] * (t[:, 1] - t[:, 0]) + r[:, 1:] * (t[:, 2] - t[:, 0])
    return P, fi, np.full(len(fi), A / max(len(fi), 1))


def dist_to_mesh(P, V, F):
    """Distance from every point to the triangle mesh (exact, libigl; else to dense surface samples)."""
    try:
        import igl
        d2 = igl.point_mesh_squared_distance(np.asarray(P, np.float64), np.asarray(V, np.float64),
                                             np.asarray(F, np.int64))[0]
        return np.sqrt(np.maximum(d2, 0.0))
    except ImportError:
        from scipy.spatial import cKDTree
        S = area_samples(V, F, max(4 * len(P), 400_000), seed=7)[0]
        return cKDTree(S).query(P, workers=-1)[0]


def thin_vertices(V, F, limit):
    """Vertices of thin parts: signs, panes, railings, fins, antennas, flags. A ray from the vertex into
    the model meets the other side within `limit`. Quadric collapse loves to shrink such double-sided
    sheets (they have no volume), so they are weighted up from the start."""
    try:
        from trimesh.ray.ray_pyembree import RayMeshIntersector
    except Exception:
        return np.zeros(len(V), bool)
    size = float(np.ptp(V, 0).max())
    n = smooth_vertex_normals(V, F, size * 1e-7)
    rmi = RayMeshIntersector(trimesh.Trimesh(V, F, process=False))
    o = V - n * size * 1e-5
    loc, ri, _ = rmi.intersects_location(o, -n, multiple_hits=False)
    thin = np.zeros(len(V), bool)
    thin[ri[np.linalg.norm(loc - o[ri], axis=1) < limit]] = True
    return thin


def sample_visibility(V, F, P, fi, n_dirs=32, seed=0):
    """Share of the open sky (cosine-weighted hemisphere, better of the two sides) each point on the surface
    sees. 0 for surfaces sandwiched inside thin slabs or walled in by other parts: they do not count. A deep
    window or a portal still sees some sky through its opening, so it does count."""
    try:
        from trimesh.ray.ray_pyembree import RayMeshIntersector
    except Exception:
        return np.ones(len(P))
    rmi = RayMeshIntersector(trimesh.Trimesh(V, F, process=False))
    fn = face_normals_areas(V, F)[0][fi]
    size = float(np.ptp(V, 0).max())
    rng = np.random.default_rng(seed)
    a = _unit(np.cross(fn, np.where(np.abs(fn[:, :1]) < 0.9, [[1.0, 0, 0]], [[0, 1.0, 0]])))
    b = np.cross(fn, a)
    best = np.zeros(len(P))
    for side in (1.0, -1.0):
        free = np.zeros(len(P))
        for _ in range(n_dirs):
            u1, u2 = rng.random(len(P)), rng.random(len(P))
            r, phi = np.sqrt(u1), 2 * math.pi * u2
            d = (a * (r * np.cos(phi))[:, None] + b * (r * np.sin(phi))[:, None]
                 + side * fn * np.sqrt(np.maximum(0, 1 - u1))[:, None])
            hit = rmi.intersects_any(P + side * fn * size * 1e-5, d)
            free += ~hit
        best = np.maximum(best, free / n_dirs)
    return best


def reduce_to_error(V, F, tau, max_faces, min_faces, bad_frac=BAD_FRAC, passes=5, step=0.8, n_samples=300_000,
                    labels=None, only_free=False):
    """Quadric edge collapse that keeps the shape within `tau` (model units) of the reference (V, F)
    everywhere that matters. In steps, stopping before more than `bad_frac` of the surface is farther than
    tau from the reference, measured both ways (reference -> result catches lost details, result ->
    reference catches bridges and flaps). Thin parts are weighted up from the start; after every pass the
    vertices of the places that are (nearly) off get a higher weight too (they are collapsed later) and the
    reduction goes on from there: small details (columns, steps, frames, signs) are the last to go instead
    of the first. Flat areas cost nothing and end with a handful of triangles. labels: one number per
    triangle (flat area or -1), carried along to the remaining triangles; only_free: only the triangles labelled
    -1 are reduced (the flat areas and their outlines stay exactly as they are). Returns (V, F, stats, labels)."""
    pml = _pymeshlab()
    if pml is None:
        if only_free:                 # no way to reduce only part of it without pymeshlab
            return V, F, None, None if labels is None else np.asarray(labels)
        v, f = decimate_fast(V, F, max_faces)
        return v, f, None, None
    from scipy.sparse import coo_matrix
    P_ref, fi_ref, a_ref = area_samples(V, F, n_samples, seed=1)
    # only what can be seen counts: surfaces sandwiched between others (inside thin slabs, behind parts)
    # may merge or vanish without anyone noticing
    vis = sample_visibility(V, F, P_ref, fi_ref)
    a_ref = a_ref * (vis > 0)        # not one of 2x32 rays gets out: sandwiched, never seen
    w = np.ones(len(V))
    lab = (np.full(len(F), -1.0) if labels is None else np.asarray(labels, np.float64))
    n_locked = int((lab >= 0).sum()) if only_free else 0   # with a selection, MeshLab's target counts selected faces
    thin = thin_vertices(V, F, 2.5 * tau)
    w[thin] = 20.0
    nbr = None
    best, start = None, min(len(F), int(max_faces))
    for it in range(passes):
        ms = pml.MeshSet()
        ms.add_mesh(pml.Mesh(vertex_matrix=np.asarray(V, np.float64), face_matrix=np.asarray(F, np.int32),
                             v_scalar_array=np.asarray(w, np.float64), f_scalar_array=lab))
        if only_free:
            ms.compute_selection_by_condition_per_face(condselect="fq < -0.5")
        target, prev, fine = start, None, False
        while True:
            if target < ms.current_mesh().face_number():
                # optimalplacement=False: a collapsed edge ends on one of its own vertices: no vertex is
                # thrown off the model (no spikes, no flaps) and vertices on a flat area stay on it.
                # qualityweight: a vertex's weight multiplies its collapse cost (only when weights differ:
                # all-equal weights make MeshLab's normalisation divide by zero)
                ms.meshing_decimation_quadric_edge_collapse(
                    targetfacenum=max(1, int(target) - n_locked), qualitythr=0.3, preserveboundary=True, boundaryweight=2.0,
                    preservenormal=True, preservetopology=True, optimalplacement=False, planarquadric=False,
                    qualityweight=bool(np.ptp(w) > 0), autoclean=True, selected=only_free)
            m = ms.current_mesh()
            Vl, Fl = m.vertex_matrix().astype(np.float64), m.face_matrix().astype(np.int64)
            d1 = dist_to_mesh(P_ref, Vl, Fl)
            P_low, _, a_low = area_samples(Vl, Fl, n_samples // 3, seed=2)
            d2 = dist_to_mesh(P_low, V, F)
            bad = max(a_ref[d1 > tau].sum() / a_ref.sum(), a_low[d2 > tau].sum() / a_low.sum())
            rms = math.sqrt(max((a_ref * d1 ** 2).sum() / a_ref.sum(), (a_low * d2 ** 2).sum() / a_low.sum()))
            ok = bad <= bad_frac and rms <= 0.5 * tau
            if not ok and prev is not None and not fine:
                # overshot with a big step: back to the last good mesh, on in small steps
                fine = True
                ms = pml.MeshSet()
                ms.add_mesh(pml.Mesh(vertex_matrix=prev[0], face_matrix=prev[1].astype(np.int32),
                                     v_scalar_array=prev[5], f_scalar_array=prev[6]))
                if only_free:
                    ms.compute_selection_by_condition_per_face(condselect="fq < -0.5")
                target = max(int(min_faces), int(len(prev[1]) * 0.92))
                continue
            if not ok and (prev is not None or best is not None):
                break
            stuck = prev is not None and len(Fl) >= len(prev[1])
            prev = (Vl, Fl, d1, bad, rms, m.vertex_scalar_array().astype(np.float64),
                    m.face_scalar_array().astype(np.float64))
            if len(Fl) <= min_faces or not ok or stuck:
                break
            target = max(int(min_faces), int(len(Fl) * (0.92 if fine else step)))
        if prev is None:              # a weighted pass that is off from the start: keep the last result
            break
        Vl, Fl, d1, bad, rms, _, labl = prev
        if best is not None and len(Fl) >= len(best[1]):
            break                     # no further progress
        best = (Vl, Fl, labl, dict(bad=bad, rms=rms, p99=float(np.percentile(d1[a_ref > 0], 99)), passes=it + 1,
                             thin=int(thin.sum()), hidden=float((vis == 0).mean())))
        badf = np.unique(fi_ref[(d1 > 0.6 * tau) & (a_ref > 0)])
        if it == passes - 1 or not len(badf) or len(Fl) <= min_faces:
            break
        # the reference vertices around the places that are (nearly) off: weighted up for the next pass,
        # which starts where this one ended
        bv = np.zeros(len(V), bool)
        bv[F[badf].ravel()] = True
        if nbr is None:
            e = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
            nbr = coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(len(V), len(V))).tocsr()
            nbr = nbr + nbr.T
        grow = bv.copy()
        for _ in range(2):
            grow |= (nbr @ grow.astype(float)) > 0
        w[grow] *= 4.0
        w[bv] *= 25.0
        start = len(Fl)
        log(f"reduce pass {it + 1}: {len(Fl):,} triangles, {bad * 100:.3f}% of the surface off; "
            f"{bv.sum():,} vertices weighted up")
    if best is None:
        return V, F, None, (None if labels is None else np.asarray(labels))
    Vl, Fl, labl, stats = best
    return Vl, Fl, stats, np.round(labl).astype(np.int64)

# ============================================================================ flat areas

def _neighbours(F):
    m = trimesh.Trimesh(np.zeros((F.max() + 1, 3)), F, process=False)
    adj = m.face_adjacency
    src = np.concatenate([adj[:, 0], adj[:, 1]])
    dst = np.concatenate([adj[:, 1], adj[:, 0]])
    o = np.argsort(src, kind="stable")
    src, dst = src[o], dst[o]
    ptr = np.searchsorted(src, np.arange(len(F) + 1))
    return ptr, dst, adj


def segment_planes(V, F, angle, tol, min_area_frac, max_spread, raw=False, group_tol=None):
    """Region growing over neighbouring triangles. Returns region per face (-1 = curved/free)."""
    fn, fa = face_normals_areas(V, F)
    fc = V[F].mean(1)
    ptr, dst, adj = _neighbours(F)
    region = -np.ones(len(F), np.int64)
    cos_t = math.cos(math.radians(angle))
    fnl, fcl = fn.tolist(), fc.tolist()
    tri = V[F]
    e2 = ((tri[:, [1, 2, 0]] - tri) ** 2).sum(2).sum(1)
    sliver = (4 * math.sqrt(3) * fa / np.maximum(e2, 1e-30) < 0.15).tolist()  # thin triangles: noisy normals
    tvl = tri.tolist()
    rid = 0
    for seed in np.argsort(-fa):
        if region[seed] >= 0:
            continue
        region[seed] = rid
        ns = fn[seed] * fa[seed]
        cs = fc[seed] * fa[seed]
        asum = fa[seed]
        nx, ny, nz = fnl[seed]
        cx, cy, cz = fcl[seed]
        q = deque([seed])
        while q:
            cur = q.popleft()
            for nb in dst[ptr[cur]:ptr[cur + 1]]:
                if region[nb] >= 0:
                    continue
                if sliver[nb]:   # judge a sliver by its corners only
                    if any(abs((p - cx) * nx + (r - cy) * ny + (s - cz) * nz) >= 0.5 * tol for p, r, s in tvl[nb]):
                        continue
                else:
                    a, b, c = fnl[nb]
                    if a * nx + b * ny + c * nz <= cos_t:
                        continue
                    p, r, s = fcl[nb]
                    if abs((p - cx) * nx + (r - cy) * ny + (s - cz) * nz) >= tol:
                        continue
                region[nb] = rid
                ns = ns + fn[nb] * fa[nb]
                cs = cs + fc[nb] * fa[nb]
                asum += fa[nb]
                L = math.sqrt(ns[0] ** 2 + ns[1] ** 2 + ns[2] ** 2) + 1e-30
                nx, ny, nz = ns[0] / L, ns[1] / L, ns[2] / L
                cx, cy, cz = cs[0] / asum, cs[1] / asum, cs[2] / asum
                q.append(nb)
        rid += 1
    if raw:
        return region, adj
    # keep regions that are big enough and really flat: a curved patch has a wide normal spread
    # and its points bulge away from the best-fit plane
    area = np.bincount(region, fa, minlength=rid)
    nsum = np.zeros((rid, 3))
    np.add.at(nsum, region, fn * fa[:, None])
    rn = _unit(nsum + 1e-30)
    cen = np.stack([np.bincount(region, fa * fc[:, k], minlength=rid) for k in range(3)], 1) / np.maximum(area, 1e-30)[:, None]
    ang = np.arccos(np.clip(np.einsum("ij,ij->i", fn, rn[region]), -1, 1))
    spread = np.degrees(np.sqrt(np.bincount(region, fa * ang ** 2, minlength=rid) / np.maximum(area, 1e-30)))
    dist = np.einsum("ij,ij->i", fc - cen[region], rn[region])
    rms = np.sqrt(np.bincount(region, fa * dist ** 2, minlength=rid) / np.maximum(area, 1e-30))
    flat = (spread <= max_spread) & (rms <= 0.7 * tol)
    # flat pieces that lie on one plane (within group_tol) count together: a facade cut into strips
    # by pilasters, or a floor split by furniture, is one flat area
    group = np.arange(rid)
    if group_tol:
        cand = np.flatnonzero(flat & (area >= 1e-5 * fa.sum()))
        if len(cand) > 4000:
            cand = cand[np.argsort(-area[cand])[:4000]]
        n, c = rn[cand], cen[cand]
        d = np.einsum("ij,ij->i", n, c)
        near = (n @ n.T > math.cos(math.radians(5.0)))
        near &= np.abs(c @ n.T - d[None, :]) <= group_tol
        near &= near.T
        ii, jj = np.nonzero(np.triu(near, 1))
        parent = np.arange(rid)

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        for a, b in zip(cand[ii], cand[jj]):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra
        group = np.array([find(i) for i in range(rid)])
    garea = np.bincount(group, area * flat, minlength=rid)
    ok = flat & (garea[group] >= min_area_frac * fa.sum())
    _, gid = np.unique(np.where(ok, group, -1), return_inverse=True)
    new_id = np.where(ok, gid - (0 if (~ok).sum() == 0 else 1), -1)
    return new_id[region], adj


def fit_planes(V, F, region, up=None, reg_angle=4.0, skip=None):
    """Least-squares plane per region (triangles in `skip` ignored); near-vertical walls made exactly
    vertical and near-level surfaces exactly level (relative to `up`)."""
    planes = []
    fn, fa = face_normals_areas(V, F)
    up = None if up is None else _unit(np.asarray(up, float))
    cr, sr = math.cos(math.radians(reg_angle)), math.sin(math.radians(reg_angle))
    if skip is not None:
        region = np.where(skip, -1, region)
    order = np.argsort(region, kind="stable")
    bounds = np.searchsorted(region[order], np.arange(region.max() + 2))
    for r in range(region.max() + 1):
        fi = order[bounds[r]:bounds[r + 1]]
        p = V[np.unique(F[fi])]
        w = fa[fi]
        c = (V[F[fi]].mean(1) * w[:, None]).sum(0) / w.sum()
        n = np.linalg.svd(p - c, full_matrices=False)[2][-1]
        if n @ (fn[fi] * w[:, None]).sum(0) < 0:
            n = -n
        if up is not None:
            d = n @ up
            if abs(d) > cr:
                n = up * np.sign(d)
            elif abs(d) < sr:
                n = _unit(n - d * up)
        planes.append([n, float(n @ c), c])
    return planes


def merge_coplanar(V, F, region, planes, tol, angle=5.0, skip=None):
    """Join flat areas that lie on one plane (a wall split in two by noise, by a pilaster or a
    window column), so they become exactly coplanar. Triangles in `skip` are not looked at."""
    nr = int(region.max() + 1)
    if nr < 2:
        return region
    full = region
    if skip is not None:
        region = np.where(skip, -1, region)
    fc = V[F].mean(1)
    fa = face_normals_areas(V, F)[1]
    order = np.argsort(region, kind="stable")
    bounds = np.searchsorted(region[order], np.arange(nr + 1))
    members = [order[bounds[r]:bounds[r + 1]] for r in range(nr)]
    N = np.array([p[0] for p in planes])
    D = np.array([p[1] for p in planes])
    C = np.array([p[2] for p in planes])
    area = np.array([fa[m].sum() for m in members])
    cos_a = math.cos(math.radians(angle))
    near = (N @ N.T > cos_a) & (np.abs(C @ N.T - D[None, :]) < 6 * tol)
    near &= near.T
    ii, jj = np.nonzero(np.triu(near, 1))
    parent = np.arange(nr)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    rng = np.random.default_rng(0)


    def pts(r):
        m = members[r]
        if not len(m):   # only added triangles: use them all
            m = np.flatnonzero(full == r)
        if len(m) > 3000:
            m = rng.choice(m, 3000, replace=False, p=fa[m] / fa[m].sum())
        return fc[m], fa[m]
    for k in np.argsort(-(area[ii] + area[jj])):
        a, b = ii[k], jj[k]
        if find(a) == find(b):
            continue
        pa, wa = pts(a)
        pb, wb = pts(b)
        P = np.concatenate([pa, pb])
        w = np.concatenate([wa / wa.sum(), wb / wb.sum()])
        c = (P * w[:, None]).sum(0) / w.sum()
        n = np.linalg.svd((P - c) * np.sqrt(w)[:, None], full_matrices=False)[2][-1]
        d = np.abs((P - c) @ n)
        if d.max() <= 3.0 * tol and np.sqrt((w * d ** 2).sum() / w.sum()) <= 1.0 * tol:
            parent[find(b)] = find(a)
    roots = np.array([find(i) for i in range(nr)])
    _, newid = np.unique(roots, return_inverse=True)
    out = full.copy()
    out[full >= 0] = newid[full[full >= 0]]
    return out


def merge_relief(V, F, region, planes, relief, angle=3.0):
    """Parallel flat areas whose planes are less than `relief` apart become one (a window pane set
    back a little, a panel, a wall a few cm off): the smaller ones go onto the biggest one's plane."""
    nr = int(region.max() + 1)
    if nr < 2:
        return region, planes
    fa = face_normals_areas(V, F)[1]
    area = np.bincount(np.maximum(region, 0), fa * (region >= 0), minlength=nr)
    N = np.array([p[0] for p in planes])
    D = np.array([p[1] for p in planes])
    C = np.array([p[2] for p in planes])
    root = -np.ones(nr, np.int64)
    cos_a = math.cos(math.radians(angle))
    for i in np.argsort(-area):
        if root[i] >= 0:
            continue
        root[i] = i
        cand = np.flatnonzero((root < 0) & (N @ N[i] > cos_a) & (np.abs(C @ N[i] - D[i]) <= relief))
        root[cand] = i
    keep, newid = np.unique(root, return_inverse=True)
    out = region.copy()
    out[region >= 0] = newid[region[region >= 0]]
    return out, [planes[k] for k in keep]


def grow_relief(V, F, region, planes, adj, relief, big_frac=0.002, away=-0.3):
    """Big flat areas take in every neighbouring triangle that stays within `relief` of their plane:
    frames, panels, bumps, shallow recesses and their side walls. Taller details are kept whole.
    Triangles facing away from the plane (the back of a thin board) are never taken."""
    if region.max() < 0:
        return region, np.zeros(len(F), bool)
    fn, fa = face_normals_areas(V, F)
    nr = region.max() + 1
    rarea = np.bincount(np.maximum(region, 0), fa * (region >= 0), minlength=nr)
    big = rarea >= big_frac * fa.sum()
    tri = V[F]
    N = np.array([p[0] for p in planes])
    D = np.array([p[1] for p in planes])
    region = region.copy()
    grown = np.zeros(len(F), bool)
    a, b = adj[:, 0], adj[:, 1]
    for _ in range(1000):
        ra, rb = region[a], region[b]
        m1 = (ra >= 0) & (rb < 0) & big[np.maximum(ra, 0)]
        m2 = (rb >= 0) & (ra < 0) & big[np.maximum(rb, 0)]
        f = np.concatenate([b[m1], a[m2]])
        r = np.concatenate([ra[m1], rb[m2]])
        if not len(f):
            break
        dist = np.abs(np.einsum("ijk,ik->ij", tri[f], N[r]) - D[r][:, None]).max(1)
        o = np.lexsort((dist, f))                      # reachable from several: the closest plane
        f, r, dist = f[o], r[o], dist[o]
        first = np.r_[True, f[1:] != f[:-1]]
        f, r, dist = f[first], r[first], dist[first]
        ok = (dist <= relief) & (np.einsum("ij,ij->i", fn[f], N[r]) > away)
        if not ok.any():
            break
        region[f[ok]] = r[ok]
        grown[f[ok]] = True
    return region, grown


def snap(V, F, region, planes, max_move):
    """Move every vertex of a flat area onto its plane; vertices shared by two planes onto their
    intersection line, by three or more onto the corner point."""
    out = V.copy()
    sel = region[np.repeat(np.arange(len(F)), 3)] >= 0
    pv = F.ravel()[sel]
    pr = region[np.repeat(np.arange(len(F)), 3)][sel]
    pairs = np.unique(np.stack([pv, pr], 1), axis=0)
    cnt = np.bincount(pairs[:, 0], minlength=len(V))
    N = np.array([p[0] for p in planes])
    D = np.array([p[1] for p in planes])
    # one plane: project
    one = pairs[cnt[pairs[:, 0]] == 1]
    vi, ri = one[:, 0], one[:, 1]
    dist = np.einsum("ij,ij->i", V[vi], N[ri]) - D[ri]
    ok = np.abs(dist) <= max_move
    out[vi[ok]] = V[vi[ok]] - dist[ok, None] * N[ri[ok]]
    # several planes: damped least squares, pulled harder towards the original point if needed
    multi = pairs[cnt[pairs[:, 0]] > 1]
    if len(multi):
        o = np.argsort(multi[:, 0], kind="stable")
        multi = multi[o]
        starts = np.flatnonzero(np.r_[True, multi[1:, 0] != multi[:-1, 0]])
        ends = np.r_[starts[1:], len(multi)]
        for s, e in zip(starts, ends):
            i = multi[s, 0]
            rs = multi[s:e, 1]
            Nm, Dm, p, lam = N[rs], D[rs], V[i], 1e-4
            for _ in range(8):
                x = np.linalg.solve(Nm.T @ Nm + lam * np.eye(3), Nm.T @ Dm + lam * p)
                if np.linalg.norm(x - p) <= max_move:
                    out[i] = x
                    break
                lam *= 10
    return out


def smooth_snap(V, F, Vs, moved, rings=4):
    """Vertices in `moved` were moved from V to Vs (onto their planes). Spread that movement smoothly
    over the next `rings` rings of vertices (harmonic interpolation), so no shard sticks out where a
    flat area meets the rest."""
    from scipy.sparse import csr_matrix, diags
    from scipy.sparse.linalg import spsolve
    nV = len(V)
    e = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    e = np.concatenate([e, e[:, ::-1]])
    A = csr_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(nV, nV))
    A.data[:] = 1.0
    ring = np.full(nV, -1)
    ring[moved] = 0
    front = moved.copy()
    for k in range(1, rings + 1):
        new = ((A @ front.astype(float)) > 0) & (ring < 0)
        ring[new] = k
        front = new
    free = ring > 0
    if not free.any():
        return Vs
    idx = np.flatnonzero(free)
    fixed = np.flatnonzero(~free)
    disp = Vs - V
    disp[~moved] = 0.0
    Lff = diags(np.asarray(A[idx].sum(1)).ravel()) - A[idx][:, idx]
    sol = spsolve(Lff.tocsc(), A[idx][:, fixed] @ disp[fixed])
    out = Vs.copy()
    out[idx] = V[idx] + np.asarray(sol).reshape(-1, 3)
    return out


def smooth_free(V, F, region, iters=3, normalthr=40):
    """Feature-preserving denoise (two-step normal smoothing) of everything that is not flat: takes the
    crumple out of the generated surface, keeps edges."""
    pml = _pymeshlab()
    if pml is None or iters <= 0:
        return V
    ms = pml.MeshSet()
    ms.add_mesh(pml.Mesh(vertex_matrix=np.asarray(V, np.float64), face_matrix=np.asarray(F, np.int32),
                         f_scalar_array=region.astype(np.float64)))
    ms.compute_selection_by_condition_per_face(condselect="fq < -0.5")
    ms.compute_selection_transfer_face_to_vertex(inclusive=False)
    ms.apply_coord_two_steps_smoothing(stepsmoothnum=int(iters), normalthr=normalthr, stepnormalnum=20,
                                       stepfitnum=20, selected=True)
    return ms.current_mesh().vertex_matrix().astype(np.float64)


def label_flat_faces(V, F, planes, tol, angle=2.0):
    """Which final triangles lie on one of the flat areas' planes (-1 = none)."""
    lab = -np.ones(len(F), np.int64)
    if not planes:
        return lab
    fn, _ = face_normals_areas(V, F)
    N = np.array([p[0] for p in planes])
    D = np.array([p[1] for p in planes])
    tri = V[F]
    best = np.full(len(F), np.inf)
    cos_a = math.cos(math.radians(angle))
    for i in range(0, len(N), 64):
        c = fn @ N[i:i + 64].T                                   # (F, k)
        d = np.abs(np.einsum("fvc,kc->fvk", tri, N[i:i + 64]) - D[i:i + 64]).max(1)
        d = np.where(c > cos_a, d, np.inf)
        k = np.argmin(d, 1)
        dk = d[np.arange(len(F)), k]
        upd = (dk < best) & (dk <= tol)
        lab[upd], best[upd] = i + k[upd], dk[upd]
    return lab

# ============================================================================ flat rebuild (-q planar)
# Generated walls are never exactly flat (the generator's grid leaves a ripple of 1-2 grid steps), so a
# plain reduction keeps thousands of triangles on a wall to follow that ripple, and every point along the
# edge of a flat area survives (fans). Here the flat areas are found with a tolerance that covers the
# ripple, and each one is rebuilt from its outline: outline put exactly onto the plane (edges between two
# planes exactly straight), simplified, filled with the fewest triangles. Only the curved rest is reduced
# by edge collapse, with the outlines locked, so the model stays closed.

def _edge_table(F, nV):
    """Undirected edges. Returns (edges (E,2), half-edge -> edge id (3F,), faces per edge as CSR (cnt, ptr, ef)).
    Half-edge h = k * nF + f is the edge from F[f, k] to F[f, (k + 1) % 3]."""
    nF = len(F)
    E = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    S = np.sort(E, 1)
    key = S[:, 0].astype(np.int64) * nV + S[:, 1]
    uk, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
    inv = inv.reshape(-1)
    edges = np.stack([uk // nV, uk % nV], 1)
    order = np.argsort(inv, kind="stable")
    ptr = np.r_[0, np.cumsum(cnt)]
    ef = np.tile(np.arange(nF), 3)[order]
    return edges, inv, cnt, ptr, ef


def _dp_keep(P, tol):
    """Douglas-Peucker on a polyline (closed when the ends are the same point): which points to keep."""
    n = len(P)
    keep = np.zeros(n, bool)
    keep[0] = keep[-1] = True
    stack = [(0, n - 1)]
    while stack:
        i, j = stack.pop()
        if j <= i + 1:
            continue
        a, b = P[i], P[j]
        ab = b - a
        L2 = float(ab @ ab)
        Q = P[i + 1:j] - a
        if L2 < 1e-30:
            d = np.linalg.norm(Q, axis=1)
        else:
            t = np.clip(Q @ ab / L2, 0.0, 1.0)
            d = np.linalg.norm(Q - t[:, None] * ab, axis=1)
        k = int(np.argmax(d))
        if d[k] > tol:
            m = i + 1 + k
            keep[m] = True
            stack += [(i, m), (m, j)]
    return keep


def _inside(Pc, P2, segs):
    """Even-odd test: which of the points Pc lie inside the polygon given by the segments."""
    A, B = P2[segs[:, 0]], P2[segs[:, 1]]
    out = np.zeros(len(Pc), bool)
    for i in range(0, len(Pc), max(1, 4_000_000 // max(len(segs), 1))):
        p = Pc[i:i + max(1, 4_000_000 // max(len(segs), 1))]
        py, px = p[:, 1:2], p[:, 0:1]
        cond = (A[None, :, 1] > py) != (B[None, :, 1] > py)
        dy = B[None, :, 1] - A[None, :, 1]
        x = A[None, :, 0] + (py - A[None, :, 1]) * (B[None, :, 0] - A[None, :, 0]) / np.where(dy == 0, 1, dy)
        out[i:i + len(p)] = ((cond & (px < x)).sum(1) % 2) == 1
    return out


def _plane_basis(n):
    n = _unit(np.asarray(n, float))
    a = np.array([1.0, 0, 0]) if abs(n[0]) < 0.9 else np.array([0, 1.0, 0])
    u = _unit(np.cross(n, a))
    return u, np.cross(n, u)


def cdt_polygon(V, loops, n, eps):
    """Fewest-triangle fill of one flat area: its outline loops (outer counter-clockwise, holes clockwise,
    seen along the plane normal n; a loop may pass a point twice where the outline touches itself)
    triangulated without adding any point (constrained Delaunay). Returns the triangles as vertex indices,
    facing n, or None when the outline is not a clean polygon (it crosses itself, ...)."""
    import triangle as tr
    if not loops or any(len(L) < 3 for L in loops):
        return None
    allv = np.concatenate(loops)
    idx, inv = np.unique(allv, return_inverse=True)
    u, w = _plane_basis(n)
    P2 = np.stack([V[idx] @ u, V[idx] @ w], 1)
    q = np.round((P2 - P2.min(0)) / eps).astype(np.int64)
    if len(np.unique(q, axis=0)) != len(q):            # two outline points on top of each other
        return None
    segs, off, A = [], 0, 0.0
    for L in loops:
        s = inv[off:off + len(L)]
        segs.append(np.stack([s, np.roll(s, -1)], 1))
        x, y = P2[s, 0], P2[s, 1]
        A += 0.5 * float((x * np.roll(y, -1) - np.roll(x, -1) * y).sum())
        off += len(L)
    segs = np.concatenate(segs)
    if A <= 0:
        return None
    try:
        t = tr.triangulate({"vertices": P2, "segments": segs}, "pcQ")
    except Exception:
        return None
    if len(t.get("vertices", [])) != len(P2) or "triangles" not in t:
        return None      # Triangle had to add points
    T = np.asarray(t["triangles"], np.int64)
    T = T[_inside(P2[T].mean(1), P2, segs)]
    if not len(T):
        return None
    a, b, c = P2[T[:, 0]], P2[T[:, 1]], P2[T[:, 2]]
    ar = 0.5 * ((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (c[:, 0] - a[:, 0]) * (b[:, 1] - a[:, 1]))
    T[ar < 0] = T[ar < 0][:, ::-1]
    if abs(np.abs(ar).sum() - A) > 1e-6 * A + eps * eps:
        return None
    return idx[T]


def _region_loops(F, faces, in_reg, he_e, cnt, ptr, ef, nF, flip=None, owners=None):
    """Outline loops (directed like the faces) of one connected flat area, or None when it has a
    non-manifold edge. Where the outline touches itself at a point, the loops are walked around the
    point face by face, so each loop turns the right way there."""
    fset = set(faces.tolist())
    if flip is None:
        flip = np.zeros(len(F), bool)
    bnd = {}                                           # boundary half-edge (a, b) -> face
    for f in faces.tolist():
        for k in range(3):
            e = he_e[_KMAP[int(flip[f])][k] * nF + f]
            c = cnt[e]
            if c > 2:
                return None
            if c == 2:
                g = ef[ptr[e]] if ef[ptr[e]] != f else ef[ptr[e] + 1]
                if in_reg[g]:
                    continue
            bnd[(int(F[f, k]), int(F[f, (k + 1) % 3]))] = f
    if not bnd:
        return []
    start_at = {}
    for (a, b) in bnd:
        start_at.setdefault(a, []).append(b)

    def nxt_of(a, b, f):
        """boundary half-edge after (a -> b) of face f: turn around b inside the area"""
        if len(start_at.get(b, ())) == 1:
            return b, start_at[b][0]
        g = f
        for _ in range(64):
            fv = F[g].tolist()
            k = fv.index(b)
            c = fv[(k + 1) % 3]                        # half-edge b -> c of face g
            if (b, c) in bnd and bnd[(b, c)] == g:
                return b, c
            e = he_e[_KMAP[int(flip[g])][k] * nF + g]
            if cnt[e] != 2:
                return None
            h = ef[ptr[e]] if ef[ptr[e]] != g else ef[ptr[e] + 1]
            if h not in fset:
                return None
            g = h
        return None

    loops, seen = [], set()
    for he in bnd:
        if he in seen:
            continue
        L, cur = [], he
        while cur not in seen:
            seen.add(cur)
            L.append(cur[0])
            nx = nxt_of(cur[0], cur[1], bnd[cur])
            if nx is None or nx not in bnd:
                return None
            cur = nx
        if cur != he:
            return None
        loops.append(np.array(L, np.int64))
    if owners is not None:
        owners.update(bnd)
    return loops


class _CollapseSide:
    """The faces that are not rebuilt (curved parts, flat pieces that could not be rebuilt): outline points
    the flat areas drop are removed here by half-edge collapses along the outline."""

    def __init__(self, V, F, size, cos_lim):
        self.V, self.F = V, F.copy()
        self.alive = np.ones(len(F), bool)
        self.vf = {}
        for f, (a, b, c) in enumerate(self.F.tolist()):
            for v in (a, b, c):
                self.vf.setdefault(v, set()).add(f)
        self.eps2 = (size * 1e-9) ** 2
        self.cos_lim = cos_lim
        self.to = {}

    def _normal(self, tri):
        p = self.V[tri]
        return np.cross(p[1] - p[0], p[2] - p[0])

    def has(self, v):
        return bool(self.vf.get(v))

    def collapse(self, r, k):
        fr, fk = self.vf.get(r, set()), self.vf.get(k, set())
        shared = fr & fk
        if len(shared) != 1:
            return False
        f0 = next(iter(shared))
        x = [v for v in self.F[f0].tolist() if v != r and v != k][0]
        nr = {v for f in fr for v in self.F[f].tolist()} - {r}
        nk = {v for f in fk for v in self.F[f].tolist()} - {k}
        if (nr & nk) != {x}:                           # link condition: would make a non-manifold edge
            return False
        new = []
        for f in fr:
            if f == f0:
                continue
            tri = self.F[f].copy()
            n0 = self._normal(tri)
            tri[tri == r] = k
            n1 = self._normal(tri)
            l0, l1 = float(n0 @ n0), float(n1 @ n1)
            if l1 <= self.eps2 * 4:
                return False                           # would be degenerate
            if l0 > self.eps2 * 4 and float(n0 @ n1) < self.cos_lim * math.sqrt(l0 * l1):
                return False                           # would flip / fold
            new.append((f, tri))
        for f, tri in new:
            self.F[f] = tri
            fk.add(f)
        self.alive[f0] = False
        for v in self.F[f0].tolist():
            if v != r:
                self.vf[v].discard(f0)
        self.vf[r] = set()
        self.to[r] = k
        return True

    def final(self, v):
        path = []
        while v in self.to:
            path.append(v)
            v = self.to[v]
        for p in path:
            self.to[p] = v
        return v


def _snap_points(V, vi, labs_of, planes, max_move):
    """Target positions of outline points: on their plane (one plane), on the line where two planes meet,
    or the corner of three or more (damped least squares, never more than max_move away)."""
    out = V.copy()
    N = np.array([p[0] for p in planes])
    D = np.array([p[1] for p in planes])
    for v, ls in zip(vi.tolist(), labs_of):
        p = V[v]
        if len(ls) == 1:
            n = N[ls[0]]
            q = p - (p @ n - D[ls[0]]) * n
            if np.linalg.norm(q - p) <= max_move:
                out[v] = q
            continue
        Nm, Dm, lam = N[ls], D[ls], 1e-4
        if np.linalg.svd(Nm, compute_uv=False)[1] < math.sin(math.radians(25)):
            q = p.copy()                               # (nearly) parallel planes: onto their average
            for _ in range(3):
                q = q - ((Nm @ q - Dm)[:, None] * Nm).mean(0)
            if np.linalg.norm(q - p) <= max_move:
                out[v] = q
            continue
        for _ in range(8):
            x = np.linalg.solve(Nm.T @ Nm + lam * np.eye(3), Nm.T @ Dm + lam * p)
            if np.linalg.norm(x - p) <= max_move:
                out[v] = x
                break
            lam *= 10
    return out


_KMAP = ((0, 1, 2), (1, 0, 2))      # half-edge k of a face wound the other way is half-edge _KMAP[1][k] of the original


def _crossing_halfedges(V, loops, n):
    """Outline segments (as (a, b) vertex pairs) that cross or touch another, non-adjacent segment in the
    plane's projection."""
    from shapely.geometry import LineString
    from shapely.strtree import STRtree
    u, w = _plane_basis(n)
    segs = []
    for L in loops:
        for i in range(len(L)):
            segs.append((int(L[i]), int(L[(i + 1) % len(L)])))
    if len(segs) < 4:
        return []
    P = {}
    for a, b in segs:
        for v in (a, b):
            if v not in P:
                P[v] = (float(V[v] @ u), float(V[v] @ w))
    geoms = [LineString([P[a], P[b]]) for a, b in segs]
    tree = STRtree(geoms)
    bad = set()
    for i, g in enumerate(geoms):
        for j in tree.query(g):
            j = int(j)
            if j <= i:
                continue
            if len({segs[i][0], segs[i][1]} & {segs[j][0], segs[j][1]}):
                continue
            if g.intersects(geoms[j]):
                bad.add(segs[i])
                bad.add(segs[j])
    return list(bad)


def _local_components(faces, he_e, cnt, ptr, ef, nF, flip):
    """Split a set of faces into edge-connected pieces."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    inset = np.zeros(nF, bool)
    inset[faces] = True
    pos = -np.ones(nF, np.int64)
    pos[faces] = np.arange(len(faces))
    e = np.concatenate([he_e[k * nF + faces] for k in range(3)])
    f = np.tile(faces, 3)
    m = cnt[e] == 2
    e, f = e[m], f[m]
    g = np.where(ef[ptr[e]] != f, ef[ptr[e]], ef[ptr[e] + 1])
    m = inset[g]
    A = coo_matrix((np.ones(int(m.sum())), (pos[f[m]], pos[g[m]])), shape=(len(faces), len(faces)))
    k, lab = connected_components(A, directed=False)
    return [faces[lab == i] for i in range(k)]


def _orient_component(F2, flip, faces, in_reg, he_e, cnt, ptr, ef, nF, V, n):
    """Wind the faces of one flat area consistently (breadth-first from its biggest face; generated meshes
    often have single flipped slivers), facing the plane normal n. Changes F2 in place; False when the area
    cannot be oriented (a Moebius-like fold)."""
    from collections import deque
    fl = faces.tolist()
    seen = {}
    area = np.linalg.norm(np.cross(V[F2[faces, 1]] - V[F2[faces, 0]], V[F2[faces, 2]] - V[F2[faces, 0]]), axis=1)
    start = int(faces[int(np.argmax(area))])
    seen[start] = True
    q = deque([start])
    while q:
        f = q.popleft()
        fv = F2[f]
        for k in range(3):
            e = he_e[_KMAP[int(flip[f])][k] * nF + f]
            if cnt[e] != 2:
                continue
            g = ef[ptr[e]] if ef[ptr[e]] != f else ef[ptr[e] + 1]
            if not in_reg[g]:
                continue
            a, b = fv[k], fv[(k + 1) % 3]
            gv = F2[g]
            kk = int(np.flatnonzero(gv == a)[0])
            agree = gv[(kk + 1) % 3] != b             # g runs the shared edge the other way: consistent
            if g in seen:
                continue                               # a fold inside: its inside is thrown away anyway
            if not agree:
                F2[g] = gv[::-1]
                flip[g] = not flip[g]
            seen[g] = True
            q.append(g)
    if len(seen) != len(fl):
        return False
    t = V[F2[faces]]
    nn = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
    if float((nn @ n).sum()) < 0:
        F2[faces] = F2[faces][:, ::-1]
        flip[faces] = ~flip[faces]
    return True


def rebuild_flat(V, F, lab, planes, tau, size, max_move, chain_tol=None, cos_free=0.5, rings=3, peel_deg=35.0,
                 absorb=0.0):
    """Flat areas (lab >= 0, one plane each) rebuilt from their outline with the fewest triangles.
    Works on the dense (not yet reduced) mesh, before anything is snapped, so the outlines are clean:
      - outline of every connected flat area that forms a clean polygon (others stay as they are)
      - outline points put onto their plane / the line where two planes meet / the corner
        (the movement spread over `rings` rings of the curved neighbours, so nothing folds there)
      - every stretch of outline between two corners simplified (Douglas-Peucker, chain_tol) - once,
        for both sides, so the model stays closed
      - the flat areas filled by a constrained Delaunay triangulation of the simplified outline (no inner
        points); on the curved side the points left out are collapsed into their neighbours along the outline
      - steep or flipped slivers on an area's edge are peeled off first, small bumps inside it (within
        `absorb` of the plane) taken in; an outline that still crosses itself is repaired, a simplified one
        that would cross is simplified less there
    Returns (V, F, label per face (-1 = curved / kept), stats)."""
    import time as _t
    t0 = _t.time()
    chain_tol = tau if chain_tol is None else chain_tol
    nV, nF = len(V), len(F)
    eps = size * 1e-9
    edges, he_e, cnt, ptr, ef = _edge_table(F, nV)
    st = dict(in_tris=nF)

    # 1. connected flat areas with a clean outline -> rebuilt ("regions"); everything else stays
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    lab = lab.copy()
    nmf = np.zeros(nF, bool)                           # faces at non-manifold edges: never rebuilt
    for e in np.flatnonzero(cnt > 2).tolist():
        nmf[ef[ptr[e]:ptr[e + 1]]] = True
    lab[nmf] = -1
    # steep or flipped slivers on the edge of a flat area fold its outline: peel them off (inside the area
    # they do no harm - the inside is thrown away)
    Np = np.array([p[0] for p in planes])
    cr = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    ln = np.linalg.norm(cr, axis=1)
    steep = (lab >= 0) & ((ln < 1e-12 * size * size) |
                          (np.einsum("ij,ij->i", cr, Np[np.maximum(lab, 0)]) < math.cos(math.radians(peel_deg)) * ln))
    two = cnt == 2
    fa, fb = ef[ptr[:-1][two]], ef[ptr[:-1][two] + 1]
    bf = np.zeros(nF, bool)
    bf[ef[ptr[:-1][cnt == 1]]] = True
    for _ in range(12):
        diff = lab[fa] != lab[fb]
        onb = bf.copy()
        onb[fa[diff]] = True
        onb[fb[diff]] = True
        peel = steep & onb & (lab >= 0)
        if not peel.any():
            break
        lab[peel] = -1
    # small bumps and dents inside a flat area (curved islands that touch only that area and stay within
    # `absorb` of its plane): part of the area - the normal map keeps their look
    n_abs = 0
    if absorb > 0:
        Dp = np.array([p[1] for p in planes])
        for _ in range(3):
            free = (lab < 0) & ~nmf
            fr2 = free[fa] & free[fb]
            g = coo_matrix((np.ones(int(fr2.sum())), (fa[fr2], fb[fr2])), shape=(nF, nF))
            _, fc = connected_components(g, directed=False)
            fc = np.where(free, fc, -1)
            mix = free[fa] != free[fb]
            isl = np.where(free[fa[mix]], fc[fa[mix]], fc[fb[mix]])
            nb = np.where(free[fa[mix]], lab[fb[mix]], lab[fa[mix]])
            pr = np.unique(np.stack([isl, nb], 1), axis=0)
            cnt_l = np.bincount(pr[:, 0], minlength=nF)
            touch_border = np.zeros(nF, bool)
            touch_border[fc[bf & free]] = True
            near_nm = (lab < 0) & nmf                  # touching a non-manifold spot: leave it alone
            nmix = (near_nm[fa] & free[fb]) | (near_nm[fb] & free[fa])
            touch_border[fc[np.where(near_nm[fa[nmix]], fb[nmix], fa[nmix])]] = True
            cand = np.flatnonzero((cnt_l == 1) & ~touch_border)
            if not len(cand):
                break
            target = -np.ones(nF, np.int64)
            target[pr[:, 0]] = pr[:, 1]
            changed = False
            fo = np.argsort(fc, kind="stable")
            fcs = fc[fo]
            for c in cand.tolist():
                lo_, hi_ = np.searchsorted(fcs, c), np.searchsorted(fcs, c, side="right")
                fs = fo[lo_:hi_]
                L_ = int(target[c])
                vv = np.unique(F[fs])
                if np.abs(V[vv] @ Np[L_] - Dp[L_]).max() <= absorb:
                    lab[fs] = L_
                    n_abs += 1
                    changed = True
            if not changed:
                break
    st["absorbed_islands"] = n_abs
    same = (lab[fa] >= 0) & (lab[fa] == lab[fb])
    g = coo_matrix((np.ones(int(same.sum())), (fa[same], fb[same])), shape=(nF, nF))
    _, comp = connected_components(g, directed=False)
    comp = np.where(lab >= 0, comp, -1)
    group = -np.ones(nF, np.int64)
    F2 = F.copy()
    flip = np.zeros(nF, bool)
    regions = []
    order = np.argsort(comp, kind="stable")
    cs = comp[order]
    starts = np.flatnonzero(np.r_[True, cs[1:] != cs[:-1]])
    ends = np.r_[starts[1:], len(cs)]
    in_reg = np.zeros(nF, bool)
    n_unclean = n_fixed = 0
    queue = [order[s:e] for s, e in zip(starts, ends) if cs[s] >= 0]
    tries = {}
    while queue:
        faces = queue.pop()
        L_ = int(lab[faces[0]])
        n = planes[L_][0]
        in_reg[faces] = True
        owners = {}
        ok = _orient_component(F2, flip, faces, in_reg, he_e, cnt, ptr, ef, nF, V, n)
        loops = _region_loops(F2, faces, in_reg, he_e, cnt, ptr, ef, nF, flip, owners) if ok else None
        in_reg[faces] = False
        if not loops:
            n_unclean += 1
            continue
        if cdt_polygon(V, loops, n, eps) is not None:
            group[faces] = len(regions)
            regions.append(dict(label=L_, faces=faces, loops=loops))
            continue
        key = int(faces.min())
        tries[key] = tries.get(key, 0) + 1
        bad = _crossing_halfedges(V, loops, n)
        if not bad or tries[key] > 6:
            n_unclean += 1
            continue
        # drop the faces on the crossing bits of outline (and their neighbours in the area), then again
        drop = {owners[h] for h in bad if h in owners}
        inset = np.zeros(nF, bool)
        inset[faces] = True
        for f in list(drop):
            for k in range(3):
                e = he_e[k * nF + f]
                if cnt[e] == 2:
                    g2 = ef[ptr[e]] if ef[ptr[e]] != f else ef[ptr[e] + 1]
                    if inset[g2]:
                        drop.add(int(g2))
        drop = np.array(sorted(drop), np.int64)
        lab[drop] = -1
        rest = faces[~np.isin(faces, drop)]
        n_fixed += 1
        if len(rest):
            for piece in _local_components(rest, he_e, cnt, ptr, ef, nF, flip):
                tries[int(piece.min())] = tries[key]
                queue.append(piece)
    st["regions"], st["unclean"], st["outline_repairs"] = len(regions), n_unclean, n_fixed

    # 2. outline points onto their planes; corners
    BORDER = -2
    f0 = ef[ptr[:-1]]
    f1 = np.where(cnt >= 2, ef[np.minimum(ptr[:-1] + 1, len(ef) - 1)], -1)
    g0 = group[f0]
    g1 = np.where(cnt >= 2, group[np.maximum(f1, 0)], BORDER)
    bnd = (cnt != 2) | (g0 != g1)
    be = np.flatnonzero(bnd)
    deg = np.bincount(edges[be].ravel(), minlength=nV)
    nm = np.zeros(nV, bool)
    nm[edges[cnt > 2].ravel()] = True
    pairs = np.unique(np.stack([F.ravel(), np.repeat(group, 3)], 1), axis=0)
    ngrp = np.bincount(pairs[:, 0], minlength=nV)
    on_border = np.zeros(nV, bool)
    on_border[edges[cnt == 1].ravel()] = True
    ngrp = ngrp + on_border
    corner = (deg > 0) & ((deg != 2) | (ngrp >= 3) | nm)
    outline = deg > 0
    rp = pairs[pairs[:, 1] >= 0]
    rp = rp[outline[rp[:, 0]]]
    labs = np.array([regions[r]["label"] for r in rp[:, 1]], np.int64) if len(rp) else np.zeros(0, np.int64)
    o = np.argsort(rp[:, 0], kind="stable")
    rp, labs = rp[o], labs[o]
    vs = np.flatnonzero(np.r_[True, rp[1:, 0] != rp[:-1, 0]]) if len(rp) else np.zeros(0, np.int64)
    ve = np.r_[vs[1:], len(rp)]
    vi = rp[vs, 0] if len(rp) else np.zeros(0, np.int64)
    labs_of = [sorted(set(labs[a:b].tolist())) for a, b in zip(vs, ve)]
    Vp = _snap_points(V, vi, labs_of, planes, max_move)
    multi = np.zeros(nV, bool)
    multi[vi[np.array([len(x) > 1 for x in labs_of], bool)]] = True if len(vi) else False
    vreg = {}
    for ri, R in enumerate(regions):
        for L in R["loops"]:
            for v in L.tolist():
                vreg.setdefault(v, set()).add(ri)
    dirty, n_rev = set(range(len(regions))), 0
    for _ in range(10):
        nd = set()
        for ri in sorted(dirty):
            R = regions[ri]
            if cdt_polygon(Vp, R["loops"], planes[R["label"]][0], eps) is not None:
                continue
            back = np.concatenate([L[multi[L] & np.any(Vp[L] != V[L], axis=1)] for L in R["loops"]])
            if not len(back):
                continue
            Vp[back] = V[back]
            n_rev += 1
            for v in back.tolist():
                nd |= vreg.get(v, set())
        dirty = nd
        if not dirty:
            break
    st["snap_reverted"] = n_rev
    moved = np.zeros(nV, bool)
    moved[vi] = True
    if rings > 0 and len(vi):
        # spread the movement over the curved neighbours (flat-area insides are dropped anyway)
        side_f = F[group < 0]
        if len(side_f):
            used = np.unique(side_f)
            loc = -np.ones(nV, np.int64)
            loc[used] = np.arange(len(used))
            Vsub = _smooth_spread(V[used], loc[side_f], Vp[used], moved[used], rings)
            Vp[used] = Vsub
    st["outline_points"] = int(outline.sum())

    # 3. outline stretches between corners ("chains"), simplified
    inc = {}
    for e in be.tolist():
        a, b = edges[e]
        inc.setdefault(int(a), []).append(e)
        inc.setdefault(int(b), []).append(e)
    visited = np.zeros(len(edges), bool)
    chains = []

    def walk(start, e):
        seq, cur = [start], start
        while True:
            visited[e] = True
            a, b = edges[e]
            nxt = int(b) if a == cur else int(a)
            seq.append(nxt)
            if corner[nxt] or nxt == start:
                return seq
            es = inc[nxt]
            e2 = es[0] if es[0] != e else es[1]
            if visited[e2]:
                return seq
            e, cur = e2, nxt

    for c in np.flatnonzero(corner).tolist():
        for e in inc.get(c, []):
            if not visited[e]:
                chains.append((walk(c, e), e))
    for e in be.tolist():                              # closed loops without a corner
        if not visited[e]:
            chains.append((walk(int(edges[e, 0]), e), e))
    removed = np.zeros(nV, bool)
    ch_seq, ch_keep, ch_tol, ch_side = [], [], [], []
    seg_chain = {}
    for seq, e in chains:
        if len(seq) < 3:
            continue
        ch_seq.append(seq)
        ch_tol.append(chain_tol)
        ch_side.append(g0[e] == -1 or g1[e] == -1)
        ch_keep.append(None)

    def simplify(ci):
        seq = ch_seq[ci]
        P = Vp[seq]
        keep = _dp_keep(P, ch_tol[ci]) if ch_tol[ci] > 0 else np.ones(len(seq), bool)
        if seq[0] == seq[-1] and keep[:-1].sum() < 3:  # a closed loop needs 3 points at least
            far = int(np.argmax(np.linalg.norm(P - P[0], axis=1)))
            keep[far] = True
            ab = P[far] - P[0]
            d = np.linalg.norm(np.cross(P - P[0], ab), axis=1)
            d[keep] = -1
            keep[int(np.argmax(d))] = True
        ch_keep[ci] = keep
        sa = np.array(seq)
        removed[sa[1:-1]] = ~keep[1:-1]
        ks = sa[keep]
        for x, y in zip(ks[:-1].tolist(), ks[1:].tolist()):
            seg_chain[(x, y)] = ci
            seg_chain[(y, x)] = ci

    for ci in range(len(ch_seq)):
        simplify(ci)
    # every flat area must still be a clean polygon: where a simplified stretch crosses another part of the
    # outline, simplify those stretches less (halve their tolerance), down to keeping all their points
    vreg2 = {}
    for ri, R in enumerate(regions):
        for L in R["loops"]:
            for v in L.tolist():
                vreg2.setdefault(v, set()).add(ri)
    dirty, n_back = set(range(len(regions))), 0
    for _ in range(40):
        nd = set()
        for ri in sorted(dirty):
            R = regions[ri]
            loops = [L[~removed[L]] for L in R["loops"]]
            if cdt_polygon(Vp, loops, planes[R["label"]][0], eps) is not None:
                continue
            bad = _crossing_halfedges(Vp, loops, planes[R["label"]][0])
            cis = {seg_chain[h] for h in bad if h in seg_chain}
            if not cis:                                # no crossing found: keep every stretch of it whole
                cis = {seg_chain[(int(L[i]), int(L[(i + 1) % len(L)]))] for L in loops for i in range(len(L))
                       if (int(L[i]), int(L[(i + 1) % len(L)])) in seg_chain}
            for ci in cis:
                if ch_tol[ci] == 0:
                    continue
                ch_tol[ci] = ch_tol[ci] / 2 if ch_tol[ci] > chain_tol / 16 else 0.0
                simplify(ci)
                n_back += 1
                for v in ch_seq[ci]:
                    nd |= vreg2.get(v, set())
        dirty = nd
        if not dirty:
            break
    side_chains = [(ch_seq[ci], ch_keep[ci]) for ci in range(len(ch_seq)) if ch_side[ci]]
    st["dp_removed"] = int(removed.sum())
    st["dp_refined"] = n_back

    # 4. the kept side: collapse its removed outline points along the outline, from both ends of a run
    side = _CollapseSide(Vp, F[group < 0], size, cos_free)
    committed = np.zeros(nV, bool)
    n_fail = 0
    for seq, keep in side_chains:
        ki = np.flatnonzero(keep)
        for i, j in zip(ki[:-1], ki[1:]):
            run = list(range(i + 1, j))
            if not run:
                continue
            order_ = []
            lo, hi = 0, len(run) - 1
            while lo <= hi:
                order_.append((run[lo], -1))
                if hi != lo:
                    order_.append((run[hi], +1))
                lo += 1
                hi -= 1
            state = {p: True for p in range(i, j + 1)}

            def neighbour(p, d):
                q = p + d
                while i <= q <= j and not state[q]:
                    q += d
                return q

            for p, d in order_:
                r = seq[p]
                if not removed[r]:
                    continue
                if not side.has(r):
                    state[p] = False
                    committed[r] = True
                    continue
                done = False
                for dd in (d, -d):
                    q = neighbour(p, dd)
                    if i <= q <= j and side.collapse(r, seq[q]):
                        done = True
                        break
                if done:
                    state[p] = False
                    committed[r] = True
                else:
                    removed[r] = False
                    n_fail += 1
    st["collapse_kept"] = n_fail

    # 5. fill every flat area from its simplified outline. A point the curved side could not give up is back
    # in the outline and may make it cross: simplify the stretches next to it less (those that have no
    # curved side); an area that still cannot be filled keeps its own triangles, put onto its plane and
    # handed to the curved-part reduction
    vreg = {}
    for ri, R in enumerate(regions):
        for L in R["loops"]:
            for v in L.tolist():
                vreg.setdefault(v, set()).add(ri)
    reg_chains = {}
    for ci, seq in enumerate(ch_seq):
        for ri in (vreg.get(seq[1], set()) if len(seq) > 2 else set()):
            reg_chains.setdefault(ri, set()).add(ci)
    tris = [None] * len(regions)
    fallback = set()
    dirty = set(range(len(regions)))
    for _ in range(60):
        nd = set()
        for ri in sorted(dirty):
            if ri in fallback:
                continue
            R = regions[ri]
            loops = [L[~removed[L]] for L in R["loops"]]
            T = cdt_polygon(Vp, loops, planes[R["label"]][0], eps)
            tris[ri] = T
            if T is not None:
                continue
            bad = _crossing_halfedges(Vp, loops, planes[R["label"]][0])
            cis = {seg_chain[h] for h in bad if h in seg_chain}
            cis = {ci for ci in cis if not ch_side[ci] and ch_tol[ci] > 0}
            if not cis:
                # cannot be filled: it keeps its own triangles, so every outline point it shares with another
                # flat area must stay in that area's outline too (else there would be a crack)
                fallback.add(ri)
                cis = {ci for ci in reg_chains.get(ri, ()) if not ch_side[ci] and ch_tol[ci] > 0}
                for ci in cis:
                    ch_tol[ci] = 0.0
                    simplify(ci)
                    for v in ch_seq[ci]:
                        nd |= vreg.get(v, set())
                continue
            for ci in cis:
                ch_tol[ci] = ch_tol[ci] / 2 if ch_tol[ci] > chain_tol / 16 else 0.0
                simplify(ci)
                for v in ch_seq[ci]:
                    nd |= vreg.get(v, set())
        dirty = nd - fallback
        if not dirty:
            break
    for ri in fallback:
        tris[ri] = None
    out_tris, out_lab, n_fb, fb_tris = [], [], 0, 0
    fb_faces = []
    for ri, R in enumerate(regions):
        T = tris[ri]
        if T is None:
            n_fb += 1
            T = F[R["faces"]].copy()
            for k in range(3):
                T[:, k] = [side.final(v) for v in T[:, k].tolist()]
            T = T[(T[:, 0] != T[:, 1]) & (T[:, 1] != T[:, 2]) & (T[:, 0] != T[:, 2])]
            inner = np.setdiff1d(np.unique(T), np.flatnonzero(outline))
            n_, d_ = planes[R["label"]][0], planes[R["label"]][1]
            Vp[inner] = Vp[inner] - (Vp[inner] @ n_ - d_)[:, None] * n_
            fb_faces.append(T)
            fb_tris += len(T)
            continue
        out_tris.append(T)
        out_lab.append(np.full(len(T), R["label"]))
    st["fallback"] = n_fb
    st["fallback_tris"] = fb_tris
    keep_side = np.concatenate([side.F[side.alive]] + fb_faces)
    Fn = np.concatenate(out_tris + [keep_side])
    labn = np.concatenate(out_lab + [np.full(len(keep_side), -1)])
    # a curved-side triangle squeezed into a corner of a flat area can end up on the same three points as one
    # of the area's new triangles: a zero-thickness fin - both go (the surface around closes over it)
    key = np.sort(Fn, 1)
    _, first, inv_, cnt_ = np.unique(key, axis=0, return_index=True, return_inverse=True, return_counts=True)
    inv_ = inv_.reshape(-1)
    dup = cnt_[inv_] > 1
    if dup.any():
        drop = np.zeros(len(Fn), bool)
        for g_ in np.unique(inv_[dup]).tolist():
            idx_ = np.flatnonzero(inv_ == g_)
            if len(idx_) == 2:                      # opposite pair (a fin) or the same face twice
                a_, b_ = Fn[idx_[0]], Fn[idx_[1]]
                k_ = int(np.flatnonzero(b_ == a_[0])[0])
                same_dir = b_[(k_ + 1) % 3] == a_[1]
                drop[idx_[1:] if same_dir else idx_] = True
            else:
                drop[idx_[1:]] = True
        Fn, labn = Fn[~drop], labn[~drop]
        st["fins_removed"] = int(drop.sum())
    used, inv = np.unique(Fn, return_inverse=True)
    Vn, Fn = Vp[used], inv.reshape(-1, 3)
    st["tris"] = len(Fn)
    st["seconds"] = round(_t.time() - t0, 1)
    return Vn, Fn, labn, st


def _smooth_spread(V, F, Vs, moved, rings):
    """Vertices in `moved` go from V to Vs; their movement is spread smoothly over the next `rings` rings
    (harmonic), so the neighbouring triangles do not fold."""
    from scipy.sparse import csr_matrix, diags
    from scipy.sparse.linalg import spsolve
    nV = len(V)
    e = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    e = np.concatenate([e, e[:, ::-1]])
    A = csr_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(nV, nV))
    A.data[:] = 1.0
    ring = np.full(nV, -1)
    ring[moved] = 0
    front = moved.copy()
    for k in range(1, rings + 1):
        new = ((A @ front.astype(float)) > 0) & (ring < 0)
        ring[new] = k
        front = new
    free = ring > 0
    out = np.where(moved[:, None], Vs, V)
    if not free.any():
        return out
    idx = np.flatnonzero(free)
    fixed = np.flatnonzero(~free)
    disp = out - V
    Lff = diags(np.asarray(A[idx].sum(1)).ravel()) - A[idx][:, idx]
    sol = spsolve(Lff.tocsc(), A[idx][:, fixed] @ disp[fixed])
    out[idx] = V[idx] + np.asarray(sol).reshape(-1, 3)
    return out


def find_flat(Vm, Fm, angle, tol, relief, min_plane, up=(0, 1, 0), merge_parallel=False, big_frac=0.0002):
    """Flat areas for the rebuild: found with a tolerance that covers the generator's noise, shallow relief
    taken in, then every face checked against its area's plane (and the plane refit) until each one is
    close enough to be put exactly onto it. Returns (region per face, planes, grown)."""
    region, adj = segment_planes(Vm, Fm, angle, tol, min_plane, max_spread=0.35 * angle, group_tol=tol)
    if region.max() < 0:
        return region, [], np.zeros(len(Fm), bool)
    planes = fit_planes(Vm, Fm, region, up=up)
    if relief > 0:
        if merge_parallel:
            region, planes = merge_relief(Vm, Fm, region, planes, relief)
        region, grown = grow_relief(Vm, Fm, region, planes, adj, relief, big_frac=big_frac)
    else:
        grown = np.zeros(len(Fm), bool)
    region = merge_coplanar(Vm, Fm, region, planes, 0.5 * tol, skip=grown)
    planes = fit_planes(Vm, Fm, region, up=up, skip=grown)
    for _ in range(4):
        N = np.array([p[0] for p in planes])
        D = np.array([p[1] for p in planes])
        r = np.maximum(region, 0)
        dev = np.abs(np.einsum("fkc,fc->fk", Vm[Fm], N[r]) - D[r][:, None]).max(1)
        bad = (region >= 0) & (dev > np.where(grown, relief + tol, 1.5 * tol))
        if not bad.any():
            break
        region = np.where(bad, -1, region)
        keep = np.unique(region[region >= 0])
        remap = -np.ones(len(planes), np.int64)
        remap[keep] = np.arange(len(keep))
        region = np.where(region >= 0, remap[np.maximum(region, 0)], -1)
        if region.max() < 0:
            return region, [], grown
        planes = fit_planes(Vm, Fm, region, up=up, skip=grown)
    return region, planes, grown


def planar_lowpoly(Vm, Fm, cfg, size, tol, relief, tau, faces, min_faces, up, mpu):
    """-q planar, steps 4 + 5: flat areas rebuilt from their outlines, the curved rest reduced to within tau
    (outlines locked). Returns (V, F, flat-area label per triangle (-1 = curved), planes, stats, tau)."""
    t0 = time.time()
    k, unit = (mpu * 100, "cm") if mpu else (1024 / size, "grid steps")
    region, planes, _ = find_flat(Vm, Fm, cfg.angle, tol, relief, cfg.min_plane, up=up)
    if not planes:
        log("flat areas: none found - reducing as usual")
        Vl, Fl, st, _ = reduce_to_error(Vm, Fm, tau, faces, min_faces, bad_frac=getattr(cfg, "bad_frac", BAD_FRAC))
        return Vl, Fl, None, [], st, tau
    Vr, Fr, labr, rs = rebuild_flat(Vm, Fm, region, planes, tau, size, max_move=relief + 1.5 * tol,
                                    chain_tol=0.75 * tol, absorb=2 * tol)
    far = face_normals_areas(Vr, Fr)[1]
    log(f"flat areas: {len(planes)} planes; {rs['regions']} areas rebuilt from their outlines = "
        f"{far[labr >= 0].sum() / far.sum() * 100:.0f}% of the surface in {(labr >= 0).sum():,} triangles "
        f"(flat within {tol * k:.0f} {unit}, relief up to {relief * k:.0f} {unit} pressed flat; "
        f"{rs['outline_repairs']} outlines repaired, {rs['unclean'] + rs['fallback']} small areas kept as they were) "
        f"({time.time() - t0:.0f}s)")
    if cfg.smooth > 0:
        Vr = smooth_free(Vr, Fr, labr, iters=cfg.smooth)
    t0 = time.time()
    Vl, Fl, st, flat = reduce_to_error(Vr, Fr, tau, faces, min_faces, bad_frac=getattr(cfg, "bad_frac", BAD_FRAC),
                                       labels=labr, only_free=True)
    if flat is None:
        flat = labr
    log(f"curved parts: {(labr < 0).sum():,} -> {(flat < 0).sum():,} triangles, within {tau * k:.1f} {unit} "
        f"({time.time() - t0:.0f}s)")
    # honest check against the detailed surface, both ways
    P, _, _ = area_samples(Vm, Fm, 200_000, seed=3)
    d1 = dist_to_mesh(P, Vl, Fl)
    P2, _, _ = area_samples(Vl, Fl, 100_000, seed=4)
    d2 = dist_to_mesh(P2, Vm, Fm)
    d = np.concatenate([d1, d2])
    st = dict(st or {}, p50=float(np.median(d)), p99=float(np.percentile(d, 99)), max=float(d.max()))
    o0, n0, _ = edge_stats(Fm)
    o1, n1, _ = edge_stats(Fl)
    log(f"low-poly: {len(Fl):,} triangles ({(flat >= 0).sum():,} flat, {(flat < 0).sum():,} curved); "
        f"vs the detailed surface: half within {st['p50'] * k:.1f} {unit}, 99% within {st['p99'] * k:.1f} {unit}; "
        f"open edges {o1:,} (input {o0:,}), non-manifold {n1:,} (input {n0:,})")
    return Vl, Fl, flat, planes, st, tau


# ============================================================================ bake

def split_hard_edges(V, F, hard_angle, flat=None, planes=None):
    """Duplicate vertices along edges sharper than hard_angle and around flat areas (so a flat wall
    shades perfectly flat); smooth normals elsewhere."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    m = trimesh.Trimesh(V, F, process=False)
    adj = m.face_adjacency
    soft = m.face_adjacency_angles < math.radians(hard_angle)
    if flat is not None:
        la, lb = flat[adj[:, 0]], flat[adj[:, 1]]
        soft &= (la == lb)
    adj, edges = adj[soft], m.face_adjacency_edges[soft]
    nF = len(F)

    def corner(fidx, v):
        k = np.argmax(F[fidx] == v[:, None], axis=1)
        return fidx * 3 + k
    c = []
    for j in (0, 1):
        c.append(np.stack([corner(adj[:, 0], edges[:, j]), corner(adj[:, 1], edges[:, j])], 1))
    c = np.concatenate(c)
    g = coo_matrix((np.ones(len(c), np.int8), (c[:, 0], c[:, 1])), shape=(nF * 3, nF * 3))
    _, lab = connected_components(g, directed=False)
    # corners of the same original vertex in the same smooth group share one new vertex
    key = F.ravel().astype(np.int64) * (lab.max() + 1) + lab
    uk, inv = np.unique(key, return_inverse=True)
    Fs = inv.reshape(-1, 3)
    Vs = V[F.ravel()][np.unique(inv, return_index=True)[1]]
    fn, fa = face_normals_areas(V, F)
    Ns = np.zeros((len(uk), 3))
    np.add.at(Ns, inv, np.repeat(fn * fa[:, None], 3, axis=0))
    Ns = _unit(Ns + 1e-30)
    if flat is not None and planes:  # flat areas shade exactly flat
        vlab = np.full(len(uk), -1, np.int64)
        vlab[inv] = np.repeat(flat, 3)
        sel = vlab >= 0
        P = np.array([p[0] for p in planes])[vlab[sel]]
        Ns[sel] = P * np.sign(np.einsum("ij,ij->i", P, Ns[sel]) + 1e-12)[:, None]
    return Vs, Fs, Ns


def _morton_pieces(V, F, k):
    """Split the triangles into k compact pieces of equal size (Z-order of the triangle centres)."""
    c = V[F].mean(1)
    q = ((c - c.min(0)) / (np.ptp(c, 0).max() + 1e-12) * 1023).astype(np.int64)

    def spread(x):
        x = x & 0x3FF
        x = (x | (x << 16)) & 0x30000FF
        x = (x | (x << 8)) & 0x300F00F
        x = (x | (x << 4)) & 0x30C30C3
        return (x | (x << 2)) & 0x9249249
    code = spread(q[:, 0]) | (spread(q[:, 1]) << 1) | (spread(q[:, 2]) << 2)
    return np.array_split(np.argsort(code, kind="stable"), k)


def unwrap(V, F, N, tex, padding, piece=20_000):
    """xatlas UVs. A big mesh is cut into compact pieces of ~`piece` triangles that xatlas charts in
    parallel and packs into one atlas: 10x faster on 250k triangles, same charts and packing."""
    import xatlas
    atlas = xatlas.Atlas()
    k = max(1, int(round(len(F) / piece)))
    pieces = _morton_pieces(V, F, k) if k > 1 else [np.arange(len(F))]
    used_list = []
    for fidx in pieces:
        used, inv = np.unique(F[fidx], return_inverse=True)
        atlas.add_mesh(V[used].astype(np.float32), inv.reshape(-1, 3).astype(np.uint32), N[used].astype(np.float32))
        used_list.append(used)
    co = xatlas.ChartOptions()        # hard edges are already split, so they are seams anyway
    co.max_cost = 4.0                 # bigger charts: far fewer of them, packing several times faster
    co.max_iterations = 1
    po = xatlas.PackOptions()
    po.resolution = int(tex)
    po.padding = int(padding)
    po.bilinear = True
    po.rotate_charts = True
    atlas.generate(co, po)
    vmaps, faces, uvs, off = [], [], [], 0
    for i, used in enumerate(used_list):
        vmap, Fo, UV = atlas[i]
        vmaps.append(used[vmap.astype(np.int64)])
        faces.append(Fo.astype(np.int64) + off)
        uvs.append(UV.astype(np.float64))
        off += len(vmap)
    return np.concatenate(vmaps), np.concatenate(faces), np.concatenate(uvs)


def smooth_vertex_normals(V, F, eps):
    """Area-weighted vertex normals that ignore texture seams (vertices at the same place share one)."""
    q = np.floor((V - V.min(0)) / eps + 0.5).astype(np.int64)
    _, grp = np.unique(q, axis=0, return_inverse=True)
    grp = grp.reshape(-1)
    fn, fa = face_normals_areas(V, F)
    acc = np.zeros((grp.max() + 1, 3))
    for k in range(3):
        np.add.at(acc, grp[F[:, k]], fn * fa[:, None])
    return _unit(acc[grp] + 1e-30)


def axis_vertex_normals(V, F, eps):
    """Smooth vertex normals that do not care which way the triangles are wound (generated models have many
    triangles wound the wrong way, and a plain average cancels out there): the main axis of the area-weighted
    face normals around the vertex (sign to be chosen by the caller)."""
    q = np.floor((V - V.min(0)) / eps + 0.5).astype(np.int64)
    _, grp = np.unique(q, axis=0, return_inverse=True)
    grp = grp.reshape(-1)
    fn, fa = face_normals_areas(V, F)
    T = np.einsum("fi,fj->fij", fn, fn) * fa[:, None, None]
    acc = np.zeros((grp.max() + 1, 3, 3))
    for k in range(3):
        np.add.at(acc, grp[F[:, k]], T)
    w, v = np.linalg.eigh(acc + np.eye(3) * 1e-30)
    return v[:, :, 2][grp]


def vertex_tangents(V, F, N, UV):
    """Per-vertex tangents (xyz = +u direction, w = handedness) from the UVs, angle-weighted and made
    orthogonal to the normals (the usual MikkTSpace-style construction)."""
    p0, p1, p2 = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    t0, t1, t2 = UV[F[:, 0]], UV[F[:, 1]], UV[F[:, 2]]
    dp1, dp2 = p1 - p0, p2 - p0
    du1, dv1 = (t1 - t0).T
    du2, dv2 = (t2 - t0).T
    det = du1 * dv2 - du2 * dv1
    r = np.where(np.abs(det) < 1e-20, 0.0, 1.0 / np.where(np.abs(det) < 1e-20, 1.0, det))
    tf = (dp1 * dv2[:, None] - dp2 * dv1[:, None]) * r[:, None]
    bf = (dp2 * du1[:, None] - dp1 * du2[:, None]) * r[:, None]
    T = np.zeros_like(V)
    B = np.zeros_like(V)
    tri = V[F]
    for k in range(3):
        e1 = tri[:, (k + 1) % 3] - tri[:, k]
        e2 = tri[:, (k + 2) % 3] - tri[:, k]
        cosang = np.einsum("ij,ij->i", e1, e2) / (np.linalg.norm(e1, axis=1) * np.linalg.norm(e2, axis=1) + 1e-30)
        w = np.arccos(np.clip(cosang, -1, 1))
        np.add.at(T, F[:, k], tf * w[:, None])
        np.add.at(B, F[:, k], bf * w[:, None])
    T = T - N * np.einsum("ij,ij->i", N, T)[:, None]
    bad = np.linalg.norm(T, axis=1) < 1e-20
    if bad.any():  # no usable UV direction: any vector perpendicular to the normal
        alt = np.where(np.abs(N[bad, 0:1]) < 0.9, [[1.0, 0, 0]], [[0, 1.0, 0]])
        T[bad] = np.cross(N[bad], alt)
    T = _unit(T)
    w = np.where(np.einsum("ij,ij->i", np.cross(N, T), B) < 0, -1.0, 1.0)
    return np.concatenate([T, w[:, None]], 1)


class Sampler:
    """Colour / roughness / metal of the original model at a surface point (by part, UV)."""

    def __init__(self, parts):
        self.parts = []
        for p in parts:
            vis = p.visual
            d = dict(uv=None, base=None, mr=None, bf=np.ones(4), mf=1.0, rf=1.0, vcol=None)
            if isinstance(vis, trimesh.visual.TextureVisuals):
                d["uv"] = None if vis.uv is None else np.asarray(vis.uv, np.float64)
                mat = vis.material
                if isinstance(mat, trimesh.visual.material.SimpleMaterial):
                    mat = mat.to_pbr()
                img = getattr(mat, "baseColorTexture", None)
                if img is not None:
                    d["base"] = np.asarray(img.convert("RGBA"))
                bf = getattr(mat, "baseColorFactor", None)
                if bf is not None:
                    bf = np.asarray(bf, np.float64)
                    d["bf"] = bf / 255.0 if bf.max() > 1.0 else bf
                mr = getattr(mat, "metallicRoughnessTexture", None)
                if mr is not None:
                    d["mr"] = np.asarray(mr.convert("RGB"))
                d["mf"] = 1.0 if getattr(mat, "metallicFactor", None) is None else float(mat.metallicFactor)
                d["rf"] = 1.0 if getattr(mat, "roughnessFactor", None) is None else float(mat.roughnessFactor)
            elif vis.kind == "vertex":
                d["vcol"] = np.asarray(vis.vertex_colors, np.float64) / 255.0
            elif vis.kind == "face":
                d["fcol"] = np.asarray(vis.face_colors, np.float64) / 255.0
            self.parts.append(d)

    @staticmethod
    def _tex(img, uv):
        import cv2
        H, W = img.shape[:2]
        mx = (uv[:, 0] * W - 0.5).astype(np.float32).reshape(-1, 1)
        my = ((1 - uv[:, 1]) * H - 0.5).astype(np.float32).reshape(-1, 1)
        n, w = len(mx), 1024
        pad = (-n) % w
        mx = np.concatenate([mx.ravel(), np.zeros(pad, np.float32)]).reshape(-1, w)
        my = np.concatenate([my.ravel(), np.zeros(pad, np.float32)]).reshape(-1, w)
        out = []
        for i in range(0, len(mx), 1024):   # remap needs fewer than 32767 rows per call
            out.append(cv2.remap(img, mx[i:i + 1024], my[i:i + 1024], cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_WRAP).reshape(-1, img.shape[2]))
        return np.concatenate(out)[:n].astype(np.float64) / 255.0

    def sample(self, part, lf, bary):
        """part: part index per sample, lf: local face index in that part, bary: (n,3)."""
        n = len(part)
        rgb = np.full((n, 3), 0.7)
        rough = np.ones(n)
        metal = np.zeros(n)
        for i, d in enumerate(self.parts):
            s = np.flatnonzero(part == i)
            if not len(s):
                continue
            faces = self._faces[i][lf[s]]
            col = np.tile(d["bf"], (len(s), 1))
            if d["uv"] is not None and d["base"] is not None:
                uv = (d["uv"][faces] * bary[s, :, None]).sum(1)
                col = col * self._tex(d["base"], uv)
            elif d.get("vcol") is not None:
                col = col * (d["vcol"][faces] * bary[s, :, None]).sum(1)
            elif d.get("fcol") is not None:
                col = col * d["fcol"][lf[s]]
            rgb[s] = col[:, :3]
            r, m = np.full(len(s), d["rf"]), np.full(len(s), d["mf"])
            if d["uv"] is not None and d["mr"] is not None:
                uv = (d["uv"][faces] * bary[s, :, None]).sum(1)
                t = self._tex(d["mr"], uv)
                r, m = r * t[:, 1], m * t[:, 2]
            rough[s], metal[s] = r, m
        return rgb, rough, metal


def _fill_gaps(img, mask, iters=12):
    """Grow the baked texels into the empty texels around them, so UV seams do not show."""
    import cv2
    img = img.astype(np.float32)
    m = mask.astype(np.float32)
    acc = img * m[..., None]
    k = np.ones((3, 3), np.float32)
    for _ in range(iters):
        s = cv2.filter2D(acc, -1, k, borderType=cv2.BORDER_CONSTANT)
        c = cv2.filter2D(m, -1, k, borderType=cv2.BORDER_CONSTANT)
        new = (c > 0) & (m == 0)
        if not new.any():
            break
        acc[new] = s[new] / c[new, None]
        m[new] = 1
    if (m == 0).any():
        acc[m == 0] = acc[m > 0].mean(0)
    return acc


def _push_pull(img, valid, todo):
    """Fill the `todo` texels of img from the `valid` ones around them, however big the area: averages
    on ever coarser levels, then back down (push-pull)."""
    import cv2
    img = img.astype(np.float32)
    w = valid.astype(np.float32)
    levels = [(img * w[..., None], w)]
    while min(levels[-1][1].shape) > 4:
        a, ww = levels[-1]
        h, wd = ww.shape
        levels.append((cv2.resize(a, (wd // 2, h // 2), interpolation=cv2.INTER_AREA),
                       cv2.resize(ww, (wd // 2, h // 2), interpolation=cv2.INTER_AREA)))
    acc, ww = levels[-1]
    col = acc / np.maximum(ww, 1e-8)[..., None]
    for a, wl in reversed(levels[:-1]):
        up = cv2.resize(col, (wl.shape[1], wl.shape[0]), interpolation=cv2.INTER_LINEAR)
        here = a / np.maximum(wl, 1e-8)[..., None]
        t = np.clip(wl, 0, 1)[..., None]
        col = here * t + up * (1 - t)
    out = img.copy()
    out[todo] = col[todo]
    return out


def _clone_fill(base, mr, nimg, mask, over, tri, fnorm, grow=8, dark=0.75):
    """Texels over closed openings (`over`): a window stays as it was seen through it (darker than the wall
    around it); a hole in a wall is covered - together with its ragged rim, `grow` texels wide - with a
    piece of the same wall taken from next to it, like a clone stamp: same chart, same direction, colour,
    roughness and normal map copied, edges blended, brightness matched, so it looks as if there never
    was a hole. Returns (holes covered, windows kept, holes left as they were)."""
    from scipy import ndimage
    T = mask.shape[0]
    chart, _ = ndimage.label(mask, structure=np.ones((3, 3)))
    lab, _n = ndimage.label(over, structure=np.ones((3, 3)))
    lum = base @ np.array([0.299, 0.587, 0.114], np.float32)
    usable = mask & ~over
    n_hole = n_win = n_left = 0
    for r, sl in enumerate(ndimage.find_objects(lab), 1):
        if sl is None:
            continue
        pad = grow + 16
        y0, y1 = max(sl[0].start - pad, 0), min(sl[0].stop + pad, T)
        x0, x1 = max(sl[1].start - pad, 0), min(sl[1].stop + pad, T)
        reg = lab[y0:y1, x0:x1] == r
        ring = ndimage.binary_dilation(reg, iterations=6) & ~reg & usable[y0:y1, x0:x1]
        if ring.sum() < 10:
            continue
        L = lum[y0:y1, x0:x1]
        if L[reg].mean() < dark * L[ring].mean():
            n_win += 1                      # darker than the wall: a window, keep it
            continue
        ch = chart[y0:y1, x0:x1]
        c = np.bincount(ch[reg & (ch > 0)]).argmax() if (reg & (ch > 0)).any() else 0
        done = False
        for g in sorted({grow, max(grow // 2, 4), 4}, reverse=True):
            # the area to cover: the cap plus its ragged rim, on the same chart only
            Fm = ndimage.binary_dilation(reg, iterations=g) & mask[y0:y1, x0:x1] & (ch == c)
            Fm |= reg
            fy, fx = np.nonzero(Fm)
            fy, fx = fy + y0, fx + x0
            ok = tri[fy, fx] >= 0
            if not ok.any():
                break
            nr = fnorm[fy[ok], fx[ok]].mean(0)
            nr = nr / (np.linalg.norm(nr) + 1e-12)
            h, w = int(fy.max() - fy.min() + 3), int(fx.max() - fx.min() + 3)
            found = None
            for k in range(1, 9):
                for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)):
                    oy, ox = dy * k * h, dx * k * w
                    sy, sx = fy + oy, fx + ox
                    if sy.min() < 0 or sx.min() < 0 or sy.max() >= T or sx.max() >= T:
                        continue
                    if not usable[sy, sx].all() or not (chart[sy, sx] == c).all():
                        continue
                    if (fnorm[sy, sx] @ nr).min() < 0.97:
                        continue
                    found = (oy, ox)
                    break
                if found:
                    break
            if found is None:
                continue
            oy, ox = found
            # blend: full copy inside, fading into the original over the outer 3 texels
            alpha = np.clip(ndimage.distance_transform_edt(Fm) / 3.0, 0, 1)[Fm][:, None].astype(np.float32)
            # match the brightness of the copied piece to the wall around it
            rb = ndimage.binary_dilation(Fm, iterations=5) & ~Fm & usable[y0:y1, x0:x1]
            ry, rx = np.nonzero(rb)
            ry, rx = ry + y0, rx + x0
            sr = (ry + oy >= 0) & (ry + oy < T) & (rx + ox >= 0) & (rx + ox < T)
            delta = 0.0
            if sr.sum() > 10:
                delta = base[ry, rx].mean(0) - base[ry[sr] + oy, rx[sr] + ox].mean(0)
            for img, adj in ((base, True), (mr, False), (nimg, False)):
                if img is None:
                    continue
                src = img[fy + oy, fx + ox] + (delta if adj else 0.0)
                img[fy, fx] = alpha * src + (1 - alpha) * img[fy, fx]
            n_hole += 1
            done = True
            break
        if not done:
            n_left += 1
    return n_hole, n_win, n_left


def bake(parts, lowV, lowF, tex, hard_angle, cage, padding=4, flat=None, planes=None, normal_map=True,
         normal_strength=1.0, chunk=1 << 20):
    """UV-unwrap the low-poly mesh and bake colour, roughness/metal and (optionally) a tangent-space normal
    map from the original detailed model. Everything in the same (levelled) frame. Texels are done in
    chunks of `chunk`, so a 4096 bake of a million-triangle model needs ~2 GB, not 6+.
    Returns (V, F, normals, uv, tangents or None, images dict)."""
    import cv2
    from PIL import Image

    t0 = time.time()
    Vs, Fs, Ns = split_hard_edges(lowV, lowF, hard_angle, flat, planes)
    vmap, Fo, UV = unwrap(Vs, Fs, Ns, tex, padding)
    Vo, No = Vs[vmap], Ns[vmap]
    T4 = vertex_tangents(Vo, Fo, No, UV) if normal_map else None
    log(f"uv: {len(Fo):,} triangles, {len(Vo):,} vertices ({time.time() - t0:.0f}s)")

    # texel -> triangle id (rasterised in texture space)
    T = int(tex)
    P = np.stack([UV[:, 0] * T, (1 - UV[:, 1]) * T], 1)
    tid = np.zeros((T, T), np.int32)
    pix = (P[Fo] * 16 - 8).round().astype(np.int32)          # texel centres, 4 bits sub-pixel
    for i in range(len(Fo)):
        cv2.polylines(tid, [pix[i]], True, i + 1, 1, lineType=cv2.LINE_8, shift=4)
    for i in range(len(Fo)):
        cv2.fillPoly(tid, [pix[i]], i + 1, lineType=cv2.LINE_8, shift=4)
    ys, xs = np.nonzero(tid)
    ys, xs = ys.astype(np.int32), xs.astype(np.int32)
    tt = (tid[ys, xs] - 1).astype(np.int32)
    del tid
    mask = np.zeros((T, T), bool)
    mask[ys, xs] = True

    # the detailed model: ray along -normal from just outside the low-poly, else the closest point
    hiV, hiF, hiP = combine(parts)
    local_face = np.concatenate([np.arange(len(p.faces)) for p in parts])
    try:
        from trimesh.ray.ray_pyembree import RayMeshIntersector
        rmi = RayMeshIntersector(trimesh.Trimesh(hiV, hiF, process=False))
    except Exception:   # without embreex ray casting is far too slow: closest point only
        rmi = None
        log("embreex is missing (pip install embreex): baking by closest point only")
    try:
        import igl
    except ImportError:
        igl = None
    kd = None
    sampler = Sampler(parts)
    sampler._faces = [np.asarray(p.faces) for p in parts]
    hn = axis_vertex_normals(hiV, hiF, float((hiV.max(0) - hiV.min(0)).max()) * 1e-5) if normal_map else None
    base = np.zeros((T, T, 3), np.float32)
    mr = np.zeros((T, T, 3), np.float32)
    nimg = np.zeros((T, T, 3), np.float32) if normal_map else None
    n_ray, n_detail = 0, 0
    gap = np.zeros((T, T), bool)
    flat_n = np.zeros((T, T), bool)
    tri = np.full((T, T), -1, np.int32)
    tri[ys, xs] = tt
    for c0 in range(0, len(tt), chunk):
        t, y, x = tt[c0:c0 + chunk], ys[c0:c0 + chunk], xs[c0:c0 + chunk]
        q = np.stack([x + 0.5, y + 0.5], 1)
        a, b, c = P[Fo[t, 0]], P[Fo[t, 1]], P[Fo[t, 2]]
        v0, v1, v2 = b - a, c - a, q - a
        den = v0[:, 0] * v1[:, 1] - v1[:, 0] * v0[:, 1]
        den = np.where(np.abs(den) < 1e-12, 1e-12, den)
        w1 = (v2[:, 0] * v1[:, 1] - v1[:, 0] * v2[:, 1]) / den
        w2 = (v0[:, 0] * v2[:, 1] - v2[:, 0] * v0[:, 1]) / den
        W = np.clip(np.stack([1 - w1 - w2, w1, w2], 1), 0, None)
        W /= W.sum(1, keepdims=True) + 1e-12
        del a, b, c, v0, v1, v2, den, w1, w2, q
        pos = (Vo[Fo[t]] * W[:, :, None]).sum(1)
        nrm = _unit((No[Fo[t]] * W[:, :, None]).sum(1) + 1e-30)
        hit_face = -np.ones(len(t), np.int64)
        hit_loc = pos.copy()
        if rmi is not None:
            o = pos + nrm * cage
            loc, ri, fi = rmi.intersects_location(o, -nrm, multiple_hits=False)
            if len(ri):
                ok = np.linalg.norm(loc - o[ri], axis=1) <= 2 * cage
                hit_face[ri[ok]] = fi[ok]
                hit_loc[ri[ok]] = loc[ok]
        miss = np.flatnonzero(hit_face < 0)
        n_ray += len(t) - len(miss)
        if len(miss):
            if igl is not None:      # exact closest point on the detailed surface
                d2, fi, cp = igl.point_mesh_squared_distance(pos[miss], hiV, np.asarray(hiF, np.int64))
                hit_face[miss], hit_loc[miss] = fi, cp
                far = miss[d2 > (2 * cage) ** 2]       # nothing of the original near: a cap over a closed hole
                flat_n[y[far], x[far]] = True
                if len(far) and rmi is not None:
                    # paint it with what was seen through the opening (a window stays a dark window)
                    loc2, ri2, fi2 = rmi.intersects_location(pos[far] + nrm[far] * cage, -nrm[far],
                                                             multiple_hits=False)
                    hit_face[far[ri2]], hit_loc[far[ri2]] = fi2, loc2
                    far = np.delete(far, ri2)
                if len(far):
                    gap[y[far], x[far]] = True
            else:                    # nearest triangle centre
                from scipy.spatial import cKDTree
                if kd is None:
                    kd = cKDTree(hiV[hiF].mean(1))
                hit_face[miss] = kd.query(pos[miss], workers=-1)[1]
        bary = trimesh.triangles.points_to_barycentric(hiV[hiF[hit_face]], hit_loc)
        bary = np.clip(np.nan_to_num(bary, nan=1 / 3), 0, None)
        bary /= bary.sum(1, keepdims=True) + 1e-12
        rgb, rough, metal = sampler.sample(hiP[hit_face], local_face[hit_face], bary)
        base[y, x] = rgb
        mr[y, x, 0], mr[y, x, 1], mr[y, x, 2] = 1.0, rough, metal
        if normal_map:
            # detailed model's (smooth) normal at the hit point, in the low-poly's tangent frame
            hc = hn[hiF[hit_face]]                                  # sign-free: turned towards the low-poly's
            hc = hc * np.sign(np.einsum("ijk,ik->ij", hc, nrm) + 1e-12)[:, :, None]   # outside first
            nh = _unit((hc * bary[:, :, None]).sum(1) + 1e-30)
            tg = (T4[Fo[t], :3] * W[:, :, None]).sum(1)
            tg = _unit(tg - nrm * np.einsum("ij,ij->i", tg, nrm)[:, None] + 1e-30)
            bt = np.cross(nrm, tg) * T4[Fo[t, 0], 3][:, None]
            loc_n = np.stack([np.einsum("ij,ij->i", nh, tg), np.einsum("ij,ij->i", nh, bt),
                              np.einsum("ij,ij->i", nh, nrm)], 1)
            loc_n[loc_n[:, 2] < 0.05] = [0.0, 0.0, 1.0]   # facing away: a wrong hit, keep the surface normal
            if normal_strength != 1.0:
                loc_n[:, :2] *= normal_strength
            loc_n = _unit(loc_n)
            n_detail += int((np.abs(loc_n[:, :2]).max(1) > 0.05).sum())
            nimg[y, x] = loc_n * 0.5 + 0.5
            fl = flat_n[y, x]
            if fl.any():
                nimg[y[fl], x[fl]] = [0.5, 0.5, 1.0]
    log(f"bake: {len(tt):,} texels, {n_ray / max(len(tt), 1) * 100:.0f}% by ray ({time.time() - t0:.0f}s)")
    if gap.any():   # closed holes: colour from the surrounding wall instead of whatever was behind the hole
        src = mask & ~gap
        base = _push_pull(base, src, gap)
        mr = _push_pull(mr, src, gap)
        if normal_map:
            nimg[gap] = [0.5, 0.5, 1.0]
        log(f"bake: {int(gap.sum()):,} texels over closed holes with nothing behind, painted from around")
    if flat_n.any():
        fn_lo = face_normals_areas(Vo, Fo)[0].astype(np.float32)
        fnorm = np.zeros((T, T, 3), np.float32)
        fnorm[ys, xs] = fn_lo[tt]
        texel = math.sqrt(face_normals_areas(Vo, Fo)[1].sum() / max(int(mask.sum()), 1))
        grow = int(np.clip(round(2 * cage / texel), 4, 48))      # the ragged rim: what the bake still reached
        n_hole, n_win, n_left = _clone_fill(base, mr, nimg, mask, flat_n, tri, fnorm, grow=grow)
        del fnorm
        log(f"bake: closed openings - {n_hole} hole(s) covered with the wall next to them, {n_win} kept as "
            f"windows (darker than the wall: what was seen through them)"
            + (f", {n_left} without a clean piece of wall to copy" if n_left else ""))
    del tri
    images = {"base": Image.fromarray(np.clip(_fill_gaps(base, mask) * 255 + 0.5, 0, 255).astype(np.uint8))}
    del base
    images["mr"] = Image.fromarray(np.clip(_fill_gaps(mr, mask) * 255 + 0.5, 0, 255).astype(np.uint8))
    del mr
    if normal_map:
        nimg = _fill_gaps(nimg, mask)
        nimg[~mask & (nimg.sum(2) == 0)] = [0.5, 0.5, 1.0]
        images["normal"] = Image.fromarray(np.clip(nimg * 255 + 0.5, 0, 255).astype(np.uint8))
        log(f"normal map: {100 * n_detail / max(len(tt), 1):.0f}% of the texels carry detail")
    return Vo, Fo, No, UV, T4, images


# ============================================================================ output

def write_glb(path, V, N, F, UV=None, T4=None, images=None, name="lowpoly"):
    """Minimal glTF 2.0 binary writer with tangents and a normal map (trimesh does not write tangents).
    UV in trimesh convention (v up); images = dict(base=PIL, mr=PIL, normal=PIL), any may be missing."""
    import io
    import json
    import struct
    chunks, views, accessors = [], [], []

    def add_view(data, target=None):
        off = sum(len(c) for c in chunks)
        chunks.append(data + b"\x00" * ((4 - len(data) % 4) % 4))
        v = {"buffer": 0, "byteOffset": off, "byteLength": len(data)}
        if target:
            v["target"] = target
        views.append(v)
        return len(views) - 1

    def add_acc(arr, ctype, typ, target, minmax=False):
        arr = np.ascontiguousarray(arr)
        a = {"bufferView": add_view(arr.tobytes(), target), "componentType": ctype, "count": int(arr.shape[0]),
             "type": typ}
        if minmax:
            a["min"] = arr.min(0).tolist()
            a["max"] = arr.max(0).tolist()
        accessors.append(a)
        return len(accessors) - 1

    attrs = {"POSITION": add_acc(V.astype(np.float32), 5126, "VEC3", 34962, True),
             "NORMAL": add_acc(_unit(N).astype(np.float32), 5126, "VEC3", 34962)}
    if T4 is not None:
        attrs["TANGENT"] = add_acc(T4.astype(np.float32), 5126, "VEC4", 34962)
    if UV is not None:
        uvg = np.stack([UV[:, 0], 1.0 - UV[:, 1]], 1).astype(np.float32)
        attrs["TEXCOORD_0"] = add_acc(uvg, 5126, "VEC2", 34962)
    idx = add_acc(F.astype(np.uint32).reshape(-1), 5125, "SCALAR", 34963)
    gl = {"asset": {"version": "2.0", "generator": "lowpoly.py"}, "scene": 0, "scenes": [{"nodes": [0]}],
          "nodes": [{"mesh": 0, "name": name}],
          "meshes": [{"name": name, "primitives": [{"attributes": attrs, "indices": idx, "material": 0}]}]}
    mat = {"name": name, "pbrMetallicRoughness": {"metallicFactor": 0.0, "roughnessFactor": 0.8}, "doubleSided": False}
    images = images or {}
    if images:
        gl["samplers"] = [{"magFilter": 9729, "minFilter": 9987, "wrapS": 33071, "wrapT": 33071}]
        gl["images"], gl["textures"] = [], []

        def add_img(img, fmt):
            buf = io.BytesIO()
            if fmt == "JPEG":
                img.convert("RGB").save(buf, format="JPEG", quality=92, subsampling=0)
            else:
                img.save(buf, format="PNG", optimize=False, compress_level=6)
            gl["images"].append({"bufferView": add_view(buf.getvalue()),
                                 "mimeType": "image/jpeg" if fmt == "JPEG" else "image/png"})
            gl["textures"].append({"sampler": 0, "source": len(gl["images"]) - 1})
            return len(gl["textures"]) - 1
        pbr = mat["pbrMetallicRoughness"]
        if images.get("base") is not None:
            pbr["baseColorTexture"] = {"index": add_img(images["base"], images.get("format", "PNG"))}
        if images.get("mr") is not None:
            pbr["metallicRoughnessTexture"] = {"index": add_img(images["mr"], images.get("format", "PNG"))}
            pbr["metallicFactor"], pbr["roughnessFactor"] = 1.0, 1.0
        if images.get("normal") is not None:
            mat["normalTexture"] = {"index": add_img(images["normal"], "PNG"), "scale": 1.0}
    gl["materials"] = [mat]
    gl["accessors"], gl["bufferViews"] = accessors, views
    binary = b"".join(chunks)
    gl["buffers"] = [{"byteLength": len(binary)}]
    js = json.dumps(gl, separators=(",", ":")).encode()
    js += b" " * ((4 - len(js) % 4) % 4)
    total = 12 + 8 + len(js) + 8 + len(binary)
    with open(path, "wb") as fh:
        fh.write(struct.pack("<III", 0x46546C67, 2, total))
        fh.write(struct.pack("<II", len(js), 0x4E4F534A) + js)
        fh.write(struct.pack("<II", len(binary), 0x004E4942) + binary)

# ============================================================================ all together

class LowPoly:
    """Result of make_lowpoly; .export(path) writes the GLB (with tangents and the normal map)."""

    def __init__(self, V, F, N, UV=None, T4=None, images=None, name="lowpoly", tex_format="PNG"):
        self.vertices, self.faces, self.normals, self.uv, self.tangents = V, F, N, UV, T4
        self.images = dict(images or {})
        self.images["format"] = tex_format
        self.name = name

    def export(self, path, **_ignored):
        write_glb(str(path), self.vertices, self.normals, self.faces, self.uv, self.tangents,
                  self.images if self.uv is not None else None, self.name)


def make_lowpoly(src, cfg, name=""):
    """src: file path / trimesh object. cfg: namespace from config(). Returns (LowPoly, info)."""
    t_all = time.time()
    parts = load_parts(src)
    V, F, _ = combine(parts)
    size = float((V.max(0) - V.min(0)).max())
    info = dict(input_tris=len(F))

    # 1. merge
    t0 = time.time()
    Vw, Fw = weld(V, F, size * 1e-5)
    log(f"input: {len(F):,} triangles, {len(Vw):,} vertices after welding ({time.time() - t0:.0f}s)")

    # 2. level (rotation R, applied to everything incl. the bake reference)
    R = np.eye(3)
    if cfg.level:
        up, tilt, s_best, s_prior = estimate_up(Vw, Fw, max_tilt=cfg.max_tilt)
        if s_best >= 0.08 and s_best - s_prior >= 0.01 and tilt > 0.3:
            R = rot_between(up, [0, 1, 0])
            log(f"level: tilted {tilt:.1f} deg, straightened ({s_prior * 100:.0f}% -> {s_best * 100:.0f}% of the "
                f"surface vertical/level)")
        else:
            log(f"level: kept as it is ({s_best * 100:.0f}% of the surface vertical/level)")
    if cfg.align_walls:
        yaw = wall_yaw(Vw @ R.T, Fw)
        c, s = math.cos(yaw), math.sin(yaw)
        R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]]) @ R
    if not np.allclose(R, np.eye(3)):
        Vw = Vw @ R.T
        T4x4 = np.eye(4)
        T4x4[:3, :3] = R
        for p in parts:
            p.apply_transform(T4x4)

    if cfg.drop_ground:  # flat pieces lying at the very bottom (generated pavement, rugs, shadows)
        comp = face_components(Fw, len(Vw))
        y = Vw[:, 1]
        h = y.max() - y.min()
        drop = [cid for cid in np.unique(comp)
                if np.ptp(y[Fw[comp == cid]]) < 0.015 * h and y[Fw[comp == cid]].max() < y.min() + 0.05 * h]
        if drop:
            Fw = Fw[~np.isin(comp, drop)]
            used, inv = np.unique(Fw, return_inverse=True)
            Vw, Fw = Vw[used], inv.reshape(-1, 3)
            log(f"ground: dropped {len(drop)} flat piece(s) at the bottom")

    # real size -> how much detail. Known size: the detail is set in metres (METRIC), so small things
    # get few triangles and big ones more. Unknown: everything relative to the model's size (PRESETS).
    ref_lo, ref_hi = Vw.min(0), Vw.max(0)
    h_units = float(Vw[:, 1].max() - Vw[:, 1].min())
    user = getattr(cfg, "user", set())
    mpu, how = None, ""                                    # metres per model unit
    if cfg.height > 0:
        mpu, how = cfg.height / h_units, "given"
    elif cfg.size > 0:
        mpu, how = cfg.size / size, "given"
    elif getattr(cfg, "units", "auto") == "m" or (getattr(cfg, "units", "auto") == "auto" and size > 1.5):
        mpu, how = 1.0, "the model is in metres"
    tol_rel, relief_rel, error_rel = cfg.tol, cfg.relief, cfg.error
    faces, min_faces, tex = cfg.faces, cfg.min_faces, cfg.tex
    met = METRIC.get(getattr(cfg, "quality", "medium"), METRIC["medium"])
    planar = bool(getattr(cfg, "planar", False))
    grid_based = planar or getattr(cfg, "quality", "") == "visual"
    if planar:
        try:
            import shapely  # noqa: F401
            import triangle  # noqa: F401
        except ImportError:
            log("the planar rebuild needs two more packages (pip install triangle shapely): reducing as usual")
            planar = False
    if getattr(cfg, "error_mm", 0) and not mpu:
        log("--error-mm needs the real size (--height, sizes.txt or a name like 'church_45m'): using --error")
    if mpu:
        if not grid_based:
            if "tol" not in user:
                tol_rel = max(met["tol"] / mpu, met["tol_rel"] * size) / size   # never below the generator's noise
            if "relief" not in user:
                relief_rel = min(met["relief"] / mpu, 0.03 * size) / size
        if getattr(cfg, "error_mm", 0):
            error_rel = cfg.error_mm / 1000 / (mpu * size)
        elif "error" not in user and not grid_based:   # metric, within min_rel..rel_error of the size
            error_rel = min(max(met["error"] / (mpu * size), met["min_rel"]), met["rel_error"])
        if "faces" not in user:
            faces = met["faces"]
        if "min_faces" not in user:
            min_faces = met["min_faces"]
        log(f"size: {h_units * mpu:.1f} m tall, {size * mpu:.1f} m across ({how}) -> shape within "
            f"{error_rel * size * mpu * 1000:.0f} mm everywhere, flat within {tol_rel * size * mpu * 100:.1f} cm, "
            f"relief up to {relief_rel * size * mpu * 100:.1f} cm flattened")
    else:
        log("size: unknown -> detail relative to the model (give --height, a sizes.txt or a name like "
            "'church_45m' for size-aware detail)")
    error_rel = max(error_rel, 0.0002)          # below ~1/5 of the generator's voxel there is only noise
    info["metres_per_unit"] = mpu

    # 3. a clean surface: the model's own triangles that can be seen from outside (hidden parts removed,
    #    small holes closed), or, for a broken shell (open, many loose edges), rebuilt on a voxel grid
    t0 = time.time()
    n_open, n_nonman, n_edges = edge_stats(Fw)
    fill = getattr(cfg, "fill_openings", False)
    solid = fill or cfg.solid == "on" or (cfg.solid == "auto" and (n_open + n_nonman) > 0.01 * n_edges)
    if solid:
        try:
            import igl  # noqa: F401  (only checks that it is installed)
        except ImportError:
            log("libigl is missing (pip install libigl): keeping the input surface instead of rebuilding it")
            solid = False
    if solid:
        Vc, Fc = clean_surface(Vw, Fw, repair=False)
        Vo, Fo, turned, dropped = orient_by_sight(Vc, Fc)
        log(f"repair: {turned * 100:.0f}% of the surface was wound the wrong way (turned), {dropped * 100:.0f}% "
            f"was hidden inner skin (dropped); input had {n_open:,} open / {n_nonman:,} non-manifold edges "
            f"({time.time() - t0:.0f}s)")
        Vs, Fs, _vox = solid_surface(Vo, Fo, res=cfg.voxel, fill=fill, covered=not fill, orient=False)
        Vs, share = project_back(Vs, Fs, Vo, Fo, _vox)
        o_, n_, _ = edge_stats(Fs)
        log(f"repair: one closed shell, every triangle facing out ({o_} open / {n_} non-manifold edges), put back "
            f"onto the original surface ({share * 100:.0f}% of its points), {len(Fs):,} triangles "
            f"({time.time() - t0:.0f}s)")
    elif cfg.solid == "off":
        Vs, Fs = clean_surface(Vw, Fw, min_part=getattr(cfg, "min_part", 0.0005))
        log(f"surface: kept as it is, {len(Fs):,} triangles ({time.time() - t0:.0f}s)")
    else:
        Vs, Fs = clean_surface(Vw, Fw, min_part=getattr(cfg, "min_part", 0.0005))
        Vs, Fs, hidden = remove_hidden(Vs, Fs)
        log(f"surface: {hidden * 100:.0f}% of it was hidden (inner skins, leftovers) and removed, hidden openings capped, "
            f"{len(Fs):,} triangles ({time.time() - t0:.0f}s)")
    if (solid or cfg.solid == "off") and trimesh.Trimesh(Vs, Fs, process=False).volume < 0:   # inside out
        Fs = Fs[:, ::-1]
    t0 = time.time()
    if _pymeshlab():
        Vm, Fm = decimate_qem(Vs, Fs, cfg.mid, quality=0.5)
    else:
        Vm, Fm = decimate_fast(Vs, Fs, cfg.mid)
        Vm, Fm = weld(Vm, Fm, size * 1e-6)
    log(f"working mesh: {len(Fm):,} triangles ({time.time() - t0:.0f}s)")

    # 4. flat areas: exactly flat, shallow details flattened into them, smooth hand-over to the rest
    t0 = time.time()
    tol, relief = tol_rel * size, relief_rel * size
    up = [0, 1, 0] if cfg.level else None
    if planar:
        Vl, Fl, flat, planes, st, tau = planar_lowpoly(Vm, Fm, cfg, size, tol, relief, error_rel * size, faces,
                                                       min_faces, up, mpu)
        info["planes"] = len(planes)
        info["tris"] = len(Fl)
        info["error"] = st
    else:
        if getattr(cfg, "flat", True) and tol > 0:
            region, adj = segment_planes(Vm, Fm, cfg.angle, tol, cfg.min_plane, max_spread=0.35 * cfg.angle,
                                         group_tol=tol)
        else:                                      # nothing flattened: the shape stays as generated
            region = -np.ones(len(Fm), np.int64)
        planes = []
        if region.max() >= 0:
            planes = fit_planes(Vm, Fm, region, up=up)
            if relief > 0:
                region, planes = merge_relief(Vm, Fm, region, planes, relief)
                region, grown = grow_relief(Vm, Fm, region, planes, adj, relief)
            else:
                grown = np.zeros(len(Fm), bool)
            region = merge_coplanar(Vm, Fm, region, planes, tol, skip=grown)
            planes = fit_planes(Vm, Fm, region, up=up, skip=grown)
            Vsnap = snap(Vm, Fm, region, planes, max(2 * tol, 1.3 * relief))
            moved = np.zeros(len(Vm), bool)
            moved[np.unique(Fm[region >= 0])] = True
            Vm = smooth_snap(Vm, Fm, Vsnap, moved, rings=2)
        fa = face_normals_areas(Vm, Fm)[1]
        flat_share = fa[region >= 0].sum() / fa.sum()
        if cfg.smooth > 0:
            Vm = smooth_free(Vm, Fm, region, iters=cfg.smooth)
        log(f"flat areas: {len(planes)} ({flat_share * 100:.0f}% of the surface)"
            f"{'; the rest smoothed' if cfg.smooth > 0 else ''} ({time.time() - t0:.0f}s)")
        info["planes"] = len(planes)
        if getattr(cfg, "keep_stages", False):
            info["stages"] = dict(Vm=Vm, Fm=Fm, region=region, planes=planes)

        # 5. as few triangles as the shape allows while it stays within --error of the reference everywhere
        t0 = time.time()
        tau = error_rel * size
        Vl, Fl, st, _ = reduce_to_error(Vm, Fm, tau, faces, min_faces, bad_frac=getattr(cfg, "bad_frac", BAD_FRAC))
        Vl, Fl = weld(Vl, Fl, size * 1e-7)
        flat = label_flat_faces(Vl, Fl, planes, 0.25 * tol, angle=2.0) if planes else None
        if st is None:
            how_close = ""
        else:
            k, unit = (mpu * 1000, "mm") if mpu else (100 / size, "% of the size")
            how_close = (f", {100 - st['bad'] * 100:.2f}% of the surface within {tau * k:.0f} {unit}, 99% within "
                         f"{st['p99'] * k:.0f} {unit}, RMS {st['rms'] * k:.1f} {unit}")
        log(f"low-poly: {len(Fl):,} triangles{how_close} ({time.time() - t0:.0f}s)")
        info["tris"] = len(Fl)
        info["error"] = st
        if "stages" in info:
            info["stages"].update(Vl=Vl, Fl=Fl, flat=flat)

    # texture size from the real surface area: about met["texel"] metres per texture pixel
    if mpu and tex > 0 and "tex" not in user:
        area_m2 = float(face_normals_areas(Vl, Fl)[1].sum()) * mpu * mpu
        want = math.sqrt(area_m2 / 0.65) / met["texel"]          # ~65% of a texture is covered by charts
        tex = int(min(max(2 ** round(math.log2(max(want, 1.0))), met["min_tex"]), met["max_tex"]))
        log(f"texture: {tex} px for {area_m2:,.0f} m2 of surface")

    # 6. bake (colour, roughness/metal, normal map)
    # the bake looks for the detailed surface this far in front of / behind the low-poly: a little more than
    # the largest gap between them (flattened relief + shape error). Too far, and rays from a wall hit a
    # sill or a pinnacle in front of it (smeared texture, wrong normal map)
    cage = cfg.cage * size if "cage" in user else float(np.clip(1.5 * relief + 3 * tau, 0.003 * size, 0.02 * size))
    if tex > 0:
        Vo, Fo, No, UV, T4, images = bake(parts, Vl, Fl, tex, cfg.hard_angle, cage, flat=flat,
                                          planes=planes, normal_map=cfg.normal_map,
                                          normal_strength=cfg.normal_strength)
    else:
        Vo, Fo, No = split_hard_edges(Vl, Fl, cfg.hard_angle, flat, planes)
        UV, T4, images = None, None, None

    # 7. place: base centred on the origin, real size - by the *input's* box, so the result sits exactly
    #    where the detailed model sits (a removed crumb or pole at one end must not shift the whole model)
    lo, hi = ref_lo, ref_hi
    Vo = Vo - [(lo[0] + hi[0]) / 2, lo[1], (lo[2] + hi[2]) / 2]
    ext = hi - lo
    if cfg.height > 0:
        Vo = Vo * (cfg.height / ext[1])
    elif cfg.size > 0:
        Vo = Vo * (cfg.size / ext.max())
    info["seconds"] = time.time() - t_all
    return LowPoly(Vo, Fo, No, UV, T4, images, name or "lowpoly", cfg.tex_format), info


def level_and_place(mesh, level=True, max_tilt=40.0, height=0.0, size=0.0):
    """Straighten and place a (textured) mesh without changing its triangles: Y up, base centred on
    the origin, optional real size. Returns the transformed copy and the tilt that was removed."""
    parts = load_parts(mesh)
    V, F, _ = combine(parts)
    tilt = 0.0
    R = np.eye(3)
    if level:
        Vw, Fw = weld(V, F, float((V.max(0) - V.min(0)).max()) * 1e-5)
        up, t, s_best, s_prior = estimate_up(Vw, Fw, max_tilt=max_tilt)
        if s_best >= 0.08 and s_best - s_prior >= 0.01 and t > 0.3:
            R, tilt = rot_between(up, [0, 1, 0]), t
    T = np.eye(4)
    T[:3, :3] = R
    Vr = V @ R.T
    lo, hi = Vr.min(0), Vr.max(0)
    ext = hi - lo
    k = height / ext[1] if height > 0 else (size / ext.max() if size > 0 else 1.0)
    S = np.eye(4)
    S[:3, :3] *= k
    S[:3, 3] = -k * np.array([(lo[0] + hi[0]) / 2, lo[1], (lo[2] + hi[2]) / 2])
    out = []
    for p in parts:
        p = p.copy()
        p.apply_transform(S @ T)
        out.append(p)
    return (out[0] if len(out) == 1 else trimesh.Scene(out)), tilt


def config(quality="visual", user=(), **over):
    """Options of a preset (default "optimize"); keyword arguments override. `user` names the options the
    user set explicitly: those win over the size-aware (METRIC) values."""
    c = dict(PRESETS[quality])
    c.update(solid="on", level=True, max_tilt=40.0, align_walls=False, drop_ground=False, hard_angle=35.0,
             cage=0.02, height=0.0, size=0.0, units="auto", normal_map=True, normal_strength=1.0, tex_format="JPEG",
             error_mm=0.0, fill_openings=False)
    c.update({k: v for k, v in over.items() if v is not None})
    c["quality"], c["user"] = quality, set(user)
    return argparse.Namespace(**c)


def main():
    p = argparse.ArgumentParser(description="Dense generated model -> clean low-poly GLB with baked textures")
    p.add_argument("inputs", nargs="+", help="GLB/OBJ/PLY files or folders")
    p.add_argument("--quality", "-q", default="visual", choices=list(PRESETS),
                   help="visual (default: looks the same as the detailed model, only what cannot be seen is "
                        "reduced), optimize (every visible detail kept, flat parts merged), high, medium, "
                        "low (far away, fewest triangles), planar (buildings: every flat wall/roof rebuilt "
                        "from its outline with the fewest triangles)")
    p.add_argument("--out", default=None,
                   help="output folder (default: <input folder>/optimized, or lowpoly / lowpoly_<quality>)")
    p.add_argument("--error-mm", type=float, default=0.0,
                   help="largest shape error in millimetres (needs the real size; medium 25, high 12, low 50): "
                        "smaller = closer to the detailed model, more triangles")
    p.add_argument("--error", type=float, default=None,
                   help="largest shape error as a fraction of the size, when the size is unknown (medium 0.0012)")
    p.add_argument("--faces", type=int, default=None, help="largest triangle count allowed")
    p.add_argument("--min-faces", type=int, default=None, help="never fewer triangles than this")
    p.add_argument("--relief", type=float, default=None,
                   help="details on a flat area shallower than this (fraction of the size) are flattened into it "
                        "and kept in the normal map and texture; 0 keeps all of them as geometry")
    p.add_argument("--smooth", type=int, default=None, help="de-noise passes on the non-flat parts (default 0: kept as generated; 1-2 for a noisy model)")
    p.add_argument("--tex", type=int, default=None, help="texture size, 0 = no textures")
    p.add_argument("--no-normal-map", dest="normal_map", action="store_false", help="do not bake a normal map")
    p.add_argument("--normal-strength", type=float, default=1.0, help="normal map strength (0.5 = half the bumps)")
    p.add_argument("--png", dest="tex_format", action="store_const", const="PNG", default="JPEG",
                   help="colour/roughness textures as PNG instead of JPEG (bigger files)")
    p.add_argument("--height", type=float, default=0.0,
                   help="real height in metres: scales the result and sets the detail by size (a sizes.txt next "
                        "to the models, or a name like 'church_45m.glb', does the same per file)")
    p.add_argument("--size", type=float, default=0.0, help="real longest side in metres (instead of --height)")
    p.add_argument("--units", default="auto", choices=["auto", "m", "unit"],
                   help="auto: a model bigger than 1.5 units is taken to be in metres already")
    p.add_argument("--plane-tol", dest="tol", type=float, default=None,
                   help="noise smaller than this (fraction of the size) still counts as flat")
    p.add_argument("--plane-angle", dest="angle", type=float, default=None,
                   help="surfaces bending less than this (degrees) can be one flat area")
    p.add_argument("--min-plane", type=float, default=None,
                   help="flat areas smaller than this share of the surface are treated as curved")
    p.add_argument("--hard-angle", type=float, default=35.0, help="edges sharper than this (degrees) shade hard")
    p.add_argument("--no-level", dest="level", action="store_false", help="do not straighten a tilted model")
    p.add_argument("--max-tilt", type=float, default=40.0, help="largest tilt (degrees) the levelling corrects")
    p.add_argument("--align-walls", action="store_true", help="also turn the main walls onto the X/Z axes")
    p.add_argument("--drop-ground", action="store_true",
                   help="remove flat pieces lying at the very bottom (generated ground, rugs, shadows)")
    p.add_argument("--solid", default="on", choices=["auto", "on", "off"],
                   help="on (default): repair first - one closed shell, every triangle facing out, gaps closed, "
                        "hollow inside filled, flat bottom, put back onto the original surface; auto: only when "
                        "the shell is badly broken (old behaviour); off: use the input as it is")
    p.add_argument("--fill-openings", action="store_true",
                   help="close holes you can look through (a hollow model with gaps in its walls) and fill the inside solid; the patches get the look of the wall next to them, windows and recesses stay "
                        "(rebuilds the surface, implies --solid on)")
    p.add_argument("--voxel", type=int, default=None, help="voxel grid for finding hidden parts / rebuilding")
    p.add_argument("--overwrite", action="store_true", help="redo files that already have an output")
    a = p.parse_args()

    files = []
    for s in a.inputs:
        s = Path(s)
        if s.is_dir():
            files += sorted(f for f in s.iterdir() if f.suffix.lower() in MESH_EXTS)
        elif s.exists():
            files.append(s)
        else:
            print(f"not found: {s}")
    if not files:
        raise SystemExit("nothing to do")
    preset = PRESETS[a.quality]
    given = {k: v for k, v in vars(a).items() if k in preset and v is not None}
    base = dict(solid=a.solid, level=a.level, max_tilt=a.max_tilt, align_walls=a.align_walls,
                drop_ground=a.drop_ground, hard_angle=a.hard_angle, units=a.units,
                normal_map=a.normal_map, normal_strength=a.normal_strength, tex_format=a.tex_format,
                error_mm=a.error_mm, fill_openings=a.fill_openings)
    print(f"quality={a.quality}{' +normal map' if a.normal_map else ''}")
    for f in files:
        out_dir = Path(a.out) if a.out else f.parent / {"optimize": "optimized", "medium": "lowpoly",
                                                                     "visual": "lowpoly_visual"}.get(
            a.quality, f"lowpoly_{a.quality}")
        out = out_dir / f"{f.stem}.glb"
        if out.exists() and not a.overwrite:
            print(f"skip {f.name} ({out} exists, use --overwrite)")
            continue
        # real height: --height / --size, else sizes.txt next to the model, else the file name
        height, size = a.height, a.size
        if height <= 0 and size <= 0:
            height = read_sizes(f.parent / "sizes.txt").get(f.stem) or height_from_name(f.stem) or 0.0
        print(f"{f.name}" + (f"  ({height:g} m tall)" if height > 0 else ""))
        cfg = config(a.quality, user=given.keys(), **{**preset, **given}, **base, height=height, size=size)
        try:
            mesh, info = make_lowpoly(f, cfg, f.stem)
        except MemoryError:
            print(f"FAIL {f.name}: out of memory, try a smaller --tex or -q low")
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        mesh.export(out)
        print(f"ok   {out}  {info['tris']:,} triangles  {info['seconds']:.0f}s  {out.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
