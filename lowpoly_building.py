"""
Low-poly building mesh from a dense TRELLIS.2 mesh (CPU, numpy/scipy).

  dense mesh (millions of triangles, often open at the bottom, hollow, with window holes)
    -> solid_proxy:     one closed solid (voxels: seal small openings, fill every floor slice,
                        keep the biggest part, marching cubes)
    -> decimate:        ~3000 triangles (quadric edge collapse)
    -> planar_lowpoly:  group triangles into planes (walls, roof faces), snap the vertices onto
                        those planes, and re-triangulate each plane from its outline only,
                        so a rectangular wall becomes 2 triangles.

The detail (windows, cornices) is not in the geometry any more; it is baked into the texture.
Coordinates are TRELLIS' frame: object inside [-0.5, 0.5]^3, z up.

Needs: scipy, scikit-image, trimesh, mapbox-earcut, and pymeshlab (falls back to
fast-simplification if pymeshlab is missing).
"""
from collections import deque

import numpy as np
import trimesh


# ----------------------------------------------------------------------------- solid proxy

def sample_surface(v, f, spacing, seed=0):
    """Points on the triangles, about 2 per spacing^2 of area, plus the vertices."""
    tri = v[f].astype(np.float64)
    area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    n = np.random.default_rng(seed).poisson(area / spacing ** 2 * 2)
    fi = np.repeat(np.arange(len(f)), n)
    r = np.random.default_rng(seed + 1).random((len(fi), 2))
    flip = r.sum(1) > 1
    r[flip] = 1 - r[flip]
    t = tri[fi]
    pts = t[:, 0] + r[:, :1] * (t[:, 1] - t[:, 0]) + r[:, 1:] * (t[:, 2] - t[:, 0])
    return np.concatenate([v, pts]).astype(np.float32)


def solid_proxy(v, f, res=128, close=2, sigma=1.0):
    """Closed single-shell version of a (possibly open, hollow) building mesh."""
    from scipy import ndimage
    from skimage import measure

    margin = close + 3
    lo, hi = v.min(0), v.max(0)
    vox = float((hi - lo).max()) / res
    lo = lo - margin * vox
    dims = np.ceil((hi - lo) / vox).astype(int) + margin + 1
    idx = np.floor((sample_surface(v, f, vox / 2) - lo) / vox).astype(int)
    surf = np.zeros(dims, bool)
    surf[idx[:, 0], idx[:, 1], idx[:, 2]] = True
    shell = ndimage.binary_dilation(surf, iterations=close)       # seal windows/doors up to ~2*close voxels
    solid = ndimage.binary_fill_holes(shell)
    for k in range(shell.shape[2]):                               # buildings are often open at the bottom
        solid[:, :, k] |= ndimage.binary_fill_holes(shell[:, :, k])
    solid = ndimage.binary_erosion(solid, iterations=close, border_value=0)
    lab, n = ndimage.label(solid)
    if n > 1:
        sizes = ndimage.sum(solid, lab, range(1, n + 1))
        solid = lab == (1 + int(np.argmax(sizes)))
    field = ndimage.gaussian_filter(solid.astype(np.float32), sigma) if sigma > 0 else solid.astype(np.float32)
    mv, mf, _, _ = measure.marching_cubes(field, 0.5)
    mv = mv * vox + lo
    return mv.astype(np.float64), mf[:, ::-1].astype(np.int64)    # outward winding


def decimate(v, f, target):
    """Quadric edge collapse to about `target` triangles (pymeshlab for small inputs, it plans
    planar areas better; fast-simplification for big ones, pymeshlab takes minutes there)."""
    try:
        if len(f) > 400_000:
            raise ImportError
        import pymeshlab
        ms = pymeshlab.MeshSet()
        ms.add_mesh(pymeshlab.Mesh(np.asarray(v, np.float64), np.asarray(f, np.int32)))
        ms.meshing_decimation_quadric_edge_collapse(
            targetfacenum=int(target), qualitythr=0.3, preserveboundary=True, preservenormal=True,
            preservetopology=True, optimalplacement=True, planarquadric=True, planarweight=0.002,
            autoclean=True)
        m = ms.current_mesh()
        return m.vertex_matrix().astype(np.float64), m.face_matrix().astype(np.int64)
    except (ImportError, AttributeError):
        import fast_simplification
        lv, lf = fast_simplification.simplify(np.asarray(v, np.float32), np.asarray(f, np.int32),
                                              target_reduction=max(0.0, 1 - target / len(f)), agg=7)
        return lv.astype(np.float64), lf.astype(np.int64)


# ----------------------------------------------------------------------------- planes

def segment_planes(v, f, angle=20.0, tol=0.008, min_area_frac=0.003, merge_angle=45.0):
    """Region growing over neighbouring triangles: same plane if the normal is within `angle`
    and the triangle is within `tol` of the region's plane. Small regions join a similar neighbour."""
    m = trimesh.Trimesh(v, f, process=False)
    fn, fa, fc = m.face_normals, m.area_faces, m.triangles_center
    nbrs = [[] for _ in range(len(f))]
    for a, b in m.face_adjacency:
        nbrs[a].append(b)
        nbrs[b].append(a)
    region = -np.ones(len(f), int)
    cos_t = np.cos(np.radians(angle))
    rid = 0
    for seed in np.argsort(-fa):
        if region[seed] >= 0:
            continue
        region[seed] = rid
        nsum, csum, asum = fn[seed] * fa[seed], fc[seed] * fa[seed], fa[seed]
        q = deque([seed])
        while q:
            cur = q.popleft()
            n, c = nsum / (np.linalg.norm(nsum) + 1e-12), csum / (asum + 1e-12)
            for nb in nbrs[cur]:
                if region[nb] < 0 and fn[nb] @ n > cos_t and abs((fc[nb] - c) @ n) < tol:
                    region[nb] = rid
                    nsum, csum, asum = nsum + fn[nb] * fa[nb], csum + fc[nb] * fa[nb], asum + fa[nb]
                    q.append(nb)
        rid += 1
    total = fa.sum()
    merge_cos = np.cos(np.radians(merge_angle))
    for _ in range(10):
        areas = np.bincount(region, weights=fa, minlength=rid)
        normals = np.zeros((rid, 3))
        np.add.at(normals, region, fn * fa[:, None])
        normals /= np.linalg.norm(normals, axis=1, keepdims=True) + 1e-12
        small = np.where((areas > 0) & (areas < min_area_frac * total))[0]
        changed = False
        for r in small[np.argsort(areas[small])]:
            faces = np.where(region == r)[0]
            cand = {region[nb] for fi in faces for nb in nbrs[fi]} - {r}
            cand = [c for c in cand if areas[c] >= areas[r] and normals[c] @ normals[r] > merge_cos]
            if cand:
                best = max(cand, key=lambda c: normals[c] @ normals[r])
                region[faces] = best
                areas[best] += areas[r]
                areas[r] = 0
                changed = True
        if not changed:
            break
    _, region = np.unique(region, return_inverse=True)
    return region


def fit_planes(v, f, region):
    planes = []
    for r in range(region.max() + 1):
        p = v[np.unique(f[region == r])]
        c = p.mean(0)
        n = np.linalg.svd(p - c, full_matrices=False)[2][-1]
        tri = v[f[region == r]]
        if n @ np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]).sum(0) < 0:
            n = -n
        planes.append((n, n @ c))
    return planes


def snap(v, f, region, planes, max_move, lam=1e-4):
    """Move every vertex onto all planes it touches (plane, edge line or corner point)."""
    vr = [set() for _ in range(len(v))]
    for fi, tri in enumerate(f):
        for x in tri:
            vr[x].add(region[fi])
    out = v.copy()
    for i, rs in enumerate(vr):
        if not rs:
            continue
        N = np.array([planes[r][0] for r in rs])
        d = np.array([planes[r][1] for r in rs])
        p, l = out[i], lam
        for _ in range(8):  # nearly parallel planes: pull harder towards the original point
            x = np.linalg.solve(N.T @ N + l * np.eye(3), N.T @ d + l * p)
            if np.linalg.norm(x - p) <= max_move:
                break
            l *= 10
        else:
            x = p
        out[i] = x
    return out


def _dp(pts, tol):
    """Douglas-Peucker; kept indices of a polyline (first and last always kept)."""
    keep = np.zeros(len(pts), bool)
    keep[0] = keep[-1] = True
    stack = [(0, len(pts) - 1)]
    while stack:
        i, j = stack.pop()
        if j <= i + 1:
            continue
        a, ab = pts[i], pts[j] - pts[i]
        L = np.linalg.norm(ab)
        seg = pts[i + 1:j] - a
        dist = np.linalg.norm(seg, axis=1) if L < 1e-12 else np.linalg.norm(np.cross(seg, ab / L), axis=1)
        k = int(np.argmax(dist))
        if dist[k] > tol:
            keep[i + 1 + k] = True
            stack += [(i, i + 1 + k), (i + 1 + k, j)]
    return np.where(keep)[0]


def _inside(pt, poly):
    x, y = pt
    px, py = poly[:, 0], poly[:, 1]
    qx, qy = np.roll(px, -1), np.roll(py, -1)
    cross = ((py > y) != (qy > y)) & (x < (qx - px) * (y - py) / (qy - py + 1e-30) + px)
    return bool(cross.sum() % 2)


def remesh_planar(v, f, region, planes, tol):
    """Re-triangulate every plane from its simplified outline. Shared outlines are simplified
    once and used by both sides, so the mesh stays closed."""
    import mapbox_earcut as earcut

    F = len(f)
    he = np.stack([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]], 1).reshape(-1, 2)
    he_face = np.repeat(np.arange(F), 3)
    twin = {(a, b): k for k, (a, b) in enumerate(he)}
    vregs = [set() for _ in range(len(v))]
    for fi in range(F):
        for x in f[fi]:
            vregs[x].add(region[fi])
    corner = np.array([len(s) >= 3 for s in vregs])
    chain_cache, out_faces = {}, []
    for r in range(region.max() + 1):
        nxt, side = {}, {}
        for k in np.where(region[he_face] == r)[0]:
            a, b = he[k]
            t = twin.get((b, a))
            other = region[he_face[t]] if t is not None else -1
            if other != r:
                nxt.setdefault(a, []).append(b)
                side[(a, b)] = other
        loops, seen = [], set()
        for a0 in list(nxt):
            for b0 in nxt[a0]:
                if (a0, b0) in seen:
                    continue
                loop, a, b = [a0], a0, b0
                while (a, b) not in seen:
                    seen.add((a, b))
                    loop.append(b)
                    cands = [c for c in nxt.get(b, []) if (b, c) not in seen]
                    if not cands:
                        break
                    a, b = b, cands[0]
                if loop[-1] == loop[0]:
                    loops.append(loop[:-1])
        polys = []
        for loop in loops:
            n = len(loop)
            cuts = [i for i in range(n)
                    if corner[loop[i]] or side.get((loop[i - 1], loop[i])) != side.get((loop[i], loop[(i + 1) % n]))]
            if len(cuts) < 2:  # island without corners: same two cut points from both sides
                s0 = int(np.argmin(loop))
                far = int(np.argmax(np.linalg.norm(v[loop] - v[loop[s0]], axis=1)))
                cuts = sorted(set([s0, far] + cuts))
            kept = []
            for ci in range(len(cuts)):
                i0, i1 = cuts[ci], cuts[(ci + 1) % len(cuts)]
                idx = [loop[(i0 + t) % n] for t in range(((i1 - i0) % n or n) + 1)]
                fwd = idx[0] < idx[-1] or (idx[0] == idx[-1] and idx[1] < idx[-2])
                key = tuple(idx) if fwd else tuple(idx[::-1])
                if key not in chain_cache:
                    chain_cache[key] = [key[j] for j in _dp(v[list(key)], tol)]
                ch = chain_cache[key]
                if ch[0] != idx[0]:
                    ch = ch[::-1]
                kept += ch[:-1]
            if len(kept) >= 3:
                polys.append(kept)
        if not polys:
            continue
        n_r = planes[r][0]
        e1 = np.cross(n_r, [1.0, 0, 0] if abs(n_r[0]) < 0.9 else [0, 1.0, 0])
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(n_r, e1)
        P2 = [np.stack([v[p] @ e1, v[p] @ e2], 1) for p in polys]
        area = [0.5 * np.sum(q[:, 0] * np.roll(q[:, 1], -1) - np.roll(q[:, 0], -1) * q[:, 1]) for q in P2]
        holes = [i for i, A in enumerate(area) if A <= 0]
        for o in [i for i, A in enumerate(area) if A > 0]:
            rings = [o] + [h for h in holes if _inside(P2[h].mean(0), P2[o])]
            pts2 = np.concatenate([P2[i] for i in rings])
            ids = np.concatenate([polys[i] for i in rings])
            ends = np.cumsum([len(P2[i]) for i in rings]).astype(np.uint32)
            for t in earcut.triangulate_float64(pts2, ends).reshape(-1, 3):
                a, b, c = ids[t]
                if len({a, b, c}) == 3:
                    nrm = np.cross(v[b] - v[a], v[c] - v[a])
                    out_faces.append((a, b, c) if nrm @ n_r >= 0 else (a, c, b))
    out_faces = np.array(out_faces, np.int64)
    used, inv = np.unique(out_faces.ravel(), return_inverse=True)
    m = trimesh.Trimesh(v[used], inv.reshape(-1, 3), process=False)
    if not m.is_watertight:  # close small gaps the polygon triangulation may leave
        m.merge_vertices()
        trimesh.repair.fill_holes(m)
        trimesh.repair.fix_normals(m)
    return np.asarray(m.vertices), np.asarray(m.faces)


# ----------------------------------------------------------------------------- all together

def building_lowpoly(v, f, proxy_res=512, close=2, mid_faces=80_000, angle=20.0, tol=0.002, min_plane=2e-5,
                     sigma=None):
    """Dense mesh (numpy, TRELLIS frame) -> (vertices, faces, number of planes).

    Defaults keep building-scale detail (pilasters, cornices, window and door recesses at
    about 0.2% of the building size, ~12 cm on a 60 m tower) and flatten everything finer:
    ~10-30k triangles for a detailed tower. For a very low-poly version use
    proxy_res=128, mid_faces=3000, tol=0.008, min_plane=0.003 (a few hundred triangles).
    """
    if sigma is None:
        sigma = 0.8 if proxy_res >= 384 else 1.0
    pv, pf = solid_proxy(np.asarray(v, np.float32), np.asarray(f, np.int64), res=proxy_res, close=close,
                         sigma=sigma)
    mv, mf = decimate(pv, pf, mid_faces)
    region = segment_planes(mv, mf, angle, tol, min_plane)
    planes = fit_planes(mv, mf, region)
    sv = snap(mv, mf, region, planes, max_move=3 * tol)
    lv, lf = remesh_planar(sv, mf, region, planes, 0.5 * tol)
    return lv, lf, region.max() + 1
