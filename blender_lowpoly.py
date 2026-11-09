"""
blender_lowpoly.py - make a light low-poly copy of a building GLB and bake its texture onto it, in Blender.

The amount of reduction depends on the shape: the script keeps removing triangles until the low poly would
move away from the original surface by more than --error. Flat walls go first, so a simple house ends up
with few triangles and a detailed church keeps more. Then it makes new UVs and bakes color + normal map
(optionally roughness) from the original, exactly like doing it by hand (Cycles, Selected to Active).

Repair first (default): the original from the generator has gaps, cracks, open edges, loose flaps and
inward-facing triangles. The script rebuilds it as one clean, closed surface (Remesh Sharp closes every
gap, a voxel remesh makes it clean), reduces that, and paints the texture from the original. Where the
original had a gap, the texture gets the colour of the nearest original surface. Every triangle faces
outward, so it also works with one-sided materials (Unreal). --no-repair keeps the original's surface
(then its holes are patched instead).

Run it with Blender's own exe (no extra packages needed), from the TRELLIS.2 folder:

  "C:\\Program Files\\Blender Foundation\\Blender 4.2\\blender.exe" -b -P blender_lowpoly.py -- glb\\high\\house.glb
  ... -- glb\\high                         all .glb files in that folder
  ... -- house.glb --error 0.001           stricter: more triangles, closer to the original
  ... -- house.glb --error 0.005           looser: fewer triangles
  ... -- house.glb --low my_low.glb        bake only: use your own low poly (made by hand)

Output: <input folder>\\lowpoly_blender\\<name>.glb  (textures inside the glb). Existing results are skipped
unless --overwrite, so a folder run can be restarted.

Main options
  --error 0.002        allowed distance from the original, as a share of the model's size (0.002 = 0.2 %,
                       e.g. 4 cm on a 20 m house). 99 % of the surface must stay within it, nothing beyond 3x.
  --error-cm 5         the same in cm (only if the model is in metres)
                       Generated models have a small ripple everywhere. The allowed error is never set below
                       3x that ripple (otherwise the low poly just copies noise); the normal map keeps the
                       ripple's look. --strict turns this off.
  --planar 5           first merge flat areas up to 5 degrees (Decimate Planar); 0 = off (default).
                       Only kept as far as the --error check allows. Helps on clean models (hand-made,
                       LoD2); on generated ones the ripple stops it, the normal reduction does the work.
  --max-faces 50000    never more triangles than this (0 = no limit)
  --min-faces 300      never fewer
  --tex 4096           texture size (default: same as the original's biggest texture, at least 2048)
  --maps color normal roughness   which textures to bake (default: color normal)
  --no-repair          don't rebuild; reduce the original surface itself (patches its holes)
  --repair-depth 10    finer rebuild (default auto: cells about the allowed error; 10 = twice as fine)
  --keep-holes         with --no-repair: don't patch the original's holes
  --smooth-angle 30    edges sharper than this stay sharp, the rest is shaded smooth
  --single-sided       one-sided material (default two-sided: generated models often have inward triangles)
  --blend              also save a .blend with original + low poly next to the glb, for checking
  --cpu                bake on the CPU (default: GPU if Cycles finds one)

Unreal: the normal map is glTF/OpenGL style. If details look lit from the wrong side, open the normal
texture and tick "Flip Green Channel".
"""
import argparse
import math
import os
import sys
import time

import bpy  # noqa: I001 - bpy first (bmesh/mathutils need it when run as a module)
import bmesh
import numpy as np
from mathutils import Matrix, Vector
from mathutils.bvhtree import BVHTree

T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


# ============================================================================ arguments

def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    p = argparse.ArgumentParser(prog="blender -b -P blender_lowpoly.py --",
                                description="Shape-adaptive low poly + texture bake for building GLBs.")
    p.add_argument("inputs", nargs="+", help=".glb files or folders with .glb files (the high poly)")
    p.add_argument("--low", help="your own low poly .glb: no reduction, only bake onto it (one input only)")
    p.add_argument("--new-uv", action="store_true", help="with --low: make new UVs even if it has some")
    p.add_argument("--error", type=float, default=0.002, help="allowed error, share of the model size")
    p.add_argument("--error-cm", type=float, default=0.0, help="allowed error in cm (model in metres)")
    p.add_argument("--strict", action="store_true",
                   help="use --error as given even if it is below the model's surface noise")
    p.add_argument("--planar", type=float, default=0.0, help="merge flat areas up to this angle first (deg)")
    p.add_argument("--keep-holes", action="store_true", help="don't close the holes of the original")
    p.add_argument("--no-repair", action="store_true",
                   help="don't rebuild a closed surface first (keeps the original's gaps and thin flaps)")
    p.add_argument("--repair-depth", type=int, default=0,
                   help="detail of the rebuilt surface: 9 = 460 cells across the model, 10 = 920 (0 = auto)")
    p.add_argument("--min-faces", type=int, default=300)
    p.add_argument("--max-faces", type=int, default=0)
    p.add_argument("--tex", type=int, default=0, help="texture size (0 = like the original, >= 2048)")
    p.add_argument("--maps", nargs="+", default=["color", "normal"], choices=["color", "normal", "roughness"])
    p.add_argument("--smooth-angle", type=float, default=30.0)
    p.add_argument("--samples", type=int, default=16, help="bake samples per pixel")
    p.add_argument("--single-sided", action="store_true")
    p.add_argument("--blend", action="store_true", help="also save a .blend for checking")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--out", help="output folder (default <input folder>/lowpoly_blender)")
    p.add_argument("--overwrite", action="store_true")
    a = p.parse_args(argv)
    if a.low and len(a.inputs) != 1:
        p.error("--low works with exactly one high poly input")
    return a


def collect_inputs(paths, out_name="lowpoly_blender"):
    files = []
    for p in paths:
        p = os.path.abspath(p)
        if os.path.isdir(p):
            files += sorted(os.path.join(p, f) for f in os.listdir(p) if f.lower().endswith(".glb"))
        elif os.path.isfile(p):
            files.append(p)
        else:
            log(f"not found, skipped: {p}")
    return [f for f in files if os.path.basename(os.path.dirname(f)) != out_name]


# ============================================================================ scene helpers

def reset_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def select_only(objs, active=None):
    for o in bpy.context.view_layer.objects:
        o.select_set(False)
    for o in objs:
        o.select_set(True)
    bpy.context.view_layer.objects.active = active or (objs[0] if objs else None)


def import_glb(path, name):
    """Import a GLB as ONE mesh object with its transform baked into the vertices."""
    before = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=path)
    new = [o for o in bpy.data.objects if o not in before]
    meshes = [o for o in new if o.type == 'MESH']
    if not meshes:
        raise RuntimeError(f"no mesh in {path}")
    for o in meshes:
        if o.data.users > 1:
            o.data = o.data.copy()
        mw = o.matrix_world.copy()
        o.parent = None
        o.data.transform(mw)
        if mw.determinant() < 0:
            o.data.flip_normals()
        o.matrix_world = Matrix.Identity(4)
    select_only(meshes)
    if len(meshes) > 1:
        bpy.ops.object.join()
    obj = bpy.context.view_layer.objects.active
    for o in new:
        if o != obj and o.name in bpy.data.objects and o.type != 'MESH':
            bpy.data.objects.remove(o)
    obj.name = name
    return obj


def setup_cycles(args):
    sc = bpy.context.scene
    sc.render.engine = 'CYCLES'
    sc.cycles.samples = args.samples
    try:
        sc.cycles.use_denoising = False
    except AttributeError:
        pass
    sc.cycles.device = 'CPU'
    if args.cpu:
        return "CPU"
    try:
        prefs = bpy.context.preferences.addons["cycles"].preferences
        for kind in ("OPTIX", "CUDA", "HIP", "ONEAPI", "METAL"):
            try:
                prefs.compute_device_type = kind
            except TypeError:
                continue
            if hasattr(prefs, "refresh_devices"):
                prefs.refresh_devices()
            else:
                prefs.get_devices()
            gpus = [d for d in prefs.devices if d.type == kind]
            if gpus:
                for d in prefs.devices:
                    d.use = d.type == kind
                sc.cycles.device = 'GPU'
                return f"GPU ({kind}: {gpus[0].name})"
    except Exception as e:  # noqa: BLE001
        log(f"no GPU for Cycles ({e}), using the CPU")
    return "CPU"


# ============================================================================ geometry

FILL = 1   # material index that marks hole patches the script added (not part of the original surface)


def mesh_arrays(obj, evaluated=False, with_fill=False):
    """(vertices Nx3, triangles Mx3[, fill mask M]) of the object, with modifiers if evaluated."""
    src = obj.evaluated_get(bpy.context.evaluated_depsgraph_get()) if evaluated else obj
    me = src.to_mesh()
    me.calc_loop_triangles()
    V = np.empty(len(me.vertices) * 3, np.float32)
    me.vertices.foreach_get("co", V)
    T = np.empty(len(me.loop_triangles) * 3, np.int32)
    me.loop_triangles.foreach_get("vertices", T)
    if with_fill:
        M = np.empty(len(me.loop_triangles), np.int32)
        me.loop_triangles.foreach_get("material_index", M)
    src.to_mesh_clear()
    V, T = V.reshape(-1, 3).astype(np.float64), T.reshape(-1, 3)
    return (V, T, M == FILL) if with_fill else (V, T)


def sample_surface(V, T, n, rng):
    """n points spread evenly (by area) over the triangles."""
    a = V[T[:, 1]] - V[T[:, 0]]
    b = V[T[:, 2]] - V[T[:, 0]]
    area = 0.5 * np.linalg.norm(np.cross(a, b), axis=1)
    if area.sum() <= 0:
        return V[:1].copy()
    idx = rng.choice(len(T), n, p=area / area.sum())
    r1, r2 = rng.random(n), rng.random(n)
    flip = r1 + r2 > 1
    r1[flip], r2[flip] = 1 - r1[flip], 1 - r2[flip]
    return V[T[idx, 0]] + r1[:, None] * a[idx] + r2[:, None] * b[idx]


def bvh(V, T):
    return BVHTree.FromPolygons(V.tolist(), T.tolist(), all_triangles=True)


def distances(tree, P):
    out = np.empty(len(P))
    nearest = tree.find_nearest
    for i, p in enumerate(P.tolist()):
        hit = nearest(p)
        out[i] = hit[3] if hit[0] is not None else np.inf
    return out


class ErrorMeter:
    """Distance between the original and a candidate low poly, both ways, on evenly spread samples."""

    def __init__(self, high, n=60000, seed=0):
        V, T = mesh_arrays(high)
        self.n = n
        self.rng = np.random.default_rng(seed)
        self.tree = bvh(V, T)
        self.high_pts = sample_surface(V, T, n, self.rng)
        k = 200000                                    # vertices follow detail: thin parts have many
        self.high_verts = V[self.rng.choice(len(V), k, replace=False)] if len(V) > k else V

    def measure(self, V, T, fill=None):
        """fill: triangles that close holes - they have no original surface to compare with"""
        d1 = distances(bvh(V, T), self.high_pts)                                # original -> low (lost detail)
        own = T if fill is None or fill.all() else T[~fill]
        d2 = distances(self.tree, sample_surface(V, own, self.n, self.rng))     # low -> original (bulges)
        d = np.concatenate([d1, d2])
        return float(np.percentile(d, 99)), float(np.percentile(d, 99.9)), float(d.mean())

    def lost_spots(self, V, T, fill, limit):
        """Places where the low poly is more than `limit` off: parts of the original it lost (thin parts,
        junctions, fins) and low-poly surface sticking out."""
        d1 = distances(bvh(V, T), self.high_verts)
        own = T if fill is None or fill.all() else T[~fill]
        S = sample_surface(V, own, self.n, self.rng)
        d2 = distances(self.tree, S)
        return np.concatenate([self.high_verts[d1 > limit], S[d2 > limit]])


def surface_noise(V, T, iterations=10):
    """Typical size of the generator's small ripple: how far vertices sit from a lightly smoothed copy
    (robust spread, so real edges and corners don't count)."""
    E = np.concatenate([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]])
    E = np.unique(np.sort(E, axis=1), axis=0)
    n = len(V)
    deg = np.bincount(E.ravel(), minlength=n).astype(np.float64)
    deg[deg == 0] = 1
    S = V.copy()
    for _ in range(iterations):
        acc = np.zeros_like(S)
        for k in range(3):
            acc[:, k] = np.bincount(E[:, 0], S[E[:, 1], k], n) + np.bincount(E[:, 1], S[E[:, 0], k], n)
        S = 0.5 * S + 0.5 * acc / deg[:, None]
    fn = np.cross(S[T[:, 1]] - S[T[:, 0]], S[T[:, 2]] - S[T[:, 0]])
    N = np.zeros_like(S)
    for k in range(3):
        for j in range(3):
            N[:, k] += np.bincount(T[:, j], fn[:, k], n)
    N /= np.maximum(np.linalg.norm(N, axis=1), 1e-30)[:, None]
    d = np.abs(((V - S) * N).sum(1))
    return float(1.4826 * np.median(d))


def smooth_by_angle(obj, angle):
    """Smooth shading, edges sharper than angle (and open/odd edges) stay sharp."""
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    thr = math.radians(angle)
    for f in bm.faces:
        f.smooth = True
    for e in bm.edges:
        e.smooth = len(e.link_faces) == 2 and e.calc_face_angle(math.pi) <= thr
    bm.to_mesh(obj.data)
    bm.free()
    obj.data.update()


def prepare_high(high, size, smooth_angle):
    """The original as bake source: generated models have patches of inward-facing triangles, whose
    normals would bake as black spots. Weld, turn all faces outward, fresh normals. (Only in memory.)"""
    me = high.data
    bm = bmesh.new()
    bm.from_mesh(me)
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=size * 1e-6)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(me)
    bm.free()
    clear_custom_normals(high)
    smooth_by_angle(high, smooth_angle)


def clean_copy(high, name, size):
    """Copy of the original for reducing: welded, no loose bits, faces turned outward, no old UVs."""
    me = high.data.copy()
    me.name = name
    low = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(low)
    bm = bmesh.new()
    bm.from_mesh(me)
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=size * 1e-5)
    bmesh.ops.dissolve_degenerate(bm, edges=bm.edges, dist=size * 1e-6)
    bmesh.ops.delete(bm, geom=[v for v in bm.verts if not v.link_faces], context='VERTS')
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(me)
    bm.free()
    while me.uv_layers:
        me.uv_layers.remove(me.uv_layers[0])
    me.materials.clear()
    clear_custom_normals(low)
    return low


def remesh_repair(high, name, size, tol, depth=0):
    """A clean, closed version of the original: Blender's Remesh (Sharp, octree dual contouring) closes
    every gap, crack and open edge and drops zero-thickness flaps; a voxel remesh then makes it one
    clean manifold surface. Cells are about the allowed error. Returns the object, cell size, depth."""
    low = clean_copy(high, name, size)
    if not depth:
        depth = int(min(11, max(8, math.ceil(math.log2(size / (0.9 * 1.25 * tol))))))
    mod = low.modifiers.new("repair", 'REMESH')
    mod.mode = 'SHARP'
    mod.octree_depth = depth
    mod.scale = 0.9
    mod.sharpness = 1.0
    mod.use_remove_disconnected = False
    try:
        mod.use_smooth_shade = False
    except AttributeError:
        pass
    apply_modifiers(low)
    # dual contouring leaves tiny tunnels and edges shared by 3+ faces, which stop the reduction;
    # a voxel remesh of the now closed shape makes one clean manifold surface
    cell = size / (2 ** depth * 0.9)
    vox = low.modifiers.new("clean", 'REMESH')
    vox.mode = 'VOXEL'
    vox.voxel_size = cell
    vox.adaptivity = 0.0
    try:
        vox.use_smooth_shade = False
    except AttributeError:
        pass
    apply_modifiers(low)
    bm = bmesh.new()
    bm.from_mesh(low.data)
    bmesh.ops.triangulate(bm, faces=[f for f in bm.faces if len(f.verts) > 3],
                          quad_method='BEAUTY', ngon_method='BEAUTY')
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(low.data)
    bm.free()
    low.data.update()
    return low, cell, depth


def clear_custom_normals(obj):
    select_only([obj])
    try:
        bpy.ops.mesh.customdata_custom_splitnormals_clear()
    except Exception:  # noqa: BLE001 - nothing to clear / operator missing in this version
        pass


def apply_planar(low, angle):
    bm = bmesh.new()
    bm.from_mesh(low.data)
    bmesh.ops.dissolve_limit(bm, angle_limit=math.radians(angle), use_dissolve_boundaries=False,
                             verts=bm.verts, edges=bm.edges, delimit={'NORMAL', 'MATERIAL'})
    bmesh.ops.triangulate(bm, faces=bm.faces, quad_method='BEAUTY', ngon_method='EAR_CLIP')
    bm.to_mesh(low.data)
    bm.free()
    low.data.update()


def boundary_loops(bm):
    """Every hole as a list of vertex indices (open edges walked round; holes that touch each other at
    a vertex are split there). Works even where neighbouring faces disagree about their winding."""
    bm.verts.index_update()
    bm.edges.index_update()
    adj = {}
    for e in bm.edges:
        if e.is_boundary:
            a, b = e.verts[0].index, e.verts[1].index
            adj.setdefault(a, []).append((b, e.index))
            adj.setdefault(b, []).append((a, e.index))
    used, loops = set(), []
    for v0 in list(adj):
        for v1, e0 in adj[v0]:
            if e0 in used:
                continue
            used.add(e0)
            walk, v = [v0], v1
            while v != v0:
                walk.append(v)
                nxt = [(a, e) for a, e in adj[v] if e not in used]
                if not nxt:
                    walk = None
                    break
                back = [(a, e) for a, e in nxt if a == v0]       # close the loop as soon as possible
                a, e = back[0] if back and len(walk) > 2 else nxt[0]
                used.add(e)
                v = a
            if walk:
                loops += split_at_repeats(walk)
    return loops


def split_at_repeats(walk):
    res, stack, pos = [], [], {}
    for v in walk:
        if v in pos:
            i = pos[v]
            res.append(stack[i:])
            for u in stack[i + 1:]:
                del pos[u]
            stack = stack[:i + 1]
        else:
            pos[v] = len(stack)
            stack.append(v)
    res.append(stack)
    return [r for r in res if len(r) >= 3]


def ear_clip(P, nrm):
    """Triangulate a closed 3D loop by cutting off the sharpest convex corner again and again
    (zips cracks shut from their ends, fills flat holes like ear clipping). Returns index triples."""
    idx = list(range(len(P)))
    tris = []
    while len(idx) > 3:
        Q = P[idx]
        a = np.roll(Q, 1, 0) - Q
        b = np.roll(Q, -1, 0) - Q
        cosang = (a * b).sum(1) / np.maximum(np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1), 1e-20)
        convex = (np.cross(b, a) @ nrm) >= 0
        score = np.where(convex, cosang, cosang - 3.0)        # prefer convex corners, then the sharpest
        k = int(np.argmax(score))
        tris.append((idx[k - 1], idx[k], idx[(k + 1) % len(idx)]))
        del idx[k]
    tris.append(tuple(idx))
    return tris


def fill_holes(obj, size):
    """Close the gaps in the surface with patches (generated models have hundreds: open dormer fronts,
    cracks at roof ridges, open bases). Not filled: outlines of single-layer sheets (the patch would lie
    on the sheet itself) and huge bent openings. Patches are tagged: the error check ignores them, the
    bake paints them from their surroundings. Returns (holes closed, holes left)."""
    me = obj.data
    V, T = mesh_arrays(obj)
    me.calc_loop_triangles()
    tri_poly = np.empty(len(me.loop_triangles), np.int32)
    me.loop_triangles.foreach_get("polygon_index", tri_poly)
    tree = bvh(V, T)
    bm = bmesh.new()
    bm.from_mesh(me)
    bm.verts.ensure_lookup_table()
    loops = [[bm.verts[i] for i in idx] for idx in boundary_loops(bm)]
    closed = left = 0
    for vs in loops:
        P = np.array([v.co[:] for v in vs])
        nrm = np.cross(P, np.roll(P, -1, 0)).sum(0)       # Newell normal: the side the hole faces
        ln = np.linalg.norm(nrm)
        diam = float(np.ptp(P, 0).max())
        if ln < 1e-12 or diam < 1e-9:
            continue
        nrm /= ln
        bent = float(np.abs((P - P.mean(0)) @ nrm).max()) > 0.2 * diam
        if diam > 0.25 * size and bent:
            left += 1
            continue
        tris = ear_clip(P, nrm)
        # a patch lying on existing faces away from the hole's rim = outline of a single-layer sheet
        rim = {v.index for v in vs}
        mids = np.array([P[list(t)].mean(0) for t in tris[:: max(1, len(tris) // 16)]])
        on_sheet = 0
        for m in mids.tolist():
            hit = tree.find_nearest(m)
            if hit[0] is not None and hit[3] < 0.002 * size:
                poly = me.polygons[int(tri_poly[hit[2]])]
                on_sheet += not any(i in rim for i in poly.vertices)
        if on_sheet > len(mids) / 2:
            left += 1
            continue
        for t in tris:
            try:
                bm.faces.new([vs[i] for i in t]).material_index = FILL
            except ValueError:                              # that face exists already
                pass
        closed += 1
    bm.to_mesh(me)
    bm.free()
    me.update()
    return closed, left


def sphere_dirs(n):
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    th = np.pi * (1 + 5 ** 0.5) * i
    return np.stack([np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)], 1)


def orient_by_sight(obj, size, n_dirs=48):
    """Turn every triangle to the side it is seen from: rays go into both half-spaces, the side where
    more of them escape is outside. Triangles seen from neither side follow their neighbours."""
    from collections import deque
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bm.faces.ensure_lookup_table()
    tree = BVHTree.FromBMesh(bm)
    dirs = [Vector(d) for d in sphere_dirs(n_dirs)]
    eps = size * 1e-4
    cast = tree.ray_cast
    score = np.zeros(len(bm.faces), np.int32)
    seen = np.zeros(len(bm.faces), bool)
    for f in bm.faces:
        c, n = f.calc_center_median(), f.normal
        s = 0
        for d in dirs:
            dn = d.dot(n)
            if abs(dn) < 0.05:
                continue
            side = 1 if dn > 0 else -1
            if cast(c + n * (eps * side), d)[0] is None:
                s += side
                seen[f.index] = True
        score[f.index] = s
    o = np.where(score < 0, -1, 1)
    known = score != 0
    queue = deque(np.nonzero(known)[0].tolist())
    while queue:
        fi = queue.popleft()
        for loop in bm.faces[fi].loops:
            other = loop.link_loop_radial_next
            if other is loop or len(loop.edge.link_faces) != 2:
                continue
            gi = other.face.index
            if not known[gi]:
                o[gi] = o[fi] * (1 if other.vert != loop.vert else -1)
                known[gi] = True
                queue.append(gi)
    flip = [bm.faces[i] for i in np.nonzero(o < 0)[0]]
    if flip:
        bmesh.ops.reverse_faces(bm, faces=flip)
    # whole pieces that no ray from outside reaches (inner shells, enclosed cavities) are dropped
    comp = np.full(len(bm.faces), -1)
    hidden = []
    for f0 in bm.faces:
        if comp[f0.index] >= 0:
            continue
        stack, members = [f0], []
        comp[f0.index] = f0.index
        while stack:
            f = stack.pop()
            members.append(f)
            for e in f.edges:
                for g in e.link_faces:
                    if comp[g.index] < 0:
                        comp[g.index] = f0.index
                        stack.append(g)
        if not seen[[f.index for f in members]].any():
            hidden += members
    if hidden and len(hidden) < len(bm.faces):
        bmesh.ops.delete(bm, geom=hidden, context='FACES')
    else:
        hidden = []
    bm.to_mesh(obj.data)
    bm.free()
    obj.data.update()
    return len(flip) / max(len(o), 1), float((~seen).mean()), len(hidden)


def orient_like(high, low, size):
    """Turn the original's triangles the same way as the nearest low-poly triangle, so its normals
    (bake source) point outward too."""
    Vl, Tl = mesh_arrays(low)
    nl = np.cross(Vl[Tl[:, 1]] - Vl[Tl[:, 0]], Vl[Tl[:, 2]] - Vl[Tl[:, 0]])
    tree = bvh(Vl, Tl)
    me = high.data
    C = np.empty(len(me.polygons) * 3, np.float32)
    me.polygons.foreach_get("center", C)
    N = np.empty(len(me.polygons) * 3, np.float32)
    me.polygons.foreach_get("normal", N)
    C, N = C.reshape(-1, 3), N.reshape(-1, 3)
    near = tree.find_nearest
    limit = 0.05 * size
    flip = np.zeros(len(C), bool)
    for i, c in enumerate(C.tolist()):
        hit = near(c, limit)
        if hit[0] is not None:
            flip[i] = float(np.dot(N[i], nl[hit[2]])) < 0
    if flip.any():
        bm = bmesh.new()
        bm.from_mesh(me)
        bm.faces.ensure_lookup_table()
        bmesh.ops.reverse_faces(bm, faces=[bm.faces[i] for i in np.nonzero(flip)[0]])
        bm.to_mesh(me)
        bm.free()
        me.update()
    return float(flip.mean())


def replace_mesh(obj, me):
    old = obj.data
    obj.data = me
    if old.users == 0:
        bpy.data.meshes.remove(old)


def apply_modifiers(obj):
    dg = bpy.context.evaluated_depsgraph_get()
    me = bpy.data.meshes.new_from_object(obj.evaluated_get(dg))
    obj.modifiers.clear()
    replace_mesh(obj, me)


def collapse_search(low, meter, tol, args, keep_group=None):
    """Decimate-Collapse ratio for the low's current mesh: halve the triangles while the error allows,
    then narrow down. Returns (ratio, {ratio: (tris, error)}, modifier)."""
    note = lambda e: f"{e[0] / tol:.2f}x / {e[1] / tol:.2f}x tol"  # noqa: E731
    F = len(low.data.polygons)
    mod = low.modifiers.get("collapse") or low.modifiers.new("collapse", 'DECIMATE')
    mod.decimate_type = 'COLLAPSE'
    mod.use_collapse_triangulate = True
    if keep_group:                                     # vertices in the group are not touched
        mod.vertex_group = keep_group
        mod.invert_vertex_group = True
        mod.vertex_group_factor = 1.0
    cache = {}

    def check(r):
        mod.ratio = r
        V, T, M = mesh_arrays(low, evaluated=True, with_fill=True)
        e = meter.measure(V, T, M)
        ok = e[0] <= tol and e[1] <= 3 * tol
        cache[r] = (len(T), e)
        log(f"  {len(T):>9,} tris: {'ok ' if ok else 'too far'} ({note(e)})")
        return ok

    r_min = min(1.0, max(args.min_faces / F, 1e-5))
    r_cap = min(1.0, args.max_faces / F) if args.max_faces else 1.0
    ok_r, bad_r = r_cap, None
    r = min(0.5, r_cap)
    while r >= r_min:
        if check(r):
            ok_r, r = r, r / 2
        else:
            bad_r = r
            break
    if bad_r is None and ok_r > r_min and r < r_min:     # never failed: try the floor
        ok_r = r_min if check(r_min) else ok_r
    if bad_r is not None and bad_r < ok_r:              # (equal: even --max-faces is too few)
        for _ in range(4):                                # narrow down between ok and too far
            mid = math.sqrt(ok_r * bad_r)
            if check(mid):
                ok_r = mid
            else:
                bad_r = mid
    if ok_r < 1.0 and ok_r not in cache:
        check(ok_r)
    if ok_r >= 1.0:
        cache[1.0] = (F, meter.measure(*mesh_arrays(low, with_fill=True)))
    return ok_r, cache, mod


def reduce_by_shape(low, meter, tol, args):
    """Smallest triangle count whose error stays within tol (99 % of the surface) and 3*tol (99.9 %).
    Then the places that lost a detail anyway (more than 2*tol off: fins, thin parts, junctions) keep
    their triangles and the rest is reduced again - so complex spots stay detailed, plain ones get few."""
    note = lambda e: f"{e[0] / tol:.2f}x / {e[1] / tol:.2f}x tol"  # noqa: E731
    F0 = len(low.data.polygons)

    # 1. optional planar merge, halved until it passes the error check
    angle = args.planar
    while angle >= 0.5:
        keep = low.data.copy()
        apply_planar(low, angle)
        e = meter.measure(*mesh_arrays(low, with_fill=True))
        if e[0] <= tol and e[1] <= 3 * tol:
            log(f"  planar {angle:g} deg: {F0:,} -> {len(low.data.polygons):,} tris ({note(e)})")
            bpy.data.meshes.remove(keep)
            break
        log(f"  planar {angle:g} deg moves the surface too much ({note(e)}), trying {angle / 2:g}")
        replace_mesh(low, keep)
        angle /= 2

    # 2. collapse over the whole model
    ok_r, cache, mod = collapse_search(low, meter, tol, args)

    # 3. keep lost details: lock the vertices around them, reduce the rest again
    from mathutils.kdtree import KDTree
    locked = set()
    rad = 2 * tol
    for rnd in range(1, 4):
        if ok_r >= 1.0:
            break
        mod.ratio = ok_r
        V, T, M = mesh_arrays(low, evaluated=True, with_fill=True)
        spots = meter.lost_spots(V, T, M, 2 * tol)
        if len(spots) < 20:
            break
        if rnd == 1:                                    # start from a safely reduced copy, not full density
            safe = [r for r, (n, e) in cache.items() if r < 1.0 and e[1] <= 0.5 * tol]
            if safe:
                mod.ratio = min(safe)
                apply_modifiers(low)
        Vb = mesh_arrays(low)[0]
        kd = KDTree(len(Vb))
        for i, v in enumerate(Vb.tolist()):
            kd.insert(v, i)
        kd.balance()
        before = len(locked)
        for p in spots.tolist():
            locked.update(i for _, i, _ in kd.find_range(p, rad))
        log(f"  round {rnd}: {len(spots):,} points lost a detail -> keeping {len(locked):,} vertices "
            f"(+{len(locked) - before:,}) of {len(Vb):,}, reducing the rest again")
        vg = low.vertex_groups.get("keep") or low.vertex_groups.new(name="keep")
        vg.add(sorted(locked), 1.0, 'REPLACE')
        ok_r, cache, mod = collapse_search(low, meter, tol, args, keep_group="keep")

    final = cache[ok_r][1]
    if ok_r < 1.0:
        mod.ratio = ok_r
        apply_modifiers(low)
    else:
        low.modifiers.clear()
    low.vertex_groups.clear()
    if args.max_faces and len(low.data.polygons) >= args.max_faces and final[0] > tol:
        log(f"  note: --max-faces {args.max_faces:,} is below what --error needs")
    return final


def finish_low(low, size):
    """Triangles only; no flat-collapsed, doubled or loose bits (reducing folds thin parts into
    triangles lying on top of each other - an exporter would delete one of them and leave a hole)."""
    bm = bmesh.new()
    bm.from_mesh(low.data)
    bmesh.ops.triangulate(bm, faces=[f for f in bm.faces if len(f.verts) > 3])
    bmesh.ops.dissolve_degenerate(bm, edges=bm.edges, dist=size * 1e-6)
    bmesh.ops.triangulate(bm, faces=[f for f in bm.faces if len(f.verts) > 3])
    bm.verts.index_update()
    groups = {}
    for f in bm.faces:
        groups.setdefault(tuple(sorted(v.index for v in f.verts)), []).append(f)
    doubled = [f for g in groups.values() if len(g) > 1 for f in g]
    if doubled:
        bmesh.ops.delete(bm, geom=doubled, context='FACES_ONLY')
    bmesh.ops.delete(bm, geom=[e for e in bm.edges if not e.link_faces], context='EDGES')
    bmesh.ops.delete(bm, geom=[v for v in bm.verts if not v.link_faces], context='VERTS')
    bm.to_mesh(low.data)
    bm.free()
    return len(doubled)


def unwrap(obj, margin):
    select_only([obj])
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.uv.smart_project(angle_limit=math.radians(66), island_margin=margin, area_weight=0.0,
                             correct_aspect=True, scale_to_bounds=False)
    try:
        bpy.ops.uv.pack_islands(rotate=True, margin=margin)
    except (RuntimeError, TypeError) as e:
        log(f"  (pack islands skipped: {e})")
    bpy.ops.object.mode_set(mode='OBJECT')


def align_low(low, high, size):
    """Bake-only mode: move the low poly onto the original if it sits somewhere else."""
    def box(o):
        V = mesh_arrays(o)[0]
        return V.min(0), V.max(0)
    lo_h, hi_h = box(high)
    lo_l, hi_l = box(low)
    shift = (lo_h + hi_h) / 2 - (lo_l + hi_l) / 2
    if np.linalg.norm(shift) > 0.01 * size:
        low.data.transform(Matrix.Translation(shift.tolist()))
        log(f"  low poly was {np.linalg.norm(shift):.3g} units away from the original: moved onto it")
    ratio = (hi_l - lo_l) / np.maximum(hi_h - lo_h, 1e-9)
    if np.any(np.abs(ratio - 1) > 0.05):
        log(f"  WARNING: low poly size differs from the original ({np.round(ratio, 2)}x): check its scale")


# ============================================================================ baking

def biggest_texture(obj):
    s = 0
    for slot in obj.material_slots:
        m = slot.material
        if m and m.use_nodes:
            for n in m.node_tree.nodes:
                if n.type == 'TEX_IMAGE' and n.image:
                    s = max(s, *n.image.size)
    return s


def principled_of(nt):
    for n in nt.nodes:
        if n.type == 'BSDF_PRINCIPLED':
            return n
    return None


def output_of(nt):
    outs = [n for n in nt.nodes if n.type == 'OUTPUT_MATERIAL']
    for n in outs:
        if n.is_active_output:
            return n
    return outs[0] if outs else None


def color_as_emission(obj):
    """Show each material's base color as pure emission: the color bake then ignores light and metal."""
    saved = []
    for slot in obj.material_slots:
        m = slot.material
        if not m or not m.use_nodes:
            continue
        nt = m.node_tree
        out = output_of(nt)
        if out is None:
            continue
        surf = out.inputs["Surface"]
        old = surf.links[0].from_socket if surf.links else None
        em = nt.nodes.new("ShaderNodeEmission")
        em.inputs["Strength"].default_value = 1.0
        bsdf = principled_of(nt)
        if bsdf is not None:
            bc = bsdf.inputs["Base Color"]
            if bc.links:
                nt.links.new(bc.links[0].from_socket, em.inputs["Color"])
            else:
                em.inputs["Color"].default_value = bc.default_value
        elif old is not None and old.node.type == 'EMISSION':      # unlit glTF material
            src = old.node.inputs["Color"]
            if src.links:
                nt.links.new(src.links[0].from_socket, em.inputs["Color"])
            else:
                em.inputs["Color"].default_value = src.default_value
        nt.links.new(em.outputs["Emission"], surf)
        saved.append((nt, out, old, em))
    return saved


def restore(saved):
    for nt, out, old, em in saved:
        nt.nodes.remove(em)
        if old is not None:
            nt.links.new(old, out.inputs["Surface"])


def bake_one(kind, high, low, node, extrusion, margin):
    select_only([high, low], active=low)
    low.active_material.node_tree.nodes.active = node
    kw = dict(use_selected_to_active=True, cage_extrusion=extrusion, max_ray_distance=0.0,
              margin=margin, use_clear=True, target='IMAGE_TEXTURES')
    if kind == "color":
        kw["type"] = 'EMIT'
    elif kind == "normal":
        kw.update(type='NORMAL', normal_space='TANGENT')
    else:
        kw["type"] = 'ROUGHNESS'
    try:
        bpy.ops.object.bake(margin_type='ADJACENT_FACES', **kw)
    except TypeError:
        bpy.ops.object.bake(**kw)
    if kind == "normal":
        n = fix_inward_normals(node.image)
        if n > 0.001:
            log(f"  {n * 100:.1f} % of the normal map pointed inward (inward triangles left): turned around")
    node.image.pack()


def fix_inward_normals(img):
    """Normal-map pixels pointing into the surface come from inward triangles of the original:
    the right normal is the opposite one."""
    px = np.empty(len(img.pixels), np.float32)
    img.pixels.foreach_get(px)
    px = px.reshape(-1, 4)
    n = px[:, :3] * 2 - 1
    bad = n[:, 2] < 0
    if bad.any():
        px[bad, :3] = (-n[bad] + 1) / 2
        img.pixels.foreach_set(px.ravel())
        img.update()
    return float(bad.mean())


def tri_uvs(obj):
    """Per loop-triangle: vertex indices (n x 3), UVs (n x 3 x 2), material index (n)."""
    me = obj.data
    me.calc_loop_triangles()
    n = len(me.loop_triangles)
    T = np.empty(n * 3, np.int32)
    me.loop_triangles.foreach_get("vertices", T)
    L = np.empty(n * 3, np.int32)
    me.loop_triangles.foreach_get("loops", L)
    M = np.empty(n, np.int32)
    me.loop_triangles.foreach_get("material_index", M)
    uv = np.zeros((len(me.loops), 2), np.float32)
    if me.uv_layers.active:
        flat = np.empty(len(me.loops) * 2, np.float32)
        me.uv_layers.active.data.foreach_get("uv", flat)
        uv = flat.reshape(-1, 2)
    return T.reshape(-1, 3), uv[L.reshape(-1, 3)], M


def barycentric(P, A, B, C):
    v0, v1, v2 = B - A, C - A, P - A
    d00, d01, d11 = (v0 * v0).sum(-1), (v0 * v1).sum(-1), (v1 * v1).sum(-1)
    d20, d21 = (v2 * v0).sum(-1), (v2 * v1).sum(-1)
    den = d00 * d11 - d01 * d01
    den = np.where(np.abs(den) < 1e-20, 1e-20, den)
    b1 = (d11 * d20 - d01 * d21) / den
    b2 = (d00 * d21 - d01 * d20) / den
    W = np.stack([1 - b1 - b2, b1, b2], -1).clip(0, 1)
    return W / np.maximum(W.sum(-1, keepdims=True), 1e-12)


class SurfaceColour:
    """The original's base colour at any point in space: nearest surface point -> its UV -> its image."""

    def __init__(self, high):
        me = high.data
        V = np.empty(len(me.vertices) * 3, np.float32)
        me.vertices.foreach_get("co", V)
        self.V = V.reshape(-1, 3).astype(np.float64)
        self.T, self.UV, self.M = tri_uvs(high)
        self.tree = bvh(self.V, self.T)
        self.images = {}
        for slot_i, slot in enumerate(high.material_slots):
            m = slot.material
            bsdf = principled_of(m.node_tree) if m and m.use_nodes else None
            if bsdf is None or not bsdf.inputs["Base Color"].links:
                continue
            node = bsdf.inputs["Base Color"].links[0].from_node
            if node.type == 'TEX_IMAGE' and node.image and node.image.size[0] > 0:
                w, h = node.image.size
                px = np.empty(w * h * 4, np.float32)
                node.image.pixels.foreach_get(px)
                self.images[slot_i] = px.reshape(h, w, 4)

    def distances(self, P):
        return distances(self.tree, P)

    def colour(self, P):
        """RGB of the nearest original surface point for each point (nan where unknown)."""
        idx = np.zeros(len(P), np.int64)
        loc = np.zeros((len(P), 3))
        near = self.tree.find_nearest
        for i, p in enumerate(P.tolist()):
            hit = near(p)
            if hit[0] is not None:
                idx[i], loc[i] = hit[2], hit[0]
        tv = self.V[self.T[idx]]
        W = barycentric(loc, tv[:, 0], tv[:, 1], tv[:, 2])
        uv = (W[:, :, None] * self.UV[idx]).sum(1) % 1.0
        out = np.full((len(P), 3), np.nan, np.float32)
        for slot_i, img in self.images.items():
            sel = self.M[idx] == slot_i
            if sel.any():
                h, w = img.shape[:2]
                x = np.clip((uv[sel, 0] * w).astype(int), 0, w - 1)
                y = np.clip((uv[sel, 1] * h).astype(int), 0, h - 1)
                out[sel] = img[y, x, :3]
        return out


def texels(tri_uv, size):
    """Pixels (x, y) of a size x size image covered by one UV triangle, with barycentric weights."""
    t = tri_uv.astype(np.float64) * size - 0.5
    lo = np.floor(t.min(0) - 1).astype(int).clip(0, size - 1)
    hi = np.ceil(t.max(0) + 1).astype(int).clip(0, size - 1)
    xs, ys = np.meshgrid(np.arange(lo[0], hi[0] + 1), np.arange(lo[1], hi[1] + 1))
    P = np.stack([xs.ravel(), ys.ravel()], 1).astype(np.float64)
    v0, v1, v2 = t[1] - t[0], t[2] - t[0], P - t[0]
    den = v0[0] * v1[1] - v1[0] * v0[1]
    if abs(den) < 1e-9:
        return xs.ravel()[:0], ys.ravel()[:0], np.zeros((0, 3))
    b1 = (v2[:, 0] * v1[1] - v1[0] * v2[:, 1]) / den
    b2 = (v0[0] * v2[:, 1] - v2[:, 0] * v0[1]) / den
    W = np.stack([1 - b1 - b2, b1, b2], 1)
    pad = 1.0 / max(1.0, np.abs(t[1:] - t[0]).max())        # about one pixel, so edges are covered
    inside = (W >= -pad).all(1)
    return xs.ravel()[inside], ys.ravel()[inside], W[inside].clip(0, 1)


def surface_gaps(low, source, tex, near_limit):
    """Texels of the low poly that sit away from the original surface (where the original had a gap,
    a crack or an open bottom, or a flap the repair removed): pixel coords, 3D points, distances."""
    me = low.data
    V = np.empty(len(me.vertices) * 3, np.float32)
    me.vertices.foreach_get("co", V)
    V = V.reshape(-1, 3).astype(np.float64)
    T, UV, _ = tri_uvs(low)
    corners = V[T]
    probes = np.concatenate([corners, corners.mean(1, keepdims=True),
                             (corners + np.roll(corners, 1, 1)) / 2], 1)          # n x 7 x 3
    d = source.distances(probes.reshape(-1, 3)).reshape(len(T), 7)
    xs, ys, pts = [], [], []
    for i in np.nonzero(d.max(1) > near_limit)[0]:
        x, y, W = texels(UV[i], tex)
        if len(x):
            xs.append(x)
            ys.append(y)
            pts.append(W @ corners[i])
    if not xs:
        return None
    x, y, P = np.concatenate(xs), np.concatenate(ys), np.concatenate(pts)
    dist = source.distances(P)
    far = dist > near_limit
    return x[far], y[far], P[far], dist[far]


def paint_gaps(img, kind, gaps, source, flat_beyond):
    """Colour: what the original looks like right next to the gap (nearest surface point), not what
    the bake saw through it. Normal map: flat where there is no original surface nearby."""
    x, y, P, dist = gaps
    W = img.size[0]
    px = np.empty(len(img.pixels), np.float32)
    img.pixels.foreach_get(px)
    px = px.reshape(W, W, 4)
    if kind == "color":
        c = source.colour(P)
        ok = ~np.isnan(c).any(1)
        px[y[ok], x[ok], :3] = c[ok]
    elif kind == "normal":
        sel = dist > flat_beyond
        px[y[sel], x[sel], :3] = (0.5, 0.5, 1.0)
    else:
        return False
    img.pixels.foreach_set(px.ravel())
    img.update()
    return True


def bake_all(high, low, name, args, size, extrusion, tol):
    tex = args.tex or min(8192, max(2048, biggest_texture(high)))
    margin = max(4, tex // 256)
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    mat.use_backface_culling = args.single_sided
    nt = mat.node_tree
    bsdf = principled_of(nt)
    bsdf.inputs["Metallic"].default_value = 0.0
    bsdf.inputs["Roughness"].default_value = 0.8
    t = time.time()
    source = SurfaceColour(high)
    gaps = surface_gaps(low, source, tex, 0.5 * tol)
    if gaps is not None:
        log(f"  {len(gaps[0]):,} texels lie over gaps of the original or away from it: they get the colour "
            f"of the nearest original surface ({time.time() - t:.0f}s)")
    low.data.polygons.foreach_set("material_index", np.zeros(len(low.data.polygons), np.int32))
    low.data.materials.clear()
    low.data.materials.append(mat)
    nodes = {}
    for i, kind in enumerate(args.maps):
        img = bpy.data.images.new(f"{name}_{kind}", tex, tex, alpha=False)
        if kind != "color":
            img.colorspace_settings.name = 'Non-Color'
        node = nt.nodes.new("ShaderNodeTexImage")
        node.image = img
        node.location = (-700, 300 - 320 * i)
        nodes[kind] = node
    log(f"bake {', '.join(args.maps)} at {tex} px, extrusion {extrusion:.4g} ({extrusion / size * 100:.2f} % of size)")
    for kind in args.maps:
        t = time.time()
        saved = color_as_emission(high) if kind == "color" else []
        try:
            bake_one(kind, high, low, nodes[kind], extrusion, margin)
        finally:
            restore(saved)
        if gaps is not None and paint_gaps(nodes[kind].image, kind, gaps, source, 2 * tol):
            nodes[kind].image.pack()
        log(f"  {kind}: {time.time() - t:.0f}s")
    links = nt.links
    if "color" in nodes:
        links.new(nodes["color"].outputs["Color"], bsdf.inputs["Base Color"])
    if "normal" in nodes:
        nm = nt.nodes.new("ShaderNodeNormalMap")
        nm.location = (-300, -350)
        links.new(nodes["normal"].outputs["Color"], nm.inputs["Color"])
        links.new(nm.outputs["Normal"], bsdf.inputs["Normal"])
    if "roughness" in nodes:
        links.new(nodes["roughness"].outputs["Color"], bsdf.inputs["Roughness"])


def export_glb(obj, path):
    select_only([obj])
    os.makedirs(os.path.dirname(path), exist_ok=True)
    bpy.ops.export_scene.gltf(filepath=path, export_format='GLB', use_selection=True)


# ============================================================================ main

def process(path, args):
    name = os.path.splitext(os.path.basename(path))[0]
    out_dir = os.path.abspath(args.out) if args.out else os.path.join(os.path.dirname(path), "lowpoly_blender")
    out = os.path.join(out_dir, name + ".glb")
    if os.path.exists(out) and not args.overwrite:
        log(f"{name}: already done ({out}), skipped (--overwrite to redo)")
        return None
    t0 = time.time()
    reset_scene()
    device = setup_cycles(args)
    high = import_glb(path, name + "_high")
    V, T = mesh_arrays(high)
    size = float(np.ptp(V, 0).max())
    metres = size > 1.5
    tol = args.error_cm / 100 if args.error_cm else args.error * size
    unit = (lambda d: f"{d * 100:.1f} cm") if metres else (lambda d: f"{d / size * 100:.2f} % of size")
    log(f"{name}: {len(T):,} tris, size {size:.3g}{' m' if metres else ''}, allowed error {unit(tol)}, bake on {device}")

    prepare_high(high, size, args.smooth_angle)
    repaired = not args.low and not args.no_repair
    if not args.low:
        noise = surface_noise(*mesh_arrays(high))
        if not args.strict and 3 * noise > tol:
            log(f"surface noise (small ripple) {unit(noise)}: allowed error raised to {unit(3 * noise)} "
                f"- below that the low poly would only copy the ripple; the normal map keeps it (--strict: don't)")
            tol = 3 * noise
    if repaired:
        t = time.time()
        low, cell, depth = remesh_repair(high, name, size, tol, args.repair_depth)
        log(f"rebuilt as a closed surface (Remesh Sharp, depth {depth}, cells {unit(cell)}): "
            f"{len(low.data.polygons):,} tris, gaps closed ({time.time() - t:.0f}s; --no-repair: keep original)")
        meter = ErrorMeter(low)                         # reduce relative to the clean shape
    else:
        meter = ErrorMeter(high)
    if args.low:
        low = import_glb(os.path.abspath(args.low), name)
        align_low(low, high, size)
        orient_like(high, low, size)
        smooth_by_angle(high, args.smooth_angle)
        if args.new_uv or not low.data.uv_layers:
            while low.data.uv_layers:
                low.data.uv_layers.remove(low.data.uv_layers[0])
        e = meter.measure(*mesh_arrays(low))
    else:
        if not repaired:
            low = clean_copy(high, name, size)
        if not args.keep_holes and not repaired:
            t = time.time()
            closed, left = fill_holes(low, size)
            if closed or left:
                log(f"holes in the surface (from the generator): {closed:,} closed, {left:,} left open "
                    f"({time.time() - t:.0f}s; --keep-holes: don't close)")
        e = reduce_by_shape(low, meter, tol, args)
        finish_low(low, size)
        if not args.keep_holes:
            fill_holes(low, size)                       # what reducing opened (removed folded bits)
        t = time.time()
        flipped, unseen, dropped = orient_by_sight(low, size)
        log(f"faces turned outward: {flipped * 100:.1f} % turned; {dropped:,} triangles of hidden inner "
            f"parts removed ({time.time() - t:.0f}s)")
        smooth_by_angle(low, args.smooth_angle)
        hf = orient_like(high, low, size)
        smooth_by_angle(high, args.smooth_angle)
        log(f"original's faces turned like the low poly: {hf * 100:.1f} %")
    tris = len(mesh_arrays(low)[1])
    log(f"low poly: {tris:,} tris ({tris / len(T) * 100:.1f} %), error 99 %: {unit(e[0])}, "
        f"99.9 %: {unit(e[1])}, mean {unit(e[2])}")

    if not low.data.uv_layers:
        t = time.time()
        unwrap(low, margin=0.003)
        log(f"new UVs: {time.time() - t:.0f}s")
    extrusion = min(max(2.0 * e[1], 0.002 * size), 0.05 * size)
    bake_all(high, low, name, args, size, extrusion, tol)

    export_glb(low, out)
    if args.blend:
        bpy.ops.wm.save_as_mainfile(filepath=os.path.join(out_dir, name + ".blend"), copy=True)
    log(f"saved {out} ({time.time() - t0:.0f}s)")
    return name, len(T), tris, unit(e[0]), time.time() - t0


def main():
    args = parse_args()
    files = collect_inputs(args.inputs)
    if not files:
        log("no .glb files found")
        return
    done = []
    for f in files:
        try:
            r = process(f, args)
            if r:
                done.append(r)
        except Exception as ex:  # noqa: BLE001 - keep going with the next building
            import traceback
            traceback.print_exc()
            log(f"FAILED {os.path.basename(f)}: {ex}")
    if done:
        print("\n  model                              original      low poly   error 99 %     time")
        for n, a, b, e, t in done:
            print(f"  {n[:32]:<32} {a:>10,} -> {b:>9,}   {e:>12}   {t:5.0f}s")


if __name__ == "__main__":
    main()
