"""
qwen_views.py - clean views of one building at set angles (0/90/180/270 ...), made by Qwen-Image-Edit-2511
in ComfyUI from your own photos, ready for make_views.py -> inference_mv.py, or for Hunyuan (hy21_mv.py).

--flat (for Hunyuan): ONE picture per photo - the same side of the building, brought down to a level (0 deg),
flat view with no shadows and a white background, like an architect's elevation. Try the methods on one photo
first, then run all photos with the best one:
    python qwen_views.py my_views_dom2 --name dom --flat --compare --views 90
       -> my_views_dom_flat\\compare\\dom_az091_compare.jpg  (photo | low | eye | prompt)
    python qwen_views.py my_views_dom2 --name dom --flat --out <Hunyuan3D-2.1>\\in\\dom      (eye + delight)
    then, in the Hunyuan3D-2.1 folder:  python hy21_mv.py in\\dom --name dom --elev 0
  eye (default)  Multiple-Angles LoRA with only its tag "front view eye-level shot wide shot": brings the camera down
                 to ground level, same side (it adds sky and ground); then a same-camera clean-up pass (surroundings,
                 shadows, dark sides -> white, even light). Best in the first test (Dom, 3D-map shots from ~30 deg
                 above, 2026-10-30): level, whole building down to its base, porch and gables in place.
  low            the same with "low-angle shot": looks up from below (converging tower), base cut off
  prompt         no LoRA, one pass: Qwen is asked in words to redraw the same side level and flat
  delight        last pass (on by default, --no-delight to skip): only removes baked sun shadows and dark sides,
                 which would end up in the texture while Unreal adds its own sun. In the Dom test it removed them all
                 and kept every detail.

Default: LEVEL, STRAIGHT-ON ELEVATIONS, like an architect's drawing rendered as a photo: no perspective, camera at
the building's mid-height, no shadows, white background, the whole building - one picture per angle, drawn from the
photos nearest to it. Works with photos from any height and crop (street photos, hand-cropped 3D-map screenshots
from above): Qwen redraws the view, so the photos do not have to line up. Then make_views.py --prep (level, no
--keep-frame, no --elev).
With several photos per view Qwen can blend them (a tower twice); and from shots taken from above it may keep the
tilt. Then use --level: every view is drawn only from the photo at its own angle, and the Multiple-Angles LoRA's
"eye-level shot" brings the camera down. Give it one photo per angle you want (e.g. the 4 --clean views):
    python qwen_views.py my_views_dom --name dom --out my_views_dom_level --level

--sheet: ONE Qwen picture with all 4 straight-on elevations in a 2x2 grid (like an architect's sheet: same scale,
same height, same ground line - the views agree with each other), upscaled x2 and cut into 4 separate pictures
<name>_az000/090/180/270.png (top-left front, top-right right side, bottom-left back, bottom-right left side);
the whole sheet is kept in <out>\\sheet\\. Qwen draws a new sheet instead of editing a photo, so it does not keep
the photos' camera: this is the mode for shots taken from above. Uses the 3 photos nearest to 0, 90 and 270.
    python qwen_views.py my_views_dom --name dom --out my_views_dom_sheet --sheet

    1. Start ComfyUI (ImageGen\\ComfyUI_windows_portable\\run_nvidia_gpu.bat) and leave it open.
    2. In the Pixal3D folder, with Windows Python (the script needs nothing but Python itself):
         python qwen_views.py photos\\church_front.jpg:0 photos\\church_side.jpg:90 photos\\church_back.jpg:200 --name church
       -> my_views_church\\church_az000.png  church_az090.png  church_az180.png  church_az270.png
    3. In WSL:  python make_views.py my_views_church --prep
                python inference_mv.py --views_dir my_views_church --output out/church_mv.glb --low_vram

Each photo gets its angle as photo.jpg:<degrees>, or from az<degrees> in its name (church_az200.jpg),
or give a folder of such names. 0 = front, 90 = the side on your right when you face the front,
180 = back, 270 = the side on your left. Rough angles are fine. Any number of photos: Qwen takes 3,
so every view uses the 3 photos closest to it (closest first).

    --views 180          only this view (or several: --views 0 180); default 0 90 180 270
    --seed 7             another try; one seed for all views keeps them alike
    --no-lightning       without the 4-step Lightning LoRA: 30 steps at cfg 4, slower, sometimes cleaner
    --lora file[:0.8]    extra LoRA from ComfyUI\\models\\loras, e.g. the Unblur-Upscale one (repeatable)
    --angles-lora        also use the Multiple-Angles LoRA; its camera tag is set relative to Picture 1 (experimental)
    --upscale            Real-ESRGAN x4, then halved: twice the size, crisper edges
    --size 1328x1328     output size in pixels (multiples of 16)
    --dry-run            only print which photos and which prompt each view gets (to paste into ComfyUI by hand)
    --overwrite          redo views that already exist (otherwise they are skipped)

--clean: every photo KEEPS ITS OWN CAMERA. Qwen only cuts the building out (neighbours, labels, trees, cars,
ground, sky, shadows removed; hidden parts filled in; texture sharpened) on white, at the same place and size.
For multi-view from your own shots, e.g. 4 Google Earth screenshots around one building at the same tilt and
range. The prompt is the same for every building and every angle; --hint names the building when it touches others.

    python qwen_views.py shots\\dom_az000.png shots\\dom_az090.png shots\\dom_az180.png shots\\dom_az270.png --clean
    python qwen_views.py shots --clean --name dom --hint "the cathedral with the tall green-spired tower"
       -> my_views_dom\\dom_az000.png ... (one per photo, same angle in the name)
    then in WSL:  python make_views.py my_views_dom --prep --keep-frame --elev 30
                  python inference_mv.py --views_dir my_views_dom --output out/dom_mv.glb --low_vram

    --refs 2|3           also show Qwen the nearest other photos (to see what is hidden; default 1 = only its own)
    --size               default in --clean: the photo's own shape at Qwen's pixel count (keeps the framing)
"""
import argparse
import json
import mimetypes
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
AZ_RE = re.compile(r"az(?:im)?_?(-?\d+(?:\.\d+)?)", re.I)          # same rule as make_views.py
NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")

# model files of Claude outputs/qwen-angles-gguf.json (ComfyUI\models\unet, text_encoders, vae, loras, upscale_models)
UNET = "qwen-image-edit-2511-Q4_K_M.gguf"
CLIP = "qwen_2.5_vl_7b_fp8_scaled.safetensors"
VAE = "qwen_image_vae.safetensors"
LIGHTNING = "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors"
ANGLES = "qwen-image-edit-2511-multiple-angles-lora.safetensors"
ESRGAN = "RealESRGAN_x4plus.pth"

SIDES = ["the front", "the front-right corner", "the right side", "the back-right corner",
         "the back", "the back-left corner", "the left side", "the front-left corner"]
TAGS = ["front view", "front-right quarter view", "right side view", "back-right quarter view",
        "back view", "back-left quarter view", "left side view", "front-left quarter view"]
STRAIGHT = {0: "the front of the building, seen straight on",
            90: "the right side of the building (the side on your right when you face the front), seen straight on",
            180: "the back of the building, seen straight on from behind",
            270: "the left side of the building (the side on your left when you face the front), seen straight on"}
STYLE = [
    "Draw it like an architectural elevation rendered as a photo: orthographic, camera exactly level with the middle "
    "of the building and very far away, no perspective, all vertical lines straight and parallel, all horizontal lines "
    "level. Even where the pictures look down on the building from above (a 3D map or a drone), show the walls straight "
    "on from ground level: the roof only as it is seen from the side, never from above.",
    "Show the whole building from the ground line to the very top of the roof, towers and spire, nothing cut off, "
    "centred, with white space all around.",
    "Keep the building exactly as in the pictures: same shape and proportions, same number and positions of windows, "
    "doors, towers and buttresses, same roof shape, materials and colours. Where no picture shows this side, continue "
    "the same materials, window style and roof so that it fits the rest of the building.",
    "Remove everything that is not this building: map labels, pins, icons, trees, bushes, cars, people, signs, "
    "street lamps, poles, wires, "
    "fences, scaffolding, neighbouring buildings, ground, road and sky. Pure white background.",
    "Even overcast light from everywhere: no shadows, no cast shadows, no shadow on the ground, no dark sides, no sun "
    "highlights; every wall equally bright, like a flat texture scan.",
    "Sharp, high-detail, clean texture even where the photos are blurry or low resolution: crisp stone, brick, plaster, "
    "roof tiles and window frames. No blur, no noise, no JPEG artefacts, no text, no labels, no frame.",
]
NEGATIVE = ("shadows, cast shadows, dark side, sunlight, trees, cars, people, signs, street, sky, ground, "
            "perspective distortion, view from above, aerial view, tilted, cropped, cut off, blurry, noise, "
            "jpeg artifacts, text, map labels, watermark, frame")

# --clean: same camera, building only. Nothing in here names a building or an angle, so it fits every view.
CLEAN = [
    "Clean up Picture 1 without changing its camera: keep exactly the same viewpoint, angle, height, perspective, "
    "framing and zoom, so the building stays at the same place and the same size in the picture. Do not rotate, "
    "move, zoom, straighten or re-frame the view.",
    "Keep only {target}, with everything that is built onto it: towers, spires, roofs, gables, dormers, chimneys, "
    "porches, stairs and side wings.",
    "Remove everything else: neighbouring buildings (cut them off cleanly where they touch it), map labels, pins, "
    "icons, text and logos, trees, bushes, grass, cars, people, signs, street lamps, poles, wires, fences, "
    "scaffolding, the ground, roads, squares, pavements, water and sky. Where something covered part of the "
    "building, continue its walls, windows and roof behind it.",
    "Keep the building exactly as it is: same shape and proportions, same number and positions of windows, doors, "
    "gables, towers, buttresses and dormers, same roof shapes, materials and colours.",
    "Pure white background all around. The building ends cleanly at its base and stands on nothing.",
    "Even soft daylight from all sides: no cast shadows on or around the building, no dark sides, no sun highlights.",
    "Sharp, clean detail: crisp stone, brick, plaster, roof tiles and window frames, straight edges; repair blurry "
    "or melted areas so they match the rest of the building. No blur, no noise, no text, no labels, no frame.",
]
CLEAN_NEGATIVE = ("shadows, cast shadows, dark side, sunlight, trees, cars, people, signs, street, sky, ground, "
                  "other buildings, map labels, icons, text, watermark, frame, blurry, noise, jpeg artifacts, "
                  "rotated view, zoomed, cropped")
QWEN_AREA = 1328 * 1328          # Qwen-Image's native pixel count

# --flat: ONE picture per photo, same side of the building, brought down to a level, flat, shadow-free view.
# Methods (try them on one photo first with --compare):
#   low     2 passes. 1) Multiple-Angles LoRA with ONLY its tag, "<sks> front view low-angle shot wide shot": the
#           LoRA treats the input as its eye-level front view, so 'low-angle' (-30 deg) brings a camera that was
#           ~30 deg above down to level, without turning around the building. 2) the --clean prompt (same camera)
#           removes surroundings, shadows and dark sides on white.
#   eye     the same with "eye-level shot" (for photos taken only a little from above)
#   prompt  1 pass, no LoRA, Qwen asked in words to redraw the same side as a level elevation
FLAT_TAG = {"low": "<sks> front view low-angle shot wide shot", "eye": "<sks> front view eye-level shot wide shot"}
FLAT_METHODS = ["low", "eye", "prompt"]
FLAT_PROMPT = [
    "Picture 1 shows a building, possibly seen from above (a 3D map or a drone) and in perspective.",
    "Redraw exactly this building as a flat architectural elevation of the SAME side: bring the camera down to ground "
    "level so it looks exactly horizontally at the building, and far back with a long telephoto lens - no "
    "perspective at all, every vertical line vertical and parallel, every horizontal line level, the walls seen "
    "straight on at their full height.",
    "Do not turn around the building: the same side faces the camera as in Picture 1, and every tower, spire, gable, "
    "porch and wing keeps its place from left to right and appears exactly once.",
    "Because the camera is level, no roof is seen from above: each roof shows only as its sloped outline above its "
    "walls; behind the front walls only what rises above them is visible (towers, spires, higher roofs).",
    "Keep the building exactly: same shape and proportions, same number and positions of windows, doors, gables, "
    "buttresses and towers, same materials and colours.",
    "Remove everything that is not this building: map labels, trees, cars, people, signs, poles, neighbouring "
    "buildings, ground, road and sky. Pure white background; the building stands on a straight ground line.",
    "Flat even overcast light from everywhere: no shadows at all, no cast shadows, no shadow on the ground, no dark "
    "sides, no sun highlights; every wall equally bright, like a flat texture scan.",
    "Sharp, clean detail: crisp stone, roof tiles and window frames. No text, no labels, no frame.",
]
# --delight: one more pass that ONLY removes the baked sun light (one job per pass works better than a long list)
DELIGHT = [
    "Relight Picture 1 with flat, completely diffuse overcast light that comes equally from every direction.",
    "Remove every shadow: the dark wedges beside buttresses and towers, the shadows under gables, eaves, porches and "
    "roof edges, shadows cast by one part of the building on another, and the darker sides of the building. Every "
    "wall gets the same brightness as the lit stone next to it; roofs keep their own colour, evenly lit.",
    "Keep everything else exactly the same: the same camera, framing, size and position of the building, the same "
    "shapes, windows, doors, ornaments, materials and colours, and the same plain white background. Do not add, "
    "remove or move anything.",
]
DELIGHT_NEGATIVE = ("shadows, cast shadows, dark side, sunlight, sun highlights, high contrast, dramatic lighting, "
                    "changed shape, moved, cropped, blurry")
FLAT_NEGATIVE = ("view from above, aerial view, bird's-eye view, roof seen from above, perspective, converging lines, "
                 "tilted, rotated view, shadows, cast shadows, dark side, sunlight, ground, sky, trees, cars, people, "
                 "text, frame, blurry")


def angdist(a, b):
    d = abs(a - b) % 360
    return min(d, 360 - d)


def sector(a):
    return int(round((a % 360) / 45.0)) % 8


def deg(a):
    return f"{a:g}"


def view_text(t):
    t %= 360
    if t in STRAIGHT:
        return STRAIGHT[t]
    return f"the building seen from {SIDES[sector(t)]} ({deg(t)}° around from the front)"


def build_prompt(t, refs, tag=None):
    n = len(refs)
    head = {1: "Picture 1 is a photo", 2: "Pictures 1 and 2 are photos"}.get(n, f"Pictures 1 to {n} are photos")
    head += " (or a 3D-map screenshot, possibly seen from above)" if n == 1 else \
        " (or 3D-map screenshots, possibly seen from above)"
    lines = [tag] if tag else []
    lines.append(f"{head} of {'a' if n == 1 else 'the same'} building. Angles go around the building to the right: 0° is the front, "
                 "90° the side on your right when you face the front, 180° the back, 270° the side on your left.")
    lines += [f"Picture {i} shows it from {SIDES[sector(a)]} (about {deg(a)}°)." for i, (_, a) in enumerate(refs, 1)]
    lines.append(f"Make one clean picture of this exact building: {view_text(t)}.")
    if angdist(refs[0][1], t) < 1:
        lines.append("Show the same side of the building as Picture 1 - do not turn around it and do not add "
                     "anything that is not in Picture 1: only bring the camera down to the middle of the building's "
                     "height and far back, so the building is seen level and straight on, not from above. "
                     "Every tower, spire and wing exactly once, where it is in Picture 1.")
    return "\n".join(lines + STYLE)


SHEET = [
    "Create a 2x2 grid of four orthographic architectural elevations of this exact building, all at the same "
    "scale, same height and same ground line:",
    "top-left = front elevation (0°, the front facing the camera),",
    "top-right = right side elevation (90°, the side on your right when you face the front),",
    "bottom-left = back elevation (180°),",
    "bottom-right = left side elevation (270°).",
    "Each view is centred in its own square cell, with a wide plain white gap between the cells; no view crosses "
    "into another cell.",
    "Camera horizontal at the building's mid-height, far away, no perspective, all vertical lines parallel. Even "
    "where the pictures look down on the building from above (a 3D map or a drone), draw every elevation level and "
    "straight on: the roofs only as they are seen from the side.",
    "Every tower, spire and wing appears once in each view, where it belongs.",
    "Keep every window, door, tower, buttress and roof shape as in the pictures; where no picture shows a side, "
    "continue the same materials and window pattern.",
    "Plain white background, no text, no labels, no frames, no trees, cars or people.",
    "Overcast sky, completely diffuse even lighting, no shadows, no dark sides, every wall equally bright, "
    "like a flat texture scan.",
]
SHEET_CELLS = [(0, 0, 0), (90, 1, 0), (180, 0, 1), (270, 1, 1)]     # angle, column, row


def sheet_refs(photos, n):
    """The photos nearest to the front, the right and the left side (then the back), at most n."""
    chosen = []
    for want in (0, 90, 270, 180):
        rest = [pa for pa in photos if pa not in chosen]
        if not rest or len(chosen) >= n:
            break
        chosen.append(min(rest, key=lambda pa: angdist(pa[1], want)))
    return chosen


def build_sheet_prompt(refs):
    n = len(refs)
    head = {1: "Picture 1 is a photo (or a 3D-map screenshot, possibly seen from above) of a building",
            2: "Pictures 1 and 2 are photos (or 3D-map screenshots, possibly seen from above) of the same building"
            }.get(n, f"Pictures 1 to {n} are photos (or 3D-map screenshots, possibly seen from above) of the same "
                     "building")
    lines = [head + ". Angles go around the building to the right: 0° is the front, 90° the side on your right "
                    "when you face the front, 180° the back, 270° the side on your left."]
    lines += [f"Picture {i} shows it from {SIDES[sector(a)]} (about {deg(a)}°)." for i, (_, a) in enumerate(refs, 1)]
    return "\n".join(lines + SHEET)


def build_clean_prompt(refs, hint=None):
    """Same-view clean-up of refs[0]; refs[1:] (other photos of the building) only help to see what is hidden."""
    target = "the main building in the centre of the picture" + (f" ({hint})" if hint else "")
    lines = []
    if len(refs) > 1:
        others, it = ("Picture 2 shows", "it") if len(refs) == 2 else (f"Pictures 2 to {len(refs)} show", "them")
        lines.append(f"{others} the same building from other sides. Use {it} only to see which parts belong to the "
                     f"building and what is hidden in Picture 1; do not take {'its' if it == 'it' else 'their'} viewpoint.")
    return "\n".join(lines + [s.format(target=target) for s in CLEAN])


def image_size(path):
    """(width, height) of a PNG, JPEG, WebP or BMP with the standard library only; None if unknown."""
    import struct
    try:
        with open(path, "rb") as fh:
            head = fh.read(64)
            if head[:8] == b"\x89PNG\r\n\x1a\n":
                return struct.unpack(">II", head[16:24])
            if head[:2] == b"BM":
                w, h = struct.unpack("<ii", head[18:26])
                return w, abs(h)
            if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
                kind = head[12:16]
                if kind == b"VP8X":
                    return 1 + int.from_bytes(head[24:27], "little"), 1 + int.from_bytes(head[27:30], "little")
                if kind == b"VP8L":
                    b = int.from_bytes(head[21:25], "little")
                    return 1 + (b & 0x3FFF), 1 + ((b >> 14) & 0x3FFF)
                if kind == b"VP8 ":
                    w, h = struct.unpack("<HH", head[26:30])
                    return w & 0x3FFF, h & 0x3FFF
                return None
            if head[:2] == b"\xff\xd8":
                fh.seek(2)
                while True:
                    m = fh.read(2)
                    if len(m) < 2 or m[0] != 0xFF:
                        return None
                    if m[1] in (0xD8, 0x01) or 0xD0 <= m[1] <= 0xD7:
                        continue
                    seglen = struct.unpack(">H", fh.read(2))[0]
                    if m[1] in (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF):
                        h, w = struct.unpack(">xHH", fh.read(5))
                        return w, h
                    fh.seek(seglen - 2, 1)
    except (OSError, struct.error):
        return None
    return None


def size_like(path, area=QWEN_AREA):
    """Output size with the photo's shape and Qwen's pixel count, multiples of 16 (keeps the framing)."""
    wh = image_size(path)
    if not wh or min(wh) <= 0:
        return 1328, 1328
    r = wh[0] / wh[1]
    return (max(256, round((area * r) ** 0.5 / 16) * 16), max(256, round((area / r) ** 0.5 / 16) * 16))


def angle_from_name(f):
    m = AZ_RE.search(f.stem)
    return float(m.group(1)) % 360 if m else None


def parse_photos(items):
    photos = []
    for s in items:
        p, a = Path(s), None
        if ":" in s:                                    # photo.jpg:90 (C:\... alone is a path)
            head, tail = s.rsplit(":", 1)
            if NUM_RE.fullmatch(tail.strip()) and head and Path(head).exists():
                p, a = Path(head), float(tail) % 360
        if p.is_dir():
            files = sorted(f for f in p.iterdir() if f.suffix.lower() in EXTS)
            if not files:
                sys.exit(f"no images in {p}")
            for f in files:
                fa = angle_from_name(f)
                if fa is None:
                    sys.exit(f"{f.name}: no angle in the name (e.g. {f.stem}_az090{f.suffix}); "
                             f"or list the photos one by one as {f.name}:90")
                photos.append((f, fa))
            continue
        if not p.is_file():
            sys.exit(f"photo not found: {s}")
        if p.suffix.lower() not in EXTS:
            sys.exit(f"{p.name}: not an image ComfyUI reads ({', '.join(sorted(EXTS))})")
        a = a if a is not None else angle_from_name(p)
        if a is None:
            sys.exit(f"{p.name}: no angle - write it as {p.name}:90, or put az090 in the name")
        photos.append((p, a))
    return photos


def parse_lora(s):
    head, _, tail = s.rpartition(":")
    if head and NUM_RE.fullmatch(tail):
        return head, float(tail)
    return s, 1.0


class Comfy:
    def __init__(self, url, timeout):
        self.url, self.timeout, self.client = url.rstrip("/"), timeout, uuid.uuid4().hex

    def call(self, path, data=None, headers=None):
        req = urllib.request.Request(self.url + path, data=data, headers=headers or {})
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            raise RuntimeError(self.explain(e.code, e.read())) from None
        except (urllib.error.URLError, ConnectionError, TimeoutError) as e:
            sys.exit(f"cannot reach ComfyUI at {self.url} ({getattr(e, 'reason', e)}).\n"
                     "Start it with run_nvidia_gpu.bat (ImageGen\\ComfyUI_windows_portable) and wait for\n"
                     "'To see the GUI go to: http://127.0.0.1:8188'. Run this script with Windows Python:\n"
                     "inside WSL, 127.0.0.1 is not the Windows ComfyUI.")

    @staticmethod
    def explain(code, body):
        try:
            j = json.loads(body)
        except ValueError:
            return f"ComfyUI answered {code}: {body[:500]!r}"
        out = [f"ComfyUI refused the job: {j.get('error', {}).get('message', j)}"]
        for nid, ne in (j.get("node_errors") or {}).items():
            for e in ne.get("errors", []):
                out.append(f"  {ne.get('class_type', nid)}: {e.get('message')} - {e.get('details', '')}")
        if any("not in" in o for o in out):
            out.append("  (a model file name differs from yours: see --unet/--clip/--vae/--lightning-lora/--esrgan,"
                       " or a custom node is missing - UnetLoaderGGUF needs ComfyUI-GGUF)")
        return "\n".join(out)

    def upload(self, path, name):
        b = uuid.uuid4().hex
        mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        body = b"".join([
            f"--{b}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n{v}\r\n".encode()
            for k, v in (("type", "input"), ("subfolder", "qwen_views"), ("overwrite", "true"))
        ] + [f"--{b}\r\nContent-Disposition: form-data; name=\"image\"; filename=\"{name}\"\r\n"
             f"Content-Type: {mime}\r\n\r\n".encode(), path.read_bytes(), f"\r\n--{b}--\r\n".encode()])
        j = json.loads(self.call("/upload/image", body, {"Content-Type": f"multipart/form-data; boundary={b}"}))
        return f"{j['subfolder']}/{j['name']}" if j.get("subfolder") else j["name"]

    def run(self, graph, save_node):
        """Queue the graph; the first image of save_node (or of each node in a list of them)."""
        nodes = [save_node] if isinstance(save_node, str) else list(save_node)
        data = json.dumps({"prompt": graph, "client_id": self.client}).encode()
        pid = json.loads(self.call("/prompt", data, {"Content-Type": "application/json"}))["prompt_id"]
        t0, last = time.time(), 0.0
        while True:
            h = json.loads(self.call(f"/history/{pid}"))
            if pid in h:
                entry = h[pid]
                status = entry.get("status") or {}
                if status.get("status_str") == "error":
                    msgs = [m[1] for m in status.get("messages", []) if m[0] == "execution_error"]
                    why = "; ".join(f"{m.get('node_type')}: {m.get('exception_message', '').strip()}" for m in msgs)
                    if re.search(r"memory|allocat", why, re.I):
                        why += "\n    (GPU memory: close Pixal3D/WSL, Unreal and browsers, or use a smaller --size like 1024x1024)"
                    raise RuntimeError(f"ComfyUI failed: {why or status}")
                outs = entry.get("outputs", {})
                imgs = [(outs.get(n) or {}).get("images") for n in nodes]
                if all(imgs):
                    print(f"\r    done in {time.time() - t0:.0f} s            ")
                    return imgs[0][0] if isinstance(save_node, str) else [i[0] for i in imgs]
                if status.get("completed"):
                    raise RuntimeError("ComfyUI finished without an image (was the job cancelled?)")
            el = time.time() - t0
            if el > self.timeout:
                raise RuntimeError(f"no result after {self.timeout} s (still in ComfyUI's queue?)")
            if el - last >= 5:
                print(f"\r    working... {el:.0f} s", end="", flush=True)
                last = el
            time.sleep(1.0)

    def download(self, img):
        q = urllib.parse.urlencode({"filename": img["filename"], "subfolder": img.get("subfolder", ""),
                                    "type": img.get("type", "output")})
        return self.call(f"/view?{q}")


def build_graph(a, images, prompt, negative, prefix, size, cells=None):
    g = {}

    def node(cls, **inputs):
        nid = str(len(g) + 1)
        g[nid] = {"class_type": cls, "inputs": inputs}
        return nid

    model = [node("UnetLoaderGGUF", unet_name=a.unet), 0]
    loras = ([(a.lightning_lora, 1.0)] if a.lightning else []) + ([(a.angles_file, 1.0)] if a.angles_lora else []) + a.lora
    for name, strength in loras:
        model = [node("LoraLoaderModelOnly", model=model, lora_name=name, strength_model=strength), 0]
    model = [node("ModelSamplingAuraFlow", model=model, shift=3.1), 0]
    model = [node("CFGNorm", model=model, strength=1.0), 0]
    clip = [node("CLIPLoader", clip_name=a.clip, type="qwen_image"), 0]
    vae = [node("VAELoader", vae_name=a.vae), 0]
    pics = {f"image{i}": [node("LoadImage", image=im), 0] for i, im in enumerate(images, 1)}
    cond = []
    for text in (prompt, negative):
        c = [node("TextEncodeQwenImageEditPlus", clip=clip, prompt=text, vae=vae, **pics), 0]
        cond.append([node("FluxKontextMultiReferenceLatentMethod", conditioning=c,
                          reference_latents_method="index_timestep_zero"), 0])
    latent = [node("EmptySD3LatentImage", width=size[0], height=size[1], batch_size=1), 0]
    samples = [node("KSampler", model=model, seed=a.seed, steps=a.steps, cfg=a.cfg, sampler_name="euler",
                    scheduler="simple", positive=cond[0], negative=cond[1], latent_image=latent, denoise=1.0), 0]
    image = [node("VAEDecode", samples=samples, vae=vae), 0]
    if a.upscale:
        up = [node("UpscaleModelLoader", model_name=a.esrgan), 0]
        image = [node("ImageUpscaleWithModel", upscale_model=up, image=image), 0]
        image = [node("ImageScaleBy", image=image, upscale_method="lanczos", scale_by=0.5), 0]
    if not cells:
        return g, node("SaveImage", images=image, filename_prefix=prefix)
    k = 2 if a.upscale else 1                       # the sheet is k x the latent size; one cell = half of it
    cw, ch = size[0] * k // 2, size[1] * k // 2
    saves = [node("SaveImage", images=image, filename_prefix=f"{prefix}_sheet")]
    for ang, col, row in cells:
        crop = [node("ImageCrop", image=image, width=cw, height=ch, x=col * cw, y=row * ch), 0]
        saves.append(node("SaveImage", images=crop, filename_prefix=f"{prefix}_az{ang:03d}"))
    return g, saves


def build_flat_graph(a, image, size, method, prefix, clean_prompt):
    """--flat: one photo -> one level, shadow-free picture of the same side. 'low'/'eye': LoRA camera pass (tag only,
    as the LoRA was trained) + --clean pass; 'prompt': one pass asked in words. Saves [final, camera pass]."""
    g = {}

    def node(cls, **inputs):
        nid = str(len(g) + 1)
        g[nid] = {"class_type": cls, "inputs": inputs}
        return nid

    unet = [node("UnetLoaderGGUF", unet_name=a.unet), 0]
    clip = [node("CLIPLoader", clip_name=a.clip, type="qwen_image"), 0]
    vae = [node("VAELoader", vae_name=a.vae), 0]

    def model(angles):
        m = unet
        for nm, st in ([(a.lightning_lora, 1.0)] if a.lightning else []) + ([(a.angles_file, 1.0)] if angles else []) + a.lora:
            m = [node("LoraLoaderModelOnly", model=m, lora_name=nm, strength_model=st), 0]
        m = [node("ModelSamplingAuraFlow", model=m, shift=3.1), 0]
        return [node("CFGNorm", model=m, strength=1.0), 0]

    def edit(mdl, img, text, negative):
        cond = []
        for t in (text, negative):
            c = [node("TextEncodeQwenImageEditPlus", clip=clip, prompt=t, vae=vae, image1=img), 0]
            cond.append([node("FluxKontextMultiReferenceLatentMethod", conditioning=c,
                              reference_latents_method="index_timestep_zero"), 0])
        lat = [node("EmptySD3LatentImage", width=size[0], height=size[1], batch_size=1), 0]
        s = [node("KSampler", model=mdl, seed=a.seed, steps=a.steps, cfg=a.cfg, sampler_name="euler",
                  scheduler="simple", positive=cond[0], negative=cond[1], latent_image=lat, denoise=1.0), 0]
        return [node("VAEDecode", samples=s, vae=vae), 0]

    src = [node("LoadImage", image=image), 0]
    extra = []
    if method == "prompt":
        img = edit(model(False), src, "\n".join(FLAT_PROMPT), "" if a.lightning else FLAT_NEGATIVE)
    else:
        cam = edit(model(True), src, FLAT_TAG[method], "")            # the LoRA's tag alone, as it was trained
        if a.keep_steps:                                              # only to look at what step 1 did
            extra.append(node("SaveImage", images=cam, filename_prefix=f"{prefix}_camera"))
        img = edit(model(False), cam, clean_prompt, "" if a.lightning else CLEAN_NEGATIVE)
    if a.delight:                                                 # shadows only, nothing else changes
        img = edit(model(False), img, "\n".join(DELIGHT), "" if a.lightning else DELIGHT_NEGATIVE)
    if a.upscale:
        up = [node("UpscaleModelLoader", model_name=a.esrgan), 0]
        img = [node("ImageUpscaleWithModel", upscale_model=up, image=img), 0]
        img = [node("ImageScaleBy", image=img, upscale_method="lanczos", scale_by=0.5), 0]
    return g, [node("SaveImage", images=img, filename_prefix=prefix)] + extra


def run_flat(a, photos, name, out, size, az_name):
    """--flat: every photo (or those at --views) -> <name>_az<angle>.png, same side, level, no shadows."""
    wanted = [v % 360 for v in a.views] if a.views else None
    sel = [(f, t) for f, t in photos if wanted is None or min(angdist(t, v) for v in wanted) <= 20]   # 90 finds az091
    if not sel:
        sys.exit("no photo at the angles given with --views")
    methods = FLAT_METHODS if a.compare else [a.flat_method]
    a.keep_steps = a.keep_steps or a.compare                         # the comparison sheet shows step 1 too
    clean_prompt = build_clean_prompt([sel[0]], a.hint)
    if a.dry_run:
        for m in methods:
            print(f"\n=== method {m}")
            if m == "prompt":
                print("\n".join(FLAT_PROMPT))
            else:
                print(f"pass 1 (Multiple-Angles LoRA): {FLAT_TAG[m]}\npass 2 (same camera):\n{clean_prompt}")
            if a.delight:
                print("last pass (--delight):\n" + "\n".join(DELIGHT))
        return
    check_mode(out, "flat", a.overwrite)
    out.mkdir(parents=True, exist_ok=True)
    cmp_dir = out / "compare"
    comfy = Comfy(a.comfy, a.timeout)
    log = out / "qwen_views.txt"
    failed, sheets = [], []
    for i, (f, t) in enumerate(sel):
        az = az_name(t)
        final = out / f"{name}_az{az}.png"
        if final.exists() and not a.overwrite and not a.compare:
            print(f"skip {final} (exists, --overwrite to redo)")
            continue
        wh = size or size_like(f)
        safe = re.sub(r"[^\w.-]+", "_", f.name)
        remote = comfy.upload(f, f"{name}_{i:02d}_{safe}")
        row = [f]
        for m in methods:
            dst = (cmp_dir / f"{name}_az{az}_{m}.png") if a.compare else final
            print(f"\n{dst.name} ({wh[0]}x{wh[1]}) from {f.name}, method {m}, seed {a.seed}")
            graph, saves = build_flat_graph(a, remote, wh, m, f"qwen_views/{name}_az{az}_{m}",
                                            build_clean_prompt([(f, t)], a.hint))
            try:
                imgs = comfy.run(graph, saves)
            except RuntimeError as e:
                print(f"    {e}")
                failed.append(dst.name)
                if "refused" in str(e):
                    break
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(comfy.download(imgs[0]))
            print(f"    -> {dst}")
            if len(imgs) > 1:                                       # the camera pass, to see what each pass did
                cam = (cmp_dir if a.compare else out / "camera_pass") / f"{name}_az{az}_{m}_camera.png"
                cam.parent.mkdir(parents=True, exist_ok=True)
                cam.write_bytes(comfy.download(imgs[1]))
                row.append(cam)
            row.append(dst)
            with log.open("a", encoding="utf-8") as fh:
                fh.write(f"## {time.strftime('%Y-%m-%d %H:%M')}  {dst.name}  seed {a.seed}  steps {a.steps}  "
                         f"cfg {a.cfg:g}  {wh[0]}x{wh[1]}  flat {m}{' delight' if a.delight else ''}  photo: {f.name}:{deg(t)}\n")
        if a.compare:
            sheets.append(compare_sheet(row, cmp_dir / f"{name}_az{az}_compare.jpg"))
    for s in filter(None, sheets):
        print(f"\ncompare: {s}  (photo | low: camera pass, result | eye: camera pass, result | prompt)")
    if failed:
        sys.exit(f"\nnot made: {', '.join(failed)}")
    if a.compare:
        print("pick the method that looks best, then run all photos with it, e.g.  --flat --flat-method eye")
    else:
        print(f"\nlook at the views in {out}; another try for one view: --views 90 --seed 7 --overwrite")
        print(f"then Hunyuan:  python hy21_mv.py {out} --name {name} --elev 0")


def compare_sheet(paths, dst, h=520):
    """photo and results side by side (needs Pillow; without it the single pictures are still there)"""
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return None
    ims = []
    for p in paths:
        im = Image.open(p).convert("RGBA")
        bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
        bg.alpha_composite(im)
        ims.append(bg.convert("RGB").resize((max(1, round(im.width * h / im.height)), h)))
    sheet = Image.new("RGB", (sum(i.width for i in ims) + 10 * len(ims), h + 30), "white")
    x, dr = 0, ImageDraw.Draw(sheet)
    for p, im in zip(paths, ims):
        sheet.paste(im, (x, 30))
        dr.text((x + 4, 8), Path(p).stem[-28:], fill=(0, 0, 0))
        x += im.width + 10
    sheet.save(dst, quality=88)
    return dst


def mode_of(a):
    return "clean" if a.clean else "sheet" if a.sheet else "flat" if a.flat else "elev"


def check_mode(out, mode, overwrite):
    """Do not mix --clean views, --sheet views and single elevations in one folder."""
    log = out / "qwen_views.txt"
    if overwrite or not log.exists():
        return
    found = set()
    for ln in log.read_text(encoding="utf-8", errors="replace").splitlines():
        if ln.startswith("## "):
            found.add("clean" if " clean " in ln else "sheet" if " sheet " in ln else "flat" if " flat " in ln
                      else "elev")
    if found and found != {mode}:
        names = {"clean": "--clean (same camera)", "sheet": "--sheet", "elev": "single elevation",
                 "flat": "--flat (level, one per photo)"}
        sys.exit(f"{out} already holds {', '.join(names[m] for m in sorted(found))} views from an earlier run; "
                 f"this run makes {names[mode]} views.\nUse another folder, e.g.  --out {out}_{mode}   (or --overwrite)")


def run_sheet(a, photos, name, out, size):
    refs = sheet_refs(photos, a.refs)
    prompt = build_sheet_prompt(refs)
    dsts = [out / f"{name}_az{ang:03d}.png" for ang, _, _ in SHEET_CELLS]
    print(f"\nsheet ({size[0]}x{size[1]}, upscaled to {2 * size[0]}x{2 * size[1]}, cut into 4) from: "
          + ", ".join(f"{f.name} ({deg(x)})" for f, x in refs))
    if a.dry_run:
        print(prompt)
        if not a.lightning:
            print(f"negative: {NEGATIVE}")
        return
    check_mode(out, "sheet", a.overwrite)
    if all(d.exists() for d in dsts) and not a.overwrite:
        print(f"skip: {out} already has all 4 views (--overwrite to redo, --seed for another try)")
        return
    out.mkdir(parents=True, exist_ok=True)
    (out / "sheet").mkdir(exist_ok=True)
    comfy = Comfy(a.comfy, a.timeout)
    remote = []
    for i, (f, _) in enumerate(refs):
        safe = re.sub(r"[^\w.-]+", "_", f.name)
        remote.append(comfy.upload(f, f"{name}_{i:02d}_{safe}"))
    print(f"uploaded {len(remote)} photo(s); seed {a.seed}, {a.steps} steps, cfg {a.cfg:g}")
    graph, saves = build_graph(a, remote, prompt, "" if a.lightning else NEGATIVE, f"qwen_views/{name}", size,
                               cells=SHEET_CELLS)
    try:
        imgs = comfy.run(graph, saves)
    except RuntimeError as e:
        sys.exit(f"    {e}")
    sheet = out / "sheet" / f"{name}_sheet_seed{a.seed}.png"
    sheet.write_bytes(comfy.download(imgs[0]))
    print(f"    -> {sheet}")
    for img, dst in zip(imgs[1:], dsts):
        dst.write_bytes(comfy.download(img))
        print(f"    -> {dst}")
    with (out / "qwen_views.txt").open("a", encoding="utf-8") as fh:
        fh.write(f"## {time.strftime('%Y-%m-%d %H:%M')}  {sheet.name}  seed {a.seed}  steps {a.steps}  cfg {a.cfg:g}  "
                 f"{size[0]}x{size[1]}  sheet  photos: {', '.join(f'{f.name}:{deg(x)}' for f, x in refs)}\n{prompt}\n\n")
    print(f"\nlook at the sheet and the 4 views in {out}; another try: --seed 7 --overwrite")
    print("if a view is cut off at a cell edge, cut the sheet with split_views.py instead (it finds the white gaps)")
    print(f"next, in WSL:  python make_views.py {out.as_posix()} --prep")


def main():
    try:
        sys.stdout.reconfigure(errors="replace")
    except AttributeError:
        pass
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("photos", nargs="+", help="photo.jpg:<angle>, a photo with az<angle> in its name, or a folder of those")
    p.add_argument("--name", help="building name for the files (default: the first photo's name)")
    p.add_argument("--out", help="output folder (default: my_views_<name>)")
    p.add_argument("--clean", action="store_true",
                   help="keep every photo's own camera: only cut the building out and clean it (one view per photo)")
    p.add_argument("--hint", help="--clean: what the building is, e.g. 'the cathedral with the tall green spire' "
                                  "(helps when it touches other buildings)")
    p.add_argument("--views", type=float, nargs="+",
                   help="angles to make (default 0 90 180 270; with --clean: only the photos at these angles)")
    p.add_argument("--refs", type=int, choices=[1, 2, 3],
                   help="photos per view, nearest first (default 3; with --clean 1 = only its own photo)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-lightning", dest="lightning", action="store_false", help="no 4-step Lightning LoRA (30 steps, cfg 4)")
    p.add_argument("--steps", type=int, help="sampling steps (default 4 with Lightning, else 30)")
    p.add_argument("--cfg", type=float, help="guidance (default 1 with Lightning, else 4)")
    p.add_argument("--lora", type=parse_lora, action="append", default=[], help="extra LoRA file[:strength] (repeatable)")
    p.add_argument("--angles-lora", action="store_true", help="add the Multiple-Angles LoRA and its <sks> camera tag")
    p.add_argument("--sheet", action="store_true",
                   help="one 2x2 sheet of 4 straight-on elevations (same scale), upscaled and cut into 4 pictures")
    p.add_argument("--level", action="store_true",
                   help="each view only from the photo at its own angle (no mixing of photos: no doubled towers) "
                        "+ the Multiple-Angles LoRA 'eye-level shot' to bring a camera from above down to level")
    p.add_argument("--flat", action="store_true",
                   help="one picture per photo: the same side, brought down to a level, flat view without shadows")
    p.add_argument("--flat-method", default="eye", choices=FLAT_METHODS,
                   help="--flat: eye (default, best in the Dom test), low (looks up from below), prompt (no LoRA)")
    p.add_argument("--keep-steps", action="store_true",
                   help="--flat: also save step 1 (the camera move, still with sky and ground) in <out>\\camera_pass")
    p.add_argument("--no-delight", dest="delight", action="store_false",
                   help="--flat: skip the last pass that removes baked shadows and dark sides (on by default)")
    p.add_argument("--compare", action="store_true",
                   help="--flat: make every method for the photos (use --views to pick one) + a comparison sheet")
    p.add_argument("--upscale", action="store_true", help="Real-ESRGAN x4 then x0.5 (twice the size)")
    p.add_argument("--size", help="output WIDTHxHEIGHT (default 1328x1328; with --clean/--flat the photo's own shape)")
    p.add_argument("--comfy", default="http://127.0.0.1:8188", help="ComfyUI address")
    p.add_argument("--timeout", type=int, default=1800, help="seconds to wait for one view (default 1800)")
    p.add_argument("--dry-run", action="store_true", help="only print the photos and prompt of every view")
    p.add_argument("--overwrite", action="store_true", help="redo views that already exist")
    p.add_argument("--unet", default=UNET)
    p.add_argument("--clip", default=CLIP)
    p.add_argument("--vae", default=VAE)
    p.add_argument("--lightning-lora", default=LIGHTNING)
    p.add_argument("--angles-file", default=ANGLES)
    p.add_argument("--esrgan", default=ESRGAN)
    a = p.parse_args()

    photos = parse_photos(a.photos)
    size = None
    if a.size:
        m = re.fullmatch(r"(\d+)\s*[xX*]\s*(\d+)", a.size.strip())
        if not m:
            sys.exit("--size must look like 1328x1328")
        size = tuple(max(256, round(int(v) / 16) * 16) for v in m.groups())
    a.steps = a.steps or (4 if a.lightning else 30)
    a.cfg = a.cfg if a.cfg is not None else (1.0 if a.lightning else 4.0)
    if a.flat and (a.clean or a.sheet or a.level or a.angles_lora):
        sys.exit("--flat is its own mode: not together with --clean, --sheet, --level or --angles-lora")
    if a.compare and not a.flat:
        sys.exit("--compare is for --flat")
    if a.sheet:
        if a.clean or a.level:
            sys.exit("--sheet makes its own level views: not together with --clean or --level")
        if a.views:
            print("[note] --sheet always makes 0, 90, 180 and 270; --views is ignored")
        a.upscale, a.angles_lora = True, True       # as the tested ComfyUI grid workflow (qwen-grid-4views-upscale)
    if a.level:
        if a.clean:
            sys.exit("--level turns the camera down; --clean keeps it. Use one or the other.")
        a.refs, a.angles_lora = a.refs or 1, True
    a.refs = a.refs or (1 if a.clean else 3)
    if a.clean and a.angles_lora:
        sys.exit("--angles-lora turns the camera; --clean keeps it. Use one or the other.")
    if a.hint and not (a.clean or a.flat):
        print("[note] --hint is only used with --clean and --flat")
    name = a.name or re.sub(r"[_\- ]*az(?:im)?_?-?\d+(?:\.\d+)?", "", photos[0][0].stem).strip("_- ") or "building"
    name = re.sub(r"[^\w\-]+", "_", name)
    out = Path(a.out or f"my_views_{name}")

    print(f"{len(photos)} photo(s):")
    for f, ang in photos:
        print(f"  {deg(ang):>5} deg  {f}")

    if a.sheet:
        return run_sheet(a, photos, name, out, size or (1328, 1328))

    def az_name(t):
        return f"{t:03.0f}" if float(t).is_integer() else f"{t:05.1f}"

    if a.flat:
        return run_flat(a, photos, name, Path(a.out or f"my_views_{name}_flat"), size, az_name)

    jobs = []
    if a.clean:                                     # one job per photo, its own angle and camera
        seen = {}
        for f, ang in photos:
            if az_name(ang) in seen:
                sys.exit(f"{f.name} and {seen[az_name(ang)].name} have the same angle ({deg(ang)}); one photo per angle")
            seen[az_name(ang)] = f
        shapes = {f.name: image_size(f) for f, _ in photos}
        ratios = [w / h for w, h in filter(None, shapes.values())]
        if len(ratios) > 1 and max(ratios) / min(ratios) > 1.01:
            print("[warning] the shots have different shapes:")
            for n, wh in shapes.items():
                print(f"    {n}: {'x'.join(map(str, wh)) if wh else '?'}")
            print("  For multi-view, take them all as full-screen captures of the same window (Win+PrtScn), so they\n"
                  "  share one scale and centre; make_views.py --keep-frame refuses pictures of different shapes.")
        wanted = [v % 360 for v in a.views] if a.views else None
        for f, t in photos:
            if wanted is not None and min(angdist(t, v) for v in wanted) > 1e-6:
                continue
            others = sorted((pa for pa in photos if pa[0] != f), key=lambda pa: angdist(pa[1], t))
            refs = [(f, t)] + others[:a.refs - 1]
            jobs.append((t, refs, build_clean_prompt(refs, a.hint), out / f"{name}_az{az_name(t)}.png",
                         size or size_like(f)))
        if not jobs:
            sys.exit("no photo at the angles given with --views")
    else:
        for t in (a.views or [0, 90, 180, 270]):
            t %= 360
            refs = sorted(photos, key=lambda pa: angdist(pa[1], t))[:a.refs]
            tag = None
            if a.angles_lora:                           # the LoRA's camera tag is relative to Picture 1
                tag = f"<sks> {TAGS[sector(t - refs[0][1])]} eye-level shot wide shot"
            jobs.append((t, refs, build_prompt(t, refs, tag), out / f"{name}_az{az_name(t)}.png", size or (1328, 1328)))

    if a.dry_run:
        for t, refs, prompt, dst, wh in jobs:
            print(f"\n=== {dst.name}  ({deg(t)} deg, {wh[0]}x{wh[1]}) uses: "
                  + ", ".join(f"{f.name} ({deg(x)})" for f, x in refs))
            print(prompt)
            if not a.lightning:
                print(f"negative: {CLEAN_NEGATIVE if a.clean else NEGATIVE}")
        return

    log = out / "qwen_views.txt"
    check_mode(out, mode_of(a), a.overwrite)
    todo = [j for j in jobs if a.overwrite or not j[3].exists()]
    for j in jobs:
        if j not in todo:
            print(f"skip {j[3]} (exists, --overwrite to redo)")
    if not todo:
        return
    out.mkdir(parents=True, exist_ok=True)
    comfy = Comfy(a.comfy, a.timeout)
    used = {f for _, refs, _, _, _ in todo for f, _ in refs}
    remote = {}
    for i, (f, _) in enumerate(photos):
        if f in used:
            safe = re.sub(r"[^\w.-]+", "_", f.name)
            remote[f] = comfy.upload(f, f"{name}_{i:02d}_{safe}")
    print(f"uploaded {len(remote)} photo(s) to ComfyUI\\input\\qwen_views; seed {a.seed}, {a.steps} steps, "
          f"cfg {a.cfg:g}" + (", same view (--clean)" if a.clean else "") + (", upscaled x2" if a.upscale else ""))

    failed = []
    negative = "" if a.lightning else (CLEAN_NEGATIVE if a.clean else NEGATIVE)   # cfg 1 ignores it anyway
    for t, refs, prompt, dst, wh in todo:
        print(f"\n{dst.name} ({wh[0]}x{wh[1]}): from " + ", ".join(f"{f.name} ({deg(x)} deg)" for f, x in refs))
        graph, save = build_graph(a, [remote[f] for f, _ in refs], prompt, negative,
                                  f"qwen_views/{name}_az{dst.stem.rsplit('_az', 1)[1]}", wh)
        try:
            img = comfy.run(graph, save)
        except RuntimeError as e:
            print(f"    {e}")
            failed.append(dst.name)
            if "refused" in str(e):                    # same graph for every view: stop early
                break
            continue
        dst.write_bytes(comfy.download(img))
        print(f"    -> {dst}")
        with log.open("a", encoding="utf-8") as fh:
            fh.write(f"## {time.strftime('%Y-%m-%d %H:%M')}  {dst.name}  seed {a.seed}  steps {a.steps}  cfg {a.cfg:g}  "
                     f"{wh[0]}x{wh[1]}{'  clean' if a.clean else ''}  "
                     f"photos: {', '.join(f'{f.name}:{deg(x)}' for f, x in refs)}\n{prompt}\n\n")

    if failed:
        sys.exit(f"\nnot made: {', '.join(failed)}")
    print(f"\nlook at the views in {out}, delete a bad one and run again (only missing views are made; --seed for another try)")
    if a.clean:
        print("check that every building sits at the same place and size as in its photo (flip between the two)")
        print(f"next, in WSL:  python make_views.py {out.as_posix()} --prep --keep-frame --elev <the shots' elevation>")
    else:
        print(f"next, in WSL:  python make_views.py {out.as_posix()} --prep")


if __name__ == "__main__":
    main()
