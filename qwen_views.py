"""
qwen_views.py - clean up photos / 3D-map screenshots of a building with Qwen-Image-Edit-2511 in ComfyUI, ready
for TRELLIS.2. Qwen cuts the building out (neighbours, labels, trees, cars, ground, sky, shadows removed; hidden
parts filled in; texture sharpened) on white, from the photo's own camera. The prompt is the same for every
building and every photo and names no angle. --hint names the building when it is not clearly the one in the
middle or touches others.

Fit (default): the building fills the picture, so every pixel Qwen reads and draws is building. Two runs per photo:
    1. find   Qwen cleans the whole photo (~1 MP, only a small preview is kept) and the script measures where the
              building is.
    2. clean  the photo is cropped around the building (frame shaped like the building, 9:16 to 16:9) and Qwen
              cleans that crop at exactly the size Qwen reads its input at (~1 MP, e.g. 1184x880), so input and
              output match pixel for pixel and the framing stays put.
    The building then fills --fill (0.92) of the picture's width or height instead of 40-70 %: ~950-1250 px on its
    long side, drawn by Qwen from a crop that shows the building ~1.5x bigger. trellis2_batch.py cuts it out and
    scales it to 983 px of TRELLIS' 1024 px input, so TRELLIS gets drawn pixels instead of upscaled ones.
    A building cut off at an edge of the result is made again once with more margin.
    Crop, building box and seed are written into the PNG (text chunk "mapgen") and into qwen_views.txt.
    --no-fit: the old single run on the whole photo (1328x1328-equivalent output).

Normally run through qwen_batch.py (every photo in photos_raw). By hand:
    1. Start ComfyUI (ImageGen\\ComfyUI_windows_portable\\run_nvidia_gpu.bat) and leave it open.
    2. Windows Python, in the TRELLIS.2 folder:
         python qwen_views.py photos_raw\\church.png:45 --name church
         python qwen_views.py shots --name dom --hint "the cathedral with the tall green-spired tower"
       -> photos_clean\\<name>_az045.png ... (one per photo)
    3. Close ComfyUI, then in WSL:  trellis -q high --in photos_clean

The angle (photo.png:45, az045 in the file name, or a folder of such names) is only a label: it goes into the
output name, so several photos of one building do not overwrite each other, and with --refs 2/3 it decides which
other photos of the building count as the nearest ones. Qwen itself gets no degrees.

    --fill 0.92          share of the picture's width or height the building fills (fit; 0.5-0.98)
    --no-fit             no find run, no crop: the whole photo as before
    --refs 2|3           also show Qwen the nearest other photos (to see what is hidden; default 1 = only its own)
    --views 45 135       only the photos at these angles
    --seed 7             another try
    --no-lightning       without the 4-step Lightning LoRA: 30 steps at cfg 4, slower, sometimes cleaner
    --lora file[:0.8]    extra LoRA from ComfyUI\\models\\loras (repeatable)
    --upscale            Real-ESRGAN x4, then halved: twice the size (needs RealESRGAN_x4plus.pth in
                         ComfyUI\\models\\upscale_models)
    --size 1184x880      output size (default: fit = Qwen's input size for the building's shape;
                         --no-fit = the photo's own shape at 1328x1328's pixel count)
    --dry-run            only print which photos and which prompt each picture gets (to paste into ComfyUI by hand)
    --overwrite          redo pictures that already exist (otherwise they are skipped)
"""
import argparse
import json
import math
import mimetypes
import re
import struct
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
import zlib
from pathlib import Path

EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
AZ_RE = re.compile(r"az(?:im)?_?(-?\d+(?:\.\d+)?)", re.I)          # same rule as qwen_batch.py
NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")

# model files (ComfyUI\models\unet, text_encoders, vae, loras, upscale_models)
UNET = "qwen-image-edit-2511-Q4_K_M.gguf"
CLIP = "qwen_2.5_vl_7b_fp8_scaled.safetensors"
VAE = "qwen_image_vae.safetensors"
LIGHTNING = "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors"
ESRGAN = "RealESRGAN_x4plus.pth"

# Same camera, building only. Nothing in here names a building or an angle, so it fits every photo.
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
# fit, clean run: Picture 1 is already cropped around the building
FIT_LINE = ("The building already fills almost all of Picture 1: keep every part of it inside the picture, "
            "nothing of it cut off at the edges.")
CLEAN_NEGATIVE = ("shadows, cast shadows, dark side, sunlight, trees, cars, people, signs, street, sky, ground, "
                  "other buildings, map labels, icons, text, watermark, frame, blurry, noise, jpeg artifacts, "
                  "rotated view, zoomed, cropped")
QWEN_AREA = 1328 * 1328          # Qwen-Image's native pixel count (--no-fit output)
REF_AREA = 1024 * 1024           # TextEncodeQwenImageEditPlus scales every input picture to this (sides x 8)
FILL = 0.92                      # building : picture, on the side where it is longest relative to the frame
ASPECT = (9 / 16, 16 / 9)        # frame shapes Qwen-Image was trained on
PREVIEW = 384                    # long side of the preview the building is measured in
INK = 24                         # a pixel this far (0-255) from the background colour is building


def angdist(a, b):
    d = abs(a - b) % 360
    return min(d, 360 - d)


def deg(a):
    return f"{a:g}"


def build_clean_prompt(refs, hint=None, fit=False):
    """Same-view clean-up of refs[0]; refs[1:] (other photos of the building) only help to see what is hidden."""
    target = "the main building in the centre of the picture" + (f" ({hint})" if hint else "")
    lines = []
    if len(refs) > 1:
        others, it = ("Picture 2 shows", "it") if len(refs) == 2 else (f"Pictures 2 to {len(refs)} show", "them")
        lines.append(f"{others} the same building from other sides. Use {it} only to see which parts belong to the "
                     f"building and what is hidden in Picture 1; do not take {'its' if it == 'it' else 'their'} viewpoint.")
    clean = [s.format(target=target) for s in CLEAN]
    if fit:
        clean.insert(1, FIT_LINE)
    return "\n".join(lines + clean)


# ----------------------------------------------------------------------------- pictures (standard library only)

def exif_orientation(seg):
    """EXIF orientation (1-8) from an APP1 / eXIf payload starting with b'Exif\\0\\0', else None"""
    try:
        if seg[:6] != b"Exif\x00\x00":
            return None
        t = seg[6:]
        e = "<" if t[:2] == b"II" else ">"
        ifd = struct.unpack(e + "I", t[4:8])[0]
        for i in range(struct.unpack(e + "H", t[ifd:ifd + 2])[0]):
            ent = t[ifd + 2 + 12 * i: ifd + 14 + 12 * i]
            if struct.unpack(e + "H", ent[:2])[0] == 0x0112:
                return struct.unpack(e + "H", ent[8:10])[0]
    except (struct.error, IndexError):
        return None
    return None


def image_size(path):
    """(width, height) of a PNG, JPEG, WebP or BMP as ComfyUI loads it (EXIF rotation applied); None if unknown."""
    try:
        with open(path, "rb") as fh:
            head = fh.read(64)
            if head[:8] == b"\x89PNG\r\n\x1a\n":
                w, h = struct.unpack(">II", head[16:24])
                fh.seek(8)
                while True:                                  # eXIf rotation (phone photos saved as PNG)
                    c = fh.read(8)
                    if len(c) < 8:
                        break
                    n, kind = struct.unpack(">I4s", c)
                    if kind in (b"IDAT", b"IEND"):
                        break
                    if kind == b"eXIf":
                        if (exif_orientation(b"Exif\x00\x00" + fh.read(n)) or 1) in (5, 6, 7, 8):
                            return h, w
                        break
                    fh.seek(n + 4, 1)
                return w, h
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
                orient = 1
                while True:
                    m = fh.read(2)
                    if len(m) < 2 or m[0] != 0xFF:
                        return None
                    if m[1] in (0xD8, 0x01) or 0xD0 <= m[1] <= 0xD7:
                        continue
                    seglen = struct.unpack(">H", fh.read(2))[0]
                    if m[1] == 0xE1:                         # APP1: EXIF (phone photos are often rotated)
                        orient = exif_orientation(fh.read(seglen - 2)) or orient
                        continue
                    if m[1] in (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF):
                        h, w = struct.unpack(">xHH", fh.read(5))
                        return (h, w) if orient in (5, 6, 7, 8) else (w, h)
                    fh.seek(seglen - 2, 1)
    except (OSError, struct.error):
        return None
    return None


def read_png(data):
    """(width, height, channels, rows) of an 8-bit PNG like the ones ComfyUI writes"""
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("not a PNG")
    pos, idat, w = 8, [], None
    while pos + 8 <= len(data):
        n, kind = struct.unpack(">I4s", data[pos:pos + 8])
        body = data[pos + 8:pos + 8 + n]
        if kind == b"IHDR":
            w, h, depth, ctype, _, _, interlace = struct.unpack(">IIBBBBB", body)
            ch = {0: 1, 2: 3, 4: 2, 6: 4}.get(ctype)
            if depth != 8 or interlace or ch is None:
                raise ValueError(f"PNG type not read here (depth {depth}, colour type {ctype}, interlace {interlace})")
        elif kind == b"IDAT":
            idat.append(body)
        elif kind == b"IEND":
            break
        pos += 12 + n
    if w is None or not idat:
        raise ValueError("broken PNG")
    raw = zlib.decompress(b"".join(idat))
    stride = w * ch
    rows, prev = [], bytearray(stride)
    for y in range(h):
        o = y * (stride + 1)
        f, line = raw[o], bytearray(raw[o + 1:o + 1 + stride])
        if f == 1:
            for i in range(ch, stride):
                line[i] = (line[i] + line[i - ch]) & 255
        elif f == 2:
            for i in range(stride):
                line[i] = (line[i] + prev[i]) & 255
        elif f == 3:
            for i in range(stride):
                line[i] = (line[i] + (((line[i - ch] if i >= ch else 0) + prev[i]) >> 1)) & 255
        elif f == 4:
            for i in range(stride):
                a = line[i - ch] if i >= ch else 0
                b, c = prev[i], (prev[i - ch] if i >= ch else 0)
                p = a + b - c
                pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
                line[i] = (line[i] + (a if pa <= pb and pa <= pc else b if pb <= pc else c)) & 255
        rows.append(line)
        prev = line
    return w, h, ch, rows


def find_building(data):
    """Where the building is in a cleaned picture (PNG bytes, white background):
    dict(box=(x0, y0, x1, y1) as shares of width / height, touch={'left', ...}, cover=share of the picture),
    or None when there is nothing on it."""
    w, h, ch, rows = read_png(data)

    def px(x, y):
        r, i = rows[y], x * ch
        return (r[i], r[i + 1], r[i + 2]) if ch >= 3 else (r[i],) * 3

    border = ([px(x, 0) for x in range(0, w, 2)] + [px(x, h - 1) for x in range(0, w, 2)]
              + [px(0, y) for y in range(0, h, 2)] + [px(w - 1, y) for y in range(0, h, 2)])
    bg = [sorted(c[k] for c in border)[len(border) // 2] for k in range(3)]
    if min(bg) < 200:                                   # the border is not background: take white
        bg = [255, 255, 255]
    ink = bytearray(w * h)
    for y in range(h):
        r = rows[y]
        for x in range(w):
            i = x * ch
            c = (r[i], r[i + 1], r[i + 2]) if ch >= 3 else (r[i],) * 3
            if max(abs(c[0] - bg[0]), abs(c[1] - bg[1]), abs(c[2] - bg[2])) > INK:
                ink[y * w + x] = 1
    # connected parts (8 neighbours): area and box of each
    seen, parts = bytearray(w * h), []
    for start in range(w * h):
        if not ink[start] or seen[start]:
            continue
        seen[start] = 1
        stack, n = [start], 0
        x0 = x1 = start % w
        y0 = y1 = start // w
        while stack:
            p = stack.pop()
            n += 1
            py, pxx = divmod(p, w)
            x0, x1, y0, y1 = min(x0, pxx), max(x1, pxx), min(y0, py), max(y1, py)
            for dy in (-1, 0, 1):
                yy = py + dy
                if 0 <= yy < h:
                    for dx in (-1, 0, 1):
                        xx = pxx + dx
                        if 0 <= xx < w:
                            q = yy * w + xx
                            if ink[q] and not seen[q]:
                                seen[q] = 1
                                stack.append(q)
        parts.append((n, x0, y0, x1, y1))
    if not parts:
        return None
    big = max(parts)
    near = 0.05 * max(w, h)
    keep = [p for p in parts if p[0] >= 0.005 * big[0]                 # real parts of the building
            or (p[0] >= 4 and p[1] <= big[3] + near and p[3] >= big[1] - near    # small bits right next to it
                and p[2] <= big[4] + near and p[4] >= big[2] - near)]        # (antenna tips, spire crosses)
    x0, y0 = min(p[1] for p in keep), min(p[2] for p in keep)
    x1, y1 = max(p[3] for p in keep) + 1, max(p[4] for p in keep) + 1
    touch = {s for s, hit in (("left", x0 <= 1), ("top", y0 <= 1), ("right", x1 >= w - 1), ("bottom", y1 >= h - 1))
             if hit}
    return dict(box=(x0 / w, y0 / h, x1 / w, y1 / h), touch=touch, cover=sum(p[0] for p in keep) / (w * h))


def png_add_text(data, key, text):
    """the PNG with a tEXt chunk key=text added before IEND"""
    end = data.rfind(b"IEND") - 4
    if end < 8:
        return data
    body = key.encode("latin-1") + b"\x00" + text.encode("latin-1", "replace")
    chunk = struct.pack(">I", len(body)) + b"tEXt" + body + struct.pack(">I", zlib.crc32(b"tEXt" + body) & 0xFFFFFFFF)
    return data[:end] + chunk + data[end:]


# ----------------------------------------------------------------------------- sizes

def node_ref_size(w, h):
    """the size TextEncodeQwenImageEditPlus scales a w x h picture to before Qwen reads it (ComfyUI nodes_qwen.py)"""
    s = math.sqrt(REF_AREA / (w * h))
    return round(w * s / 8.0) * 8, round(h * s / 8.0) * 8


def qwen_size(aspect):
    """Output size (sides x 16) closest to this shape that Qwen's input scaling leaves unchanged: the photo given
    at this size is read by Qwen pixel for pixel, and the output has the same size, so nothing shifts or zooms."""
    best, w0 = None, (REF_AREA * aspect) ** 0.5
    for w in range(max(256, int(w0 / 16) * 16 - 96), int(w0 / 16) * 16 + 112, 16):
        h0 = w / aspect
        for h in range(max(256, int(h0 / 16) * 16 - 64), int(h0 / 16) * 16 + 80, 16):
            if node_ref_size(w, h) == (w, h):
                err = abs(math.log(w / h / aspect)) + 0.25 * abs(math.log(w * h / REF_AREA))
                if best is None or err < best[0]:
                    best = (err, w, h)
    if best is None:                                    # not seen for any shape; just in case
        return max(256, round(w0 / 16) * 16), max(256, round(w0 / aspect / 16) * 16)
    return best[1], best[2]


def plan_crop(box, W, H, fill, size=None):
    """Crop (x, y, w, h) of a W x H photo around box (shares) in which the building fills `fill` of the width or
    the height, shaped like the output size (default: qwen_size of the building's shape, 9:16 to 16:9).
    Returns (crop, output size)."""
    bx0, by0, bx1, by1 = box[0] * W, box[1] * H, box[2] * W, box[3] * H
    bw, bh = max(1.0, bx1 - bx0), max(1.0, by1 - by0)
    out = size or qwen_size(min(max(bw / bh, ASPECT[0]), ASPECT[1]))
    A = out[0] / out[1]
    fw = max(bw, A * bh) / fill
    fh = fw / A
    s = min(1.0, W / fw, H / fh)                        # never bigger than the photo
    fw, fh = fw * s, fh * s
    cx = min(max((bx0 + bx1) / 2, fw / 2), W - fw / 2)
    cy = min(max((by0 + by1) / 2, fh / 2), H - fh / 2)
    w, h = max(16, min(W, round(fw))), max(16, min(H, round(fh)))
    x, y = min(max(0, round(cx - fw / 2)), W - w), min(max(0, round(cy - fh / 2)), H - h)
    return (x, y, w, h), out


def size_like(path, area=QWEN_AREA):
    """--no-fit output size: the photo's shape at Qwen's pixel count, multiples of 16 (keeps the framing)."""
    wh = image_size(path)
    if not wh or min(wh) <= 0:
        return 1328, 1328
    r = wh[0] / wh[1]
    return (max(256, round((area * r) ** 0.5 / 16) * 16), max(256, round((area / r) ** 0.5 / 16) * 16))


def pct(v):
    return f"{100 * v:.0f} %"


# ----------------------------------------------------------------------------- photos

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


# ----------------------------------------------------------------------------- ComfyUI

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

    def has_node(self, name):
        try:
            return name in json.loads(self.call(f"/object_info/{urllib.parse.quote(name)}"))
        except (RuntimeError, ValueError):
            return False

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
                        why += "\n    (GPU memory: close WSL/TRELLIS, Unreal and browsers, or use a smaller --size like 1024x1024)"
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


def build_graph(a, images, prompt, negative, size, prefix=None, preview=False, upscale=None):
    """images: photo names in ComfyUI's input folder, or (name, crop (x, y, w, h) or None, scale (w, h) or None).
    Output nodes: [SaveImage] if prefix, + [PreviewImage of a small copy] if preview."""
    g = {}

    def node(cls, **inputs):
        nid = str(len(g) + 1)
        g[nid] = {"class_type": cls, "inputs": inputs}
        return nid

    model = [node("UnetLoaderGGUF", unet_name=a.unet), 0]
    for name, strength in ([(a.lightning_lora, 1.0)] if a.lightning else []) + a.lora:
        model = [node("LoraLoaderModelOnly", model=model, lora_name=name, strength_model=strength), 0]
    model = [node("ModelSamplingAuraFlow", model=model, shift=3.1), 0]
    model = [node("CFGNorm", model=model, strength=1.0), 0]
    clip = [node("CLIPLoader", clip_name=a.clip, type="qwen_image"), 0]
    vae = [node("VAELoader", vae_name=a.vae), 0]
    pics = {}
    for i, im in enumerate(images, 1):
        name, crop, scale = (im, None, None) if isinstance(im, str) else im
        pic = [node("LoadImage", image=name), 0]
        if crop:
            x, y, w, h = crop
            if getattr(a, "crop_node", "ImageCrop") == "ImageCropV2":
                pic = [node("ImageCropV2", image=pic, crop_region=dict(x=x, y=y, width=w, height=h)), 0]
            else:
                pic = [node("ImageCrop", image=pic, width=w, height=h, x=x, y=y), 0]
        if scale:                                       # lanczos here, so Qwen's own scaling has nothing left to do
            pic = [node("ImageScale", image=pic, upscale_method="lanczos", width=scale[0], height=scale[1],
                        crop="disabled"), 0]
        pics[f"image{i}"] = pic
    cond = []
    for text in (prompt, negative):
        c = [node("TextEncodeQwenImageEditPlus", clip=clip, prompt=text, vae=vae, **pics), 0]
        cond.append([node("FluxKontextMultiReferenceLatentMethod", conditioning=c,
                          reference_latents_method="index_timestep_zero"), 0])
    latent = [node("EmptySD3LatentImage", width=size[0], height=size[1], batch_size=1), 0]
    samples = [node("KSampler", model=model, seed=a.seed, steps=a.steps, cfg=a.cfg, sampler_name="euler",
                    scheduler="simple", positive=cond[0], negative=cond[1], latent_image=latent, denoise=1.0), 0]
    image = [node("VAEDecode", samples=samples, vae=vae), 0]
    if a.upscale if upscale is None else upscale:
        up = [node("UpscaleModelLoader", model_name=a.esrgan), 0]
        image = [node("ImageUpscaleWithModel", upscale_model=up, image=image), 0]
        image = [node("ImageScaleBy", image=image, upscale_method="lanczos", scale_by=0.5), 0]
    outs = []
    if prefix:
        outs.append(node("SaveImage", images=image, filename_prefix=prefix))
    if preview:                                         # small copy to measure the building in (temp folder)
        wide = size[0] >= size[1]
        small = [node("ImageScale", image=image, upscale_method="area", width=PREVIEW if wide else 0,
                      height=0 if wide else PREVIEW, crop="disabled"), 0]
        outs.append(node("PreviewImage", images=small))
    return g, outs


# ----------------------------------------------------------------------------- main

def main():
    try:
        sys.stdout.reconfigure(errors="replace")
    except AttributeError:
        pass
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("photos", nargs="+", help="photo.png:<angle>, a photo with az<angle> in its name, or a folder of those")
    p.add_argument("--name", help="building name for the files (default: the first photo's name)")
    p.add_argument("--out", help="output folder (default: photos_clean)")
    p.add_argument("--clean", action="store_true", help=argparse.SUPPRESS)   # accepted and ignored (qwen_batch.py passes it)
    p.add_argument("--hint", help="what the building is, e.g. 'the cathedral with the tall green spire' "
                                  "(helps when it touches other buildings)")
    p.add_argument("--fill", type=float, default=FILL,
                   help=f"share of the picture's width or height the building fills (default {FILL})")
    p.add_argument("--no-fit", dest="fit", action="store_false",
                   help="no find run and no crop: clean the whole photo as it is (the old way)")
    p.add_argument("--views", type=float, nargs="+", help="only the photos at these angles")
    p.add_argument("--refs", type=int, choices=[1, 2, 3], default=1,
                   help="photos Qwen sees per picture, its own first (default 1 = only its own)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-lightning", dest="lightning", action="store_false", help="no 4-step Lightning LoRA (30 steps, cfg 4)")
    p.add_argument("--steps", type=int, help="sampling steps (default 4 with Lightning, else 30)")
    p.add_argument("--cfg", type=float, help="guidance (default 1 with Lightning, else 4)")
    p.add_argument("--lora", type=parse_lora, action="append", default=[], help="extra LoRA file[:strength] (repeatable)")
    p.add_argument("--upscale", action="store_true", help="Real-ESRGAN x4 then x0.5 (twice the size)")
    p.add_argument("--size", help="output WIDTHxHEIGHT (default: fit = Qwen's input size for the building's shape, "
                                  "--no-fit = the photo's shape at 1328x1328's pixel count)")
    p.add_argument("--comfy", default="http://127.0.0.1:8188", help="ComfyUI address")
    p.add_argument("--timeout", type=int, default=1800, help="seconds to wait for one picture (default 1800)")
    p.add_argument("--dry-run", action="store_true", help="only print the photos and prompt of every picture")
    p.add_argument("--overwrite", action="store_true", help="redo pictures that already exist")
    p.add_argument("--unet", default=UNET)
    p.add_argument("--clip", default=CLIP)
    p.add_argument("--vae", default=VAE)
    p.add_argument("--lightning-lora", default=LIGHTNING)
    p.add_argument("--esrgan", default=ESRGAN)
    a = p.parse_args()

    if not 0.5 <= a.fill <= 0.98:
        sys.exit("--fill must be between 0.5 and 0.98")
    photos = parse_photos(a.photos)
    size = None
    if a.size:
        m = re.fullmatch(r"(\d+)\s*[xX*]\s*(\d+)", a.size.strip())
        if not m:
            sys.exit("--size must look like 1184x880")
        size = tuple(max(256, round(int(v) / 16) * 16) for v in m.groups())
    a.steps = a.steps or (4 if a.lightning else 30)
    a.cfg = a.cfg if a.cfg is not None else (1.0 if a.lightning else 4.0)
    name = a.name or re.sub(r"[_\- ]*az(?:im)?_?-?\d+(?:\.\d+)?", "", photos[0][0].stem).strip("_- ") or "building"
    name = re.sub(r"[^\w\-]+", "_", name)
    out = Path(a.out or "photos_clean")

    print(f"{len(photos)} photo(s):")
    for f, ang in photos:
        print(f"  {deg(ang):>5} deg  {f}")

    def az_name(t):
        return f"{t:03.0f}" if float(t).is_integer() else f"{t:05.1f}"

    seen = {}
    for f, ang in photos:
        if az_name(ang) in seen:
            sys.exit(f"{f.name} and {seen[az_name(ang)].name} have the same angle ({deg(ang)}); one photo per angle")
        seen[az_name(ang)] = f
    jobs = []                                       # one job per photo, its own camera
    wanted = [v % 360 for v in a.views] if a.views else None
    for f, t in photos:
        if wanted is not None and min(angdist(t, v) for v in wanted) > 1e-6:
            continue
        others = sorted((pa for pa in photos if pa[0] != f), key=lambda pa: angdist(pa[1], t))
        refs = [(f, t)] + others[:a.refs - 1]
        jobs.append((t, refs, build_clean_prompt(refs, a.hint, a.fit), out / f"{name}_az{az_name(t)}.png",
                     size if (size or a.fit) else size_like(f)))
    if not jobs:
        sys.exit("no photo at the angles given with --views")

    if a.dry_run:
        if a.fit:
            print(f"\nfit: every photo first goes through Qwen whole (find run, prompt as without fit, ~1 MP, "
                  f"preview only); then a crop around the building, {pct(a.fill)} filled, is cleaned at Qwen's "
                  f"input size for its shape" + (f" ({size[0]}x{size[1]})" if size else ""))
        for t, refs, prompt, dst, wh in jobs:
            print(f"\n=== {dst.name}" + (f"  ({wh[0]}x{wh[1]})" if wh else "") + " uses: "
                  + ", ".join(f"{f.name} ({deg(x)})" for f, x in refs))
            print(prompt)
            if not a.lightning:
                print(f"negative: {CLEAN_NEGATIVE}")
        return

    log = out / "qwen_views.txt"
    todo = [j for j in jobs if a.overwrite or not j[3].exists()]
    for j in jobs:
        if j not in todo:
            print(f"skip {j[3]} (exists, --overwrite to redo)")
    if not todo:
        return
    out.mkdir(parents=True, exist_ok=True)
    comfy = Comfy(a.comfy, a.timeout)
    used = [f for f, _ in photos if any(f == r for _, refs, _, _, _ in todo for r, _ in refs)]
    remote = {}
    for i, (f, _) in enumerate(photos):
        if f in used:
            safe = re.sub(r"[^\w.-]+", "_", f.name)
            remote[f] = comfy.upload(f, f"{name}_{i:02d}_{safe}")
    print(f"uploaded {len(remote)} photo(s) to ComfyUI\\input\\qwen_views; seed {a.seed}, {a.steps} steps, "
          f"cfg {a.cfg:g}, same camera" + (f", fit {pct(a.fill)}" if a.fit else "") + (", upscaled x2" if a.upscale else ""))
    negative = "" if a.lightning else CLEAN_NEGATIVE   # cfg 1 ignores it anyway

    # ---- fit, step 1: find the building in every photo used (Qwen on the whole photo, preview only)
    found = {}                                      # photo -> (photo size, find_building result or None)
    if a.fit:
        a.crop_node = "ImageCrop" if comfy.has_node("ImageCrop") else "ImageCropV2"
        if a.crop_node == "ImageCropV2" and not comfy.has_node("ImageCropV2"):
            sys.exit("this ComfyUI has neither ImageCrop nor ImageCropV2 - update ComfyUI or use --no-fit")
        print(f"\nfind the building in {len(used)} photo(s): Qwen on the whole photo, only a preview is kept")
        for f in used:
            wh0 = image_size(f)
            if not wh0:
                print(f"  {f.name}: size not readable here -> cleaned whole, without fit")
                found[f] = (None, None)
                continue
            s1 = qwen_size(wh0[0] / wh0[1])
            print(f"  {f.name} ({wh0[0]}x{wh0[1]} -> {s1[0]}x{s1[1]})")
            graph, outs = build_graph(a, [(remote[f], None, s1)], build_clean_prompt([(f, 0)], a.hint), negative,
                                      s1, preview=True, upscale=False)
            try:
                small = comfy.run(graph, outs)[0]
                fb = find_building(comfy.download(small))
                if fb is None:
                    print("    no building on the cleaned photo")
            except (RuntimeError, ValueError) as e:
                print(f"    {e}")
                if "refused" in str(e):
                    sys.exit("stopped: ComfyUI refused the graph (see above)")
                fb = None
            found[f] = (wh0, fb)
            if fb is None:
                print("    -> this photo is cleaned whole, without fit")
                continue
            x0, y0, x1, y1 = fb["box"]
            print(f"    building at x {pct(x0)}-{pct(x1)}, y {pct(y0)}-{pct(y1)} of the photo"
                  + (f" (touches the photo's {', '.join(sorted(fb['touch']))} edge)" if fb["touch"] else ""))

    def fit_inputs(refs, fill):
        """ComfyUI inputs for refs (crop + scale each), and (crop, photo size, output size) of refs[0]"""
        images, main = [], (None, None, None)
        for k, (f, _) in enumerate(refs):
            wh0, fb = found.get(f, (None, None))
            crop, scale, outsize = None, None, None
            if fb is not None:
                crop, outsize = plan_crop(fb["box"], wh0[0], wh0[1], fill, size if k == 0 else None)
                scale = outsize if not (k == 0 and size) else node_ref_size(crop[2], crop[3])
            elif wh0:
                outsize = scale = size if (k == 0 and size) else qwen_size(wh0[0] / wh0[1])
            images.append((remote[f], crop, scale))
            if k == 0:
                main = (crop, wh0, outsize or size or (1024, 1024))
        return images, main

    # ---- clean every photo
    failed, stop = [], False
    for t, refs, prompt, dst, wh in todo:
        f = refs[0][0]
        prefix = f"qwen_views/{name}_az{dst.stem.rsplit('_az', 1)[1]}"
        fill, meta, note, res = a.fill, {}, "", None
        for attempt in (1, 2):
            if a.fit:
                images, (crop, wh0, wh) = fit_inputs(refs, fill)
            else:
                images, crop, wh0 = [remote[r] for r, _ in refs], None, image_size(f)
            print(f"\n{dst.name} ({wh[0]}x{wh[1]}): from " + ", ".join(f"{r.name} ({deg(x)} deg)" for r, x in refs))
            if crop:
                print(f"    crop {crop[2]}x{crop[3]} px at ({crop[0]}, {crop[1]}) of the {wh0[0]}x{wh0[1]} photo, "
                      f"building {pct(fill)} of the frame")
                if attempt == 1 and max(crop[2], crop[3]) < 0.5 * max(wh):
                    print(f"    (small in the photo: Qwen draws it {max(wh) / max(crop[2], crop[3]):.1f}x bigger and "
                          f"invents much of the detail - a closer screenshot gives a truer result)")
            graph, outs = build_graph(a, images, prompt, negative, wh, prefix=prefix, preview=bool(a.fit))
            try:
                res = comfy.run(graph, outs)
            except RuntimeError as e:
                print(f"    {e}")
                res = None
                stop = "refused" in str(e)              # same graph for every picture: stop early
                break
            img = res[0]
            fb = None
            if a.fit:
                try:
                    fb = find_building(comfy.download(res[1]))
                except (RuntimeError, ValueError) as e:
                    print(f"    could not measure the result ({e})")
            if fb is None:
                break
            k = 2 if a.upscale else 1
            x0, y0, x1, y1 = fb["box"]
            W, H = wh[0] * k, wh[1] * k
            long_px = max((x1 - x0) * W, (y1 - y0) * H)
            print(f"    building {pct(x1 - x0)} of the width, {pct(y1 - y0)} of the height, {long_px:.0f} px long side")
            # an edge the building touches although the crop had room there = Qwen drew it bigger / cut it off
            at_photo = set()
            if crop:
                at_photo = {s for s, hit in (("left", crop[0] == 0), ("top", crop[1] == 0),
                                             ("right", crop[0] + crop[2] >= wh0[0]),
                                             ("bottom", crop[1] + crop[3] >= wh0[1])) if hit}
            cut = sorted(fb["touch"] - at_photo)
            meta = dict(photo=f.name, photo_size=list(wh0) if wh0 else None, crop=list(crop) if crop else None,
                        size=[W, H], building=[round(x0 * W), round(y0 * H), round(x1 * W), round(y1 * H)],
                        fill=round(max(x1 - x0, y1 - y0), 3), angle=t, seed=a.seed,
                        refs=[r.name for r, _ in refs[1:]])
            if cut and attempt == 1 and crop:
                fill = max(0.6, fill - 0.12)
                print(f"    cut off at the {', '.join(cut)} edge -> again with more margin ({pct(fill)})")
                continue
            if cut:
                note = f"  CUT at {', '.join(cut)}"
                print(f"    still touches the {', '.join(cut)} edge - look at it; --fill 0.8 or --seed 7 for another try")
            break
        if res is None:
            failed.append(dst.name)
            if stop:
                break
            continue
        data = comfy.download(img)
        if meta:
            data = png_add_text(data, "mapgen", json.dumps(meta))
        dst.write_bytes(data)
        print(f"    -> {dst}")
        with log.open("a", encoding="utf-8") as fh:
            fit_txt = (f"fit crop {meta['crop']} of {meta['photo_size']}, building {meta['building']} "
                       f"fill {meta['fill']:.2f}{note}" if meta.get("crop") else ("fit: none" if a.fit else "clean"))
            fh.write(f"## {time.strftime('%Y-%m-%d %H:%M')}  {dst.name}  seed {a.seed}  steps {a.steps}  cfg {a.cfg:g}  "
                     f"{wh[0]}x{wh[1]}  {fit_txt}  "
                     f"photos: {', '.join(f'{r.name}:{deg(x)}' for r, x in refs)}\n{prompt}\n\n")

    if failed:
        sys.exit(f"\nnot made: {', '.join(failed)}")
    print(f"\nlook at the views in {out}, delete a bad one and run again (only missing ones are made; --seed for another try)")
    print("check that every building is whole and looks like its photo (flip between the two)")
    print(f"next, in WSL:  trellis -q high --in {out.as_posix()}")


if __name__ == "__main__":
    main()
