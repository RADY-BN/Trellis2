"""
qwen_views.py - clean up photos / 3D-map screenshots of a building with Qwen-Image-Edit-2511 in ComfyUI, ready
for TRELLIS.2. Every photo KEEPS ITS OWN CAMERA: Qwen only cuts the building out (neighbours, labels, trees, cars,
ground, sky, shadows removed; hidden parts filled in; texture sharpened) on white, at the same place and size.
The prompt is the same for every building and every photo and names no angle. --hint names the building when it
is not clearly the one in the middle or touches others.

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

    --refs 2|3           also show Qwen the nearest other photos (to see what is hidden; default 1 = only its own)
    --views 45 135       only the photos at these angles
    --seed 7             another try
    --no-lightning       without the 4-step Lightning LoRA: 30 steps at cfg 4, slower, sometimes cleaner
    --lora file[:0.8]    extra LoRA from ComfyUI\\models\\loras (repeatable)
    --upscale            Real-ESRGAN x4, then halved: twice the size, crisper edges
    --size 1328x1328     output size (default: the photo's own shape at Qwen's pixel count, keeps the framing)
    --dry-run            only print which photos and which prompt each picture gets (to paste into ComfyUI by hand)
    --overwrite          redo pictures that already exist (otherwise they are skipped)
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
CLEAN_NEGATIVE = ("shadows, cast shadows, dark side, sunlight, trees, cars, people, signs, street, sky, ground, "
                  "other buildings, map labels, icons, text, watermark, frame, blurry, noise, jpeg artifacts, "
                  "rotated view, zoomed, cropped")
QWEN_AREA = 1328 * 1328          # Qwen-Image's native pixel count


def angdist(a, b):
    d = abs(a - b) % 360
    return min(d, 360 - d)


def deg(a):
    return f"{a:g}"


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


def build_graph(a, images, prompt, negative, prefix, size):
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
    return g, node("SaveImage", images=image, filename_prefix=prefix)


def main():
    try:
        sys.stdout.reconfigure(errors="replace")
    except AttributeError:
        pass
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("photos", nargs="+", help="photo.png:<angle>, a photo with az<angle> in its name, or a folder of those")
    p.add_argument("--name", help="building name for the files (default: the first photo's name)")
    p.add_argument("--out", help="output folder (default: photos_clean)")
    p.add_argument("--clean", action="store_true", help=argparse.SUPPRESS)   # the only mode now; old commands still work
    p.add_argument("--hint", help="what the building is, e.g. 'the cathedral with the tall green spire' "
                                  "(helps when it touches other buildings)")
    p.add_argument("--views", type=float, nargs="+", help="only the photos at these angles")
    p.add_argument("--refs", type=int, choices=[1, 2, 3], default=1,
                   help="photos Qwen sees per picture, its own first (default 1 = only its own)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-lightning", dest="lightning", action="store_false", help="no 4-step Lightning LoRA (30 steps, cfg 4)")
    p.add_argument("--steps", type=int, help="sampling steps (default 4 with Lightning, else 30)")
    p.add_argument("--cfg", type=float, help="guidance (default 1 with Lightning, else 4)")
    p.add_argument("--lora", type=parse_lora, action="append", default=[], help="extra LoRA file[:strength] (repeatable)")
    p.add_argument("--upscale", action="store_true", help="Real-ESRGAN x4 then x0.5 (twice the size)")
    p.add_argument("--size", help="output WIDTHxHEIGHT (default: the photo's own shape at Qwen's pixel count)")
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

    photos = parse_photos(a.photos)
    size = None
    if a.size:
        m = re.fullmatch(r"(\d+)\s*[xX*]\s*(\d+)", a.size.strip())
        if not m:
            sys.exit("--size must look like 1328x1328")
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
        jobs.append((t, refs, build_clean_prompt(refs, a.hint), out / f"{name}_az{az_name(t)}.png",
                     size or size_like(f)))
    if not jobs:
        sys.exit("no photo at the angles given with --views")

    if a.dry_run:
        for t, refs, prompt, dst, wh in jobs:
            print(f"\n=== {dst.name}  ({wh[0]}x{wh[1]}) uses: " + ", ".join(f"{f.name} ({deg(x)})" for f, x in refs))
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
    used = {f for _, refs, _, _, _ in todo for f, _ in refs}
    remote = {}
    for i, (f, _) in enumerate(photos):
        if f in used:
            safe = re.sub(r"[^\w.-]+", "_", f.name)
            remote[f] = comfy.upload(f, f"{name}_{i:02d}_{safe}")
    print(f"uploaded {len(remote)} photo(s) to ComfyUI\\input\\qwen_views; seed {a.seed}, {a.steps} steps, "
          f"cfg {a.cfg:g}, same camera" + (", upscaled x2" if a.upscale else ""))

    failed = []
    negative = "" if a.lightning else CLEAN_NEGATIVE   # cfg 1 ignores it anyway
    for t, refs, prompt, dst, wh in todo:
        print(f"\n{dst.name} ({wh[0]}x{wh[1]}): from " + ", ".join(f"{f.name} ({deg(x)} deg)" for f, x in refs))
        graph, save = build_graph(a, [remote[f] for f, _ in refs], prompt, negative,
                                  f"qwen_views/{name}_az{dst.stem.rsplit('_az', 1)[1]}", wh)
        try:
            img = comfy.run(graph, save)
        except RuntimeError as e:
            print(f"    {e}")
            failed.append(dst.name)
            if "refused" in str(e):                    # same graph for every picture: stop early
                break
            continue
        dst.write_bytes(comfy.download(img))
        print(f"    -> {dst}")
        with log.open("a", encoding="utf-8") as fh:
            fh.write(f"## {time.strftime('%Y-%m-%d %H:%M')}  {dst.name}  seed {a.seed}  steps {a.steps}  cfg {a.cfg:g}  "
                     f"{wh[0]}x{wh[1]}  clean  "
                     f"photos: {', '.join(f'{f.name}:{deg(x)}' for f, x in refs)}\n{prompt}\n\n")

    if failed:
        sys.exit(f"\nnot made: {', '.join(failed)}")
    print(f"\nlook at the views in {out}, delete a bad one and run again (only missing ones are made; --seed for another try)")
    print("check that every building sits at the same place and size as in its photo (flip between the two)")
    print(f"next, in WSL:  trellis -q high --in {out.as_posix()}")


if __name__ == "__main__":
    main()
