r"""
qwen_batch.py - every photo in photos_raw -> a cleaned picture in photos_clean (Qwen, qwen_views.py --clean),
ready for TRELLIS. You do not have to give any angle: the script gives every photo one.

    1. Start ComfyUI (ImageGen\ComfyUI_windows_portable\run_nvidia_gpu.bat) and leave it open.
    2. Windows Python (not WSL), in the TRELLIS.2 folder:
           python qwen_batch.py
       -> photos_clean\<building>_az045.png for every building (the same camera as the photo, only the building
          left, on white). Done pictures are skipped, so a stopped run goes on where it was.
    3. Close ComfyUI (frees the GPU), then in WSL:
           trellis -q high --in photos_clean                    every cleaned picture -> glb\high\
           python trellis_lod2.py <lod2.glb> photos_clean/<building>_az045.png --follow 0     with real size + place

What is one building:
    photos_raw\Screenshot 104050.png      a loose photo = its own building (name = file name)
    photos_raw\church.png                 rename it to give the building a name -> photos_clean\church_az045.png
    photos_raw\church_az045.png           loose photos with the same name and an angle (az...) = one building
    photos_raw\church_az135.png
    photos_raw\dom\*.png                  a folder = one building with several photos (name = folder name)

Angles: from "az<degrees>" in the file name if it is there, else given here: 45, 135, 225, 315, 0, 90, 180, 270, ...
in file-name order (screenshots: in the order you took them). For TRELLIS the angle is only a label in the name;
it matters only for several photos of one building with --refs 2/3 (then name them: 0 = front, 90 = right side,
180 = back, 270 = left side).

Hint (optional, when the building is not clearly the one in the middle or touches others): a text file next to the
photo with the same name (photos_raw\church.txt), or hint.txt in a building's folder, e.g.
    the white high-rise hotel with the red sign in the middle

Options (everything else is passed on to qwen_views.py, e.g. --seed 7 --overwrite --refs 2 --upscale --dry-run):
    --only church dom      only these buildings
    --list                 only show the buildings, their photos and angles
    --raw / --clean-dir    other folders (default photos_raw / photos_clean next to this script)
"""
import argparse
import re
import subprocess
import sys
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}          # what qwen_views.py takes
AZ_RE = re.compile(r"az(?:im)?_?(-?\d+(?:\.\d+)?)", re.I)                # same rule as qwen_views.py
AUTO = [45, 135, 225, 315, 0, 90, 180, 270] + [a for a in range(10, 360, 10) if a % 45]


def clean_name(s):
    """building name as qwen_views.py writes it"""
    s = re.sub(r"[_\- ]*az(?:im)?_?-?\d+(?:\.\d+)?", "", s).strip("_- ")
    return re.sub(r"[^\w\-]+", "_", s).strip("_") or "building"


def angle_of(f):
    m = AZ_RE.search(f.stem)
    return float(m.group(1)) % 360 if m else None


def az_tag(a):
    return f"{a:03.0f}" if float(a).is_integer() else f"{a:05.1f}"         # as qwen_views.py names the files


def give_angles(files):
    """[(file, angle, 'name'|'auto')]: angle from the name, else the next free one of AUTO (file-name order)"""
    out, used = [], set()
    for f in files:
        a = angle_of(f)
        if a is not None:
            if az_tag(a) in used:
                sys.exit(f"{f}: two photos with the angle {a:g} in one building - rename one")
            used.add(az_tag(a))
            out.append((f, a, "name"))
    for f in files:
        if angle_of(f) is None:
            a = next((c for c in AUTO if az_tag(c) not in used), None)
            if a is None:
                sys.exit(f"{f.parent}: too many photos for one building")
            used.add(az_tag(a))
            out.append((f, float(a), "auto"))
    return sorted(out, key=lambda t: t[0].name.lower())


def read_hint(p):
    if p.is_file():
        t = " ".join(p.read_text(encoding="utf-8", errors="replace").split())
        return t or None
    return None


def find_buildings(raw):
    """{name: (files, hint)} - folders = one building each, loose photos grouped by their name without az..."""
    found = {}

    def add(name, files, hint, where):
        if name in found:
            sys.exit(f"two buildings would both be called '{name}' ({where} and {found[name][2]}) - rename one")
        found[name] = (files, hint, where)

    loose = {}
    for p in sorted(raw.iterdir(), key=lambda p: p.name.lower()):
        if p.is_dir():
            files = sorted((f for f in p.iterdir() if f.is_file() and f.suffix.lower() in EXTS),
                           key=lambda f: f.name.lower())
            if files:
                add(clean_name(p.name), files, read_hint(p / "hint.txt"), p.name + "\\")
        elif p.suffix.lower() in EXTS:
            loose.setdefault(clean_name(p.stem), []).append(p)
    for name, files in loose.items():
        hints = [h for h in (read_hint(f.with_suffix(".txt")) for f in files) if h]
        add(name, files, hints[0] if hints else None, ", ".join(f.name for f in files))
    return {k: (v[0], v[1]) for k, v in found.items()}


SKIP = ("next, in WSL", "check that every building sits", "look at the views in")   # qwen_views' own hints


def run_quiet_hints(cmd):
    """run qwen_views.py, show its output live, but leave out its next-step hints for Pixal3D; returns exit code"""
    import os
    env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=0, env=env)
    buf = b""
    while True:
        chunk = p.stdout.read(512)
        if not chunk:
            break
        buf += chunk
        *lines, buf = buf.split(b"\n")
        for ln in lines:
            t = ln.decode("utf-8", "replace").rstrip("\r")
            if not t.lstrip("\r").startswith(SKIP):
                print(t, flush=True)
        if b"\r" in buf:                                  # "working... 12 s" progress line
            print(buf.decode("utf-8", "replace"), end="", flush=True)
            buf = b""
    if buf:
        print(buf.decode("utf-8", "replace"), flush=True)
    return p.wait()


def main():
    try:
        sys.stdout.reconfigure(errors="replace")
    except AttributeError:
        pass
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", default=str(HERE / "photos_raw"), help="input folder (default: photos_raw)")
    ap.add_argument("--clean-dir", default=str(HERE / "photos_clean"), help="output folder (default: photos_clean)")
    ap.add_argument("--only", nargs="+", help="only these buildings (names as --list shows them)")
    ap.add_argument("--list", action="store_true", help="only show buildings, photos and angles")
    a, passthrough = ap.parse_known_args()
    for bad in ("--clean", "--name", "--out", "--hint"):
        if bad in passthrough:
            sys.exit(f"{bad} is set by qwen_batch.py itself")

    raw, out = Path(a.raw).resolve(), Path(a.clean_dir).resolve()
    qv = HERE / "qwen_views.py"
    if not qv.is_file():
        sys.exit(f"qwen_views.py not found next to this script ({HERE})")
    if not raw.is_dir():
        sys.exit(f"no folder {raw} - put your photos there")
    buildings = find_buildings(raw)
    if a.only:
        want = {clean_name(n) for n in a.only}
        missing = want - set(buildings)
        if missing:
            sys.exit(f"not in {raw.name}: {', '.join(sorted(missing))}  (see: python qwen_batch.py --list)")
        buildings = {k: v for k, v in buildings.items() if k in want}
    if not buildings:
        sys.exit(f"no photos in {raw}")

    plan = []
    for name, (files, hint) in buildings.items():
        shots = give_angles(files)
        done = all((out / f"{name}_az{az_tag(ang)}.png").exists() for _, ang, _ in shots)
        plan.append((name, shots, hint, done))
    print(f"{len(plan)} building(s) in {raw}:")
    for name, shots, hint, done in plan:
        print(f"  {name}{'   (done)' if done and '--overwrite' not in passthrough else ''}"
              + (f"   hint: {hint}" if hint else ""))
        for f, ang, how in shots:
            print(f"      {f.name}  ->  {name}_az{az_tag(ang)}.png" + ("  (angle given here)" if how == "auto" else ""))
    if a.list:
        return

    out.mkdir(parents=True, exist_ok=True)
    todo = [p for p in plan if not p[3] or "--overwrite" in passthrough or "--dry-run" in passthrough]
    for i, (name, shots, hint, _) in enumerate(todo, 1):
        print(f"\n===== [{i}/{len(todo)}] {name}", flush=True)
        cmd = [sys.executable, "-B", str(qv)] + [f"{f}:{ang:g}" for f, ang, _ in shots] + \
              ["--clean", "--name", name, "--out", str(out)] + (["--hint", hint] if hint else []) + passthrough
        if run_quiet_hints(cmd):
            sys.exit(f"\nstopped at '{name}' (see above). Run the same command again: finished buildings are skipped.")
    if "--dry-run" in passthrough:
        return
    print(f"\nall done -> {out}\nlook at every picture next to its photo; a bad one: delete it, or run again with "
          f"--only <name> --seed 7 --overwrite")
    print("next: close ComfyUI, then in WSL:  trellis -q high --in photos_clean")


if __name__ == "__main__":
    main()
