"""Walk an image folder and emit ``file_to_date.json`` from EXIF DateTimeOriginal.

Camera-trap COCO exports sometimes carry stale or wrong ``date_captured``
(e.g. Labelbox-side bugs, or capture-vs-upload time confusion). The visible
timestamp burned into the JPEG is usually right; the EXIF ``DateTimeOriginal``
field on the JPEG is the same value the camera wrote and is read in <2 ms
per image. OCR'ing the burn-in is multiple seconds per image and impractical
at million-image scale, so we trust EXIF.

Usage:
    python scripts/build_exif_dates.py \\
        --images /path/to/image_folder \\
        --out    /path/to/file_to_date.json \\
        [--recursive]                         # default: on
        [--resume]                            # skip files already in --out

Output is a dict {file_basename: 'YYYY-MM-DD'}. Files with no readable EXIF
date are recorded as empty strings — eval falls back to COCO's
``date_captured`` for those.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--images", required=True,
                   help="Folder of images.")
    p.add_argument("--out", required=True,
                   help="Path to file_to_date.json. Existing entries are "
                        "kept when --resume is set.")
    p.add_argument("--recursive", action="store_true", default=True,
                   help="Walk subfolders (default true).")
    p.add_argument("--no-recursive", action="store_false", dest="recursive",
                   help="Don't walk subfolders.")
    p.add_argument("--resume", action="store_true",
                   help="If --out already exists, keep its entries and only "
                        "process files not in it. Useful for adding new "
                        "deployments to an existing dataset.")
    p.add_argument("--progress-every", type=int, default=1000,
                   help="Log a status line every N images (default 1000).")
    return p.parse_args()


def parse_exif_date(path: Path) -> str:
    """Return 'YYYY-MM-DD' from the JPEG's EXIF DateTimeOriginal, or '' if
    not present / unparseable."""
    try:
        from PIL import Image
    except ImportError:
        print("ERROR: Pillow not installed. `pip install Pillow`",
              file=sys.stderr)
        sys.exit(2)

    try:
        with Image.open(path) as img:
            exif = img.getexif()
            if not exif:
                return ""
            # Walk the IFDs to find DateTimeOriginal (tag id 36867 = 0x9003).
            for sub_ifd_id in (0x8769, 0x9003, 0x132):
                pass  # fall through to IFD lookup below
            # Most cameras put it in the Exif IFD (0x8769).
            try:
                exif_ifd = exif.get_ifd(0x8769)
                raw = exif_ifd.get(0x9003) or exif_ifd.get(0x9004)
            except Exception:
                raw = None
            # Fallback: top-level DateTime tag (0x0132)
            if not raw:
                raw = exif.get(0x0132) or exif.get(0x9003) or exif.get(0x9004)
            if not raw:
                return ""
            # EXIF format: "YYYY:MM:DD HH:MM:SS"
            raw = str(raw).strip()
            if len(raw) >= 10:
                date_part = raw[:10].replace(":", "-")
                # Sanity-check it parses as ISO date.
                from datetime import date as _date
                try:
                    _date.fromisoformat(date_part)
                    return date_part
                except ValueError:
                    return ""
            return ""
    except Exception:
        return ""


def main() -> int:
    args = parse_args()
    images_dir = Path(args.images)
    out_path = Path(args.out)

    file_to_date: dict[str, str] = {}
    if args.resume and out_path.exists():
        with open(out_path) as f:
            file_to_date = json.load(f)
        print(f"[resume] loaded {len(file_to_date)} cached entries from {out_path}")

    pattern = "**/*" if args.recursive else "*"
    candidates = sorted(p for p in images_dir.glob(pattern)
                        if p.is_file()
                        and p.suffix.lower() in IMAGE_EXTS
                        and not p.name.startswith("."))
    print(f"found {len(candidates)} image files in {images_dir}")

    todo = [p for p in candidates if p.name not in file_to_date]
    if not todo:
        print("nothing to do — all files already in cache.")
        return 0
    print(f"reading EXIF for {len(todo)} new files...")

    t0 = time.time()
    n_ok = n_missing = 0
    for i, p in enumerate(todo, 1):
        d = parse_exif_date(p)
        file_to_date[p.name] = d
        if d:
            n_ok += 1
        else:
            n_missing += 1

        if i % args.progress_every == 0 or i == len(todo):
            elapsed = time.time() - t0
            rate = i / elapsed if elapsed > 0 else 0
            eta = (len(todo) - i) / rate if rate > 0 else 0
            print(f"  {i}/{len(todo)}  rate={rate:6.0f}/s  "
                  f"eta={int(eta)}s  ok={n_ok}  no-date={n_missing}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(file_to_date, f, indent=2, sort_keys=True)
    print(f"wrote {out_path}  ({len(file_to_date)} entries; {n_ok} with dates, "
          f"{n_missing} missing)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
