"""OCR burned-in timestamps from camera-trap JPEGs to recover capture dates.

Used when EXIF DateTimeOriginal has been stripped (common after Labelbox
re-encoding) but the camera burned a visible timestamp into the image. Crops
narrow top + bottom strips, OCRs them with Tesseract, regex-matches multiple
date formats, validates year ranges.

Designed to be parallelizable (multiprocessing.Pool) so it scales to large
deployments. Output format matches scripts/build_exif_dates.py: a single
``file_to_date.json`` keyed by file basename.

Tesseract is required:
    module load tesseract                    # ARCC HPC
    # or:
    sudo apt install tesseract-ocr           # Debian/Ubuntu
    pip install pytesseract Pillow

Usage:
    python scripts/ocr_burnin_dates.py \\
        --images /path/to/images \\
        --out    /path/to/file_to_date.json \\
        [--workers 8] [--limit 100] [--resume] [--recursive]

The --limit flag is useful for a quick sanity check before running on the
full dataset: try `--limit 50` first, eyeball the success rate in the log,
then re-run without --limit to fill in the rest. --resume skips files that
already have non-empty entries in --out.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".tif", ".tiff", ".png"}

# Date format patterns ordered by specificity. Each tuple is (regex,
# component-order, optional-month-name). The regex captures groups that
# parse_date_groups maps to (year, month, day).
DATE_PATTERNS = [
    # 2024-05-22 / 2024/05/22 / 2024.05.22
    (re.compile(r"\b(20\d{2})[-/.](\d{1,2})[-/.](\d{1,2})\b"), "ymd"),
    # 05-11-2024 / 5/11/2024 / 05.11.2024 (assume MDY for US camera traps)
    (re.compile(r"\b(\d{1,2})[-/.](\d{1,2})[-/.](20\d{2})\b"), "mdy"),
    # 20240522 (compact ISO)
    (re.compile(r"\b(20\d{2})(\d{2})(\d{2})\b"), "ymd"),
]

# Plausible camera-trap date range. Narrow if you know your deployment range.
MIN_YEAR = 2015
MAX_YEAR = 2030


def parse_date_groups(groups: tuple[str, ...], order: str) -> str:
    if order == "ymd":
        y, m, d = groups
    elif order == "mdy":
        m, d, y = groups
    else:
        return ""
    try:
        dt = datetime(int(y), int(m), int(d))
    except ValueError:
        return ""
    if not (MIN_YEAR <= dt.year <= MAX_YEAR):
        return ""
    return dt.date().isoformat()


def extract_date(text: str) -> str:
    """Find the first plausible date in OCR text."""
    for pattern, order in DATE_PATTERNS:
        for match in pattern.finditer(text):
            d = parse_date_groups(match.groups(), order)
            if d:
                return d
    return ""


def ocr_image(path: Path,
              top_frac: float = 0.12,
              bottom_frac: float = 0.12) -> str:
    """OCR top + bottom strips of one image. Returns ISO date or ''.

    Both strips are checked because cameras vary: Reconyx burns at top,
    Bushnell at bottom. We accept whichever yields a plausible date first.
    """
    try:
        from PIL import Image
        import pytesseract
    except ImportError as e:
        print(f"missing dep: {e}; install pytesseract + tesseract binary",
              file=sys.stderr)
        return ""

    try:
        img = Image.open(path).convert("L")  # grayscale; OCR doesn't need color
    except Exception:
        return ""

    W, H = img.size
    top = img.crop((0, 0, W, max(20, int(H * top_frac))))
    bot = img.crop((0, H - max(20, int(H * bottom_frac)), W, H))

    # PSM 6: uniform block of text (timestamp burns are wide single rows or
    # a small block). Whitelist digits + common date separators + a few chars
    # that show up adjacent ("T", ":", space). Tesseract is much faster and
    # more accurate when it doesn't have to consider the full alphabet.
    config = ("--psm 6 -c "
              "tessedit_char_whitelist=0123456789-/.:T ")

    text = ""
    try:
        text = (pytesseract.image_to_string(top, config=config)
                + " "
                + pytesseract.image_to_string(bot, config=config))
    except Exception:
        return ""

    return extract_date(text)


def worker(path_str: str) -> tuple[str, str]:
    p = Path(path_str)
    return (p.name, ocr_image(p))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--images", required=True,
                   help="Folder of images to OCR.")
    p.add_argument("--out", required=True,
                   help="Path to file_to_date.json. With --resume, "
                        "non-empty existing entries are kept.")
    p.add_argument("--recursive", action="store_true", default=True,
                   help="Walk subfolders (default true).")
    p.add_argument("--no-recursive", action="store_false", dest="recursive",
                   help="Don't walk subfolders.")
    p.add_argument("--resume", action="store_true",
                   help="Skip files already in --out (with non-empty date).")
    p.add_argument("--limit", type=int,
                   help="Process at most this many files. Useful for "
                        "validating the OCR pipeline before a full run.")
    p.add_argument("--workers", type=int, default=os.cpu_count() or 1,
                   help="Number of parallel OCR worker processes "
                        "(default: cpu_count()).")
    p.add_argument("--progress-every", type=int, default=500,
                   help="Log a status line every N images processed.")
    p.add_argument("--checkpoint-every", type=int, default=5000,
                   help="Persist results to --out every N images so a job "
                        "interrupted halfway can resume.")
    p.add_argument("--top-frac", type=float, default=0.12,
                   help="Fraction of image height used for the top strip "
                        "(default 0.12 = 12%%). Bigger if your camera burns "
                        "in a deeper band.")
    p.add_argument("--bottom-frac", type=float, default=0.12,
                   help="Same for the bottom strip.")
    return p.parse_args()


def _check_tesseract() -> None:
    try:
        import pytesseract  # noqa
    except ImportError:
        print("ERROR: pytesseract not installed. `pip install pytesseract`",
              file=sys.stderr)
        sys.exit(2)
    try:
        import pytesseract
        version = pytesseract.get_tesseract_version()
        print(f"tesseract version: {version}")
    except pytesseract.TesseractNotFoundError:
        print("ERROR: tesseract binary not on PATH. On ARCC try "
              "`module load tesseract`. On Debian/Ubuntu: "
              "`sudo apt install tesseract-ocr`.",
              file=sys.stderr)
        sys.exit(2)


def main() -> int:
    args = parse_args()
    _check_tesseract()

    images_dir = Path(args.images)
    out_path = Path(args.out)

    cache: dict[str, str] = {}
    if args.resume and out_path.exists():
        with open(out_path) as f:
            cache = json.load(f)
        n_with = sum(1 for v in cache.values() if v)
        print(f"[resume] loaded {len(cache)} cached entries "
              f"({n_with} with dates) from {out_path}")

    pattern = "**/*" if args.recursive else "*"
    candidates = sorted(p for p in images_dir.glob(pattern)
                        if p.is_file()
                        and p.suffix.lower() in IMAGE_EXTS
                        and not p.name.startswith("."))
    print(f"found {len(candidates)} image files in {images_dir}")

    todo = [str(p) for p in candidates if not cache.get(p.name)]
    if args.limit:
        todo = todo[:args.limit]
    if not todo:
        print("nothing to do.")
        return 0
    print(f"OCRing {len(todo)} files with {args.workers} worker(s)...")

    t0 = time.time()
    n_ok = n_missing = 0
    last_checkpoint = 0
    use_pool = args.workers > 1 and len(todo) > 1
    pool_ctx = mp.Pool(args.workers) if use_pool else None
    try:
        iterator = (pool_ctx.imap_unordered(worker, todo, chunksize=8)
                    if use_pool else (worker(p) for p in todo))
        for i, (fname, date) in enumerate(iterator, 1):
            cache[fname] = date
            if date:
                n_ok += 1
            else:
                n_missing += 1
            if i % args.progress_every == 0 or i == len(todo):
                elapsed = time.time() - t0
                rate = i / elapsed if elapsed > 0 else 0
                eta = (len(todo) - i) / rate if rate > 0 else 0
                pct_ok = (n_ok / i * 100) if i else 0
                print(f"  {i}/{len(todo)}  rate={rate:6.0f}/s  "
                      f"eta={int(eta)}s  ok={n_ok}({pct_ok:.0f}%)  "
                      f"no-date={n_missing}")
            if i - last_checkpoint >= args.checkpoint_every:
                _save(cache, out_path)
                last_checkpoint = i
    finally:
        if pool_ctx:
            pool_ctx.close()
            pool_ctx.join()

    _save(cache, out_path)
    n_total_ok = sum(1 for v in cache.values() if v)
    print(f"wrote {out_path}  ({len(cache)} entries; {n_total_ok} with dates, "
          f"{len(cache) - n_total_ok} missing)")
    return 0


def _save(cache: dict[str, str], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(cache, f, indent=2, sort_keys=True)
    tmp.replace(out_path)


if __name__ == "__main__":
    sys.exit(main())
