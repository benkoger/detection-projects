"""OCR burned-in timestamps from camera-trap JPEGs to recover capture dates.

Used when EXIF DateTimeOriginal has been stripped (common after Labelbox
re-encoding) but the camera burned a visible timestamp into the image. Crops
narrow top + bottom strips and OCRs them; regex-matches multiple date
formats; validates year ranges.

Default engine is **EasyOCR** because it pip-installs cleanly with no system
binary required, runs on GPU when available, and is robust on the
small-band timestamps these cameras burn in. Pass ``--engine tesseract`` if
you have the tesseract binary on PATH and want CPU multiprocessing instead.

Output format matches scripts/eval_pipeline.py expectations: a single
``file_to_date.json`` keyed by file basename. Resumable (re-runs skip files
already in --out with non-empty values).

Setup:
    pip install easyocr           # default engine
    pip install pytesseract       # only if --engine tesseract

Usage:
    # Sanity check on a sample first
    python scripts/ocr_burnin_dates.py \\
        --images /path/to/images \\
        --out    /tmp/test.json \\
        --limit 50

    # Full run
    python scripts/ocr_burnin_dates.py \\
        --images /path/to/images \\
        --out    /path/to/file_to_date.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".tif", ".tiff", ".png"}

# Date format patterns ordered by specificity. Each tuple is (regex,
# component-order). The regex captures groups that parse_date_groups maps to
# (year, month, day).
DATE_PATTERNS = [
    # 2024-05-22 / 2024/05/22 / 2024.05.22
    (re.compile(r"\b(20\d{2})[-/.](\d{1,2})[-/.](\d{1,2})\b"), "ymd"),
    # 05-11-2024 / 5/11/2024 / 05.11.2024 (assume MDY for US camera traps)
    (re.compile(r"\b(\d{1,2})[-/.](\d{1,2})[-/.](20\d{2})\b"), "mdy"),
    # 20240522 (compact ISO)
    (re.compile(r"\b(20\d{2})(\d{2})(\d{2})\b"), "ymd"),
]

# Plausible camera-trap date range. Tighten if you know your deployment.
MIN_YEAR = 2015
MAX_YEAR = 2030

# Characters EasyOCR / Tesseract may emit for timestamp burn-ins.
ALLOWLIST = "0123456789-/.:T "


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


# ---------------------- engine: EasyOCR ----------------------

class EasyOCREngine:
    """Single-process EasyOCR reader. Loads the recognition model once;
    reads text from a numpy strip per call. Uses GPU when available."""

    def __init__(self):
        try:
            import easyocr  # noqa
            import numpy as np  # noqa
        except ImportError as e:
            raise RuntimeError(
                f"EasyOCR not installed: {e}. `pip install easyocr`"
            ) from e
        try:
            import torch
            self.gpu = bool(torch.cuda.is_available())
        except ImportError:
            self.gpu = False
        # `verbose=False` suppresses model-download chatter during init.
        # `allowlist` is enforced at recognition time.
        import easyocr
        self.reader = easyocr.Reader(["en"], gpu=self.gpu, verbose=False)
        print(f"easyocr ready (gpu={self.gpu})")

    def ocr(self, np_image) -> str:
        """np_image: HxWx3 uint8 RGB or HxW grayscale. Returns concatenated
        text recognized in the strip."""
        try:
            # detail=0 → list of strings (no boxes); paragraph=False keeps
            # short tokens separate so the regex can match dates inside them.
            chunks = self.reader.readtext(
                np_image, allowlist=ALLOWLIST,
                detail=0, paragraph=False,
            )
            return " ".join(chunks)
        except Exception:
            return ""


# ---------------------- engine: Tesseract ----------------------

class TesseractEngine:
    """CPU Tesseract via pytesseract; uses multiprocessing externally.
    Each worker process imports this and runs OCR on one image at a time."""

    def __init__(self):
        try:
            import pytesseract  # noqa
        except ImportError as e:
            raise RuntimeError(
                f"pytesseract not installed: {e}. `pip install pytesseract`"
            ) from e
        try:
            import pytesseract
            pytesseract.get_tesseract_version()
        except Exception as e:
            raise RuntimeError(
                f"tesseract binary not on PATH ({e}). On ARCC: try "
                "`module load tesseract`. On Debian/Ubuntu: "
                "`sudo apt install tesseract-ocr`."
            ) from e

    def ocr(self, np_image) -> str:
        try:
            from PIL import Image
            import pytesseract
            img = Image.fromarray(np_image)
            config = f"--psm 6 -c tessedit_char_whitelist={ALLOWLIST}"
            return pytesseract.image_to_string(img, config=config)
        except Exception:
            return ""


# ---------------------- per-image driver ----------------------

def crop_strips(path: Path,
                top_frac: float, bottom_frac: float):
    """Returns (top_np, bottom_np) as numpy uint8 arrays (or None on failure)."""
    try:
        from PIL import Image
        import numpy as np
    except ImportError as e:
        print(f"missing dep: {e}", file=sys.stderr)
        return None, None
    try:
        img = Image.open(path).convert("RGB")
    except Exception:
        return None, None
    W, H = img.size
    top = img.crop((0, 0, W, max(20, int(H * top_frac))))
    bot = img.crop((0, H - max(20, int(H * bottom_frac)), W, H))
    return np.asarray(top), np.asarray(bot)


def ocr_image(engine, path: Path,
              top_frac: float, bottom_frac: float) -> str:
    top, bot = crop_strips(path, top_frac, bottom_frac)
    if top is None:
        return ""
    text = engine.ocr(top) + " " + engine.ocr(bot)
    return extract_date(text)


# ---------------------- main / CLI ----------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--images", required=True,
                   help="Folder of images to OCR.")
    p.add_argument("--out", required=True,
                   help="Path to file_to_date.json. With --resume, "
                        "non-empty existing entries are kept.")
    p.add_argument("--engine", default="easyocr",
                   choices=["easyocr", "tesseract"],
                   help="OCR engine (default: easyocr — pure pip, GPU). "
                        "tesseract uses CPU multiprocessing and needs the "
                        "system binary.")
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
                   help="Tesseract only: number of CPU workers "
                        "(default: cpu_count()). Ignored for easyocr.")
    p.add_argument("--progress-every", type=int, default=500,
                   help="Log a status line every N images processed.")
    p.add_argument("--checkpoint-every", type=int, default=5000,
                   help="Persist results to --out every N images so a job "
                        "interrupted halfway can resume.")
    p.add_argument("--top-frac", type=float, default=0.12,
                   help="Fraction of image height for the top strip "
                        "(default 0.12 = 12%%).")
    p.add_argument("--bottom-frac", type=float, default=0.12,
                   help="Same for the bottom strip.")
    return p.parse_args()


def main() -> int:
    args = parse_args()

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

    todo = [p for p in candidates if not cache.get(p.name)]
    if args.limit:
        todo = todo[:args.limit]
    if not todo:
        print("nothing to do.")
        return 0

    if args.engine == "easyocr":
        engine = EasyOCREngine()
        return _run_serial(engine, todo, cache, out_path, args)
    else:
        return _run_tesseract_pool(todo, cache, out_path, args)


def _save(cache: dict[str, str], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(cache, f, indent=2, sort_keys=True)
    tmp.replace(out_path)


def _run_serial(engine, todo, cache, out_path, args) -> int:
    print(f"OCRing {len(todo)} files with engine={args.engine}...")
    t0 = time.time()
    n_ok = n_missing = 0
    last_checkpoint = 0
    for i, path in enumerate(todo, 1):
        date = ocr_image(engine, path,
                         top_frac=args.top_frac,
                         bottom_frac=args.bottom_frac)
        cache[path.name] = date
        if date:
            n_ok += 1
        else:
            n_missing += 1
        if i % args.progress_every == 0 or i == len(todo):
            elapsed = time.time() - t0
            rate = i / elapsed if elapsed > 0 else 0
            eta = (len(todo) - i) / rate if rate > 0 else 0
            pct = (n_ok / i * 100) if i else 0
            print(f"  {i}/{len(todo)}  rate={rate:5.1f}/s  "
                  f"eta={int(eta)}s  ok={n_ok}({pct:.0f}%)  "
                  f"no-date={n_missing}")
        if i - last_checkpoint >= args.checkpoint_every:
            _save(cache, out_path)
            last_checkpoint = i

    _save(cache, out_path)
    n_total_ok = sum(1 for v in cache.values() if v)
    print(f"wrote {out_path}  ({len(cache)} entries; {n_total_ok} with dates, "
          f"{len(cache) - n_total_ok} missing)")
    return 0


# Tesseract path uses multiprocessing — each worker loads its own engine.
_tess_engine = None  # per-process

def _tess_worker(arg) -> tuple[str, str]:
    global _tess_engine
    path_str, top_frac, bottom_frac = arg
    if _tess_engine is None:
        _tess_engine = TesseractEngine()
    return (Path(path_str).name,
            ocr_image(_tess_engine, Path(path_str), top_frac, bottom_frac))


def _run_tesseract_pool(todo, cache, out_path, args) -> int:
    import multiprocessing as mp
    print(f"OCRing {len(todo)} files with engine=tesseract, "
          f"workers={args.workers}...")
    t0 = time.time()
    n_ok = n_missing = 0
    last_checkpoint = 0
    work = [(str(p), args.top_frac, args.bottom_frac) for p in todo]
    with mp.Pool(args.workers) as pool:
        for i, (fname, date) in enumerate(
                pool.imap_unordered(_tess_worker, work, chunksize=8), 1):
            cache[fname] = date
            if date:
                n_ok += 1
            else:
                n_missing += 1
            if i % args.progress_every == 0 or i == len(todo):
                elapsed = time.time() - t0
                rate = i / elapsed if elapsed > 0 else 0
                eta = (len(todo) - i) / rate if rate > 0 else 0
                pct = (n_ok / i * 100) if i else 0
                print(f"  {i}/{len(todo)}  rate={rate:5.1f}/s  "
                      f"eta={int(eta)}s  ok={n_ok}({pct:.0f}%)  "
                      f"no-date={n_missing}")
            if i - last_checkpoint >= args.checkpoint_every:
                _save(cache, out_path)
                last_checkpoint = i

    _save(cache, out_path)
    n_total_ok = sum(1 for v in cache.values() if v)
    print(f"wrote {out_path}  ({len(cache)} entries; {n_total_ok} with dates, "
          f"{len(cache) - n_total_ok} missing)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
