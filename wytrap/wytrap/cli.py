"""argparse CLI: `wytrap detect ...`, `wytrap species ...`."""

from __future__ import annotations

import argparse
import sys

from wytrap import __version__
from wytrap.species_lists import BUILTIN_LISTS, load_species


def _add_detect_parser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "detect",
        help="Run MegaDetector v6 + BioCLIP-2 over a folder of images.",
    )
    p.add_argument("--input", "-i", required=True,
                   help="Folder of input images.")
    p.add_argument("--output", "-o", required=True,
                   help="Folder where per-image JSON results are written.")
    p.add_argument("--species", default="wyoming_all",
                   help="Builtin name (e.g. wyoming_all, wyoming_mammals, "
                        "ynp_testbed) or path to a newline-delimited species file.")
    p.add_argument("--det-threshold", type=float, default=0.75,
                   help="MegaDetector confidence cutoff (default: 0.75). "
                        "SpeciesNet-recommended range for production. "
                        "Lower (0.50, 0.30) if your camera deployments have "
                        "many tiny far-shot animals at the cost of more FPs.")
    p.add_argument("--cls-topk", type=int, default=5,
                   help="BioCLIP top-k labels to record per detection (default: 5).")
    p.add_argument("--batch-size", type=int, default=8,
                   help="MegaDetector batch size (default: 8). Currently unused "
                        "in the per-image loop; reserved for future batched mode.")
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"],
                   help="Compute device (default: auto).")
    p.add_argument("--recursive", action="store_true",
                   help="Walk subfolders of --input.")
    p.add_argument("--resume", action="store_true",
                   help="Skip images that already have an output JSON.")
    p.add_argument("--jsonl",
                   help="Optional path to also append all records to a single JSONL file.")
    p.add_argument("--min-pixel-side", type=int, default=60,
                   help="Boxes with shorter side below this many pixels are "
                        "tagged quality='low_pixels' and dropped (BioCLIP "
                        "needs ~60+ px on a side for usable signal). "
                        "Default 60.")
    p.add_argument("--border-overlap-truncated", type=float, default=0.20,
                   help="Boxes whose perimeter overlaps the image edge by at "
                        "least this fraction are tagged quality='truncated' "
                        "(animal extends out of frame). Treated as PASCAL "
                        "VOC 'difficult' at eval time — kept but not counted "
                        "as TP/FP/FN. Default 0.20.")
    p.add_argument("--max-aspect-ratio", type=float, default=8.0,
                   help="Boxes with longer-side / shorter-side above this "
                        "are tagged quality='thin' and dropped. Default 8.0; "
                        "loose enough to keep snakes and elongated angles.")
    p.add_argument("--skip-classification-when-bad", action="store_true",
                   help="Skip BioCLIP entirely on boxes whose quality is not "
                        "'ok'. Saves compute when you only care about clean "
                        "detections.")
    p.add_argument("--log-file",
                   help="Persistent log file. Default: <output>/wytrap.log. "
                        "Logs are appended in addition to stdout, so SLURM "
                        ".out and the run folder both keep a copy.")
    p.add_argument("--tile", action="store_true",
                   help="Enable SAHI sliced detection. Off by default after "
                        "tiny-eval A/B showed it hurt both precision and "
                        "recall on YNP-BisonGraze (full-image MegaDetector "
                        "alone outperforms full + tiles + NMS). Worth "
                        "re-enabling per-camera if you have very "
                        "zoomed-out deployments where small far-shot "
                        "animals are the priority.")
    p.add_argument("--tile-size", type=int, default=480,
                   help="Tile edge length in pixels (default: 480; "
                        "ignored unless --tile is set).")
    p.add_argument("--tile-overlap", type=float, default=0.2,
                   help="Fractional overlap between adjacent tiles "
                        "(default: 0.2 = 20%%).")
    p.add_argument("--no-multiscale", action="store_true",
                   help="Disable multi-scale BioCLIP classification. By "
                        "default each detection is classified at three "
                        "crop scales (tight / padded / whole-image) and "
                        "the highest-scoring scale wins — improves "
                        "extreme close-ups and far-shots.")
    p.add_argument("--multiscale-pad", type=float, default=2.0,
                   help="Padded-crop expansion factor for multi-scale "
                        "(default: 2.0 = 2x the box, clamped).")


def _add_species_parser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "species",
        help="Inspect a builtin or file-based species list.",
    )
    p.add_argument("--list", default="wyoming_all",
                   help=f"One of {sorted(BUILTIN_LISTS)} or a path to a species file.")
    p.add_argument("--count-only", action="store_true",
                   help="Print only the species count.")


def _cmd_detect(args: argparse.Namespace) -> int:
    # Lazy import: keeps `wytrap --version` and `wytrap species` fast.
    from wytrap.run import process_folder

    summary = process_folder(
        input_dir=args.input,
        output_dir=args.output,
        species=args.species,
        det_threshold=args.det_threshold,
        cls_topk=args.cls_topk,
        batch_size=args.batch_size,
        device=args.device,
        recursive=args.recursive,
        resume=args.resume,
        jsonl_path=args.jsonl,
        min_pixel_side=args.min_pixel_side,
        border_overlap_truncated=args.border_overlap_truncated,
        max_aspect_ratio=args.max_aspect_ratio,
        skip_classification_when_bad=args.skip_classification_when_bad,
        tile=args.tile,
        tile_size=args.tile_size,
        tile_overlap=args.tile_overlap,
        multiscale=not args.no_multiscale,
        multiscale_pad=args.multiscale_pad,
        log_file=args.log_file,
    )
    return 0 if summary["failed"] == 0 else 1


def _cmd_species(args: argparse.Namespace) -> int:
    species = load_species(args.list)
    if args.count_only:
        print(len(species))
    else:
        for s in species:
            if s["scientific"] == s["common"]:
                print(s["common"])
            else:
                print(f"{s['scientific']} | {s['common']}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="wytrap",
        description="Wyoming camera-trap detection + classification pipeline.",
    )
    parser.add_argument("--version", action="version",
                        version=f"wytrap {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)
    _add_detect_parser(sub)
    _add_species_parser(sub)

    args = parser.parse_args(argv)

    if args.command == "detect":
        return _cmd_detect(args)
    if args.command == "species":
        return _cmd_species(args)
    parser.error(f"unknown command {args.command!r}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
