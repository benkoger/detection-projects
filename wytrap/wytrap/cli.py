"""The `wytrap` command.

    wytrap ingest     arrange a raw camera dump into images/ + labels.json + manifest.json
    wytrap detect     images -> MegaDetector -> a classifier -> records
    wytrap classify   re-label an earlier run's boxes with another classifier
    wytrap calibrate  BioCLIP prior correction from a run's prompt scores
    wytrap eval       score a run against image-level labels
    wytrap merge      side-by-side table of several classifiers on the same boxes
    wytrap census     what a run saw per camera, and what the vocabulary is missing
    wytrap vocab      write the candidate lists a vocabulary implies per model
    wytrap species    inspect a built-in or file-based species list

`wytrap <command> --help` lists every option.
"""

from __future__ import annotations

import argparse
import sys

from wytrap import __version__
from wytrap.classifiers import CLASSIFIERS
from wytrap.species_lists import BUILTIN_LISTS, load_species

DEFAULT_DETECTOR = "MDV1000-redwood"   # Hugging Face hosted; what AddaxAI Connect runs


# --------------------------------------------------------------------------
# option groups
# --------------------------------------------------------------------------

def _add_classifier_args(p: argparse.ArgumentParser, default: str = "bioclip") -> None:
    g = p.add_argument_group("classifier")
    g.add_argument("--classifier", "-c", default=default, choices=CLASSIFIERS,
                   help=f"species model (default {default}). "
                        "bioclip: BioCLIP 2 zero-shot; speciesnet: SpeciesNet per box; "
                        "speciesnet-ensemble: SpeciesNet with roll-up and geofence; "
                        "addax: an AddaxAI zoo model (needs --model); none: detection only.")
    g.add_argument("--model", "-m", default=None,
                   help="checkpoint for the classifier: an AddaxAI repo id "
                        "(Addax-Data-Science/WUSA-SDZWA-v1), a SpeciesNet id "
                        "(hf:Addax-Data-Science/SPECIESNET-v4-0-2-A), or an open_clip "
                        "model string for BioCLIP. Each classifier has a default.")
    g.add_argument("--vocab", default=None,
                   help="taxon-node vocabulary CSV (taxonomy/wyoming_vocab.csv). Restricts "
                        "every classifier to the same candidate set: BioCLIP uses its "
                        "prompts, SpeciesNet and zoo models renormalise over the classes "
                        "that fall inside it.")
    g.add_argument("--cls-topk", type=int, default=5,
                   help="labels to record per box (default 5)")
    g.add_argument("--batch-size", type=int, default=32,
                   help="crops per forward pass for SpeciesNet / zoo models (default 32)")
    g.add_argument("--skip-classification-when-bad", action="store_true",
                   help="do not classify boxes whose quality tag is not 'ok'")
    b = p.add_argument_group("bioclip")
    b.add_argument("--species", default="wyoming_all",
                   help="BioCLIP candidate list when no --vocab: a built-in name "
                        f"({', '.join(sorted(BUILTIN_LISTS))}) or a file of "
                        "'Scientific name | common name' lines. Default wyoming_all.")
    b.add_argument("--prompt-bias", default=None,
                   help="prompt_bias.json from `wytrap calibrate`: per-prompt log-space "
                        "biases subtracted before ranking (prior correction)")
    b.add_argument("--no-multiscale", action="store_true",
                   help="classify the tight crop only (default: tight, padded and whole "
                        "frame, best scale wins)")
    b.add_argument("--multiscale-pad", type=float, default=2.0,
                   help="padded-crop expansion factor (default 2.0)")
    s = p.add_argument_group("speciesnet-ensemble")
    s.add_argument("--country", default="USA", help="ISO3 country for the geofence (default USA)")
    s.add_argument("--admin1", default="WY", help="state for the geofence (default WY)")
    s.add_argument("--no-geofence", action="store_true", help="disable SpeciesNet's geofence")


def _build_classifier(args: argparse.Namespace):
    from wytrap.classifiers import build_classifier
    return build_classifier(
        args.classifier, model=args.model, vocab=args.vocab, device=args.device,
        topk=args.cls_topk, batch_size=args.batch_size,
        skip_bad=args.skip_classification_when_bad,
        species=args.species, prompt_bias=args.prompt_bias,
        multiscale=not args.no_multiscale, multiscale_pad=args.multiscale_pad,
        country=args.country, admin1=args.admin1, geofence=not args.no_geofence)


def _add_detector_args(p: argparse.ArgumentParser) -> None:
    d = p.add_argument_group("detector")
    d.add_argument("--detector", default=DEFAULT_DETECTOR,
                   help="MegaDetector checkpoint: MDV1000-redwood (default; what AddaxAI "
                        "Connect runs), MDV5a, MDV5b (Hugging Face), or an MDv6 variant "
                        "(MDV6-yolov9-c/e, MDV6-yolov10-c/e, MDV6-rtdetr-c; from Zenodo)")
    d.add_argument("--det-threshold", type=float, default=0.20,
                   help="confidence floor for recorded boxes (default 0.20). Record low "
                        "and raise the floor at eval or merge time; 0.30 was the best "
                        "value on Idaho.")
    d.add_argument("--det-imgsz", type=int, default=None,
                   help="detector input size, long side px (multiple of 32). Default the "
                        "model's native 1280; 1920/2560 recover small animals at 2-4x cost.")
    d.add_argument("--keep-labels", default="animal,person,vehicle",
                   help="MegaDetector classes to record (default all three). Only animal "
                        "boxes are classified; people and vehicles keep the detector's label.")
    d.add_argument("--tile", action="store_true",
                   help="SAHI sliced detection (off by default: it hurt both precision and "
                        "recall on full-frame camera-trap images)")
    d.add_argument("--tile-size", type=int, default=480)
    d.add_argument("--tile-overlap", type=float, default=0.2)
    q = p.add_argument_group("box quality")
    q.add_argument("--min-pixel-side", type=int, default=60,
                   help="boxes with a shorter side below this are tagged low_pixels (default 60)")
    q.add_argument("--border-overlap-truncated", type=float, default=0.20,
                   help="perimeter fraction on the image edge that tags a box truncated (default 0.20)")
    q.add_argument("--max-aspect-ratio", type=float, default=8.0,
                   help="aspect ratio above which a box is tagged thin (default 8.0)")


def _add_common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--log-file", default=None,
                   help="persistent log (default <output>/wytrap.log); appended to")


# --------------------------------------------------------------------------
# commands
# --------------------------------------------------------------------------

def _cmd_detect(args: argparse.Namespace) -> int:
    from wytrap.run import process_folder
    classifier = _build_classifier(args)
    summary = process_folder(
        input_dir=args.input, output_dir=args.output, classifier=classifier,
        detector_version=args.detector, det_threshold=args.det_threshold,
        det_imgsz=args.det_imgsz,
        keep_labels=tuple(s.strip() for s in args.keep_labels.split(",") if s.strip()),
        device=args.device, recursive=not args.no_recursive, resume=not args.no_resume,
        jsonl_path=args.jsonl or f"{args.output}/all_records.jsonl",
        min_pixel_side=args.min_pixel_side,
        border_overlap_truncated=args.border_overlap_truncated,
        max_aspect_ratio=args.max_aspect_ratio,
        tile=args.tile, tile_size=args.tile_size, tile_overlap=args.tile_overlap,
        log_file=args.log_file)
    return 0 if summary["failed"] == 0 else 1


def _cmd_classify(args: argparse.Namespace) -> int:
    from wytrap.run import reclassify
    classifier = _build_classifier(args)
    if classifier is None:
        raise SystemExit("wytrap classify needs a classifier other than 'none'")
    summary = reclassify(records_path=args.records, output_dir=args.output,
                         classifier=classifier, jsonl_path=args.jsonl, log_file=args.log_file)
    return 0 if summary["failed"] == 0 else 1


def _cmd_species(args: argparse.Namespace) -> int:
    species = load_species(args.list)
    if args.count_only:
        print(len(species))
    else:
        for s in species:
            print(s["common"] if s["scientific"] == s["common"]
                  else f"{s['scientific']} | {s['common']}")
    return 0


def _cmd_vocab(args: argparse.Namespace) -> int:
    from wytrap.vocab import write_lists
    write_lists(args.vocab, speciesnet_model=args.speciesnet_model, zoo=args.zoo)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="wytrap", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--version", action="version", version=f"wytrap {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("detect", help="MegaDetector + a classifier over a folder of images",
                       formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--input", "-i", required=True, help="folder of images (walked recursively)")
    p.add_argument("--output", "-o", required=True, help="run folder for records and logs")
    p.add_argument("--jsonl", default=None, help="aggregate JSONL (default <output>/all_records.jsonl)")
    p.add_argument("--no-recursive", action="store_true", help="top-level images only")
    p.add_argument("--no-resume", action="store_true",
                   help="redo images that already have an output JSON (default: skip them)")
    _add_classifier_args(p)
    _add_detector_args(p)
    _add_common_args(p)
    p.set_defaults(func=_cmd_detect)

    p = sub.add_parser("classify", help="another classifier on the boxes of an earlier run",
                       formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--records", "-r", required=True,
                   help="all_records.jsonl of the run whose boxes to reuse")
    p.add_argument("--output", "-o", required=True)
    p.add_argument("--jsonl", default=None, help="aggregate JSONL (default <output>/all_records.jsonl)")
    _add_classifier_args(p, default="speciesnet-ensemble")
    _add_common_args(p)
    p.set_defaults(func=_cmd_classify)

    from wytrap import calibrate, census, evaluate, ingest, merge
    p = sub.add_parser("census", help="tally a run's labels per camera and audit them against a vocabulary",
                       description=census.__doc__,
                       formatter_class=argparse.RawDescriptionHelpFormatter)
    census.add_arguments(p)
    p.set_defaults(func=census.run)

    p = sub.add_parser("ingest", help="arrange a raw camera dump into images/ + labels.json + manifest.json",
                       description=ingest.__doc__,
                       formatter_class=argparse.RawDescriptionHelpFormatter)
    ingest.add_arguments(p)
    p.set_defaults(func=ingest.run)

    p = sub.add_parser("calibrate", help="BioCLIP prior correction (label-free)",
                       description=calibrate.__doc__,
                       formatter_class=argparse.RawDescriptionHelpFormatter)
    calibrate.add_arguments(p)
    p.set_defaults(func=calibrate.run)

    p = sub.add_parser("eval", help="score a run against image-level labels",
                       description=evaluate.__doc__,
                       formatter_class=argparse.RawDescriptionHelpFormatter)
    evaluate.add_arguments(p)
    p.set_defaults(func=evaluate.run)

    p = sub.add_parser("merge", help="side-by-side table of several runs on the same boxes",
                       description=merge.__doc__,
                       formatter_class=argparse.RawDescriptionHelpFormatter)
    merge.add_arguments(p)
    p.set_defaults(func=lambda a, _p=p: merge.run(_p, a))

    p = sub.add_parser("vocab", help="write the per-model candidate lists of a vocabulary")
    p.add_argument("vocab", help="taxonomy/*.csv")
    p.add_argument("--speciesnet-model", default=None,
                   help="also write the SpeciesNet target list (hf:Addax-Data-Science/SPECIESNET-v4-0-2-A)")
    p.add_argument("--zoo", nargs="*", default=[],
                   help="AddaxAI repos to report kept/masked classes for")
    p.set_defaults(func=_cmd_vocab)

    p = sub.add_parser("species", help="inspect a built-in or file-based species list")
    p.add_argument("--list", default="wyoming_all",
                   help=f"one of {sorted(BUILTIN_LISTS)} or a path to a species file")
    p.add_argument("--count-only", action="store_true")
    p.set_defaults(func=_cmd_species)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
