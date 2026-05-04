"""Result record schema and (de)serialization."""

from dataclasses import dataclass, asdict, field
from pathlib import Path
import json


@dataclass
class DetectionRecord:
    box_xyxy: list[int]            # [x1, y1, x2, y2] absolute pixels
    det_score: float               # MegaDetector confidence
    det_label: str                 # "animal" / "person" / "vehicle"
    label: str                     # canonical/merged class (== fine_label by default)
    fine_label: str                # BioCLIP top-1 common name
    scientific_label: str          # BioCLIP top-1 scientific name (the prompt)
    cls_score: float               # BioCLIP top-1 score
    topk: list[dict]               # [{"common": ..., "scientific": ..., "score": ...}, ...]
    quality: str = "ok"            # "ok" | "edge" | "small" | "thin" | "skipped"
    quality_reason: str = ""       # short detail when quality != "ok"


@dataclass
class ImageRecord:
    image_path: str
    image_size: list[int]          # [W, H]
    detections: list[DetectionRecord] = field(default_factory=list)
    error: str | None = None       # populated if the image failed to process


def save_record(record: ImageRecord, out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(asdict(record), f, indent=2)


def append_jsonl(record: ImageRecord, out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "a") as f:
        f.write(json.dumps(asdict(record)) + "\n")


def load_record(path: str | Path) -> ImageRecord:
    with open(path, "r") as f:
        d = json.load(f)
    detections = []
    for det in d.get("detections", []):
        # Be tolerant of older records that lack the new fields.
        det.setdefault("scientific_label", det.get("fine_label", ""))
        det.setdefault("quality", "ok")
        det.setdefault("quality_reason", "")
        # topk migrated from [[name, score], ...] to [{...}, ...].
        topk = det.get("topk", [])
        if topk and isinstance(topk[0], (list, tuple)):
            det["topk"] = [{"common": name, "scientific": name, "score": score}
                           for name, score in topk]
        detections.append(DetectionRecord(**det))
    return ImageRecord(
        image_path=d["image_path"],
        image_size=d["image_size"],
        detections=detections,
        error=d.get("error"),
    )


def output_path_for(image_path: str | Path, out_dir: str | Path,
                    input_root: str | Path | None = None) -> Path:
    """Mirror the input folder structure under out_dir, swapping suffix to .json."""
    image_path = Path(image_path)
    out_dir = Path(out_dir)
    if input_root is not None:
        try:
            rel = image_path.relative_to(Path(input_root))
        except ValueError:
            rel = Path(image_path.name)
    else:
        rel = Path(image_path.name)
    return out_dir / rel.with_suffix(".json")
