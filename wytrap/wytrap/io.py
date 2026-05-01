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
    fine_label: str                # raw BioCLIP top-1
    cls_score: float               # BioCLIP top-1 score
    topk: list[list]               # [[name, score], ...]


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
    detections = [DetectionRecord(**det) for det in d.get("detections", [])]
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
