"""link_cultivars.py — tag each bud crop in `labels.csv` with its cultivar.

The bud-instances dataset has no cultivar information: a crop is named
`<frame>_b<x>_<y>.png`, where `<frame>.png` is the source frame it was cut from.
In the *split-views* (full-flower) dataset, every frame lives inside a subfolder
named after the cultivar. So the cultivar of a crop = the split-views subfolder
that contains its source frame.

This is a one-time, re-runnable prep step: it reads `<dataset>/labels.csv`
(`Filename;LabelIndex`) and rewrites it with an added `Cultivar` column. Run it
again after regenerating the crops. The cultivar is metadata only (a tag for
the viewer filter and the leakage-safe split) — it is **not** the class/target,
which stays the ripeness `LabelIndex` (1/3/5).

Fails loud (Rule 12): if any crop's source frame is not found under the
split-views tree, nothing is written and the offending crops are listed.

    python link_cultivars.py --dataset /path/to/bud-instances \
                             --split-views /path/to/...-split-views
"""

import argparse
import csv
import re
from pathlib import Path

_CROP_SUFFIX = re.compile(r"_b\d+_\d+(\.[^.]+)$")  # `_b<x>_<y>.png` → source frame ext


def _frame_to_cultivar(split_views: Path) -> dict[str, str]:
    """Map each frame filename → its cultivar (the subfolder it sits in).

    Raises if the same frame name appears under two different cultivar folders —
    that would make the crop→cultivar link ambiguous.
    """
    mapping: dict[str, str] = {}
    for png in split_views.rglob("*.png"):
        cultivar = png.parent.name
        prior = mapping.get(png.name)
        if prior is not None and prior != cultivar:
            raise ValueError(
                f"frame {png.name!r} appears under both {prior!r} and {cultivar!r}; "
                "cultivar link is ambiguous."
            )
        mapping[png.name] = cultivar
    if not mapping:
        raise ValueError(f"no .png frames found under {split_views}")
    return mapping


def _source_frame(crop_name: str) -> str:
    """`<frame>_b<x>_<y>.png` → `<frame>.png` (the split-views source frame)."""
    return _CROP_SUFFIX.sub(r"\1", crop_name)


def link(dataset: Path, split_views: Path) -> None:
    """Rewrite `<dataset>/labels.csv` with a `Cultivar` column; fail loud on gaps."""
    frame_to_cultivar = _frame_to_cultivar(split_views)

    labels_path = dataset / "labels.csv"
    with open(labels_path, encoding="utf-8-sig", newline="") as f:
        has_sep = f.readline().lower().startswith("sep=")
        if not has_sep:
            f.seek(0)
        rows = list(csv.DictReader(f, delimiter=";"))

    unlinked: list[str] = []
    for row in rows:
        cultivar = frame_to_cultivar.get(_source_frame(row["Filename"].strip()))
        if cultivar is None:
            unlinked.append(row["Filename"])
        row["Cultivar"] = cultivar or ""
    if unlinked:
        raise ValueError(
            f"{len(unlinked)} crop(s) have no source frame under {split_views} — "
            f"wrong split-views dataset? e.g. {unlinked[:10]}"
        )

    with open(labels_path, "w", encoding="utf-8-sig", newline="") as f:
        if has_sep:
            f.write("sep=;\n")
        writer = csv.DictWriter(f, fieldnames=["Filename", "LabelIndex", "Cultivar"], delimiter=";")
        writer.writeheader()
        writer.writerows(rows)
    print(f"tagged {len(rows)} crop(s) across {len(set(frame_to_cultivar.values()))} "
          f"cultivar(s) → {labels_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", "-d", required=True, help="bud-instances folder (has labels.csv)")
    ap.add_argument("--split-views", "-s", required=True,
                    help="split-views (full-flower) folder; subfolders are cultivars")
    a = ap.parse_args()
    link(Path(a.dataset).resolve(), Path(a.split_views).resolve())
