"""Build manifest CSVs (path, label, group, dataset) from the supported data sources.

Label 1 = AI-generated, 0 = human. `group` is the TTS system for AI clips and the
book/speaker for human clips; it is used to split data without leakage and to
break down results.

    python -m voice_detector.data mlaad data/MLAAD-tiny
    python -m voice_detector.data parquet in_the_wild "data/hf/InTheWild/data/test-*.parquet"
    python -m voice_detector.data folder my_data path/to/dir_with_real_and_fake_subfolders
"""

import argparse
import glob
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import soundfile as sf
from tqdm import tqdm

from .audio import load_audio

AUDIO_EXTS = {".wav", ".flac", ".mp3", ".ogg", ".m4a", ".opus"}
MANIFEST_DIR = Path("data/manifests")


def write_manifest(df: pd.DataFrame, name: str, out_dir: Path = MANIFEST_DIR) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    df = df.reset_index(drop=True)
    df.to_csv(out_dir / f"{name}.csv", index=False)
    counts = df.label.value_counts().reindex([0, 1], fill_value=0)
    print(f"{name:>16}: {len(df):6d} clips  human={counts[0]:5d}  ai={counts[1]:5d}  groups={df.group.nunique()}")


def split_groups(groups: pd.Series, fractions: dict[str, float], seed: int) -> pd.Series:
    """Assign whole groups to splits so that each split gets roughly `fractions` of the rows."""
    sizes = groups.value_counts().sample(frac=1, random_state=seed)
    bounds = np.cumsum(list(fractions.values())) * len(groups)
    assignment, filled = {}, 0
    for group, size in sizes.items():
        split = next(n for n, b in zip(fractions, bounds) if filled < b)
        assignment[group] = split
        filled += size
    return groups.map(assignment)


def mlaad(root: Path, seed: int) -> None:
    """MLAAD-tiny: English is split into train/val/test, German is a cross-language test set.

    AI clips are split by TTS system and human clips by book, so the val and test
    sets contain only voice generators and recordings the model never trained on.
    """
    rows = []
    for path in sorted(root.glob("*/*/**/*.wav")):
        kind, lang = path.relative_to(root).parts[:2]
        if kind == "fake":
            group = path.relative_to(root).parts[2]
        else:
            group = re.sub(r"_\d+_f\d+$", "", path.stem)  # book title
        rows.append({"path": str(path), "label": int(kind == "fake"), "group": group, "dataset": f"mlaad-{lang}"})
    df = pd.DataFrame(rows)

    en = df[df.dataset == "mlaad-en"].copy()
    fractions = {"train": 0.7, "val": 0.1, "test": 0.2}
    en["split"] = ""
    for label in (0, 1):
        mask = en.label == label
        en.loc[mask, "split"] = split_groups(en.loc[mask, "group"], fractions, seed)
    for split, name in {"train": "train", "val": "val", "test": "test_mlaad_en"}.items():
        write_manifest(en[en.split == split].drop(columns="split"), name)
    write_manifest(df[df.dataset == "mlaad-de"], "test_mlaad_de")


def parquet(name: str, pattern: str, per_class: int, seed: int) -> None:
    """Export a balanced sample from a SpeechAntiSpoofingBenchmarks-style parquet dataset.

    Expected columns: path, audio {bytes, ...}, label (0 = bonafide, 1 = spoof), notes (JSON).
    """
    files = sorted(glob.glob(pattern))
    meta = pd.concat([pq.read_table(f, columns=["path", "label", "notes"]).to_pandas().assign(file=f) for f in files])
    meta["row"] = meta.groupby("file").cumcount()
    sample = pd.concat(g.sample(min(per_class, len(g)), random_state=seed) for _, g in meta.groupby("label"))
    out_dir = Path("data/benchmarks") / name
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for file, picked in sample.groupby("file"):
        table = pq.read_table(file, columns=["audio"]).column("audio")
        for r in tqdm(picked.itertuples(), total=len(picked), desc=Path(file).name):
            out = out_dir / f"{Path(r.path).stem}.flac"
            if not out.exists():
                sf.write(out, load_audio(table[r.row].as_py()["bytes"]), 16_000)
            notes = json.loads(r.notes) if r.notes else {}
            group = notes.get("speaker") or notes.get("speaker_id") or ""
            rows.append({"path": str(out), "label": int(r.label), "group": group, "dataset": name})
    write_manifest(pd.DataFrame(rows).sort_values("path"), f"test_{name}")


def folder(name: str, root: Path) -> None:
    """Any directory with `real/` (or `human/`) and `fake/` (or `ai/`) subfolders, searched recursively."""
    rows = []
    for sub, label in [("real", 0), ("human", 0), ("fake", 1), ("ai", 1)]:
        for path in sorted((root / sub).rglob("*")):
            if path.suffix.lower() in AUDIO_EXTS:
                group = path.parent.name if path.parent != root / sub else sub
                rows.append({"path": str(path), "label": label, "group": group, "dataset": name})
    if not rows:
        raise SystemExit(f"No audio found under {root}/real, {root}/human, {root}/fake or {root}/ai")
    write_manifest(pd.DataFrame(rows), name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=0)
    sub = parser.add_subparsers(dest="source", required=True)
    p = sub.add_parser("mlaad")
    p.add_argument("root", type=Path)
    p = sub.add_parser("parquet")
    p.add_argument("name")
    p.add_argument("pattern")
    p.add_argument("--per-class", type=int, default=1000)
    p = sub.add_parser("folder")
    p.add_argument("name")
    p.add_argument("root", type=Path)
    args = parser.parse_args()

    if args.source == "mlaad":
        mlaad(args.root, args.seed)
    elif args.source == "parquet":
        parquet(args.name, args.pattern, args.per_class, args.seed)
    else:
        folder(args.name, args.root)


if __name__ == "__main__":
    main()
