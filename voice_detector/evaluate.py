"""Evaluate a trained checkpoint on cached test sets.

Reports threshold-free metrics (EER, AUC) and accuracy at the threshold chosen on
the validation set, plus how often each TTS system / human group is classified
correctly. Writes results/<checkpoint>.json and results/<checkpoint>.md.

    python -m voice_detector.evaluate test_mlaad_en test_mlaad_de test_in_the_wild
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .extract import load_features
from .metrics import summarize
from .model import DetectorHead
from .train import CONFIG_KEYS, predict


def fmt(value, pct=True) -> str:
    if value is None:
        return "–"
    return f"{value:.1%}" if pct else f"{value:.3f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("names", nargs="+", help="names of cached feature sets")
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/detector.pt"))
    parser.add_argument("--out-dir", type=Path, default=Path("results"))
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    head = DetectorHead(**ckpt["config"]["head"])
    head.load_state_dict(ckpt["head"])
    threshold = ckpt["threshold"]

    results = {"checkpoint": str(args.checkpoint), "threshold": threshold, "val": ckpt["val"], "sets": {}}
    for name in args.names:
        feats, df, meta = load_features(name)
        if any(meta[k] != ckpt["config"][k] for k in CONFIG_KEYS):
            raise SystemExit(f"{name} was extracted with different encoder settings than the checkpoint")
        scores = predict(head, torch.from_numpy(np.asarray(feats, dtype=np.float32)))
        df = df.assign(score=scores, correct=(scores >= threshold) == df.label.astype(bool))
        per_group = (df.groupby(["label", "group"]).correct.agg(["mean", "size"])
                     .reset_index().rename(columns={"mean": "accuracy", "size": "n"}))
        results["sets"][name] = {
            **summarize(df.label.values, scores, threshold),
            "groups": per_group.to_dict(orient="records"),
        }

    args.out_dir.mkdir(exist_ok=True)
    stem = args.out_dir / args.checkpoint.stem
    stem.with_suffix(".json").write_text(json.dumps(results, indent=2))

    lines = [f"Threshold {threshold:.3f} (equal-error point on validation set)\n",
             "| set | clips (human / AI) | EER | AUC | accuracy | human correct | AI caught |",
             "|---|---|---|---|---|---|---|"]
    for name, r in {"val": ckpt["val"], **results["sets"]}.items():
        lines.append(f"| {name} | {r['n_human']} / {r['n_ai']} | {fmt(r.get('eer'))} | {fmt(r.get('auc'), False)} "
                     f"| {fmt(r['accuracy'])} | {fmt(r['human_recall'])} | {fmt(r['ai_recall'])} |")
    for name, r in results["sets"].items():
        groups = [g for g in r["groups"] if g["label"] == 1]
        if len(groups) > 1:
            lines += [f"\n**{name}: share of each AI system's clips caught**\n", "| system | clips | caught |", "|---|---|---|"]
            lines += [f"| {g['group']} | {g['n']} | {g['accuracy']:.0%} |"
                      for g in sorted(groups, key=lambda g: g["accuracy"])]
    stem.with_suffix(".md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
