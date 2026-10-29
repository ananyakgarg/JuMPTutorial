"""Train the detector head on cached encoder features.

Model selection uses the validation set, which (for MLAAD) contains only TTS
systems and human recordings that are absent from training, so the checkpoint is
chosen for how well it generalises rather than how well it memorises.

    python -m voice_detector.train --train train --val val
    python -m voice_detector.train --train train train_aug   # add augmented copies
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .extract import load_features
from .metrics import equal_error_rate, summarize
from .model import DetectorHead

CONFIG_KEYS = ("encoder", "encoder_layers", "clip_seconds")


def to_tensors(feats: np.ndarray, manifest) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.from_numpy(np.asarray(feats, dtype=np.float32)), torch.tensor(manifest.label.values, dtype=torch.float32)


@torch.inference_mode()
def predict(head: DetectorHead, x: torch.Tensor, batch_size: int = 1024) -> np.ndarray:
    head.eval()
    return torch.cat([torch.sigmoid(head(x[i:i + batch_size])) for i in range(0, len(x), batch_size)]).numpy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train", nargs="+", default=["train"], help="name(s) of cached training features")
    parser.add_argument("--val", default="val", help="name of cached validation features")
    parser.add_argument("--out", type=Path, default=Path("checkpoints/detector.pt"))
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    torch.manual_seed(args.seed)

    loaded = {name: load_features(name) for name in [*args.train, args.val]}
    configs = {name: {k: meta[k] for k in CONFIG_KEYS} for name, (_, _, meta) in loaded.items()}
    config = configs[args.val]
    if any(c != config for c in configs.values()):
        raise SystemExit(f"feature sets were extracted with different settings: {configs}")
    x_train, y_train = map(torch.cat, zip(*(to_tensors(*loaded[name][:2]) for name in args.train)))
    x_val, y_val = to_tensors(*loaded[args.val][:2])
    print(f"train: {len(x_train)} clips ({int(y_train.sum())} AI)   val: {len(x_val)} clips ({int(y_val.sum())} AI)")

    num_layers, dim = x_train.shape[1:]
    head_config = {"num_layers": num_layers, "dim": dim, "hidden": args.hidden, "dropout": args.dropout}
    head = DetectorHead(**head_config)
    head.feat_mean.copy_(x_train.mean(0))
    head.feat_std.copy_(x_train.std(0).clamp_min(1e-4))

    pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW([
        {"params": [head.layer_logits], "weight_decay": 0.0},
        {"params": head.mlp.parameters()},
    ], lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best, best_state, stale = (np.inf, np.inf), None, 0
    for epoch in range(1, args.epochs + 1):
        head.train()
        total = 0.0
        for idx in torch.randperm(len(x_train)).split(args.batch_size):
            loss = loss_fn(head(x_train[idx]), y_train[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item() * len(idx)
        scheduler.step()

        val_scores = predict(head, x_val)
        val_eer = equal_error_rate(y_val.numpy(), val_scores)[0]
        val_loss = nn.functional.binary_cross_entropy(torch.from_numpy(val_scores).clamp(1e-6, 1 - 1e-6), y_val).item()
        print(f"epoch {epoch:3d}  train_loss {total / len(x_train):.4f}  val_loss {val_loss:.4f}  val_EER {val_eer:.2%}")
        if (val_eer, val_loss) < best:
            best, best_state, stale = (val_eer, val_loss), {k: v.clone() for k, v in head.state_dict().items()}, 0
        else:
            stale += 1
            if stale >= args.patience:
                break

    head.load_state_dict(best_state)
    val_scores = predict(head, x_val)
    _, threshold = equal_error_rate(y_val.numpy(), val_scores)
    val_summary = summarize(y_val.numpy(), val_scores, threshold)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "config": {**config, "head": head_config},
        "head": head.state_dict(),
        "threshold": threshold,
        "val": val_summary,
        "train_features": args.train,
    }, args.out)

    weights = " ".join(f"{w:.2f}" for w in head.layer_weights().tolist())
    print(f"\nbest val EER {val_summary['eer']:.2%}  AUC {val_summary['auc']:.4f}  threshold {threshold:.3f}")
    print(f"layer weights: {weights}")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
