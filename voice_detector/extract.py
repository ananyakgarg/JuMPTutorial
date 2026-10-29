"""Run the frozen encoder over every clip in a manifest and cache the features.

The encoder is the slow part, so features are computed once and reused for every
training run. Writes features/<manifest>.npy (float16, [clips, layers, dim]) plus a
JSON sidecar recording how they were made. Work is split into chunks spread over
worker processes; interrupted runs resume from the last finished chunk.

    python -m voice_detector.extract data/manifests/train.csv data/manifests/val.csv
    python -m voice_detector.extract --augment data/manifests/train.csv   # -> features/train_aug
"""

import argparse
import json
import multiprocessing as mp
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoConfig

from .audio import SAMPLE_RATE, load_audio, preprocess
from .augment import degrade
from .model import SpeechEncoder

FEATURE_DIR = Path("features")
DEFAULT_ENCODER = "facebook/wav2vec2-xls-r-300m"
DEFAULT_LAYERS = 12
DEFAULT_CLIP_SECONDS = 4.0
CHUNK_SIZE = 50


def load_features(name: str) -> tuple[np.ndarray, pd.DataFrame, dict]:
    """Return (features, manifest, extraction config) for a cached manifest."""
    meta = json.loads((FEATURE_DIR / f"{name}.json").read_text())
    if len(meta["done_chunks"]) < num_chunks(meta["rows"]):
        raise RuntimeError(f"features/{name} is incomplete; rerun extract")
    feats = np.load(FEATURE_DIR / f"{name}.npy", mmap_mode="r")
    return feats, pd.read_csv(meta["manifest"]), meta


def num_chunks(rows: int) -> int:
    return -(-rows // CHUNK_SIZE)


def output_shape(config: dict) -> tuple[int, int]:
    model_config = AutoConfig.from_pretrained(config["encoder"])
    layers = config["encoder_layers"] or model_config.num_hidden_layers
    return layers + 1, 2 * model_config.hidden_size


_encoder: SpeechEncoder | None = None


def _init_worker(config: dict, threads: int) -> None:
    global _encoder
    torch.set_num_threads(threads)
    _encoder = SpeechEncoder(config["encoder"], config["encoder_layers"])


def _encode_chunk(job: tuple[int, list[str], dict]) -> tuple[int, np.ndarray]:
    chunk, paths, config = job
    max_samples = int(config["clip_seconds"] * SAMPLE_RATE)
    feats = []
    for i, path in enumerate(paths):
        wav = preprocess(load_audio(path))[:max_samples]
        if config["augment"]:
            wav = degrade(wav, np.random.default_rng([chunk, i]))
        feats.append(_encoder(wav).numpy())
    return chunk, np.stack(feats).astype(np.float16)


def extract(manifest: Path, pool: mp.Pool, config: dict) -> None:
    name = manifest.stem + ("_aug" if config["augment"] else "")
    df = pd.read_csv(manifest)
    npy, sidecar = FEATURE_DIR / f"{name}.npy", FEATURE_DIR / f"{name}.json"
    meta = {**config, "manifest": str(manifest), "rows": len(df), "done_chunks": []}

    if sidecar.exists() and npy.exists():
        previous = json.loads(sidecar.read_text())
        if all(previous.get(k) == v for k, v in meta.items() if k != "done_chunks"):
            meta["done_chunks"] = previous["done_chunks"]
    done = set(meta["done_chunks"])
    todo = [c for c in range(num_chunks(len(df))) if c not in done]
    if not todo:
        print(f"{name}: already extracted")
        return
    mode = "r+" if meta["done_chunks"] else "w+"
    feats = np.lib.format.open_memmap(npy, mode=mode, dtype=np.float16, shape=(len(df), *output_shape(config)))

    jobs = [(c, df.path[c * CHUNK_SIZE:(c + 1) * CHUNK_SIZE].tolist(), config) for c in todo]
    start = time.time()
    with tqdm(total=len(df), initial=len(df) - sum(len(j[1]) for j in jobs), desc=name, unit="clip") as bar:
        for chunk, chunk_feats in pool.imap_unordered(_encode_chunk, jobs):
            feats[chunk * CHUNK_SIZE:chunk * CHUNK_SIZE + len(chunk_feats)] = chunk_feats
            feats.flush()
            meta["done_chunks"].append(chunk)
            sidecar.write_text(json.dumps(meta, indent=2))
            bar.update(len(chunk_feats))
    print(f"{name}: {sum(len(j[1]) for j in jobs)} clips in {time.time() - start:.0f}s")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("manifests", nargs="+", type=Path)
    parser.add_argument("--encoder", default=DEFAULT_ENCODER, help="Hugging Face speech model id")
    parser.add_argument("--encoder-layers", type=int, default=DEFAULT_LAYERS,
                        help="keep only the first N transformer layers (0 = all)")
    parser.add_argument("--clip-seconds", type=float, default=DEFAULT_CLIP_SECONDS,
                        help="use at most this much audio per clip (after trimming silence)")
    parser.add_argument("--augment", action="store_true",
                        help="degrade each clip first (codec, noise, phone band, reverb); saves as <name>_aug")
    parser.add_argument("--workers", type=int, default=os.cpu_count(),
                        help="encoder processes; on CPU several single-threaded workers beat one multi-threaded one")
    args = parser.parse_args()

    FEATURE_DIR.mkdir(exist_ok=True)
    config = {"encoder": args.encoder, "encoder_layers": args.encoder_layers or None,
              "clip_seconds": args.clip_seconds, "augment": args.augment}
    threads = max(1, os.cpu_count() // args.workers)
    with mp.get_context("spawn").Pool(args.workers, _init_worker, (config, threads)) as pool:
        for manifest in args.manifests:
            extract(manifest, pool, config)


if __name__ == "__main__":
    main()
