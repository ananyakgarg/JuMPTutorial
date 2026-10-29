"""Detector = frozen self-supervised speech encoder + small trainable head."""

import numpy as np
import torch
from torch import nn
from transformers import AutoFeatureExtractor, AutoModel
from transformers.utils import logging as hf_logging

from .audio import SAMPLE_RATE, load_audio, preprocess, windows

hf_logging.set_verbosity_error()


class SpeechEncoder(nn.Module):
    """Frozen pretrained speech model that summarises a clip as per-layer statistics.

    Returns a [num_layers + 1, 2 * hidden_size] tensor: the mean and standard
    deviation over time of every hidden layer. The lower transformer layers carry
    most of the fine acoustic detail that gives synthetic speech away, so the
    deeper ones can be dropped with `num_layers` to save compute.
    """

    def __init__(self, name: str, num_layers: int | None = None):
        super().__init__()
        self.name = name
        self.processor = AutoFeatureExtractor.from_pretrained(name)
        self.model = AutoModel.from_pretrained(name)
        if num_layers is not None:
            self.model.encoder.layers = self.model.encoder.layers[:num_layers]
            self.model.config.num_hidden_layers = num_layers
        self.model.eval().requires_grad_(False)

    @property
    def output_shape(self) -> tuple[int, int]:
        return len(self.model.encoder.layers) + 1, 2 * self.model.config.hidden_size

    @torch.inference_mode()
    def forward(self, wav: np.ndarray) -> torch.Tensor:
        inputs = self.processor(wav, sampling_rate=SAMPLE_RATE, return_tensors="pt")
        hidden = self.model(inputs.input_values, output_hidden_states=True).hidden_states
        hidden = torch.stack(hidden).squeeze(1)  # [layers, frames, dim]
        return torch.cat([hidden.mean(1), hidden.std(1)], dim=-1)


class DetectorHead(nn.Module):
    """Learned softmax mix over encoder layers, then a small MLP.

    Outputs one logit per clip; positive means AI-generated.
    """

    def __init__(self, num_layers: int, dim: int, hidden: int = 256, dropout: float = 0.3):
        super().__init__()
        # Per-(layer, dim) standardisation, fitted on the training set.
        self.register_buffer("feat_mean", torch.zeros(num_layers, dim))
        self.register_buffer("feat_std", torch.ones(num_layers, dim))
        self.layer_logits = nn.Parameter(torch.zeros(num_layers))
        self.mlp = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def layer_weights(self) -> torch.Tensor:
        return self.layer_logits.softmax(0)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        feats = (feats - self.feat_mean) / self.feat_std
        mixed = torch.einsum("bld,l->bd", feats, self.layer_weights())
        return self.mlp(mixed).squeeze(-1)


class Detector:
    """A trained checkpoint ready to score audio.

    Scores are the probability that the voice is AI-generated. Long recordings are
    split into windows of the training clip length and the window scores averaged.
    """

    def __init__(self, checkpoint: str):
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)
        self.config = ckpt["config"]
        self.threshold = ckpt["threshold"]
        self.encoder = SpeechEncoder(self.config["encoder"], self.config["encoder_layers"])
        self.head = DetectorHead(**self.config["head"])
        self.head.load_state_dict(ckpt["head"])
        self.head.eval()

    @torch.inference_mode()
    def score(self, wav: np.ndarray) -> tuple[float, list[float]]:
        """Return (overall score, per-window scores) for a 16 kHz mono waveform."""
        chunks = windows(preprocess(wav), self.config["clip_seconds"])
        feats = torch.stack([self.encoder(chunk) for chunk in chunks])
        probs = torch.sigmoid(self.head(feats)).tolist()
        return float(np.mean(probs)), probs

    def score_file(self, path: str) -> tuple[float, list[float]]:
        return self.score(load_audio(path))
