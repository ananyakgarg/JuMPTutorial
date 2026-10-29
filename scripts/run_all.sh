#!/usr/bin/env bash
# Download data, extract features, train and evaluate, end to end.
# Takes ~2 hours on a 4-core CPU, most of it in feature extraction.
set -euo pipefail
cd "$(dirname "$0")/.."

# Training data: MLAAD-tiny (~4 GB). Real speech from M-AILABS audiobooks, AI speech from 79 TTS systems.
hf download mueller91/MLAAD-tiny --repo-type dataset --local-dir data/MLAAD-tiny

# Out-of-domain test sets (a slice of each is enough for a reliable estimate).
hf download SpeechAntiSpoofingBenchmarks/InTheWild --repo-type dataset --local-dir data/hf/InTheWild \
    --include "data/test-00000-of-00008.parquet" "data/test-00001-of-00008.parquet"
hf download SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA --repo-type dataset --local-dir data/hf/ASVspoof2019_LA \
    --include "data/test-00000-of-00009.parquet"

python -m voice_detector.data mlaad data/MLAAD-tiny
python -m voice_detector.data parquet in_the_wild "data/hf/InTheWild/data/test-*.parquet" --per-class 600
python -m voice_detector.data parquet asvspoof2019 "data/hf/ASVspoof2019_LA/data/test-*.parquet" --per-class 600

python -m voice_detector.extract data/manifests/{train,val,test_mlaad_en,test_mlaad_de,test_in_the_wild,test_asvspoof2019}.csv
python -m voice_detector.train
python -m voice_detector.evaluate test_mlaad_en test_mlaad_de test_in_the_wild test_asvspoof2019
