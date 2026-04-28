#!/usr/bin/env bash
# ============================================================
# run.sh — Generate ESM-2 embeddings for the ASTRAL SCOP 40%
#           dataset on an NVIDIA GPU (tested with RTX 5090).
#
# Usage:
#   chmod +x run.sh
#   ./run.sh
# ============================================================
set -euo pipefail

FASTA="astral-scopedom-seqres-gd-sel-gs-bib-40-2.03.fa.txt"
OUTPUT="embeddings_esm2_650M.pt"
MODEL="esm2_t33_650M_UR50D"   # 1280-dim, 650M params
BATCH=32                       # fits comfortably on an RTX 5090 (24 GB+)
MAX_LEN=1022                   # ESM-2 hard limit; 8 seqs will be truncated

echo "========================================================"
echo "  ESM-2 Embedding Generator"
echo "  Dataset : $FASTA"
echo "  Model   : $MODEL"
echo "  Output  : $OUTPUT"
echo "========================================================"

# ── Install dependencies (skip if already installed) ────────
echo ""
echo "[1/3] Installing Python dependencies …"
pip install --quiet fair-esm torch numpy tqdm

# ── Run embedding generation ─────────────────────────────────
echo ""
echo "[2/3] Generating embeddings …"
echo "      (First run downloads ~1.2 GB model weights once)"
echo ""
python generate_embeddings.py \
    --input    "$FASTA"   \
    --output   "$OUTPUT"  \
    --model    "$MODEL"   \
    --batch-size "$BATCH" \
    --max-len  "$MAX_LEN" \
    --per-sequence

# ── Quick verification ───────────────────────────────────────
echo ""
echo "[3/3] Verifying output …"
python - << PYEOF
import torch
emb = torch.load("$OUTPUT", map_location="cpu")
keys   = list(emb.keys())
sample = emb[keys[0]]
print(f"  Sequences embedded : {len(emb)}")
print(f"  Embedding shape    : {tuple(sample.shape)}")
print(f"  dtype              : {sample.dtype}")
print(f"  Sample ID          : {keys[0]}")
print(f"  Sample values      : {sample[:5].tolist()}")
PYEOF

echo ""
echo "✅  Done! Embeddings saved to: $OUTPUT"
echo ""
echo "  Load in Python:"
echo "    import torch"
echo "    embeddings = torch.load('$OUTPUT')"
echo "    # embeddings['d1dlwa_']  →  Tensor(1280,)"
