# seqBased_Embeddings

Generate sequence-based protein embeddings using **ESM-2** (Meta AI's protein language model).

---

## Dataset

This project uses the **ASTRAL SCOP FASTA** dataset:

```
astral-scopedom-seqres-gd-sel-gs-bib-40-2.03.fa.txt
```

Located at: `/home/shikhar-rana/Downloads/dataset/`

---

## Installation

```bash
pip install -r requirements.txt
```

> **GPU note:** If you have a CUDA-capable GPU, install the matching `torch` version from https://pytorch.org/get-started/locally/ for faster inference.

---

## Usage

```bash
python generate_embeddings.py \
    --input /home/shikhar-rana/Downloads/dataset/astral-scopedom-seqres-gd-sel-gs-bib-40-2.03.fa.txt \
    --output embeddings.pt
```

### All options

| Flag | Default | Description |
|------|---------|-------------|
| `--input` / `-i` | *(required)* | Path to input FASTA file |
| `--output` / `-o` | `embeddings.pt` | Output file (PyTorch `.pt` dict) |
| `--model` / `-m` | `esm2_t33_650M_UR50D` | ESM-2 model variant (see below) |
| `--batch-size` / `-b` | `8` | Sequences per forward pass (reduce if OOM) |
| `--repr-layer` | `-1` | Transformer layer to extract (`-1` = last layer) |
| `--per-sequence` | off | Mean-pool residues to one vector per sequence |
| `--max-len` | `1022` | Truncate sequences longer than this |
| `--device` | auto | `cpu`, `cuda`, `cuda:0`, etc. |

### Examples

**Per-sequence embeddings (one vector per protein):**
```bash
python generate_embeddings.py \
    --input /home/shikhar-rana/Downloads/dataset/astral-scopedom-seqres-gd-sel-gs-bib-40-2.03.fa.txt \
    --output embeddings_per_seq.pt \
    --per-sequence
```

**Smaller/faster model (8 M params, good for CPU):**
```bash
python generate_embeddings.py \
    --input /home/shikhar-rana/Downloads/dataset/astral-scopedom-seqres-gd-sel-gs-bib-40-2.03.fa.txt \
    --output embeddings.pt \
    --model esm2_t6_8M_UR50D \
    --per-sequence
```

---

## Available ESM-2 Models

| Model name | Parameters | Embedding dim | Notes |
|---|---|---|---|
| `esm2_t6_8M_UR50D` | 8 M | 320 | Fastest, CPU-friendly |
| `esm2_t12_35M_UR50D` | 35 M | 480 | Small, CPU-friendly |
| `esm2_t30_150M_UR50D` | 150 M | 640 | Medium |
| `esm2_t33_650M_UR50D` | 650 M | 1280 | **Recommended** |
| `esm2_t36_3B_UR50D` | 3 B | 2560 | GPU recommended |
| `esm2_t48_15B_UR50D` | 15 B | 5120 | Needs multi-GPU |

---

## Loading the Embeddings

```python
import torch

embeddings = torch.load("embeddings.pt")

# embeddings is a dict: { sequence_id (str) -> torch.Tensor }
# Per-residue:  shape (seq_len, embed_dim)
# Per-sequence: shape (embed_dim,)

for seq_id, emb in embeddings.items():
    print(seq_id, emb.shape)
```

---

## References

- ESM-2 paper: [Lin et al., 2023](https://www.science.org/doi/10.1126/science.ade2574)
- [fair-esm GitHub](https://github.com/facebookresearch/esm)
- [ASTRAL SCOP database](https://scop.berkeley.edu/)
