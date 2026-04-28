# seqBased_Embeddings

Extract **sequence-based protein embeddings** from the [ESM-2](https://github.com/facebookresearch/esm) protein language model (Meta AI Research) and save them as NumPy arrays for downstream machine-learning tasks.

---

## Table of Contents

1. [Requirements](#requirements)  
2. [Installation](#installation)  
3. [Prepare your dataset](#prepare-your-dataset)  
4. [Extract embeddings](#extract-embeddings)  
5. [Output files](#output-files)  
6. [Using the embeddings in Python](#using-the-embeddings-in-python)  
7. [Model variants](#model-variants)  
8. [Tips](#tips)  

---

## Requirements

| Dependency | Minimum version |
|------------|----------------|
| Python     | 3.8             |
| PyTorch    | 1.12.0          |
| fair-esm   | 2.0.0           |
| NumPy      | 1.22.0          |
| tqdm       | 4.64.0          |

A CUDA-capable GPU is **optional** — CPU inference works but is slower for large datasets or bigger models.

---

## Installation

```bash
# 1. Clone this repository
git clone https://github.com/shikharrana/seqBased_Embeddings-.git
cd seqBased_Embeddings-

# 2. (Recommended) Create and activate a virtual environment
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

# 3. Install all dependencies
pip install -r requirements.txt
```

> **Note:** On first run the selected ESM-2 weights are downloaded automatically  
> from the [ESM model hub](https://dl.fbaipublicfiles.com/fair-esm/models/)  
> and cached in `~/.cache/torch/hub/checkpoints/`.

---

## Prepare your dataset

Your protein sequences must be in **FASTA format** (`.fasta` or `.fa`):

```
>ProteinA some description
MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAP
>ProteinB
MVHLTPEEKSAVTALWGKVNVDEVGGEALGRLL
```

A sample file is already included:

```
example_sequences.fasta   # 5 human proteins (EGFR, p53, HBB, Insulin, SOD1)
```

---

## Extract embeddings

```bash
python extract_embeddings.py --fasta example_sequences.fasta
```

### All options

```
usage: extract_embeddings.py [-h]
                             --fasta FASTA
                             [--model MODEL]
                             [--batch_size BATCH_SIZE]
                             [--repr_layer REPR_LAYER]
                             [--max_seq_len MAX_SEQ_LEN]
                             [--device DEVICE]
                             [--output_dir OUTPUT_DIR]
                             [--save_csv]

optional arguments:
  --fasta          Path to the input FASTA file (required)
  --model          ESM-2 model variant (default: esm2_t33_650M_UR50D)
  --batch_size     Sequences per batch (default: 4)
  --repr_layer     Layer to extract representations from; -1 = last (default: -1)
  --max_seq_len    Truncate sequences longer than this (default: 1022)
  --device         'auto', 'cpu', 'cuda', or 'cuda:N' (default: auto)
  --output_dir     Directory for output files (default: embeddings/)
  --save_csv       Also write embeddings.csv (large for big datasets)
```

### Example with a custom model and GPU

```bash
python extract_embeddings.py \
    --fasta my_proteins.fasta \
    --model esm2_t12_35M_UR50D \
    --batch_size 8 \
    --device cuda \
    --output_dir results/
```

---

## Output files

After a successful run, the `--output_dir` directory contains:

| File             | Description                                                   |
|------------------|---------------------------------------------------------------|
| `embeddings.npy` | NumPy array of shape `(N, D)` — N sequences, D embedding dim |
| `headers.txt`    | One FASTA header per line, matching the row order in the array|
| `embeddings.csv` | *(optional)* CSV with a `header` column + one column per dim  |

---

## Using the embeddings in Python

```python
import numpy as np

embeddings = np.load("embeddings/embeddings.npy")   # shape (N, D)

with open("embeddings/headers.txt") as fh:
    headers = [line.strip() for line in fh]

print(f"Loaded {len(headers)} embeddings of dimension {embeddings.shape[1]}")

# Example: use in scikit-learn
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
reduced = pca.fit_transform(embeddings)
```

---

## Model variants

| Model name              | Params | Embedding dim | Notes                         |
|-------------------------|--------|---------------|-------------------------------|
| `esm2_t6_8M_UR50D`      |  8 M   | 320           | Fastest; good for prototyping |
| `esm2_t12_35M_UR50D`    | 35 M   | 480           |                               |
| `esm2_t30_150M_UR50D`   | 150 M  | 640           |                               |
| `esm2_t33_650M_UR50D`   | 650 M  | 1280          | **Default — best quality/speed balance** |
| `esm2_t36_3B_UR50D`     |  3 B   | 2560          | Requires ≥16 GB RAM/VRAM     |
| `esm2_t48_15B_UR50D`    | 15 B   | 5120          | Requires high-end GPU         |

---

## Tips

* **Memory issues?** Reduce `--batch_size` to 1 or switch to a smaller model.  
* **Speed?** Use `--device cuda` and increase `--batch_size` if your GPU has enough VRAM.  
* **Long sequences?** ESM-2 supports up to 1022 residues; longer sequences are truncated automatically.  
* **Reproducibility:** Model weights are deterministic — the same sequence always yields the same embedding.

---

## Running tests

```bash
pip install pytest
pytest test_extract_embeddings.py -v
```
