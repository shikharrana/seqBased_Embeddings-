"""
extract_embeddings.py
---------------------
Extract sequence-level embeddings from protein sequences using the ESM-2
language model (Meta / Facebook Research).

Supported ESM-2 model names (choose based on available RAM/VRAM):
  esm2_t6_8M_UR50D      –  8 M params, embedding dim 320  (fastest / smallest)
  esm2_t12_35M_UR50D    – 35 M params, embedding dim 480
  esm2_t30_150M_UR50D   – 150 M params, embedding dim 640
  esm2_t33_650M_UR50D   – 650 M params, embedding dim 1280  (recommended)
  esm2_t36_3B_UR50D     –  3 B params, embedding dim 2560
  esm2_t48_15B_UR50D    – 15 B params, embedding dim 5120  (requires high-end GPU)

Usage:
  python extract_embeddings.py --fasta example_sequences.fasta

Run `python extract_embeddings.py --help` for full option list.
"""

import argparse
import os
import sys

import numpy as np
from tqdm import tqdm

# ---------------------------------------------------------------------------
# ESM-2 model identifiers and their number of transformer layers
# ---------------------------------------------------------------------------
ESM2_MODELS = {
    "esm2_t6_8M_UR50D": 6,
    "esm2_t12_35M_UR50D": 12,
    "esm2_t30_150M_UR50D": 30,
    "esm2_t33_650M_UR50D": 33,
    "esm2_t36_3B_UR50D": 36,
    "esm2_t48_15B_UR50D": 48,
}


# ---------------------------------------------------------------------------
# FASTA parser (no external dependency)
# ---------------------------------------------------------------------------
def parse_fasta(fasta_path: str):
    """Yield (header, sequence) tuples from a FASTA file."""
    header, seq_parts = None, []
    with open(fasta_path, "r") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(seq_parts)
                header = line[1:]  # drop the leading '>'
                seq_parts = []
            else:
                seq_parts.append(line)
    if header is not None:
        yield header, "".join(seq_parts)


# ---------------------------------------------------------------------------
# Embedding extraction
# ---------------------------------------------------------------------------
def load_model(model_name: str, device):
    """Download (if needed) and load an ESM-2 model + its batch converter."""
    try:
        import esm as esm_lib
    except ImportError:
        sys.exit(
            "ERROR: 'fair-esm' is not installed.\n"
            "Install it with:  pip install fair-esm"
        )

    loader = getattr(esm_lib.pretrained, model_name, None)
    if loader is None:
        sys.exit(
            f"ERROR: Unknown model '{model_name}'.\n"
            f"Choose from: {', '.join(ESM2_MODELS)}"
        )

    print(f"Loading model '{model_name}' …")
    model, alphabet = loader()
    model = model.to(device)
    model.eval()
    batch_converter = alphabet.get_batch_converter()
    return model, alphabet, batch_converter


def extract_embeddings(
    fasta_path: str,
    model_name: str = "esm2_t33_650M_UR50D",
    batch_size: int = 4,
    repr_layer: int = -1,          # -1  → last transformer layer
    max_seq_len: int = 1022,       # ESM-2 positional limit (1024 - 2 special tokens)
    device_str: str = "auto",
    output_dir: str = "embeddings",
    save_csv: bool = False,
):
    # ------------------------------------------------------------------
    # Device selection
    # ------------------------------------------------------------------
    try:
        import torch
    except ImportError:
        sys.exit(
            "ERROR: 'torch' is not installed.\n"
            "Install it with:  pip install torch"
        )

    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)
    print(f"Using device: {device}")

    # ------------------------------------------------------------------
    # Resolve repr_layer index
    # ------------------------------------------------------------------
    num_layers = ESM2_MODELS[model_name]
    if repr_layer == -1:
        repr_layer = num_layers
    if not (1 <= repr_layer <= num_layers):
        sys.exit(
            f"ERROR: repr_layer must be between 1 and {num_layers} "
            f"for model '{model_name}'."
        )

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    model, alphabet, batch_converter = load_model(model_name, device)

    # ------------------------------------------------------------------
    # Read & validate sequences
    # ------------------------------------------------------------------
    sequences = list(parse_fasta(fasta_path))
    if not sequences:
        sys.exit(f"ERROR: No sequences found in '{fasta_path}'.")

    # Truncate sequences that exceed ESM-2's positional embedding limit
    truncated = 0
    clean_sequences = []
    for header, seq in sequences:
        if len(seq) > max_seq_len:
            seq = seq[:max_seq_len]
            truncated += 1
        clean_sequences.append((header, seq))

    if truncated:
        print(
            f"Warning: {truncated} sequence(s) were truncated to {max_seq_len} residues."
        )

    print(f"Processing {len(clean_sequences)} sequence(s) in batches of {batch_size} …")

    # ------------------------------------------------------------------
    # Create output directory
    # ------------------------------------------------------------------
    os.makedirs(output_dir, exist_ok=True)

    all_headers = []
    all_embeddings = []

    # ------------------------------------------------------------------
    # Batch processing
    # ------------------------------------------------------------------
    for batch_start in tqdm(range(0, len(clean_sequences), batch_size)):
        batch = clean_sequences[batch_start : batch_start + batch_size]
        batch_labels, batch_strs, batch_tokens = batch_converter(batch)
        batch_tokens = batch_tokens.to(device)

        with torch.no_grad():
            results = model(
                batch_tokens,
                repr_layers=[repr_layer],
                return_contacts=False,
            )

        token_reps = results["representations"][repr_layer]  # (B, L+2, D)

        for i, (header, seq) in enumerate(batch):
            # Slice off <cls> (index 0) and <eos> (last) tokens; mean-pool
            seq_len = len(seq)
            embedding = token_reps[i, 1 : seq_len + 1].mean(0).cpu().numpy()
            all_headers.append(header)
            all_embeddings.append(embedding)

    # ------------------------------------------------------------------
    # Save embeddings
    # ------------------------------------------------------------------
    embeddings_array = np.stack(all_embeddings, axis=0)  # (N, D)

    npy_path = os.path.join(output_dir, "embeddings.npy")
    np.save(npy_path, embeddings_array)
    print(f"Saved embeddings → {npy_path}  shape: {embeddings_array.shape}")

    headers_path = os.path.join(output_dir, "headers.txt")
    with open(headers_path, "w") as fh:
        fh.write("\n".join(all_headers) + "\n")
    print(f"Saved sequence headers → {headers_path}")

    if save_csv:
        csv_path = os.path.join(output_dir, "embeddings.csv")
        import csv

        with open(csv_path, "w", newline="") as fh:
            writer = csv.writer(fh)
            dim = embeddings_array.shape[1]
            writer.writerow(["header"] + [f"dim_{j}" for j in range(dim)])
            for header, emb in zip(all_headers, embeddings_array):
                writer.writerow([header] + emb.tolist())
        print(f"Saved embeddings as CSV → {csv_path}")

    return embeddings_array, all_headers


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract ESM-2 sequence-based protein embeddings.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--fasta",
        required=True,
        help="Path to the input FASTA file containing protein sequences.",
    )
    parser.add_argument(
        "--model",
        default="esm2_t33_650M_UR50D",
        choices=list(ESM2_MODELS.keys()),
        help="ESM-2 model variant to use.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Number of sequences processed per batch.",
    )
    parser.add_argument(
        "--repr_layer",
        type=int,
        default=-1,
        help=(
            "Transformer layer whose representations are extracted. "
            "-1 selects the last layer."
        ),
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=1022,
        help="Maximum sequence length (longer sequences are truncated).",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Compute device: 'auto', 'cpu', 'cuda', or 'cuda:N'.",
    )
    parser.add_argument(
        "--output_dir",
        default="embeddings",
        help="Directory where output files are saved.",
    )
    parser.add_argument(
        "--save_csv",
        action="store_true",
        help="Also save embeddings as a CSV file (large for big datasets).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    extract_embeddings(
        fasta_path=args.fasta,
        model_name=args.model,
        batch_size=args.batch_size,
        repr_layer=args.repr_layer,
        max_seq_len=args.max_seq_len,
        device_str=args.device,
        output_dir=args.output_dir,
        save_csv=args.save_csv,
    )
