"""
Generate sequence-based embeddings using ESM-2 from a FASTA file.

Usage:
    python generate_embeddings.py --input <path_to_fasta> --output <output.pt> \
        [--model esm2_t33_650M_UR50D] [--batch-size 32] [--repr-layer -1] \
        [--per-sequence] [--device cpu]

Output:
    A PyTorch .pt file containing a dict:
        {
            "<sequence_id>": torch.Tensor  # shape: (seq_len, embed_dim) or (embed_dim,)
        }
"""

import argparse
import torch
import esm
from pathlib import Path
from tqdm import tqdm


# ---------------------------------------------------------------------------
# FASTA parsing
# ---------------------------------------------------------------------------

def read_fasta(fasta_path: str) -> list[tuple[str, str]]:
    """Parse a FASTA file and return a list of (label, sequence) tuples."""
    sequences = []
    label = None
    seq_parts = []

    with open(fasta_path, "r") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if label is not None:
                    sequences.append((label, "".join(seq_parts)))
                label = line[1:].split()[0]  # use first token of header as ID
                seq_parts = []
            else:
                seq_parts.append(line.upper())
        if label is not None:
            sequences.append((label, "".join(seq_parts)))

    return sequences


# ---------------------------------------------------------------------------
# ESM-2 model loading
# ---------------------------------------------------------------------------

AVAILABLE_MODELS = {
    "esm2_t6_8M_UR50D": esm.pretrained.esm2_t6_8M_UR50D,
    "esm2_t12_35M_UR50D": esm.pretrained.esm2_t12_35M_UR50D,
    "esm2_t30_150M_UR50D": esm.pretrained.esm2_t30_150M_UR50D,
    "esm2_t33_650M_UR50D": esm.pretrained.esm2_t33_650M_UR50D,
    "esm2_t36_3B_UR50D": esm.pretrained.esm2_t36_3B_UR50D,
    "esm2_t48_15B_UR50D": esm.pretrained.esm2_t48_15B_UR50D,
}


def load_model(model_name: str, device: torch.device):
    if model_name not in AVAILABLE_MODELS:
        raise ValueError(
            f"Unknown model '{model_name}'. Choose from: {list(AVAILABLE_MODELS)}"
        )
    print(f"Loading ESM-2 model: {model_name} ...")
    model, alphabet = AVAILABLE_MODELS[model_name]()
    model = model.to(device)
    model.eval()
    return model, alphabet


# ---------------------------------------------------------------------------
# Embedding generation
# ---------------------------------------------------------------------------

def generate_embeddings(
    sequences: list[tuple[str, str]],
    model,
    alphabet,
    device: torch.device,
    batch_size: int,
    repr_layer: int,
    per_sequence: bool,
) -> dict:
    """
    Run ESM-2 inference and return a dict of {label: embedding_tensor}.

    Args:
        sequences:    List of (label, sequence) tuples.
        model:        Loaded ESM-2 model.
        alphabet:     ESM-2 alphabet.
        device:       torch.device.
        batch_size:   Number of sequences per forward pass.
        repr_layer:   Which transformer layer to extract representations from.
                      Negative values count from the last layer (-1 = last).
        per_sequence: If True, mean-pool over residues to get one vector per
                      sequence. If False, return per-residue representations.
    """
    batch_converter = alphabet.get_batch_converter()
    num_layers = model.num_layers
    layer_idx = repr_layer if repr_layer >= 0 else num_layers + 1 + repr_layer

    embeddings = {}

    for start in tqdm(range(0, len(sequences), batch_size), desc="Batches"):
        batch = sequences[start: start + batch_size]
        _, _, tokens = batch_converter(batch)
        tokens = tokens.to(device)

        with torch.no_grad():
            results = model(tokens, repr_layers=[layer_idx], return_contacts=False)

        representations = results["representations"][layer_idx]  # (B, L, D)

        for i, (label, seq) in enumerate(batch):
            # Slice off BOS/EOS tokens; seq_len == len(seq)
            rep = representations[i, 1: len(seq) + 1]  # (seq_len, D)
            if per_sequence:
                rep = rep.mean(dim=0)  # (D,)
            embeddings[label] = rep.cpu()

    return embeddings


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate ESM-2 sequence embeddings from a FASTA file."
    )
    parser.add_argument(
        "--input", "-i", required=True,
        help="Path to input FASTA file (e.g. sequences.fa or sequences.fa.txt).",
    )
    parser.add_argument(
        "--output", "-o", default="embeddings.pt",
        help="Path for the output embeddings file (PyTorch .pt). Default: embeddings.pt",
    )
    parser.add_argument(
        "--model", "-m", default="esm2_t33_650M_UR50D",
        choices=list(AVAILABLE_MODELS),
        help="ESM-2 model variant. Default: esm2_t33_650M_UR50D (650 M params).",
    )
    parser.add_argument(
        "--batch-size", "-b", type=int, default=8,
        help="Sequences per forward pass. Reduce if you run out of memory. Default: 8.",
    )
    parser.add_argument(
        "--repr-layer", type=int, default=-1,
        help="Transformer layer index to extract. -1 = last layer. Default: -1.",
    )
    parser.add_argument(
        "--per-sequence", action="store_true",
        help="Mean-pool over residues to produce one vector per sequence. "
             "Without this flag, per-residue embeddings are saved.",
    )
    parser.add_argument(
        "--device", default=None,
        help="Compute device: 'cpu', 'cuda', 'cuda:0', etc. "
             "Auto-detected (CUDA if available) when not specified.",
    )
    parser.add_argument(
        "--max-len", type=int, default=1022,
        help="Sequences longer than this are truncated before inference "
             "(ESM-2 supports up to 1022 residues by default). Default: 1022.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Device selection
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load sequences
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    print(f"Reading sequences from: {input_path}")
    sequences = read_fasta(str(input_path))
    print(f"  Found {len(sequences)} sequences.")

    # Truncate long sequences
    truncated = 0
    processed = []
    for label, seq in sequences:
        if len(seq) > args.max_len:
            seq = seq[: args.max_len]
            truncated += 1
        processed.append((label, seq))
    if truncated:
        print(f"  Truncated {truncated} sequences to {args.max_len} residues.")

    # Load model
    model, alphabet = load_model(args.model, device)

    # Generate embeddings
    print(
        f"Generating {'per-sequence' if args.per_sequence else 'per-residue'} "
        f"embeddings (repr_layer={args.repr_layer}, batch_size={args.batch_size}) ..."
    )
    embeddings = generate_embeddings(
        processed, model, alphabet, device,
        batch_size=args.batch_size,
        repr_layer=args.repr_layer,
        per_sequence=args.per_sequence,
    )

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(embeddings, str(output_path))
    print(f"Embeddings saved to: {output_path}")

    # Summary
    sample_key = next(iter(embeddings))
    print(f"  Sample — ID: '{sample_key}', shape: {embeddings[sample_key].shape}")


if __name__ == "__main__":
    main()
