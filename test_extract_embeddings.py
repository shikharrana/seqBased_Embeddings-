"""
test_extract_embeddings.py
--------------------------
Unit tests for the helper utilities in extract_embeddings.py.
These tests do NOT require the ESM-2 model weights to be downloaded;
they exercise the FASTA parser and the embedding-aggregation logic only.
"""

import os
import tempfile

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Import the module under test
# ---------------------------------------------------------------------------
from extract_embeddings import parse_fasta, ESM2_MODELS


# ---------------------------------------------------------------------------
# FASTA parser tests
# ---------------------------------------------------------------------------
FASTA_CONTENT = """\
>seq1 first protein
ACDEFGHIKLMNPQRSTVWY
>seq2 second protein
MKTAYIAKQRQISFVK
>seq3 multiline
ACDEF
GHIKL
"""


def _write_fasta(content: str) -> str:
    """Write *content* to a temporary file and return its path."""
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".fasta", delete=False
    )
    tmp.write(content)
    tmp.flush()
    tmp.close()
    return tmp.name


class TestParseFasta:
    def test_number_of_records(self):
        path = _write_fasta(FASTA_CONTENT)
        try:
            records = list(parse_fasta(path))
        finally:
            os.unlink(path)
        assert len(records) == 3

    def test_headers_stripped(self):
        path = _write_fasta(FASTA_CONTENT)
        try:
            records = list(parse_fasta(path))
        finally:
            os.unlink(path)
        headers = [h for h, _ in records]
        assert headers[0] == "seq1 first protein"
        assert headers[1] == "seq2 second protein"
        assert headers[2] == "seq3 multiline"

    def test_sequence_content(self):
        path = _write_fasta(FASTA_CONTENT)
        try:
            records = list(parse_fasta(path))
        finally:
            os.unlink(path)
        assert records[0][1] == "ACDEFGHIKLMNPQRSTVWY"
        assert records[1][1] == "MKTAYIAKQRQISFVK"

    def test_multiline_sequence_joined(self):
        path = _write_fasta(FASTA_CONTENT)
        try:
            records = list(parse_fasta(path))
        finally:
            os.unlink(path)
        assert records[2][1] == "ACDEFGHIKL"

    def test_empty_file(self):
        path = _write_fasta("")
        try:
            records = list(parse_fasta(path))
        finally:
            os.unlink(path)
        assert records == []

    def test_blank_lines_ignored(self):
        content = ">prot1\n\nACDE\n\nFGHI\n"
        path = _write_fasta(content)
        try:
            records = list(parse_fasta(path))
        finally:
            os.unlink(path)
        assert len(records) == 1
        assert records[0][1] == "ACDEFGHI"


# ---------------------------------------------------------------------------
# ESM2_MODELS constant tests
# ---------------------------------------------------------------------------
class TestEsm2ModelsConstant:
    def test_contains_expected_models(self):
        expected = [
            "esm2_t6_8M_UR50D",
            "esm2_t12_35M_UR50D",
            "esm2_t30_150M_UR50D",
            "esm2_t33_650M_UR50D",
            "esm2_t36_3B_UR50D",
            "esm2_t48_15B_UR50D",
        ]
        for name in expected:
            assert name in ESM2_MODELS, f"'{name}' missing from ESM2_MODELS"

    def test_layer_counts_are_positive_integers(self):
        for name, layers in ESM2_MODELS.items():
            assert isinstance(layers, int) and layers > 0, (
                f"Layer count for '{name}' must be a positive integer"
            )

    def test_layer_counts_match_model_names(self):
        # The layer count is encoded in the model name (tN)
        for name, layers in ESM2_MODELS.items():
            # e.g. esm2_t33_650M_UR50D → 33
            encoded = int(name.split("_t")[1].split("_")[0])
            assert encoded == layers, (
                f"Layer mismatch for '{name}': name says {encoded}, "
                f"dict says {layers}"
            )


# ---------------------------------------------------------------------------
# Embedding aggregation logic (mean-pool) — no model required
# ---------------------------------------------------------------------------
class TestMeanPooling:
    """Verify that mean-pooling a fake token representation works correctly."""

    def test_mean_pool_shape(self):
        seq_len = 10
        embed_dim = 320
        # Simulate token representations: shape (seq_len + 2, embed_dim)
        # index 0 = <cls>, indices 1..seq_len = residues, -1 = <eos>
        token_reps = np.random.rand(seq_len + 2, embed_dim).astype(np.float32)
        embedding = token_reps[1 : seq_len + 1].mean(axis=0)
        assert embedding.shape == (embed_dim,)

    def test_mean_pool_values(self):
        seq_len = 4
        embed_dim = 2
        token_reps = np.array(
            [
                [0.0, 0.0],   # <cls>  — should be ignored
                [1.0, 2.0],   # residue 1
                [3.0, 4.0],   # residue 2
                [5.0, 6.0],   # residue 3
                [7.0, 8.0],   # residue 4
                [0.0, 0.0],   # <eos>  — should be ignored
            ],
            dtype=np.float32,
        )
        embedding = token_reps[1 : seq_len + 1].mean(axis=0)
        expected = np.array([4.0, 5.0], dtype=np.float32)
        np.testing.assert_allclose(embedding, expected)
