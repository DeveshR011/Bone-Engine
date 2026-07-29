"""Tests for SPLADE sparse-vector key handling.

These exercise the ID/token mapping without downloading a model, using a
stub tokenizer whose subword behaviour mirrors BERT's: ``decode()`` strips the
``##`` continuation marker, while ``convert_ids_to_tokens`` preserves it.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.encoder.splade_encoder import SpladeEncoder


class StubTokenizer:
    """Minimal BERT-like tokenizer with a lossy decode(), as the real one has."""

    vocab = {"retrieval": 100, "##ing": 101, "sparse": 102, "[UNK]": 1}
    unk_token_id = 1

    def convert_ids_to_tokens(self, ids):
        rev = {v: k for k, v in self.vocab.items()}
        return [rev.get(i) for i in ids]

    def convert_tokens_to_ids(self, tokens):
        return [self.vocab.get(t, self.unk_token_id) for t in tokens]

    def decode(self, ids):
        # Mirrors the real lossy behaviour: the '##' marker is dropped.
        return " ".join(t.lstrip("#") for t in self.convert_ids_to_tokens(ids) if t)

    def encode(self, text, add_special_tokens=True):
        return [self.vocab.get(text, self.unk_token_id)]


@pytest.fixture
def encoder():
    """A SpladeEncoder with only the attributes these tests touch."""
    enc = SpladeEncoder.__new__(SpladeEncoder)
    enc.tokenizer = StubTokenizer()
    enc.vocab_size = 200
    return enc


class TestTokenIdMapping:
    def test_subword_token_survives_round_trip(self, encoder):
        """Regression: decode()->encode() corrupted '##ing' into a new term."""
        id_dict = {101: 4.2}

        tokens = encoder._ids_to_tokens(id_dict)
        assert tokens == {"##ing": 4.2}

        dense = encoder.sparse_dicts_to_dense([tokens])
        assert dense[0, 101] == pytest.approx(4.2)

    def test_old_decode_path_would_have_lost_the_marker(self, encoder):
        """Documents why the round trip changed — the stub reproduces the bug."""
        assert encoder.tokenizer.decode([101]) == "ing"
        assert encoder.tokenizer.encode("ing")[0] == encoder.tokenizer.unk_token_id

    def test_multiple_terms_round_trip(self, encoder):
        id_dict = {100: 1.5, 101: 2.5, 102: 3.5}
        dense = encoder.sparse_dicts_to_dense([encoder._ids_to_tokens(id_dict)])

        for token_id, weight in id_dict.items():
            assert dense[0, token_id] == pytest.approx(weight)
        assert np.count_nonzero(dense) == 3

    def test_unknown_tokens_are_dropped_not_misplaced(self, encoder):
        dense = encoder.sparse_dicts_to_dense([{"not_in_vocab": 9.9}])
        assert np.count_nonzero(dense) == 0

    def test_empty_inputs(self, encoder):
        assert encoder._ids_to_tokens({}) == {}
        assert encoder.sparse_dicts_to_dense([{}]).shape == (1, 200)
        assert np.count_nonzero(encoder.sparse_dicts_to_dense([{}])) == 0

    def test_dense_matrix_shape_matches_batch(self, encoder):
        dense = encoder.sparse_dicts_to_dense([{"sparse": 1.0}, {"retrieval": 2.0}])
        assert dense.shape == (2, 200)
        assert dense[0, 102] == pytest.approx(1.0)
        assert dense[1, 100] == pytest.approx(2.0)


class TestQuantization:
    def test_scales_to_8bit_range(self, encoder):
        encoder.global_max = 10.0
        out = encoder._quantize({1: 10.0, 2: 5.0})
        assert out[1] == 255.0
        assert out[2] == 127.0

    def test_noop_without_global_max(self, encoder):
        encoder.global_max = None
        assert encoder._quantize({1: 1.0}) == {1: 1.0}
