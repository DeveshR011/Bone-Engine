"""Tests for HNSW sparse-vector dimension mapping.

The mapping must be an exact inverse of the tokenizer's ID->token mapping and,
without a tokenizer, must be stable across processes — an index is built in one
run and queried in another.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

# hnswlib needs a C++ toolchain to build and is not required by the BEIR
# benchmark, so this module skips rather than breaking collection without it.
pytest.importorskip("hnswlib", reason="hnswlib not installed")

from src.backends.hnsw_backend import HNSWSparseBackend  # noqa: E402


class StubTokenizer:
    vocab = {"retrieval": 100, "##ing": 101}
    unk_token_id = 1

    def convert_tokens_to_ids(self, token):
        return self.vocab.get(token, self.unk_token_id)

    def encode(self, text, add_special_tokens=True):
        return [self.vocab.get(text, self.unk_token_id)]


class TestTokenToIndex:
    def test_subword_token_maps_to_its_vocab_id(self):
        assert HNSWSparseBackend._token_to_index("##ing", StubTokenizer()) == 101

    def test_plain_token_maps_to_its_vocab_id(self):
        assert HNSWSparseBackend._token_to_index("retrieval", StubTokenizer()) == 100

    def test_unknown_token_is_rejected(self):
        assert HNSWSparseBackend._token_to_index("absent", StubTokenizer()) == -1

    def test_fallback_is_in_range(self):
        idx = HNSWSparseBackend._token_to_index("retrieval", None)
        assert 0 <= idx < 30522

    def test_fallback_is_deterministic_within_process(self):
        a = HNSWSparseBackend._token_to_index("retrieval", None)
        b = HNSWSparseBackend._token_to_index("retrieval", None)
        assert a == b

    def test_fallback_is_stable_across_processes(self):
        """Regression: builtin hash() is randomized per interpreter, so an
        index built in one process mapped tokens to different dimensions when
        queried from another."""
        expected = HNSWSparseBackend._token_to_index("retrieval", None)

        code = (
            "import sys; sys.path.insert(0, '.');"
            "from src.backends.hnsw_backend import HNSWSparseBackend as H;"
            "print(H._token_to_index('retrieval', None))"
        )
        out = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True, text=True, timeout=120,
        )
        assert out.returncode == 0, out.stderr
        assert int(out.stdout.strip()) == expected
