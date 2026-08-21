# data/

## What is in here

`documents.jsonl`, `queries.jsonl`, and `qrels.jsonl` are a **15-document,
5-query smoke-test fixture**. They exist so the legacy Elasticsearch /
PostgreSQL pipeline can be run end to end without external data. They are
**not an evaluation set**, and numbers produced from them are not benchmark
results.

Why they cannot measure retrieval quality: with 15 documents, a Recall@10
cutoff returns two-thirds of the entire collection, so every method converges
toward perfect recall regardless of how good it is. With 5 queries, one query
changing rank moves any average by 20 percentage points. Differences between
systems on this fixture are noise.

A historical example of exactly that failure: an earlier version of this repo
reported Recall@10 = 1.000 for three separate backends on this fixture, and
ranked Hybrid Fusion *last* of seven methods. The first was an artifact of
corpus size; the second was a genuine bug in score normalisation (since fixed
and covered by regression tests in `tests/test_fusion.py`). On real corpora
fusion now beats every individual retriever it combines.

## Measured results live elsewhere

Retrieval quality is measured on BEIR over full corpora:

```
python scripts/run_beir.py --datasets scifact nfcorpus fiqa trec-covid
```

That writes `results/beir_results.json` and is the only path whose numbers
belong in the README. See the repository README for the current results table.

## Generated files (gitignored)

`beir/` holds downloaded BEIR corpora; `*.index`, `*.npy`, and
`splade_sparse.jsonl` are build artifacts. All are regenerated on demand and
are not tracked.
