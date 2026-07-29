"""
BEIR Dataset Loader
===================
Downloads, caches, and parses BEIR benchmark datasets — the standard
zero-shot IR benchmark that published retrieval results are reported on.

Datasets are fetched from the official public mirror and cached under
``data/beir/<name>/``. Nothing is re-downloaded once extracted.

BEIR on-disk layout::

    <name>/corpus.jsonl     {"_id", "title", "text"}
    <name>/queries.jsonl    {"_id", "text"}
    <name>/qrels/test.tsv   query-id \t corpus-id \t score   (with header)

Only queries that appear in qrels are kept: BEIR ships the full query pool for
some datasets while judging a subset, and scoring unjudged queries would
silently drag every metric toward zero.
"""

from __future__ import annotations

import json
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

from ..utils.logging_utils import Timer, get_logger

logger = get_logger(__name__)

BEIR_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{name}.zip"

# Approximate corpus sizes, used only for logging and runtime expectations.
DATASET_SIZES: dict[str, int] = {
    "scifact": 5_183,
    "nfcorpus": 3_633,
    "fiqa": 57_638,
    "trec-covid": 171_332,
    "scidocs": 25_657,
    "arguana": 8_674,
    "quora": 522_931,
    "climate-fever": 5_416_593,
    "dbpedia-entity": 4_635_922,
    "hotpotqa": 5_233_329,
    "nq": 2_681_468,
    "msmarco": 8_841_823,
}

# The laptop-friendly subset: all four fit comfortably in 6 GB VRAM and finish
# in minutes to low hours, while remaining directly comparable to published
# BEIR tables.
DEFAULT_SUBSET = ["scifact", "nfcorpus", "fiqa", "trec-covid"]


@dataclass
class BeirDataset:
    """A loaded BEIR dataset."""

    name: str
    corpus: dict[str, dict[str, str]] = field(default_factory=dict)
    queries: dict[str, str] = field(default_factory=dict)
    qrels: dict[str, dict[str, int]] = field(default_factory=dict)

    @property
    def doc_ids(self) -> list[str]:
        return list(self.corpus.keys())

    @property
    def query_ids(self) -> list[str]:
        return list(self.queries.keys())

    def doc_texts(self, doc_ids: list[str] | None = None) -> list[str]:
        """Concatenate title and body, the standard BEIR document rendering."""
        ids = doc_ids if doc_ids is not None else self.doc_ids
        texts = []
        for did in ids:
            doc = self.corpus[did]
            title = (doc.get("title") or "").strip()
            body = (doc.get("text") or "").strip()
            texts.append(f"{title} {body}".strip() if title else body)
        return texts

    def __repr__(self) -> str:
        judged = sum(len(v) for v in self.qrels.values())
        return (
            f"BeirDataset({self.name}: {len(self.corpus):,} docs, "
            f"{len(self.queries):,} queries, {judged:,} judgments)"
        )


def download_dataset(name: str, cache_dir: str | Path = "data/beir") -> Path:
    """Download and extract a BEIR dataset if not already cached.

    Returns the path to the extracted dataset directory.
    """
    cache_dir = Path(cache_dir)
    dataset_dir = cache_dir / name

    if (dataset_dir / "corpus.jsonl").exists():
        logger.info("Dataset '%s' already cached at %s", name, dataset_dir)
        return dataset_dir

    cache_dir.mkdir(parents=True, exist_ok=True)
    zip_path = cache_dir / f"{name}.zip"
    url = BEIR_URL.format(name=name)

    size_hint = DATASET_SIZES.get(name)
    logger.info(
        "Downloading '%s'%s from %s",
        name,
        f" (~{size_hint:,} docs)" if size_hint else "",
        url,
    )

    with Timer("download") as t:
        try:
            urllib.request.urlretrieve(url, zip_path)
        except Exception as e:
            raise RuntimeError(
                f"Failed to download BEIR dataset '{name}' from {url}. "
                f"Check the dataset name and network access. Original error: {e}"
            ) from e
    logger.info("Downloaded in %.1f s", t.elapsed)

    logger.info("Extracting to %s", cache_dir)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(cache_dir)
    zip_path.unlink()

    if not (dataset_dir / "corpus.jsonl").exists():
        raise RuntimeError(
            f"Extracted archive for '{name}' but found no corpus.jsonl in {dataset_dir}"
        )

    return dataset_dir


def load_dataset(
    name: str,
    split: str = "test",
    cache_dir: str | Path = "data/beir",
    max_corpus_size: int | None = None,
) -> BeirDataset:
    """Load a BEIR dataset, downloading it on first use.

    Args:
        name: BEIR dataset name, e.g. ``"scifact"``.
        split: qrels split to score against — usually ``"test"``.
        cache_dir: Where datasets are cached.
        max_corpus_size: Optional cap on corpus size for smoke tests. Judged
            documents are always retained, so metrics stay well-defined, but
            a truncated corpus is an easier retrieval task and its scores are
            NOT comparable to published BEIR numbers.

    Returns:
        A populated :class:`BeirDataset`.
    """
    dataset_dir = download_dataset(name, cache_dir)

    # --- qrels -------------------------------------------------------
    qrels_path = dataset_dir / "qrels" / f"{split}.tsv"
    if not qrels_path.exists():
        available = [p.stem for p in (dataset_dir / "qrels").glob("*.tsv")]
        raise FileNotFoundError(
            f"No '{split}' split for '{name}'. Available splits: {available}"
        )

    qrels: dict[str, dict[str, int]] = {}
    with open(qrels_path, encoding="utf-8") as f:
        header = f.readline()  # query-id \t corpus-id \t score
        if not header.lower().startswith("query-id"):
            f.seek(0)  # some mirrors ship headerless qrels
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            qid, did, score = parts[0], parts[1], int(parts[2])
            # BEIR uses graded relevance; 0 means explicitly judged
            # non-relevant and must not be treated as a positive.
            if score > 0:
                qrels.setdefault(qid, {})[did] = score

    # --- queries (only those with judgments) -------------------------
    queries: dict[str, str] = {}
    with open(dataset_dir / "queries.jsonl", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            if obj["_id"] in qrels:
                queries[obj["_id"]] = obj["text"]

    # --- corpus ------------------------------------------------------
    judged_docs = {did for rels in qrels.values() for did in rels}
    corpus: dict[str, dict[str, str]] = {}

    with open(dataset_dir / "corpus.jsonl", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            did = obj["_id"]
            if (
                max_corpus_size is not None
                and len(corpus) >= max_corpus_size
                and did not in judged_docs
            ):
                continue
            corpus[did] = {
                "title": obj.get("title", ""),
                "text": obj.get("text", ""),
            }

    dataset = BeirDataset(name=name, corpus=corpus, queries=queries, qrels=qrels)

    # A query whose judged documents were dropped would be unscoreable.
    missing = judged_docs - corpus.keys()
    if missing:
        logger.warning(
            "%d judged documents are absent from the corpus for '%s'",
            len(missing), name,
        )

    if max_corpus_size is not None:
        logger.warning(
            "Corpus truncated to %d docs — scores are NOT comparable to "
            "published BEIR results.", len(corpus),
        )

    logger.info("Loaded %s", dataset)
    return dataset
