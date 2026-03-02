"""Structured logging, timing utilities, and experiment result logging."""

from __future__ import annotations

import csv
import logging
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Create a consistently-formatted logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = logging.Formatter(
            "[%(asctime)s] %(name)s — %(levelname)s — %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


# ---------------------------------------------------------------------------
# Timer
# ---------------------------------------------------------------------------

class Timer:
    """Reusable high-resolution timer with lap support.

    Usage:
        timer = Timer()
        timer.start()
        ...
        timer.stop()
        print(timer.elapsed_ms)

    Or as a context manager:
        with Timer() as t:
            ...
        print(t.elapsed_ms)
    """

    def __init__(self, name: str = ""):
        self.name = name
        self._start: float | None = None
        self._end: float | None = None
        self.laps: list[float] = []

    def start(self) -> "Timer":
        self._start = time.perf_counter()
        self._end = None
        return self

    def stop(self) -> float:
        self._end = time.perf_counter()
        elapsed = self._end - self._start
        self.laps.append(elapsed)
        return elapsed

    def lap(self) -> float:
        """Record a lap without stopping the timer."""
        now = time.perf_counter()
        elapsed = now - self._start
        self.laps.append(elapsed)
        return elapsed

    @property
    def elapsed(self) -> float:
        """Elapsed time in seconds."""
        if self._end is not None:
            return self._end - self._start
        if self._start is not None:
            return time.perf_counter() - self._start
        return 0.0

    @property
    def elapsed_ms(self) -> float:
        return self.elapsed * 1000.0

    def __enter__(self) -> "Timer":
        self.start()
        return self

    def __exit__(self, *args: Any) -> None:
        self.stop()

    def __repr__(self) -> str:
        label = f" [{self.name}]" if self.name else ""
        return f"Timer{label}: {self.elapsed_ms:.2f} ms"


# ---------------------------------------------------------------------------
# Experiment result logging
# ---------------------------------------------------------------------------

@dataclass
class ExperimentResult:
    """Single experiment result row."""
    experiment: str = ""
    system: str = ""            # e.g. "sparse", "dense", "hybrid", "graph"
    variant: str = ""           # e.g. "quantized", "cpu", "gpu"
    recall_at_5: float = 0.0
    recall_at_10: float = 0.0
    recall_at_20: float = 0.0
    ndcg_at_10: float = 0.0
    mrr: float = 0.0
    query_latency_ms: float = 0.0
    index_size_mb: float = 0.0
    memory_usage_mb: float = 0.0
    indexing_time_s: float = 0.0
    docs_per_sec: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    notes: str = ""


def log_experiment(result: ExperimentResult, results_dir: str = "results") -> Path:
    """Append an experiment result to the CSV log file.

    Returns the path to the CSV file.
    """
    os.makedirs(results_dir, exist_ok=True)
    csv_path = Path(results_dir) / "experiments.csv"

    row = asdict(result)
    file_exists = csv_path.exists()

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    return csv_path


def load_config(config_path: str = "config.yaml") -> dict:
    """Load YAML configuration."""
    import yaml

    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
