"""Dataset loading for benchmark evaluation."""

from .beir_loader import DEFAULT_SUBSET, BeirDataset, download_dataset, load_dataset

__all__ = ["BeirDataset", "load_dataset", "download_dataset", "DEFAULT_SUBSET"]
