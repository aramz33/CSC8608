import random
import time
import os
import numpy as np
import torch


def get_device(preferred: str = "cuda") -> torch.device:
    if preferred == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class Timer:
    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *_):
        self.elapsed = time.perf_counter() - self.start


def accuracy(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> float:
    preds = logits[mask].argmax(dim=-1)
    correct = (preds == labels[mask]).sum().item()
    return correct / mask.sum().item()


def macro_f1(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> float:
    preds = logits[mask].argmax(dim=-1).cpu().numpy()
    targets = labels[mask].cpu().numpy()
    classes = np.unique(targets)
    f1_sum = 0.0
    for c in classes:
        tp = ((preds == c) & (targets == c)).sum()
        fp = ((preds == c) & (targets != c)).sum()
        fn = ((preds != c) & (targets == c)).sum()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_c = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        f1_sum += f1_c
    return float(f1_sum / len(classes))


def compute_metrics(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor):
    return {
        "acc": accuracy(logits, labels, mask),
        "f1": macro_f1(logits, labels, mask),
    }
