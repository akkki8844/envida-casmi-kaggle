"""MRR@25 exactly as the competition describes it."""
from __future__ import annotations

from typing import Iterable, Mapping, Sequence

from .chem import key14


def reciprocal_rank(guesses: Sequence[str], truth_key: str, k: int = 25) -> float:
    """1/rank of the first guess whose key matches; 0 if none in the first k."""
    for i, smi in enumerate(guesses[:k], start=1):
        if key14(smi) == truth_key:
            return 1.0 / i
    return 0.0


def mrr_at_25(
    predictions: Mapping[str, Iterable[str]], truth: Mapping[str, str], k: int = 25
) -> float:
    """predictions: molecule_id -> ranked SMILES; truth: molecule_id -> SMILES.

    Missing molecules score 0. Both sides go through key14.
    """
    total = 0.0
    for mol_id, smi in truth.items():
        tk = key14(smi)
        total += reciprocal_rank(list(predictions.get(mol_id, [])), tk, k)
    return total / max(len(truth), 1)


def parse_submission_field(field: str) -> list[str]:
    return [s for s in field.split(";") if s]
