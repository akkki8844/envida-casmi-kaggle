"""Chemistry helpers: official-style structure key, adduct masses, fingerprints."""
from __future__ import annotations

from functools import lru_cache

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, inchi, rdFingerprintGenerator
from rdkit.Chem.MolStandardize import rdMolStandardize

RDLogger.DisableLog("rdApp.*")

# Monoisotopic masses (Da)
M_H = 1.00782503207
M_PROTON = 1.00727646688
M_ELECTRON = 0.00054857990946
M_O = 15.99491461956
M_C = 12.0
M_N = 14.0030740048
M_NA = 22.9897692809
M_K = 38.96370668
M_CL = 34.96885268
M_H2O = 2 * M_H + M_O
M_NH4 = M_N + 4 * M_H - M_ELECTRON  # NH4+ cation
M_FORMIC = M_C + 2 * M_H + 2 * M_O  # CH2O2

# precursor_mz = neutral_mass + ADDUCT_DELTA[adduct]
ADDUCT_DELTA = {
    "[M+H]+": M_PROTON,
    "[M+NH4]+": M_NH4,
    "[M-H2O+H]+": M_PROTON - M_H2O,
    "[M-2H2O+H]+": M_PROTON - 2 * M_H2O,
    "[M+Na]+": M_NA - M_ELECTRON,
    "[M+K]+": M_K - M_ELECTRON,
    "[M-H]-": -M_PROTON,
    "[M-H2O-H]-": -M_PROTON - M_H2O,
    "[M+CH2O2-H]-": M_FORMIC - M_PROTON,
    "[M+Cl]-": M_CL + M_ELECTRON,
}
TEST_ADDUCTS = tuple(ADDUCT_DELTA)

_TAUT = rdMolStandardize.TautomerEnumerator()
_MORGAN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
FP_BITS = 2048


@lru_cache(maxsize=500_000)
def key14(smiles: str) -> str | None:
    """InChIKey14 after RDKit tautomer canonicalisation (the official matching key).

    Falls back to the un-canonicalised molecule if tautomer canonicalisation fails.
    Returns None if the SMILES cannot be parsed.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        canon = _TAUT.Canonicalize(mol)
    except Exception:
        canon = mol
    try:
        return inchi.MolToInchiKey(canon)[:14]
    except Exception:
        return None


def neutral_mass(precursor_mz: float, adduct: str) -> float | None:
    delta = ADDUCT_DELTA.get(adduct)
    return None if delta is None else precursor_mz - delta


def exact_mass(smiles: str) -> float | None:
    mol = Chem.MolFromSmiles(smiles)
    return None if mol is None else Descriptors.ExactMolWt(mol)


def morgan_bits(smiles: str) -> np.ndarray:
    """2048-bit Morgan (r=2) fingerprint as uint8; all zeros if unparsable."""
    mol = Chem.MolFromSmiles(smiles)
    out = np.zeros(FP_BITS, dtype=np.uint8)
    if mol is not None:
        out[list(_MORGAN.GetFingerprint(mol).GetOnBits())] = 1
    return out
