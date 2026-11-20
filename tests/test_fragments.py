import numpy as np

from casmi.fragments import FEATURES, FragmentScorer, fragment_masses

PARACETAMOL = "CC(=O)Nc1ccc(O)cc1"         # C8H9NO2, [M+H]+ 152.0706
ISOMER = "COC(=O)c1ccc(N)cc1"               # methyl 4-aminobenzoate, same formula


def test_fragment_masses_contain_amide_cleavage():
    m, c = fragment_masses(PARACETAMOL)
    # HO-C6H4-NH (keeps the hydrogens it had in the molecule) = C6H6NO, one acyclic bond broken
    hit = np.isclose(m, 108.0449, atol=1e-3)
    assert hit.any() and c[hit].min() == 1
    assert (m >= 25).all() and np.all(np.diff(m) > 0) and m.size == c.size


def test_isomers_are_separated_by_fragment_explanation():
    sc = FragmentScorer(np.array([PARACETAMOL, ISOMER]))
    # paracetamol's characteristic fragment [4-aminophenol+H]+ 110.0600 and the ketene loss product's ring ion 93.06
    peaks = [(np.array([65.0386, 93.0573, 110.0600, 152.0706]), np.array([0.1, 0.2, 1.0, 0.5]), 152.0706, "[M+H]+")]
    X = sc.match(peaks, 0, np.array([0, 1]))
    assert X.shape == (2, len(FEATURES))
    assert X[0, FEATURES.index("fx_cheap")] > X[1, FEATURES.index("fx_cheap")]
    assert X[0, FEATURES.index("fx_score")] > X[1, FEATURES.index("fx_score")]
    assert X[0, FEATURES.index("fx_rank")] == 0.0 and X[0, FEATURES.index("fx_rel")] == 0.0


def test_no_peaks_gives_nan():
    sc = FragmentScorer(np.array([PARACETAMOL]))
    X = sc.match([(np.array([152.07]), np.array([1.0]), 152.0706, "[M+H]+")], 0, np.array([0]))
    assert np.isnan(X).all()
