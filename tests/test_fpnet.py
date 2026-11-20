import numpy as np

from casmi.fpnet import FEATURES, N_BITS, N_NL, N_PK, V, candidate_features, tokens
from casmi.spectra import spectrum_vectors, stack_csr


def test_tokens_pool_bins_and_mode():
    mz = np.array([100.013, 100.091, 250.5])
    p, l = spectrum_vectors(mz, np.array([1.0, 0.5, 0.2]), 300.0)
    X = tokens(stack_csr([p]), stack_csr([l]), [1])
    assert X.shape == (1, V)
    cols = X.indices
    assert 1000 in cols and 2505 in cols            # 0.1 Da peak bins
    assert N_PK + N_NL + 1 in cols                   # negative-mode token
    assert ((cols >= N_PK) & (cols < N_PK + N_NL)).any()  # neutral-loss tokens present


def test_candidate_features_prefer_matching_bits():
    rng = np.random.default_rng(0)
    truth = (rng.random(N_BITS) < 0.03).astype(np.uint8)
    other = (rng.random(N_BITS) < 0.03).astype(np.uint8)
    p = np.where(truth == 1, 0.8, 0.02).astype(np.float32)
    X = candidate_features(p, np.stack([other, truth]))
    assert X.shape == (2, len(FEATURES))
    for name in ("fp_ll", "fp_tani", "fp_cos"):
        assert X[1, FEATURES.index(name)] > X[0, FEATURES.index(name)]
    assert X[1, FEATURES.index("fp_rank")] == 0.0 and X[1, FEATURES.index("fp_ll_rel")] == 0.0
