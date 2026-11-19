import math

from casmi.chem import ADDUCT_DELTA, key14, neutral_mass
from casmi.metric import mrr_at_25, reciprocal_rank

GLUCOSE_A = "OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O"
GLUCOSE_B = "OCC1OC(O)C(O)C(O)C1O"


def test_glucose_example_from_overview_matches():
    # The Overview page states both reduce to WQZGKKKJIJFFOK.
    assert key14(GLUCOSE_A) == "WQZGKKKJIJFFOK"
    assert key14(GLUCOSE_B) == "WQZGKKKJIJFFOK"


def test_tautomers_match():
    # keto/enol of acetone should collapse after tautomer canonicalisation
    assert key14("CC(C)=O") == key14("CC(=C)O")


def test_reciprocal_rank_positions():
    truth = key14(GLUCOSE_A)
    assert reciprocal_rank([GLUCOSE_B], truth) == 1.0
    assert reciprocal_rank(["CCO", GLUCOSE_B], truth) == 0.5
    assert math.isclose(reciprocal_rank(["CCO"] * 24 + [GLUCOSE_B], truth), 1 / 25)
    assert reciprocal_rank(["CCO"] * 25 + [GLUCOSE_B], truth) == 0.0


def test_mrr_missing_molecule_scores_zero():
    truth = {"a": GLUCOSE_A, "b": "CCO"}
    preds = {"a": [GLUCOSE_B]}
    assert mrr_at_25(preds, truth) == 0.5


def test_neutral_mass_protonated_glucose():
    # glucose C6H12O6 = 180.063388; [M+H]+ = 181.070665
    assert abs(neutral_mass(181.070665, "[M+H]+") - 180.063388) < 1e-4


def test_all_ten_test_adducts_defined():
    assert len(ADDUCT_DELTA) == 10
