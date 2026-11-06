import math

import pytest

from src.food11.promote import comparison_problem, gate, standard_error


def test_standard_error():
    assert standard_error(0.75, 1096) == pytest.approx(math.sqrt(0.75 * 0.25 / 1096))


def test_clearly_better_candidate_is_promoted():
    promote, threshold, se = gate(acc_candidate=0.80, acc_champion=0.75, n=1096)
    assert promote
    assert threshold == pytest.approx(0.75 + 2 * se)


def test_slightly_better_candidate_is_not_promoted():
    # +1 point is inside the noise of a ~1100-image test set (2*se ~ 2.6 points)
    promote, _, _ = gate(acc_candidate=0.76, acc_champion=0.75, n=1096)
    assert not promote


def test_worse_candidate_is_not_promoted():
    promote, _, _ = gate(acc_candidate=0.40, acc_champion=0.75, n=1096)
    assert not promote


def test_same_test_set_is_comparable():
    params = {"dataset": "mini", "test_set_md5": "abc"}
    assert comparison_problem(params, dict(params)) is None


def test_runs_without_fingerprint_stay_comparable():
    # two older runs, both measured before test_set_md5 existed
    assert comparison_problem({"dataset": "mini"}, {"dataset": "mini"}) is None


def test_different_test_sets_are_not_comparable():
    assert comparison_problem(
        {"dataset": "mini", "test_set_md5": "old"}, {"dataset": "mini", "test_set_md5": "new"}
    )
    # old champion without fingerprint vs new candidate with one (data changed)
    assert comparison_problem({"dataset": "mini"}, {"dataset": "mini", "test_set_md5": "new"})
    assert comparison_problem({"dataset": "mini"}, {"dataset": "processed"})
