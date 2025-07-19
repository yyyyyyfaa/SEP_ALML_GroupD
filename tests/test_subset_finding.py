"""Unit tests for the subset_finding function."""

from __future__ import annotations

import numpy as np
import pytest
from shapiq.interaction_values import InteractionValues

from shapiq_student.subset_finding import (
    greedy_extreme_max,
    greedy_extreme_min,
    subset_finding,
    v_hat,
)


@pytest.fixture
def example_weights():
    return {
        frozenset(): 0.1,
        frozenset([0]): 1.0,
        frozenset([1]): 2.0,
        frozenset([0, 1]): 3.0,
        frozenset([2]): -1.0,
        frozenset([1, 2]): 0.5,
    }

def test_v_hat(example_weights):
    val = v_hat([0, 1], example_weights, k_max = 2)
    expected = 0.1 + 1.0 + 2.0 + 3.0 #empty + [0] + [1] + [0, 1]
    assert np.isclose(val, expected)

def test_greedy_extreme_max(example_weights):
    N = [0, 1, 2]
    result = greedy_extreme_max(2, N, example_weights, k_max = 2)
    k = 2
    assert len(result) == k
    assert 1 in result #1 has the highest individual weight

def test_greedy_extreme_min(example_weights):
    N = [0, 1, 2]
    result = greedy_extreme_min(2, N, example_weights, k_max = 2)
    k = 2
    assert len(result) == k
    assert k in result #2 has the lowest individual weight

def test_v_hat_with_empty_players():
    assert v_hat([], {}, 2) == 0.0

def test_v_hat_with_k_max_zero():
    weights = {
        frozenset(): 1.0,
        frozenset([0]): 2.0
    }
    assert v_hat([0], weights, 0) == 1.0

def test_v_hat_with_unknown_weights():
    assert v_hat([0, 1], {}, 2) == 0.0

def test_greedy_extreme_max_with_zero_length():
    result = greedy_extreme_max(0, [0, 1], {}, 2)
    assert result == set()

def test_greedy_extreme_min_with_zero_length():
    result = greedy_extreme_min(0, [0, 1], {}, 2)
    assert result == set()

def test_greedy_extreme_max_exceeds_number_of_players():
    with pytest.raises(ValueError, match = "Requested coalition length exceeds number of players"):
        greedy_extreme_max(3, [0, 1], {}, 2)

def test_greedy_extreme_min_with_empty_candidates():
    result = greedy_extreme_min(0, [], {}, 1)
    assert result == set()

def test_subset_finding_with_simple_imput():
    iv = InteractionValues(
        index = "all",
        values = [],
        n_players = 0,
        max_order = 0,
        min_order = 0,
        baseline_value = 0.0
    )

    result = subset_finding(iv, max_size = 2)
    assert result.values == []
