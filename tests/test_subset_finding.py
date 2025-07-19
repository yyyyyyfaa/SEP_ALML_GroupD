"""Unit tests for the subset_finding function."""

from __future__ import annotations

from itertools import chain, combinations

import numpy as np
import pytest
from shapiq.interaction_values import InteractionValues

from shapiq_student.subset_finding import (
    greedy_extreme_max,
    greedy_extreme_min,
    subset_finding,
    v_hat,
)


def generate_interaction_index(n_players: int, max_order: int, min_order: int):
    return list(chain.from_iterable(
        combinations(range(n_players), k)
        for k in range(min_order, max_order + 1)
    ))


@pytest.fixture
def example_weights():
    """Fixture providing example weights for subset finding tests.

    Returns:
    -------
    dict
        A dictionary mapping frozensets of player indices to their weights.
    """
    return {
        frozenset(): 0.1,
        frozenset([0]): 1.0,
        frozenset([1]): 2.0,
        frozenset([0, 1]): 3.0,
        frozenset([2]): -1.0,
        frozenset([1, 2]): 0.5,
    }

def test_v_hat(example_weights):
    """Test the v_hat function with a specific example of weights."""
    val = v_hat([0, 1], example_weights, k_max = 2)
    expected = 0.1 + 1.0 + 2.0 + 3.0 #empty + [0] + [1] + [0, 1]
    assert np.isclose(val, expected)

def test_greedy_extreme_max(example_weights):
    """Test the greedy_extreme_max function to ensure it selects the correct players."""
    N = [0, 1, 2]
    result = greedy_extreme_max(2, N, example_weights, k_max = 2)
    k = 2
    assert len(result) == k
    assert 1 in result #1 has the highest individual weight

def test_greedy_extreme_min(example_weights):
    """Test the greedy_extreme_min function to ensure it selects the correct players."""
    N = [0, 1, 2]
    result = greedy_extreme_min(2, N, example_weights, k_max = 2)
    k = 2
    assert len(result) == k
    assert k in result #2 has the lowest individual weight

def test_subset_finding(example_weights):
    index = generate_interaction_index(n_players = 3, max_order = 2, min_order = 1)
    index = frozenset(index)
    iv = InteractionValues(
        index = index,
        values = [1.0, 2.0, -1.0, 3.0, 0.0, 0.0],
        n_players = 3,
        max_order = 2,
        min_order = 1,
        baseline_value = 0.0
    )

    result = subset_finding(iv, max_size = 2)

    assert isinstance(result.values, list)
    assert len(result.values) == len(iv.values)

    expected_preserved = [0, 1, 3]
    for i in expected_preserved:
        assert result.values[i] != 0.0


def test_v_hat_with_empty_players():
    """Test v_hat returns 0.0 when given empty players and weights."""
    assert v_hat([], {}, 2) == 0.0

def test_v_hat_with_k_max_zero():
    """Test v_hat returns correct value when k_max is zero."""
    weights = {
        frozenset(): 1.0,
        frozenset([0]): 2.0
    }
    assert v_hat([0], weights, 0) == 1.0

def test_v_hat_with_unknown_weights():
    """Test v_hat returns 0.0 when weights are unknown (empty dictionary)."""
    assert v_hat([0, 1], {}, 2) == 0.0

def test_greedy_extreme_max_with_zero_length():
    """Test that greedy_extreme_max returns an empty set when the requested length is zero."""
    result = greedy_extreme_max(0, [0, 1], {}, 2)
    assert result == set()

def test_greedy_extreme_min_with_zero_length():
    """Test that greedy_extreme_min returns an empty set when the requested length is zero."""
    result = greedy_extreme_min(0, [0, 1], {}, 2)
    assert result == set()

def test_greedy_extreme_max_exceeds_number_of_players():
    """Test that greedy_extreme_max raises ValueError when requested coalition length exceeds number of players."""
    with pytest.raises(ValueError, match = "Requested coalition length exceeds number of players"):
        greedy_extreme_max(3, [0, 1], {}, 2)

def test_greedy_extreme_min_with_empty_candidates():
    """Test that greedy_extreme_min returns an empty set when candidates list is empty."""
    result = greedy_extreme_min(0, [], {}, 1)
    assert result == set()

def test_subset_finding_with_all_zero_values():
    index = generate_interaction_index(n_players = 2, max_order = 2, min_order = 1)
    index = frozenset(index)
    values = [0.0, 0.0, 0.0]
    iv = InteractionValues(
        index = index,
        values = values,
        n_players = 2,
        max_order = 2,
        min_order = 1,
        baseline_value = 0.0
    )
    result = subset_finding(iv, max_size = 2)
    assert all(val == 0.0 for val in result.values)
