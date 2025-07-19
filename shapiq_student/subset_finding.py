"""Coalition finding algorithms and utilities.

This module provides functions for computing coalition values and finding coalitions
that maximize or minimize a value function using greedy algorithms.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
from shapiq import InteractionValues


def v_hat(Players: list, e_weights: dict, k_max: int) -> float:
    """Compute the value function for a set of players up to a maximum coalition size.

    Parameters
    ----------
    Players : list
        List of players in the coalition.
    e_weights : dict
        Dictionary mapping frozensets of players to their associated weights.
    k_max : int
        Maximum coalition size to consider.

    Returns:
    -------
    float
        The computed value for the coalition.
    """
    total = e_weights.get(frozenset(), 0.0)
    for r in range(1, k_max + 1):
        for T in combinations(Players, r):
            total += e_weights.get(frozenset(T), 0.0)
    return total

def greedy_extreme_max(length: int, N: list, e_weights: dict, k_max: int) -> set:
    """Find a coalition of the given length that maximizes the value function using a greedy algorithm.

    Parameters
    ----------
    length : int
        Desired size of the coalition.
    N : list
        List of players.
    e_weights : dict
        Dictionary mapping frozensets of players to their associated weights.
    k_max : int
        Maximum coalition size to consider.

    Returns:
    -------
    set
        Set of players forming the coalition with the maximum value.
    """
    if length > len(N):
        msg = "Requested coalition length exceeds number of players"
        raise ValueError(msg)
    Players = set()
    candidates = set(N)
    while len(Players) < length:
        best_score = -np.inf
        best_elem = None
        for i in candidates:
            val = v_hat(Players.union({i}), e_weights, k_max)
            if val > best_score:
                best_score = val
                best_elem = i
        Players.add(best_elem)
        candidates.remove(best_elem)
    return Players

def greedy_extreme_min(length: int, N: list, e_weights: dict, k_max: int) -> set:
    """Find a coalition of the given length that minimizes the value function using a greedy algorithm.

    Parameters
    ----------
    length : int
        Desired size of the coalition.
    N : list
        List of players.
    e_weights : dict
        Dictionary mapping frozensets of players to their associated weights.
    k_max : int
        Maximum coalition size to consider.

    Returns:
    -------
    set
        Set of players forming the coalition with the minimum value.
    """
    Players = set()
    candidates = set(N)
    while len(Players) < length:
        best_score = np.inf
        best_elem = None
        for i in candidates:
            val = v_hat(Players.union({i}), e_weights, k_max)
            if val < best_score:
                best_score = val
                best_elem = i
        Players.add(best_elem)
        candidates.remove(best_elem)
    return Players

def subset_finding(interaction_values: InteractionValues, max_size: int) -> dict:
    """Compute new interaction values based on greedy subset finding.

    Parameters
    ----------
    interaction_values : InteractionValues
        The original Interaction values object.
    k_max : int
        Maximum coalition size to consider.

    Returns:
    -------
    InteractionValues
        A new InteractionValues object with filtered/support-sellected entries.
    """
    weights = interaction_values.values
    index = interaction_values.index
    n_players = interaction_values.n_players

    #converting weights into dict from frozenset(inices)
    e_weights = {frozenset(coal): value for coal, value in zip(index, weights, strict=False)}

    N = list(range(n_players))
    selected_coalitions = set()

    #finding coalitions for all lengths
    for length in range(1, n_players + 1):
        s_max = greedy_extreme_max(length, N, e_weights, max_size)
        s_min = greedy_extreme_min(length, N, e_weights, max_size)

        #back to String coalitions
        str_max = tuple(sorted(s_max))
        str_min = tuple(sorted(s_min))


        selected_coalitions.add(str_max)
        selected_coalitions.add(str_min)

    #empty coalition
    if frozenset() in e_weights:
        selected_coalitions.add("")

    #filter values
    new_values = []
    epsilon = 1e-3

    for coal in index:
        value = e_weights.get(frozenset(coal), 0.0)
        if (tuple(sorted(coal)) in selected_coalitions) or (abs(value) > epsilon):
            new_values.append(value)
        else:
            new_values.append(0.0)

    return InteractionValues(
        index = interaction_values.index,
        values = new_values,
        n_players = n_players,
        max_order = max_size,
        min_order = interaction_values.min_order,
        baseline_value = interaction_values.baseline_value)
