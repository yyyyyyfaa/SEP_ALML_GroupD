"""Coalition finding algorithms and utilities.

This module provides functions for computing coalition values and finding coalitions
that maximize or minimize a value function using greedy algorithms.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np


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

def greedy_coalition_finding(N: list, e_weights: dict, k_max: int) -> dict:
    """Find coalitions using a greedy algorithm that maximizes or minimizes a value function.

    Parameters
    ----------
    N : list
        List of players.
    e_weights : dict
        Dictionary mapping frozensets of players to their associated weights.
    k_max : int
        Maximum coalition size to consider.

    Returns:
    -------
    dict
        Dictionary mapping coalition size to a tuple of (max coalition, min coalition).
    """
    result = {}
    for length in range(1, len(N) + 1):
        s_max = greedy_extreme_max(length, N, e_weights, k_max)
        s_min = greedy_extreme_min(length, N, e_weights, k_max)
        result[length] = (s_max, s_min)
    return result
