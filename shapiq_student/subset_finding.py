"""Functions for filtering InteractionValues objects by interaction order.

This module provides:
- subset_finding: filters an InteractionValues object to include only interactions up to a specified maximum size.
"""

from __future__ import annotations

from shapiq import InteractionValues


def subset_finding(interaction_values: InteractionValues, max_size: int) -> InteractionValues:
    """Filters an InteractionValues object to include only interactions up to max_size.

    Parameters
    ----------
    interaction_values : InteractionValues
        The original interaction values.
    max_size : int
        Maximum size/order of interactions to retain.

    Returns:
    -------
    InteractionValues
        A new InteractionValues object containing only interactions of order ≤ max_size.
    """
    filtered_values = {
        T: value
        for T, value in interaction_values.items()
        if len(T) <= max_size
    }

    return InteractionValues(
        values=filtered_values,
        index=interaction_values.index,
        n_players=interaction_values.n_players,
        max_order=max_size
    )
