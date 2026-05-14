"""
CO.py
-----
Attribute grouping using Pearson linear correlation (SIFCO method).
Groups attributes whose absolute correlation exceeds the given threshold.
"""

import numpy as np
from utils import get_numeric_features


def attribute_grouping(df, target_col, threshold=0.7):
    """
    Group correlated attributes using Pearson correlation.
    The target column is excluded from the computation.

    Algorithm (connected components):
        - Build the absolute Pearson correlation matrix
        - For each unvisited attribute, expand a group by adding
          any attribute correlated above the threshold (BFS/DFS)

    Parameters
    ----------
    df         : pd.DataFrame — training dataset
    target_col : str          — name of the target column (excluded)
    threshold  : float        — correlation threshold in [0, 1]

    Returns
    -------
    groups : list of lists   — each sublist is a group of attribute names
    corr   : pd.DataFrame    — absolute Pearson correlation matrix
    """
    # Keep only numeric columns, exclude target
    feature_cols = get_numeric_features(df, target_col)
    df_num = df[feature_cols]

    # Absolute Pearson correlation matrix
    corr = df_num.corr(method="pearson").abs()
    features = corr.columns.tolist()

    visited = set()
    groups = []

    for f in features:
        if f in visited:
            continue
        # Start a new group from feature f
        group = {f}
        stack = [f]
        while stack:
            current = stack.pop()
            for other in features:
                if other not in group and corr.loc[current, other] >= threshold:
                    group.add(other)
                    stack.append(other)
        visited |= group
        groups.append(sorted(group))

    return groups, corr
