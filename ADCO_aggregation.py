"""
ADCO_aggregation.py
-------------------
Aggregation functions for combining confidence scores from multiple groups.

Each function takes a dict of {class: [confidence_from_G1, confidence_from_G2, ...]}
and returns the predicted class.

Available methods:
    - sum_rule     : sum of confidences per class (Kittler et al., 1998)
    - max_rule     : maximum confidence per class
    - product_rule : product of confidences per class
    - s_norm      : probabilistic s-norm (t-conorm) (a + b - a*b), stays in [0, 1]
                     (Dubois & Prade, 1985)
"""

import math


def _aggregate(scores_per_class, combine_fn):
    """
    Generic aggregation helper.

    Parameters
    ----------
    scores_per_class : dict  {class: [conf_G1, conf_G2, ...]}
    combine_fn       : function([float]) → float

    Returns
    -------
    predicted_class : str or int
    final_scores    : dict {class: float}
    """
    final_scores = {
        cls: combine_fn(confs)
        for cls, confs in scores_per_class.items()
    }
    predicted_class = max(final_scores, key=final_scores.get)
    return predicted_class, final_scores


def sum_rule(scores_per_class):
    """
    Sum Rule: sum of confidences per class.
    Most commonly used combination rule (Kittler et al., 1998).
    Note: result may exceed 1 — not a probability, but a relative score.

    Parameters
    ----------
    scores_per_class : dict  {class: [conf_G1, conf_G2, ...]}

    Returns
    -------
    predicted_class : str or int
    final_scores    : dict {class: float}

    Example
    -------
    {"versicolor": [0.90, 0.75], "virginica": [0.95]}
    → versicolor: 1.65, virginica: 0.95  → versicolor wins
    """
    return _aggregate(scores_per_class, sum)


def max_rule(scores_per_class):
    """
    Max Rule: maximum confidence per class.
    The most confident group decides.

    Parameters
    ----------
    scores_per_class : dict  {class: [conf_G1, conf_G2, ...]}

    Returns
    -------
    predicted_class : str or int
    final_scores    : dict {class: float}

    Example
    -------
    {"versicolor": [0.90, 0.75], "virginica": [0.95]}
    → versicolor: 0.90, virginica: 0.95  → virginica wins
    """
    return _aggregate(scores_per_class, max)


def product_rule(scores_per_class):
    """
    Product Rule: product of confidences per class.
    Stays in [0, 1] but sensitive to low confidence values (near 0).

    Parameters
    ----------
    scores_per_class : dict  {class: [conf_G1, conf_G2, ...]}

    Returns
    -------
    predicted_class : str or int
    final_scores    : dict {class: float}

    Example
    -------
    {"versicolor": [0.90, 0.75], "virginica": [0.95]}
    → versicolor: 0.675, virginica: 0.95  → virginica wins
    """
    return _aggregate(scores_per_class, lambda confs: round(math.prod(confs), 4))


def s_norm(scores_per_class):
    """
    Probabilistic T-conorm (S-norm): a + b - a*b
    Applied iteratively for more than 2 groups.
    Always stays in [0, 1] (Dubois & Prade, 1985).

    S(a, b) = a + b - a*b
    For 3 values: S(S(a, b), c)

    Parameters
    ----------
    scores_per_class : dict  {class: [conf_G1, conf_G2, ...]}

    Returns
    -------
    predicted_class : str or int
    final_scores    : dict {class: float}

    Example
    -------
    {"versicolor": [0.90, 0.75], "virginica": [0.95]}
    → versicolor: 0.90 + 0.75 - 0.90*0.75 = 0.975
    → virginica:  0.95
    → versicolor wins
    """
    def s_norm(confs):
        result = confs[0]
        for c in confs[1:]:
            result = result + c - result * c
        return round(result, 4)

    return _aggregate(scores_per_class, s_norm)


# Registry of available aggregation methods
AGGREGATION_METHODS = {
    "sum":       sum_rule,
    "max":       max_rule,
    "product":   product_rule,
    "s_norm":  s_norm,
}


def aggregate(scores_per_class, method="sum"):
    """
    Apply the chosen aggregation method.

    Parameters
    ----------
    scores_per_class : dict  {class: [conf_G1, conf_G2, ...]}
    method           : str   — one of "sum", "max", "product", "s_norm"

    Returns
    -------
    predicted_class : str or int
    final_scores    : dict {class: float}

    Raises
    ------
    ValueError if method is unknown
    """
    if method not in AGGREGATION_METHODS:
        raise ValueError(
            f"Unknown aggregation method '{method}'. "
            f"Available: {list(AGGREGATION_METHODS.keys())}"
        )
    return AGGREGATION_METHODS[method](scores_per_class)
