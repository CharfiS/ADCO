"""
utils
-----
Shared algorithmic utilities: feature selection, condition simplification.

Code implementation was assisted by Claude (Anthropic, 2026) under the author's direction. All architectural decisions, scientific choices (attribute grouping strategy, confidence degree definition, aggregation methods), code and algorithmic corrections and validations were made by the author.
"""

import numpy as np


def get_numeric_features(df, target_col):
    """
    Return the list of numeric feature column names,
    excluding the target column.

    Parameters
    ----------
    df         : pd.DataFrame
    target_col : str — name of the target column

    Returns
    -------
    feature_cols : list of str

    Example
    -------
    On Iris: ["sepal length (cm)", "sepal width (cm)",
              "petal length (cm)", "petal width (cm)"]
    """
    return (
        df.drop(columns=[target_col])
          .select_dtypes(include=[np.number])
          .columns.tolist()
    )


def simplify_conditions(conditions):
    """
    Generic simplification of classification rule conditions.
    Works for any classifier producing conditions of the form:
        feature  op  threshold
    where op is one of: ">", ">=", "<", "<="

    For each attribute:
        - Lower bounds (>, >=) : keep the largest threshold
            * If two conditions have the same threshold,
              ">" is more restrictive than ">="
        - Upper bounds (<, <=) : keep the smallest threshold
            * If two conditions have the same threshold,
              "<" is more restrictive than "<="

    Contradiction detection:
        - lower > upper                        → always impossible
        - lower == upper AND any op is strict  → impossible
          e.g. x > 5 AND x <= 5  → impossible
               x >= 5 AND x < 5  → impossible
               x > 5  AND x < 5  → impossible
        - lower == upper AND both inclusive    → valid (x = 5)
          e.g. x >= 5 AND x <= 5 → valid

    Parameters
    ----------
    conditions : list of dicts
        {"feature": str, "op": str, "threshold": float}
        op must be one of ">", ">=", "<", "<="

    Returns
    -------
    simplified : list of dicts, or None if contradiction detected

    Examples
    --------
    x > 2, x > 5, x <= 10, x <= 8  →  x > 5, x <= 8
    x > 5, x < 5                    →  None (contradiction)
    x >= 5, x <= 5                  →  x >= 5, x <= 5 (x = 5, valid)
    """
    lower     = {}  # feature → largest lower bound value
    lower_op  = {}  # feature → operator of the winning lower bound
    upper     = {}  # feature → smallest upper bound value
    upper_op  = {}  # feature → operator of the winning upper bound

    # ── Extract most restrictive bounds ───────────────────────────────────
    for c in conditions:
        feat   = c["feature"]
        op     = c["op"]
        thresh = c["threshold"]

        if op in [">", ">="]:
            current = lower.get(feat, float("-inf"))
            if thresh > current:
                lower[feat]    = thresh
                lower_op[feat] = op
            elif thresh == current:
                # Same threshold: ">" is stricter than ">="
                if op == ">":
                    lower_op[feat] = ">"

        elif op in ["<", "<="]:
            current = upper.get(feat, float("inf"))
            if thresh < current:
                upper[feat]    = thresh
                upper_op[feat] = op
            elif thresh == current:
                # Same threshold: "<" is stricter than "<="
                if op == "<":
                    upper_op[feat] = "<"

    # ── Detect contradictions ─────────────────────────────────────────────
    for feat in set(lower.keys()) & set(upper.keys()):
        lo    = lower[feat]
        up    = upper[feat]
        lo_op = lower_op[feat]
        up_op = upper_op[feat]

        if lo > up:
            return None  # Always impossible (e.g. x > 8 AND x <= 5)

        if lo == up:
            # Valid only if both bounds are inclusive (>= AND <=)
            if lo_op == ">" or up_op == "<":
                return None

    # ── Rebuild in original feature order ────────────────────────────────
    ordered_features = []
    for c in conditions:
        if c["feature"] not in ordered_features:
            ordered_features.append(c["feature"])

    simplified = []
    for feat in ordered_features:
        if feat in lower:
            simplified.append({
                "feature":   feat,
                "op":        lower_op[feat],
                "threshold": round(lower[feat], 4)
            })
        if feat in upper:
            simplified.append({
                "feature":   feat,
                "op":        upper_op[feat],
                "threshold": round(upper[feat], 4)
            })

    return simplified


def simplify_conditions_cart(conditions):
    """
    Simplify conditions for CART (sklearn) rules.

    Calls the generic simplify_conditions() function then validates that
    only ">" and "<=" operators are present, as CART only produces these.

    Parameters
    ----------
    conditions : list of dicts
        {"feature": str, "op": str, "threshold": float}

    Returns
    -------
    simplified : list of dicts with only ">" and "<=" operators,
                 or None if contradiction detected

    Raises
    ------
    ValueError if a non-CART operator (">=" or "<") is found after simplification
    """
    simplified = simplify_conditions(conditions)
    if simplified is None:
        return None

    # Validate that only CART operators are present
    for c in simplified:
        if c["op"] not in [">", "<="]:
            raise ValueError(
                f"Unexpected operator '{c['op']}' after simplification. "
                f"CART only produces '>' and '<='."
            )
    return simplified
