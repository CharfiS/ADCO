"""
ADCO_confidence
---------------
Ensemble prediction using confidence-based aggregation strategies.

Code implementation was assisted by Claude (Anthropic, 2026) under the author's direction. All architectural decisions, scientific choices (attribute grouping strategy, confidence degree definition, aggregation methods), code and algorithmic corrections and validations were made by the author.
"""

from ADCO_aggregation import aggregate


def rule_matches(rule, observation):
    """
    Check whether a rule's conditions are all satisfied by an observation.

    Parameters
    ----------
    rule        : dict          — a rule from the global rule base GR
    observation : pd.Series     — attribute values of one sample

    Returns
    -------
    bool : True if all conditions are satisfied
    """
    for condition in rule["conditions"]:
        feat   = condition["feature"]
        op     = condition["op"]
        thresh = condition["threshold"]
        value  = observation[feat]

        if op == "<=" and not (value <= thresh):
            return False
        if op == ">"  and not (value >  thresh):
            return False

    return True


def predict_one_ensemble(GR, observation, method="sum"):
    """
    Predict the class of a single observation.

    For each group:
        - Find all matching rules
        - Keep the one with the highest confidence
    Then aggregate across groups using the chosen method.

    Parameters
    ----------
    GR          : list of dicts — global rule base from ADCO_EM.build_global_rules()
    observation : pd.Series
    method      : str           — aggregation method ("sum", "max", "product", "s_norm")

    Returns
    -------
    predicted_class  : str or int, or None if no rule matches
    final_scores     : dict  {class: aggregated_score}
    activated_rules  : list of dicts — best rule per group
    """
    # Group rules by source group
    groups = {}
    for rule in GR:
        g = rule["group"]
        if g not in groups:
            groups[g] = []
        groups[g].append(rule)

    # For each group, find the best matching rule
    scores_per_class = {}   # {class: [conf_G1, conf_G2, ...]}
    activated_rules  = []

    for group_name, rules in groups.items():
        matching = [r for r in rules if rule_matches(r, observation)]
        if not matching:
            continue

        # Keep the rule with the highest confidence
        best_rule = max(matching, key=lambda r: r["confidence"])
        activated_rules.append({**best_rule, "group": group_name})

        # Accumulate confidence per class
        cls = best_rule["class"]
        if cls not in scores_per_class:
            scores_per_class[cls] = []
        scores_per_class[cls].append(best_rule["confidence"])

    if not scores_per_class:
        return None, {}, []

    # Aggregate scores across groups
    predicted_class, final_scores = aggregate(scores_per_class, method=method)

    # Tie-breaking: if two classes have the same score,
    # keep the one with the highest total support (Basma, 2012)
    max_score = max(final_scores.values())
    tied = [cls for cls, score in final_scores.items() if score == max_score]
    if len(tied) > 1:
        # Sum support across activated rules for each tied class
        support_per_class = {}
        for rule in activated_rules:
            cls = rule["class"]
            if cls in tied:
                support_per_class[cls] = (
                    support_per_class.get(cls, 0) + rule["support"]
                )
        predicted_class = max(tied, key=lambda c: support_per_class.get(c, 0))

    return predicted_class, final_scores, activated_rules


def predict_ensemble(GR, df_test, target_col, method="sum"):
    """
    Predict classes for all samples in the test set.

    Parameters
    ----------
    GR         : list of dicts — global rule base
    df_test    : pd.DataFrame
    target_col : str
    method     : str           — aggregation method

    Returns
    -------
    results : list of dicts
        {
            "true_class":       str,
            "predicted_class":  str or None,
            "final_scores":     dict,
            "activated_rules":  list,
            "correct":          bool
        }
    """
    results = []
    for _, row in df_test.iterrows():
        predicted, scores, activated = predict_one_ensemble(GR, row, method=method)
        true_class = row[target_col]
        results.append({
            "true_class":      true_class,
            "predicted_class": predicted,
            "final_scores":    scores,
            "activated_rules": activated,
            "correct":         predicted == true_class
        })
    return results


def accuracy_ensemble(results):
    """
    Compute classification accuracy.

    Parameters
    ----------
    results : list of dicts — output of predict_ensemble()

    Returns
    -------
    float : accuracy in [0, 1]
    """
    if not results:
        return 0.0
    correct = sum(1 for r in results if r["correct"])
    return round(correct / len(results), 4)
