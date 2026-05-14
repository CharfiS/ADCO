"""
ADCO_EM.py
----------
Ensemble Method based on Decision Trees (CART).
Trains one decision tree per attribute group (via ADCO_tree)
and builds the global rule base GR.
"""

from ADCO_tree import train_tree, extract_rules


def train_trees(df_train, groups, target_col, max_depth=None, random_state=42):
    """
    Train one CART decision tree per attribute group.

    Parameters
    ----------
    df_train     : pd.DataFrame — training set only
    groups       : list of lists — attribute groups from CO.attribute_grouping()
    target_col   : str
    max_depth    : int or None  — maximum tree depth (None = unlimited)
    random_state : int

    Returns
    -------
    trees : list of dicts  {"group": [...], "tree": DecisionTreeClassifier}
    """
    trees = []
    for group in groups:
        trees.append(train_tree(df_train, group, target_col, max_depth, random_state))
    return trees


def build_global_rules(trees):
    """
    Build the global rule base GR by merging rules from all groups.
    Each rule keeps a reference to its source group.

    Parameters
    ----------
    trees : list of dicts  {"group": [...], "tree": DecisionTreeClassifier}

    Returns
    -------
    GR : list of dicts
        {
            "group":      str,   e.g. "G1"
            "conditions": [...],
            "class":      str,
            "confidence": float,
            "support":    int
        }
    """
    GR = []
    for i, tree_dict in enumerate(trees, 1):
        for rule in extract_rules(tree_dict):
            rule["group"] = f"G{i}"
            GR.append(rule)
    return GR
