"""
ADCO_tree
---------
Single CART decision tree: training and rule extraction.

Code implementation was assisted by Claude (Anthropic, 2026) under the author's direction. All architectural decisions, scientific choices (attribute grouping strategy, confidence degree definition, aggregation methods), code and algorithmic corrections and validations were made by the author.
"""

from sklearn.tree import DecisionTreeClassifier, _tree
from utils import simplify_conditions_cart


def train_tree(df_train, features, target_col, max_depth=None, random_state=42):
    """
    Train a single CART decision tree on the given feature columns.

    Parameters
    ----------
    df_train     : pd.DataFrame — training set
    features     : list of str  — feature columns to use
    target_col   : str          — target column name
    max_depth    : int or None  — maximum tree depth (None = unlimited)
    random_state : int

    Returns
    -------
    tree_dict : dict  {"group": features, "tree": DecisionTreeClassifier}
    """
    X = df_train[features]
    y = df_train[target_col]

    clf = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)
    clf.fit(X, y)

    return {"group": features, "tree": clf}


def extract_rules(tree_dict):
    """
    Extract classification rules from a trained decision tree.

    Traverses the tree recursively. Each leaf produces one rule.
    Conditions are simplified via simplify_conditions_cart().
    Contradictory rules are discarded.

    Confidence = proportion of training samples in the leaf
                 that belong to the predicted class (leaf purity).
    Support    = total number of training samples reaching the leaf.

    Parameters
    ----------
    tree_dict : dict  {"group": list of str, "tree": DecisionTreeClassifier}

    Returns
    -------
    rules : list of dicts
        {
            "conditions": [{"feature": str, "op": str, "threshold": float}, ...],
            "class":      str or int,
            "confidence": float,
            "support":    int
        }
    """
    clf           = tree_dict["tree"]
    feature_names = tree_dict["group"]
    tree          = clf.tree_
    class_names   = clf.classes_

    rules = []

    def recurse(node, conditions):
        if tree.feature[node] == _tree.TREE_UNDEFINED:
            # ── Leaf node ────────────────────────────────────────────────
            values         = tree.value[node][0]       # raw class counts
            total          = values.sum()
            best_class_idx = values.argmax()
            # Confidence = proportion of majority class in the leaf
            confidence = float(values[best_class_idx] / total) if total > 0 else 0.0
            # Support = true sample count via n_node_samples
            support    = int(tree.n_node_samples[node])

            simplified = simplify_conditions_cart(list(conditions))
            if simplified is None:
                return  # Contradictory rule → skip

            rules.append({
                "conditions": simplified,
                "class":      class_names[best_class_idx],
                "confidence": round(confidence, 4),
                "support":    support
            })
        else:
            # ── Decision node ─────────────────────────────────────────────
            feat   = feature_names[tree.feature[node]]
            thresh = round(float(tree.threshold[node]), 4)

            # Use sklearn's actual child indices (not node*2+1)
            left_child  = tree.children_left[node]
            right_child = tree.children_right[node]

            recurse(left_child,
                    conditions + [{"feature": feat, "op": "<=", "threshold": thresh}])
            recurse(right_child,
                    conditions + [{"feature": feat, "op": ">",  "threshold": thresh}])

    recurse(0, [])
    return rules
