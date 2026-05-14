"""
ADCO_one.py
-----------
Baseline method: trains a single CART decision tree on all attributes.
Uses sklearn's built-in predict() for classification.
Used to compare performance against the ADCO ensemble approach.
"""

from utils import get_numeric_features
from ADCO_tree import train_tree, extract_rules

# Alias for clarity
extract_rules_single = extract_rules


def train(df_train, target_col, max_depth=None, random_state=42):
    """
    Train a single decision tree on all numeric attributes.

    Parameters
    ----------
    df_train     : pd.DataFrame — training set
    target_col   : str
    max_depth    : int or None
    random_state : int

    Returns
    -------
    tree_dict : dict  {"group": list of feature names, "tree": DecisionTreeClassifier}
    """
    feature_cols = get_numeric_features(df_train, target_col)
    return train_tree(df_train, feature_cols, target_col, max_depth, random_state)


def predict_single(tree_dict, df_test, target_col):
    """
    Predict classes for all samples in the test set using sklearn's predict().
    No need for manual rule matching since there is only one tree.

    Parameters
    ----------
    tree_dict  : dict          — output of train()
    df_test    : pd.DataFrame  — test set
    target_col : str

    Returns
    -------
    results : list of dicts
        {
            "true_class":      str,
            "predicted_class": str,
            "correct":         bool
        }
    """
    clf    = tree_dict["tree"]
    X_test = df_test[tree_dict["group"]]
    y_pred = clf.predict(X_test)
    y_true = df_test[target_col].values

    return [
        {
            "true_class":      true,
            "predicted_class": pred,
            "correct":         true == pred
        }
        for true, pred in zip(y_true, y_pred)
    ]


def accuracy_single(results):
    """
    Compute classification accuracy.

    Parameters
    ----------
    results : list of dicts — output of predict_single()

    Returns
    -------
    float : accuracy in [0, 1]
    """
    if not results:
        return 0.0
    correct = sum(1 for r in results if r["correct"])
    return round(correct / len(results), 4)
