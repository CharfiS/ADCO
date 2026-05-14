"""
data_loader.py
--------------
Handles dataset loading and train/test splitting.
"""

import pandas as pd
from sklearn import datasets as skdatasets
from sklearn.model_selection import train_test_split


# Known datasets available via sklearn
SKLEARN_DATASETS = {
    "iris":          skdatasets.load_iris,
    "wine":          skdatasets.load_wine,
    "breast_cancer": skdatasets.load_breast_cancer,
}


def load_dataset(source):
    """
    Load a dataset from a known sklearn name or a CSV file path.

    Parameters
    ----------
    source : str
        Either a known dataset name ("iris", "wine", "breast_cancer")
        or a path to a CSV file.

    Returns
    -------
    df : pd.DataFrame
    """
    key = source.lower().strip()
    if key in SKLEARN_DATASETS:
        bunch = SKLEARN_DATASETS[key](as_frame=True)
        df = bunch.frame.copy()
        # Replace numeric target values with class names
        if hasattr(bunch, "target_names"):
            df["target"] = [bunch.target_names[i] for i in bunch.target]
        return df
    else:
        return pd.read_csv(source)


def split_dataset(df, target_col, test_size=0.2, random_state=42):
    """
    Split the dataset into train and test sets.
    Uses stratification to preserve class proportions.

    Parameters
    ----------
    df           : pd.DataFrame
    target_col   : str   — name of the target column
    test_size    : float — proportion of the test set (default 0.2)
    random_state : int   — random seed for reproducibility

    Returns
    -------
    df_train, df_test : pd.DataFrame
    """
    df_train, df_test = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df[target_col]
    )
    return df_train.reset_index(drop=True), df_test.reset_index(drop=True)
