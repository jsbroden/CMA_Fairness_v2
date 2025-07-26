# put into universe.py????

from __future__ import annotations
from typing import Tuple, List, Optional
import pandas as pd
from sklearn.model_selection import train_test_split


def _build_strat_vector(
    df: pd.DataFrame,
    stratify_option: str,
    target_col: str,
    protected_col: Optional[str] = None,
) -> Optional[pd.Series]:
    """Return the column to pass to `stratify=` according to the option requested."""
    if stratify_option == "none":
        return None
    if stratify_option == "target":
        return df[target_col]
    if stratify_option == "protected-attribute":
        if protected_col is None:
            raise ValueError("`protected_col` must be given for this stratification option.")
        return df[protected_col]
    if stratify_option == "both":
        if protected_col is None:
            raise ValueError("`protected_col` must be given for this stratification option.")
        # cast to string so that the cartesian product becomes a single categorical column
        return df[protected_col].astype("category").astype(str) + "-" + df[target_col].astype("category").astype(str)

    raise ValueError(f"Unknown stratification option: {stratify_option}")


def longitudinal_split(
    data: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    year_col: str,
    protected_col: Optional[str] = None,
    *,
    stratify_option: str = "none",  # "none", "target", "protected-attribute", "both"
    test_size: float = 0.20,
    calib_size: float = 0.10,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.Series,
           pd.DataFrame, pd.Series,
           pd.DataFrame, pd.Series]:
    """
    Split a *longitudinal* data set into train‑calib‑test, year‑by‑year, with optional stratification.

    Parameters
    ----------
    data            : tidy DataFrame that still contains all columns, incl. year, target, protected.
    feature_cols    : list of column names that form X.
    target_col      : column with the prediction target.
    year_col        : column that identifies calendar year.
    protected_col   : optional column for the sensitive / protected attribute.
    stratify_option : "none" | "target" | "protected-attribute" | "both"
    test_size       : share of each year that goes into the final test set (0.2 → 20 %).
    calib_size      : share (of *the remaining data after test has been removed*)
                      that goes into the calibration set.
                      NB: calibration share is relative to the *non‑test* remainder.
    random_state    : seed for deterministic reproducibility.

    Returns
    -------
    X_train, y_train, X_calib, y_calib, X_test, y_test
    """
    train_chunks, calib_chunks, test_chunks = [], [], []

    for yr, yearly_df in data.groupby(year_col, sort=True):
        # ---------------- 1st split: temp (train+calib) vs. test ----------------
        strat_all = _build_strat_vector(
            yearly_df, stratify_option, target_col, protected_col
        )
        temp_df, test_df = train_test_split(
            yearly_df,
            test_size=test_size,
            stratify=strat_all,
            random_state=random_state,
        )

        # ---------------- 2nd split: train vs. calibration ----------------
        #   The calibration share is relative to what is *left* after the test has been peeled off.
        calib_fraction = calib_size / (1.0 - test_size)
        strat_temp = _build_strat_vector(
            temp_df, stratify_option, target_col, protected_col
        )
        train_df, calib_df = train_test_split(
            temp_df,
            test_size=calib_fraction,
            stratify=strat_temp,
            random_state=random_state,
        )

        train_chunks.append(train_df)
        calib_chunks.append(calib_df)
        test_chunks.append(test_df)

    # Shuffle once more so that years are interleaved; keeps reproducibility via random_state
    train_df = pd.concat(train_chunks).sample(
        frac=1.0, random_state=random_state
    ).reset_index(drop=True)
    calib_df = pd.concat(calib_chunks).sample(
        frac=1.0, random_state=random_state
    ).reset_index(drop=True)
    test_df = pd.concat(test_chunks).sample(
        frac=1.0, random_state=random_state
    ).reset_index(drop=True)

    # Split into X / y
    X_train, y_train = train_df[feature_cols], train_df[target_col]
    X_calib, y_calib = calib_df[feature_cols], calib_df[target_col]
    X_test,  y_test  = test_df[feature_cols],  test_df[target_col]

    return X_train, y_train, X_calib, y_calib, X_test, y_test
