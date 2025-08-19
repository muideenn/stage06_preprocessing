"""
Reusable data cleaning utilities for Stage 6: Data Preprocessing.

Functions
---------
fill_missing_median(df, columns=None)
    Fill NaNs in numeric columns with the median of each column.
drop_missing(df, how="any", thresh=None, subset=None)
    Drop rows with missing values, with flexible options.
normalize_data(df, columns=None, method="standard")
    Scale/normalize selected numeric columns using standardization or min-max.

Notes
-----
- Functions return a NEW dataframe; originals are not mutated (copy-on-write).
- For reproducibility, each function logs basic actions via print().
"""
from __future__ import annotations
from typing import Iterable, List, Optional, Tuple
import pandas as pd
import numpy as np


def _coerce_columns(df: pd.DataFrame, columns: Optional[Iterable[str]]) -> List[str]:
    if columns is None:
        # default to numeric columns only
        return df.select_dtypes(include=[np.number]).columns.tolist()
    return [c for c in columns if c in df.columns]


def fill_missing_median(df: pd.DataFrame, columns: Optional[Iterable[str]] = None) -> pd.DataFrame:
    """
    Fill missing values in the specified numeric columns with each column's median.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    columns : iterable of str or None, optional
        Columns to fill. Defaults to all numeric columns.

    Returns
    -------
    pd.DataFrame
        New dataframe with NaNs filled.
    """
    cols = _coerce_columns(df, columns)
    out = df.copy()
    medians = out[cols].median(numeric_only=True)
    out[cols] = out[cols].fillna(medians)
    print(f"[fill_missing_median] Filled columns: {cols} with medians: {medians.to_dict()}")
    return out


def drop_missing(
    df: pd.DataFrame,
    how: str = "any",
    thresh: Optional[int] = None,
    subset: Optional[Iterable[str]] = None
) -> pd.DataFrame:
    """
    Drop rows that contain missing values.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    how : {"any", "all"}, default "any"
        - "any": drop a row if ANY of the subset columns is NaN
        - "all": drop a row if ALL of the subset columns are NaN
    thresh : int, optional
        Minimum number of non-NA values required to keep a row.
        If provided, `how` is ignored.
    subset : iterable of str, optional
        Columns to consider. If None, all columns are considered.

    Returns
    -------
    pd.DataFrame
        New dataframe with rows dropped according to the rule.
    """
    out = df.copy()
    before = len(out)
    out = out.dropna(how=how, thresh=thresh, subset=subset)
    after = len(out)
    print(f"[drop_missing] Dropped {before - after} rows (from {before} to {after}). "
          f"Params -> how={how}, thresh={thresh}, subset={list(subset) if subset else None}")
    return out


def normalize_data(
    df: pd.DataFrame,
    columns: Optional[Iterable[str]] = None,
    method: str = "standard"
) -> Tuple[pd.DataFrame, dict]:
    """
    Normalize/scale numeric data.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    columns : iterable of str or None, optional
        Columns to scale. Defaults to all numeric columns.
    method : {"standard", "minmax"}, default "standard"
        - "standard": z-score (x - mean) / std
        - "minmax": (x - min) / (max - min)

    Returns
    -------
    (pd.DataFrame, dict)
        - New dataframe with normalized columns.
        - Fitted parameters per column (means/stds or mins/maxs) for reproducibility.
    """
    cols = _coerce_columns(df, columns)
    out = df.copy()
    params = {}

    if method not in {"standard", "minmax"}:
        raise ValueError('method must be one of {"standard", "minmax"}')

    for c in cols:
        x = out[c].astype(float)
        if method == "standard":
            mean = float(x.mean())
            std = float(x.std(ddof=0))
            if std == 0:
                # Avoid divide by zero; leave column as zeros
                out[c] = 0.0
                params[c] = {"method": "standard", "mean": mean, "std": std}
            else:
                out[c] = (x - mean) / std
                params[c] = {"method": "standard", "mean": mean, "std": std}
        else:  # minmax
            mn = float(x.min())
            mx = float(x.max())
            rng = mx - mn
            if rng == 0:
                out[c] = 0.0
                params[c] = {"method": "minmax", "min": mn, "max": mx}
            else:
                out[c] = (x - mn) / rng
                params[c] = {"method": "minmax", "min": mn, "max": mx}

    print(f"[normalize_data] Normalized columns: {cols} using method={method}")
    return out, params
