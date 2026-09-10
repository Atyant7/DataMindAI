"""
Feature engineering utilities for DataMindAI.

This module provides deterministic DataFrame transformations for creating new features:
datetime decomposition, interaction terms, polynomial features, and numerical binning.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def extract_datetime_features(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    *,
    drop_original: bool = False,
) -> pd.DataFrame:
    """
    Extract calendar and cyclical features from datetime columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    columns : list of str, optional
        Datetime columns to decompose. If None, auto-detected datetime columns are used.
    drop_original : bool, default False
        Whether to drop the original datetime column after extraction.

    Returns
    -------
    pd.DataFrame
        DataFrame with new datetime feature columns.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")

    result = df.copy()

    if columns is None:
        target_cols = [
            col for col in result.columns
            if pd.api.types.is_datetime64_any_dtype(result[col])
        ]
        # Also attempt to find string columns that look like dates if none found
        if not target_cols:
            for col in result.select_dtypes(include="object").columns:
                try:
                    converted = pd.to_datetime(result[col], errors="coerce")
                    if converted.notna().sum() / len(result) > 0.8:
                        result[col] = converted
                        target_cols.append(col)
                except Exception:
                    continue
    else:
        target_cols = [col for col in columns if col in result.columns]

    for col in target_cols:
        dt_series = pd.to_datetime(result[col], errors="coerce")

        result[f"{col}_year"] = dt_series.dt.year
        result[f"{col}_month"] = dt_series.dt.month
        result[f"{col}_day"] = dt_series.dt.day
        result[f"{col}_dayofweek"] = dt_series.dt.dayofweek
        result[f"{col}_is_weekend"] = dt_series.dt.dayofweek.isin([5, 6]).astype(int)

        # Only extract hour/minute if they contain non-zero variability
        if dt_series.dt.hour.nunique() > 1:
            result[f"{col}_hour"] = dt_series.dt.hour

        if drop_original:
            result = result.drop(columns=[col])

    return result


def create_interaction_features(
    df: pd.DataFrame,
    col1: str,
    col2: str,
    *,
    operation: str = "multiply",
    new_col_name: str | None = None,
) -> pd.DataFrame:
    """
    Create an interaction feature between two numerical columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    col1 : str
        First column name.
    col2 : str
        Second column name.
    operation : str, default 'multiply'
        'multiply', 'ratio', 'add', or 'subtract'.
    new_col_name : str, optional
        Custom name for the new interaction column.

    Returns
    -------
    pd.DataFrame
        DataFrame with new interaction column.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")

    if col1 not in df.columns:
        raise ValueError(f"Column '{col1}' does not exist.")
    if col2 not in df.columns:
        raise ValueError(f"Column '{col2}' does not exist.")

    result = df.copy()

    if operation == "multiply":
        name = new_col_name or f"{col1}_x_{col2}"
        result[name] = result[col1] * result[col2]
    elif operation == "ratio":
        name = new_col_name or f"{col1}_per_{col2}"
        # Prevent division by zero
        denom = result[col2].replace(0, np.nan)
        result[name] = result[col1] / denom
    elif operation == "add":
        name = new_col_name or f"{col1}_plus_{col2}"
        result[name] = result[col1] + result[col2]
    elif operation == "subtract":
        name = new_col_name or f"{col1}_minus_{col2}"
        result[name] = result[col1] - result[col2]
    else:
        raise ValueError(
            f"Unsupported interaction operation: '{operation}'. "
            "Choose 'multiply', 'ratio', 'add', or 'subtract'."
        )

    return result


def create_polynomial_features(
    df: pd.DataFrame,
    columns: list[str],
    *,
    degree: int = 2,
) -> pd.DataFrame:
    """
    Create polynomial power terms for specified numerical columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    columns : list of str
        Numerical columns to raise to polynomial powers.
    degree : int, default 2
        Maximum polynomial degree (e.g. 2 for squared terms).

    Returns
    -------
    pd.DataFrame
        DataFrame with added polynomial columns.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")

    result = df.copy()

    for col in columns:
        if col not in result.columns:
            continue
        for d in range(2, degree + 1):
            result[f"{col}_pow_{d}"] = result[col] ** d

    return result


def bin_numerical_feature(
    df: pd.DataFrame,
    column: str,
    *,
    n_bins: int = 5,
    strategy: str = "quantile",
    labels: list[str] | None = None,
    new_col_name: str | None = None,
) -> pd.DataFrame:
    """
    Discretize a continuous numerical feature into discrete bins.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    column : str
        Column to bin.
    n_bins : int, default 5
        Number of bins.
    strategy : str, default 'quantile'
        'quantile' (equal frequency) or 'uniform' (equal width).
    labels : list of str, optional
        Custom bin names.
    new_col_name : str, optional
        Name of the created binned column.

    Returns
    -------
    pd.DataFrame
        DataFrame with binned column.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")

    if column not in df.columns:
        raise ValueError(f"Column '{column}' does not exist.")

    result = df.copy()
    target_name = new_col_name or f"{column}_binned"

    if strategy == "quantile":
        result[target_name] = pd.qcut(
            result[column],
            q=n_bins,
            labels=labels,
            duplicates="drop",
        )
    elif strategy == "uniform":
        result[target_name] = pd.cut(
            result[column],
            bins=n_bins,
            labels=labels,
        )
    else:
        raise ValueError(
            f"Unsupported binning strategy: '{strategy}'. Choose 'quantile' or 'uniform'."
        )

    return result
