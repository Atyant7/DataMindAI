"""
Backend data preprocessing and cleaning utilities for DataMindAI.

This module provides deterministic DataFrame transformations for data cleaning,
missing value handling, outlier mitigation, categorical encoding, and numerical scaling.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def clean_dataset(
    df: pd.DataFrame,
    *,
    drop_duplicates: bool = True,
    strip_strings: bool = True,
    clean_column_names: bool = False,
) -> pd.DataFrame:
    """
    Perform general cleaning on a dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    drop_duplicates : bool, default True
        Whether to drop duplicate rows.
    strip_strings : bool, default True
        Whether to strip leading/trailing whitespace from string entries.
    clean_column_names : bool, default False
        Whether to normalize column names into lowercase snake_case.

    Returns
    -------
    pd.DataFrame
        Cleaned copy of the DataFrame.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")

    cleaned = df.copy()

    if clean_column_names:
        cleaned.columns = [
            str(col).strip().lower().replace(" ", "_").replace("-", "_")
            for col in cleaned.columns
        ]

    if strip_strings:
        string_cols = cleaned.select_dtypes(include=["object", "string"]).columns
        for col in string_cols:
            cleaned[col] = cleaned[col].apply(
                lambda val: val.strip() if isinstance(val, str) else val
            )

    if drop_duplicates:
        cleaned = cleaned.drop_duplicates().reset_index(drop=True)

    return cleaned


def handle_missing_values(
    df: pd.DataFrame,
    *,
    numeric_strategy: str = "median",
    categorical_strategy: str = "most_frequent",
    custom_strategies: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """
    Impute missing values across numerical and categorical columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    numeric_strategy : str, default 'median'
        Strategy for numerical columns: 'median', 'mean', or 'zero'.
    categorical_strategy : str, default 'most_frequent'
        Strategy for categorical columns: 'most_frequent' or 'missing'.
    custom_strategies : dict, optional
        Per-column custom values or functions.

    Returns
    -------
    pd.DataFrame
        DataFrame with missing values handled.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")

    result = df.copy()
    custom_strategies = custom_strategies or {}

    # Handle custom strategies first
    for col, strategy in custom_strategies.items():
        if col in result.columns:
            if callable(strategy):
                result[col] = strategy(result[col])
            else:
                result[col] = result[col].fillna(strategy)

    # Handle numeric columns
    numeric_cols = [
        col for col in result.select_dtypes(include="number").columns
        if col not in custom_strategies
    ]
    for col in numeric_cols:
        if result[col].isna().any():
            if numeric_strategy == "median":
                val = result[col].median()
            elif numeric_strategy == "mean":
                val = result[col].mean()
            elif numeric_strategy == "zero":
                val = 0
            else:
                raise ValueError(f"Unknown numeric strategy: {numeric_strategy}")
            result[col] = result[col].fillna(val)

    # Handle categorical / object columns
    cat_cols = [
        col for col in result.select_dtypes(include=["object", "category", "string"]).columns
        if col not in custom_strategies
    ]
    for col in cat_cols:
        if result[col].isna().any():
            if categorical_strategy == "most_frequent":
                mode_vals = result[col].mode(dropna=True)
                fill_val = mode_vals.iloc[0] if not mode_vals.empty else "Unknown"
            elif categorical_strategy == "missing":
                fill_val = "Missing"
            else:
                raise ValueError(f"Unknown categorical strategy: {categorical_strategy}")
            result[col] = result[col].fillna(fill_val)

    return result


def remove_outliers_iqr(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    *,
    factor: float = 1.5,
    method: str = "clip",
) -> pd.DataFrame:
    """
    Detect and handle outliers using the Interquartile Range (IQR) method.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    columns : list of str, optional
        Numerical columns to check. If None, all numerical columns are used.
    factor : float, default 1.5
        IQR multiplier defining outlier bounds.
    method : str, default 'clip'
        'clip' to cap values at upper/lower fences, or 'filter' to drop outlier rows.

    Returns
    -------
    pd.DataFrame
        DataFrame with outliers treated.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")

    result = df.copy()

    if columns is None:
        target_cols = result.select_dtypes(include="number").columns.tolist()
    else:
        target_cols = [col for col in columns if col in result.columns]

    if method == "clip":
        for col in target_cols:
            q1 = result[col].quantile(0.25)
            q3 = result[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - (factor * iqr)
            upper_bound = q3 + (factor * iqr)
            result[col] = result[col].clip(lower=lower_bound, upper=upper_bound)
        return result

    elif method == "filter":
        mask = pd.Series(True, index=result.index)
        for col in target_cols:
            q1 = result[col].quantile(0.25)
            q3 = result[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - (factor * iqr)
            upper_bound = q3 + (factor * iqr)
            mask &= (result[col] >= lower_bound) & (result[col] <= upper_bound)
        return result[mask].reset_index(drop=True)

    else:
        raise ValueError(f"Unsupported outlier handling method: '{method}'. Choose 'clip' or 'filter'.")


def encode_categoricals(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    *,
    method: str = "onehot",
    drop_first: bool = False,
) -> pd.DataFrame:
    """
    Encode categorical features.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    columns : list of str, optional
        Categorical columns to encode. If None, all object/category columns are used.
    method : str, default 'onehot'
        'onehot' or 'label'.
    drop_first : bool, default False
        Whether to drop the first category in one-hot encoding.

    Returns
    -------
    pd.DataFrame
        DataFrame with encoded features.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")

    result = df.copy()

    if columns is None:
        target_cols = result.select_dtypes(include=["object", "category", "string"]).columns.tolist()
    else:
        target_cols = [col for col in columns if col in result.columns]

    if not target_cols:
        return result

    if method == "onehot":
        return pd.get_dummies(
            result,
            columns=target_cols,
            drop_first=drop_first,
            dtype=int,
        )
    elif method == "label":
        for col in target_cols:
            categories = {val: idx for idx, val in enumerate(result[col].dropna().unique())}
            result[col] = result[col].map(categories)
        return result
    else:
        raise ValueError(f"Unsupported encoding method: '{method}'. Choose 'onehot' or 'label'.")


def scale_features(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    *,
    method: str = "standard",
) -> pd.DataFrame:
    """
    Scale numerical features.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    columns : list of str, optional
        Numerical columns to scale. If None, all numerical columns are used.
    method : str, default 'standard'
        'standard' (zero mean, unit variance) or 'minmax' ([0, 1] range).

    Returns
    -------
    pd.DataFrame
        DataFrame with scaled numerical features.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")

    result = df.copy()

    if columns is None:
        target_cols = result.select_dtypes(include="number").columns.tolist()
    else:
        target_cols = [col for col in columns if col in result.columns]

    if not target_cols:
        return result

    if method == "standard":
        scaler = StandardScaler()
    elif method == "minmax":
        scaler = MinMaxScaler()
    else:
        raise ValueError(f"Unsupported scaling method: '{method}'. Choose 'standard' or 'minmax'.")

    result[target_cols] = scaler.fit_transform(result[target_cols])
    return result
