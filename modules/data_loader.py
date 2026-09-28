"""
Data Loader Module for Applied Time Series Analysis (ATSA).

Provides utility functions to:
- Ingest uploaded CSV and Excel datasets.
- Auto-detect candidate date and numeric columns.
- Parse datetime series, sort chronologically, and set datetime indices.
- Produce clean pandas DataFrame/Series ready for time series analysis.
"""

from typing import List, Optional, Tuple, Union
import io
import re
import numpy as np
import pandas as pd


def load_uploaded_file(uploaded_file) -> pd.DataFrame:
    """
    Load an uploaded file (CSV or Excel) into a pandas DataFrame.

    Parameters
    ----------
    uploaded_file : UploadedFile or io.BytesIO
        Streamlit UploadedFile or file-like object containing dataset.

    Returns
    -------
    pd.DataFrame
        Raw loaded pandas DataFrame.

    Raises
    ------
    ValueError
        If the file format is unsupported or file cannot be parsed.
    """
    if uploaded_file is None:
        raise ValueError("No file provided to load.")

    file_name = uploaded_file.name.lower()

    try:
        if file_name.endswith(".csv"):
            return pd.read_csv(uploaded_file)
        elif file_name.endswith((".xlsx", ".xls")):
            return pd.read_excel(uploaded_file)
        else:
            raise ValueError(f"Unsupported file format for '{uploaded_file.name}'. Please upload a CSV or Excel file.")
    except Exception as exc:
        raise ValueError(f"Error reading file '{uploaded_file.name}': {str(exc)}") from exc


def detect_date_columns(df: pd.DataFrame) -> List[str]:
    """
    Auto-detect candidate date/time columns in a DataFrame.

    Checks:
    1. Columns with explicit datetime dtypes.
    2. Column names matching common temporal keywords (date, time, timestamp, year, month, etc.).
    3. Non-numeric object columns whose initial sample values can be parsed as dates.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.

    Returns
    -------
    List[str]
        List of column names identified as candidate date columns, ordered by likelihood.
    """
    candidates: List[str] = []
    col_names = list(df.columns)

    # 1. Existing datetime columns
    for col in col_names:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            candidates.append(col)

    # 2. Heuristic based on column naming patterns
    date_pattern = re.compile(
        r"(date|time|timestamp|year|period|day|month|ds|datetime|dt)",
        re.IGNORECASE
    )
    for col in col_names:
        if col not in candidates and date_pattern.search(str(col)):
            candidates.append(col)

    # 3. Sample parsing check on remaining object / string columns
    for col in col_names:
        if col in candidates:
            continue
        if df[col].dtype == object or pd.api.types.is_string_dtype(df[col]):
            sample = df[col].dropna().head(10)
            if not sample.empty:
                try:
                    parsed = pd.to_datetime(sample, errors="coerce")
                    if parsed.notna().mean() > 0.8:
                        candidates.append(col)
                except Exception:
                    pass

    return candidates


def detect_numeric_columns(df: pd.DataFrame) -> List[str]:
    """
    Detect all numeric columns in a DataFrame suitable for time series modeling.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.

    Returns
    -------
    List[str]
        List of column names with numeric dtypes.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    return numeric_cols


def prepare_time_series(
    df: pd.DataFrame,
    date_col: str,
    target_col: Optional[str] = None,
    sort_index: bool = True,
    drop_na_dates: bool = True,
) -> Union[pd.DataFrame, pd.Series]:
    """
    Parse the designated date column, set it as the DatetimeIndex, and clean the dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Input raw DataFrame.
    date_col : str
        Name of the column containing date/time information.
    target_col : Optional[str], default=None
        Optional specific target numeric column to isolate as a pd.Series.
        If None, returns the entire DataFrame with the DatetimeIndex.
    sort_index : bool, default=True
        Whether to sort the data chronologically by date.
    drop_na_dates : bool, default=True
        Whether to drop rows where the date could not be parsed.

    Returns
    -------
    Union[pd.DataFrame, pd.Series]
        Cleaned time series DataFrame or Series indexed by DatetimeIndex.

    Raises
    ------
    ValueError
        If the date column is missing or cannot be converted to valid dates.
    """
    if date_col not in df.columns:
        raise ValueError(f"Specified date column '{date_col}' not found in DataFrame.")

    cleaned_df = df.copy()

    # Parse datetime
    cleaned_df[date_col] = pd.to_datetime(cleaned_df[date_col], errors="coerce")

    if drop_na_dates:
        cleaned_df = cleaned_df.dropna(subset=[date_col])

    if cleaned_df.empty:
        raise ValueError(f"All values in date column '{date_col}' were unparseable or empty.")

    # Set DatetimeIndex
    cleaned_df = cleaned_df.set_index(date_col)
    cleaned_df.index.name = "date"

    if sort_index:
        cleaned_df = cleaned_df.sort_index()

    if target_col is not None:
        if target_col not in cleaned_df.columns:
            raise ValueError(f"Target column '{target_col}' not found in DataFrame.")
        series = cleaned_df[target_col].copy()
        series.name = target_col
        return series

    return cleaned_df


def get_series_summary(ts: Union[pd.DataFrame, pd.Series]) -> dict:
    """
    Extract essential high-level metadata and statistics for the time series.

    Parameters
    ----------
    ts : Union[pd.DataFrame, pd.Series]
        Time series with DatetimeIndex.

    Returns
    -------
    dict
        Dictionary containing length, date span, detected frequency, and missing values.
    """
    if not isinstance(ts.index, pd.DatetimeIndex):
        raise ValueError("Provided object must have a pandas DatetimeIndex.")

    inferred_freq = pd.infer_freq(ts.index) if len(ts.index) >= 3 else None

    return {
        "num_observations": len(ts),
        "start_date": str(ts.index.min()),
        "end_date": str(ts.index.max()),
        "inferred_frequency": inferred_freq or "Irregular / Undetermined",
        "missing_values": int(ts.isna().sum() if isinstance(ts, pd.Series) else ts.isna().sum().sum()),
    }
