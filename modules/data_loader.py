"""
Data Loader Module for Applied Time Series Analysis (ATSA).

Provides utility functions to:
- Ingest uploaded CSV and Excel datasets.
- Auto-detect candidate date and numeric columns using strict heuristics.
- Parse datetime series, handle year-only integer columns, and set datetime indices.
- Regularize time series: merge duplicates, enforce uniform frequency with alias compatibility,
  check timestamp alignment (fallback to resample on mismatch), and fill missing values.
- Produce clean pandas DataFrame/Series ready for time series analysis.
"""

from typing import List, Optional, Tuple, Union
import io
import re
import numpy as np
import pandas as pd
from packaging.version import Version

PANDAS_GE_2_2 = Version(pd.__version__) >= Version("2.2.0")


def normalize_pandas_freq(freq_str: Optional[str]) -> Optional[str]:
    """
    Map frequency string to modern or legacy pandas offset aliases based on installed pandas version.

    Parameters
    ----------
    freq_str : Optional[str]
        Raw frequency string or friendly label (e.g. 'ME - Month End').

    Returns
    -------
    Optional[str]
        Normalized pandas frequency alias.
    """
    if not freq_str or freq_str == "Auto-detect":
        return None

    # Extract base token before any hyphen or description
    base_freq = freq_str.split()[0].strip()

    modern_map = {
        "M": "ME",
        "Q": "QE",
        "H": "h",
        "Y": "YE",
        "A": "YE",
        "AS": "YS",
    }
    legacy_map = {
        "ME": "M",
        "QE": "Q",
        "h": "H",
        "YE": "Y",
    }

    if PANDAS_GE_2_2:
        return modern_map.get(base_freq, base_freq)
    else:
        return legacy_map.get(base_freq, base_freq)


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
    Auto-detect candidate date/time columns in a DataFrame using strict heuristics.

    Checks:
    1. Columns with explicit datetime dtypes.
    2. Column names matching whole words or common token boundaries for temporal terms:
       (date, datetime, timestamp, time, year, month, period, ds).
    3. Integer/float columns whose non-null values fall entirely within the 1800-2100 year range.
    4. Non-numeric object columns whose initial sample values can be parsed as dates.

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

    # 2. Strict regex matching whole words or token boundaries (not arbitrary substrings)
    date_pattern = re.compile(
        r"(?:^|[_\s-])(date|datetime|timestamp|time|year|month|period|ds)(?:$|[_\s-])",
        re.IGNORECASE,
    )
    for col in col_names:
        if col not in candidates and date_pattern.search(str(col)):
            candidates.append(col)

    # 3. Check for integer-like year columns (1800 to 2100)
    for col in col_names:
        if col in candidates:
            continue
        try:
            non_na = df[col].dropna()
            if not non_na.empty:
                numeric_vals = pd.to_numeric(non_na, errors="coerce")
                if (
                    numeric_vals.notna().all()
                    and (numeric_vals.astype(int) == numeric_vals).all()
                    and numeric_vals.between(1800, 2100).all()
                ):
                    candidates.append(col)
        except Exception:
            pass

    # 4. Sample parsing check on remaining object / string columns
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
    dayfirst: bool = False,
) -> Union[pd.DataFrame, pd.Series]:
    """
    Parse the designated date column, set it as the DatetimeIndex, and clean the dataset.

    Handles integer-like Year values (1800-2100) with format="%Y" rather than epoch timestamps.
    Respects dayfirst parameter for international date formats (dd/mm/yyyy).

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
    dayfirst : bool, default=False
        Whether date strings have day first (dd/mm/yyyy).

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

    # Check if column is integer-like Year values (1800 to 2100)
    is_year_col = False
    try:
        non_na = cleaned_df[date_col].dropna()
        if not non_na.empty:
            num_vals = pd.to_numeric(non_na, errors="coerce")
            if (
                num_vals.notna().all()
                and (num_vals.astype(int) == num_vals).all()
                and num_vals.between(1800, 2100).all()
            ):
                is_year_col = True
    except Exception:
        is_year_col = False

    if is_year_col:
        cleaned_df[date_col] = pd.to_datetime(
            pd.to_numeric(cleaned_df[date_col], errors="coerce").astype("Int64").astype(str),
            format="%Y",
            errors="coerce",
        )
    else:
        cleaned_df[date_col] = pd.to_datetime(
            cleaned_df[date_col],
            dayfirst=dayfirst,
            errors="coerce",
        )

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


def regularize_series(
    series: pd.Series,
    freq: Optional[str] = None,
    fill_method: str = "interpolate",
) -> Tuple[pd.Series, dict]:
    """
    Regularize a time series by handling duplicates, applying frequency with alias compatibility,
    validating timestamp alignment (falling back to resample aggregation if asfreq yields >50% new NaNs),
    and filling missing values.

    Parameters
    ----------
    series : pd.Series
        Time series indexed with DatetimeIndex.
    freq : Optional[str], default=None
        Target frequency string ('D', 'W', 'MS', 'ME'/'M', 'Q'/'QE', 'QS', 'YS', 'h'/'H') or None/'Auto-detect'.
    fill_method : str, default='interpolate'
        Method to fill missing observations: 'interpolate', 'ffill', 'bfill', or 'drop'.

    Returns
    -------
    Tuple[pd.Series, dict]
        Cleaned, regularized pd.Series and a report dict with:
        {'duplicates_merged': int, 'rows_added': int, 'nans_filled': int, 'freq_used': str,
         'method': str, 'warning': Optional[str]}
    """
    if not isinstance(series.index, pd.DatetimeIndex):
        raise ValueError("Series index must be a pandas DatetimeIndex.")

    s = series.copy()

    # 1. Handle duplicate dates by aggregating by mean
    duplicates_merged = 0
    if s.index.has_duplicates:
        s_dedup = s.groupby(level=0).mean()
        duplicates_merged = len(s) - len(s_dedup)
        s = s_dedup

    # 2. Determine frequency
    normalized_target_freq = normalize_pandas_freq(freq)
    freq_used = normalized_target_freq

    if freq_used is None:
        inferred = pd.infer_freq(s.index)
        if inferred:
            freq_used = normalize_pandas_freq(inferred)
        elif len(s.index) >= 2:
            # Fallback heuristic for frequency detection when missing dates exist
            diffs = (s.index[1:] - s.index[:-1]).total_seconds()
            min_sec = diffs.min()
            if 3500 <= min_sec <= 3700:
                freq_used = "h" if PANDAS_GE_2_2 else "H"
            elif 85000 <= min_sec <= 87000:
                freq_used = "D"
            elif 6 * 86400 <= min_sec <= 8 * 86400:
                freq_used = "W"
            elif 27 * 86400 <= min_sec <= 32 * 86400:
                is_month_start = all(d.day == 1 for d in s.index[:5])
                freq_used = "MS" if is_month_start else ("ME" if PANDAS_GE_2_2 else "M")
            elif 85 * 86400 <= min_sec <= 95 * 86400:
                is_q_start = all(d.day == 1 for d in s.index[:5])
                freq_used = "QS" if is_q_start else ("QE" if PANDAS_GE_2_2 else "Q")
            elif 360 * 86400 <= min_sec <= 370 * 86400:
                freq_used = "YS"

    # 3. Apply frequency alignment with alignment check
    rows_added = 0
    method_used = "none"
    warning_msg = None

    if freq_used is not None:
        try:
            s_asfreq = s.asfreq(freq_used)
            new_nans = int(s_asfreq.isna().sum()) - int(s.isna().sum())

            # Alignment check: if asfreq would produce mostly NaN (>50% of resulting rows are new NaNs)
            if len(s_asfreq) > 0 and (new_nans / len(s_asfreq)) > 0.5:
                s_resampled = s.resample(freq_used).mean()
                method_used = "resample"
                warning_msg = (
                    f"Selected frequency '{freq_used}' does not align with timestamps (asfreq would produce "
                    f">50% missing values). Automatically fell back to resample().mean() aggregation."
                )
                rows_added = max(0, len(s_resampled) - len(s))
                s = s_resampled
            else:
                method_used = "asfreq"
                rows_added = len(s_asfreq) - len(s)
                s = s_asfreq
        except Exception as asfreq_err:
            try:
                s_resampled = s.resample(freq_used).mean()
                method_used = "resample"
                warning_msg = f"asfreq('{freq_used}') failed ({asfreq_err}); fell back to resample().mean()."
                rows_added = max(0, len(s_resampled) - len(s))
                s = s_resampled
            except Exception as resample_err:
                raise ValueError(
                    f"Frequency alignment failed for frequency '{freq_used}'. "
                    f"asfreq error: {asfreq_err}; resample error: {resample_err}"
                ) from resample_err

    # 4. Fill missing values
    nans_before = int(s.isna().sum())
    nans_filled = 0

    if nans_before > 0:
        if fill_method == "interpolate":
            try:
                s = s.interpolate(method="time").ffill().bfill()
            except Exception:
                s = s.interpolate(method="linear").ffill().bfill()
        elif fill_method == "ffill":
            s = s.ffill().bfill()
        elif fill_method == "bfill":
            s = s.bfill().ffill()
        elif fill_method == "drop":
            s = s.dropna()
        else:
            raise ValueError(
                f"Unknown fill method '{fill_method}'. Must be interpolate, ffill, bfill, or drop."
            )

        nans_after = int(s.isna().sum())
        nans_filled = nans_before - nans_after

    report = {
        "duplicates_merged": int(duplicates_merged),
        "rows_added": int(rows_added),
        "nans_filled": int(nans_filled),
        "freq_used": str(freq_used) if freq_used else "None (Irregular)",
        "method": method_used,
        "warning": warning_msg,
    }

    return s, report


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

    inferred_freq = ts.index.freqstr if ts.index.freqstr else pd.infer_freq(ts.index)

    return {
        "num_observations": len(ts),
        "start_date": str(ts.index.min()),
        "end_date": str(ts.index.max()),
        "inferred_frequency": inferred_freq or "Irregular / Undetermined",
        "missing_values": int(ts.isna().sum() if isinstance(ts, pd.Series) else ts.isna().sum().sum()),
    }
