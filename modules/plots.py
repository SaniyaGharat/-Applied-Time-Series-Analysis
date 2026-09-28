"""
Plotly Figure Builders Module for Applied Time Series Analysis (ATSA).

Pure visualization module (no Streamlit imports).
All functions return `plotly.graph_objects.Figure` instances formatted with `template='plotly_white'`.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import norm, probplot
from statsmodels.tsa.stattools import acf


def plot_time_series(
    series: pd.Series,
    title: Optional[str] = None,
    yaxis_title: Optional[str] = None,
    show_rangeslider: bool = True,
    line_color: str = "#2563eb",
) -> go.Figure:
    """
    Build an interactive time series line plot.

    Parameters
    ----------
    series : pd.Series
        Time series indexed by DatetimeIndex or integers.
    title : Optional[str]
        Figure title.
    yaxis_title : Optional[str]
        Y-axis label.
    show_rangeslider : bool, default=True
        Whether to display x-axis rangeslider.
    line_color : str, default='#2563eb'
        Line color.

    Returns
    -------
    go.Figure
        Interactive Plotly figure.
    """
    clean_s = series.dropna()
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=clean_s.index,
            y=clean_s.values,
            mode="lines+markers" if len(clean_s) <= 100 else "lines",
            name=clean_s.name or "Value",
            line=dict(color=line_color, width=2),
            marker=dict(size=4),
            hovertemplate="<b>Date</b>: %{x}<br><b>Value</b>: %{y:.3f}<extra></extra>",
        )
    )

    fig.update_layout(
        template="plotly_white",
        title=title or (f"{clean_s.name} Series" if clean_s.name else "Time Series Plot"),
        xaxis_title="Date / Time",
        yaxis_title=yaxis_title or clean_s.name or "Value",
        hovermode="x unified",
        margin=dict(l=40, r=40, t=50, b=40),
        xaxis=dict(rangeslider=dict(visible=show_rangeslider)),
    )

    return fig


def plot_rolling_stats(
    series: pd.Series,
    rolling_df: pd.DataFrame,
    window: int,
) -> go.Figure:
    """
    Plot the original series with rolling mean and rolling standard deviation overlays.

    Parameters
    ----------
    series : pd.Series
        Original time series.
    rolling_df : pd.DataFrame
        DataFrame containing 'rolling_mean' and 'rolling_std'.
    window : int
        Rolling window size.

    Returns
    -------
    go.Figure
        Plotly figure.
    """
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=series.index,
            y=series.values,
            mode="lines",
            name="Original",
            line=dict(color="#94a3b8", width=1.5),
            opacity=0.7,
        )
    )

    if "rolling_mean" in rolling_df.columns:
        fig.add_trace(
            go.Scatter(
                x=rolling_df.index,
                y=rolling_df["rolling_mean"],
                mode="lines",
                name=f"Rolling Mean (w={window})",
                line=dict(color="#2563eb", width=2.5),
            )
        )

    if "rolling_std" in rolling_df.columns:
        fig.add_trace(
            go.Scatter(
                x=rolling_df.index,
                y=rolling_df["rolling_std"],
                mode="lines",
                name=f"Rolling Std (w={window})",
                line=dict(color="#dc2626", width=2, dash="dash"),
            )
        )

    fig.update_layout(
        template="plotly_white",
        title=f"Rolling Statistics (Window = {window})",
        xaxis_title="Date / Time",
        yaxis_title="Value",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=60, b=40),
    )

    return fig


def plot_stem_correlation(
    lags: np.ndarray,
    values: np.ndarray,
    conf_bound: float,
    title: str = "Autocorrelation",
    yaxis_title: str = "Correlation",
    bar_color: str = "#2563eb",
) -> go.Figure:
    """
    Create a stem plot for ACF or PACF with confidence interval bounds.

    Parameters
    ----------
    lags : np.ndarray
        Array of lag indices.
    values : np.ndarray
        Correlation values.
    conf_bound : float
        Critical threshold for 95% confidence (+/- 1.96 / sqrt(N)).
    title : str
        Figure title.
    yaxis_title : str
        Y-axis label.
    bar_color : str
        Color of stem bars and points.

    Returns
    -------
    go.Figure
        Plotly figure.
    """
    fig = go.Figure()

    # Vertical stems
    fig.add_trace(
        go.Bar(
            x=lags,
            y=values,
            width=0.2,
            marker_color=bar_color,
            name=title,
            showlegend=False,
            hovertemplate="<b>Lag %{x}</b>: %{y:.3f}<extra></extra>",
        )
    )

    # Point markers on stems
    fig.add_trace(
        go.Scatter(
            x=lags,
            y=values,
            mode="markers",
            marker=dict(size=7, color=bar_color),
            showlegend=False,
            hoverinfo="skip",
        )
    )

    # 95% confidence bounds
    fig.add_hline(
        y=conf_bound,
        line=dict(color="#ef4444", dash="dash", width=1.5),
        annotation_text="+95% Conf",
        annotation_position="top right",
    )
    fig.add_hline(
        y=-conf_bound,
        line=dict(color="#ef4444", dash="dash", width=1.5),
        annotation_text="-95% Conf",
        annotation_position="bottom right",
    )
    fig.add_hline(y=0, line=dict(color="#64748b", width=1))

    # Shaded band
    fig.add_hrect(
        y0=-conf_bound,
        y1=conf_bound,
        fillcolor="#ef4444",
        opacity=0.08,
        line_width=0,
    )

    fig.update_layout(
        template="plotly_white",
        title=title,
        xaxis_title="Lag",
        yaxis_title=yaxis_title,
        yaxis=dict(range=[-1.05, 1.05]),
        margin=dict(l=40, r=40, t=50, b=40),
    )

    return fig


def plot_acf_pacf_combined(
    lags: np.ndarray,
    acf_vals: np.ndarray,
    pacf_vals: np.ndarray,
    conf_bound: float,
) -> go.Figure:
    """
    Plot ACF and PACF side-by-side in a 1x2 subplot.

    Returns
    -------
    go.Figure
        Combined Plotly figure.
    """
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=["Autocorrelation (ACF)", "Partial Autocorrelation (PACF)"],
    )

    # ACF stems
    fig.add_trace(
        go.Bar(x=lags, y=acf_vals, width=0.2, marker_color="#2563eb", name="ACF", showlegend=False),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=lags, y=acf_vals, mode="markers", marker=dict(size=6, color="#2563eb"), showlegend=False),
        row=1,
        col=1,
    )

    # PACF stems
    fig.add_trace(
        go.Bar(x=lags, y=pacf_vals, width=0.2, marker_color="#7c3aed", name="PACF", showlegend=False),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(x=lags, y=pacf_vals, mode="markers", marker=dict(size=6, color="#7c3aed"), showlegend=False),
        row=1,
        col=2,
    )

    # Horizontal zero and bound lines
    for col_idx in [1, 2]:
        fig.add_hline(y=conf_bound, line=dict(color="#ef4444", dash="dash", width=1.2), row=1, col=col_idx)
        fig.add_hline(y=-conf_bound, line=dict(color="#ef4444", dash="dash", width=1.2), row=1, col=col_idx)
        fig.add_hline(y=0, line=dict(color="#64748b", width=1), row=1, col=col_idx)

    fig.update_layout(
        template="plotly_white",
        yaxis1=dict(range=[-1.05, 1.05], title="ACF"),
        yaxis2=dict(range=[-1.05, 1.05], title="PACF"),
        xaxis1=dict(title="Lag"),
        xaxis2=dict(title="Lag"),
        margin=dict(l=40, r=40, t=50, b=40),
    )

    return fig


def plot_decomposition(decomp: Dict[str, Any], title: Optional[str] = None) -> go.Figure:
    """
    Build a 4-panel decomposition plot (Observed, Trend, Seasonal, Residual) with shared X axes.

    Parameters
    ----------
    decomp : Dict[str, Any]
        Output dictionary from diagnostics.decompose().
    title : Optional[str]
        Overall figure title.

    Returns
    -------
    go.Figure
        4-panel subplot figure.
    """
    observed = decomp["observed"]
    trend = decomp["trend"]
    seasonal = decomp["seasonal"]
    resid = decomp["resid"]

    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        subplot_titles=["Observed", "Trend", "Seasonal", "Residual"],
    )

    fig.add_trace(
        go.Scatter(x=observed.index, y=observed.values, mode="lines", name="Observed", line=dict(color="#0f172a")),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(x=trend.index, y=trend.values, mode="lines", name="Trend", line=dict(color="#2563eb", width=2)),
        row=2,
        col=1,
    )

    fig.add_trace(
        go.Scatter(x=seasonal.index, y=seasonal.values, mode="lines", name="Seasonal", line=dict(color="#10b981")),
        row=3,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=resid.index,
            y=resid.values,
            mode="lines+markers" if len(resid) <= 100 else "lines",
            name="Residual",
            line=dict(color="#f59e0b"),
            marker=dict(size=3),
        ),
        row=4,
        col=1,
    )

    fig.add_hline(y=0 if decomp.get("model") == "additive" else 1, line=dict(color="#94a3b8", dash="dash"), row=4, col=1)

    chart_title = title or (
        f"Time Series Decomposition ({decomp.get('method', '').upper()}, {decomp.get('model', '').capitalize()} Model, "
        f"Period = {decomp.get('period')})"
    )

    fig.update_layout(
        template="plotly_white",
        height=750,
        title=chart_title,
        showlegend=False,
        hovermode="x unified",
        margin=dict(l=50, r=40, t=60, b=40),
    )

    return fig


def plot_seasonal_box(
    seasonal_df: pd.DataFrame,
    period_name: Optional[str] = None,
) -> go.Figure:
    """
    Build a seasonal distribution box plot grouped by cycle (month, weekday, quarter, or period).

    Parameters
    ----------
    seasonal_df : pd.DataFrame
        DataFrame from diagnostics.seasonal_subseries().
    period_name : Optional[str]
        Title descriptor.

    Returns
    -------
    go.Figure
        Plotly box plot.
    """
    fig = px.box(
        seasonal_df,
        x="cycle_label",
        y="value",
        color="cycle_label",
        points="all",
        template="plotly_white",
        title=f"Seasonal Distribution ({period_name or 'By Cycle'})",
    )

    fig.update_layout(
        xaxis_title="Cycle / Season",
        yaxis_title="Value",
        showlegend=False,
        margin=dict(l=40, r=40, t=50, b=40),
    )

    return fig


def plot_lag(series: pd.Series, lag: int = 1) -> go.Figure:
    """
    Build a lag scatter plot (y_t vs y_{t-k}) with diagonal reference line.

    Parameters
    ----------
    series : pd.Series
        Time series.
    lag : int, default=1
        Lag order k.

    Returns
    -------
    go.Figure
        Plotly scatter figure.
    """
    clean_s = series.dropna()
    y_t = clean_s.iloc[lag:].values
    y_lag = clean_s.iloc[:-lag].values

    corr = float(np.corrcoef(y_lag, y_t)[0, 1]) if len(y_t) > 1 else 0.0

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=y_lag,
            y=y_t,
            mode="markers",
            marker=dict(color="#2563eb", size=6, opacity=0.7),
            name=f"Lag {lag}",
            hovertemplate="y(t-%{x}): %{x:.3f}<br>y(t): %{y:.3f}<extra></extra>",
        )
    )

    # 45-degree diagonal reference line
    min_val = min(np.min(y_lag), np.min(y_t))
    max_val = max(np.max(y_lag), np.max(y_t))
    fig.add_trace(
        go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode="lines",
            line=dict(color="#94a3b8", dash="dash"),
            name="y = x",
        )
    )

    fig.update_layout(
        template="plotly_white",
        title=f"Lag Plot (k = {lag}, Pearson r = {corr:.3f})",
        xaxis_title=f"y(t - {lag})",
        yaxis_title="y(t)",
        margin=dict(l=40, r=40, t=50, b=40),
    )

    return fig


def plot_distribution_and_qq(series: pd.Series) -> go.Figure:
    """
    Build a 1x2 subplot with histogram + normal PDF fit on the left, and normal Q-Q plot on the right.

    Parameters
    ----------
    series : pd.Series
        Time series or residual data.

    Returns
    -------
    go.Figure
        Plotly 1x2 figure.
    """
    clean_s = series.dropna().values
    n = len(clean_s)

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=["Value Distribution & Normal Fit", "Normal Q-Q Plot"],
    )

    # 1. Histogram + Normal curve
    fig.add_trace(
        go.Histogram(
            x=clean_s,
            histnorm="probability density",
            marker_color="#93c5fd",
            name="Empirical Density",
            opacity=0.75,
        ),
        row=1,
        col=1,
    )

    mean_val = float(np.mean(clean_s))
    std_val = float(np.std(clean_s, ddof=1)) if n > 1 else 1.0

    if std_val > 0:
        x_grid = np.linspace(np.min(clean_s), np.max(clean_s), 150)
        pdf_vals = norm.pdf(x_grid, loc=mean_val, scale=std_val)
        fig.add_trace(
            go.Scatter(
                x=x_grid,
                y=pdf_vals,
                mode="lines",
                name="Fitted Normal",
                line=dict(color="#1d4ed8", width=2.5),
            ),
            row=1,
            col=1,
        )

    # 2. Q-Q Plot
    osm, osr = probplot(clean_s, dist="norm")[0]
    slope, intercept, r = probplot(clean_s, dist="norm")[1]

    fig.add_trace(
        go.Scatter(
            x=osm,
            y=osr,
            mode="markers",
            marker=dict(color="#7c3aed", size=5),
            name="Sample Quantiles",
            showlegend=False,
            hovertemplate="Theoretical: %{x:.2f}<br>Sample: %{y:.2f}<extra></extra>",
        ),
        row=1,
        col=2,
    )

    qq_x = np.array([np.min(osm), np.max(osm)])
    qq_y = slope * qq_x + intercept
    fig.add_trace(
        go.Scatter(
            x=qq_x,
            y=qq_y,
            mode="lines",
            line=dict(color="#ef4444", width=2),
            name="Reference Line",
            showlegend=False,
        ),
        row=1,
        col=2,
    )

    fig.update_layout(
        template="plotly_white",
        title=f"Distribution Diagnostics (Mean = {mean_val:.2f}, Std = {std_val:.2f})",
        xaxis1=dict(title="Value"),
        yaxis1=dict(title="Density"),
        xaxis2=dict(title="Theoretical Quantiles (Normal)"),
        yaxis2=dict(title="Ordered Values"),
        margin=dict(l=40, r=40, t=50, b=40),
    )

    return fig


def plot_residual_diagnostics(residuals: pd.Series) -> go.Figure:
    """
    Build a comprehensive 2x2 residual diagnostics figure:
    - (1, 1): Residual line plot over time
    - (1, 2): Histogram of residuals with standard normal fit
    - (2, 1): Normal Q-Q plot
    - (2, 2): ACF of residuals with 95% confidence bounds

    Parameters
    ----------
    residuals : pd.Series
        Residual series from a model or transformation.

    Returns
    -------
    go.Figure
        2x2 diagnostics Plotly figure.
    """
    clean_res = residuals.dropna()
    n = len(clean_res)
    std_res = (clean_res - clean_res.mean()) / (clean_res.std(ddof=1) if clean_res.std(ddof=1) > 0 else 1.0)

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[
            "Standardized Residuals",
            "Histogram & Normal Curve",
            "Normal Q-Q Plot",
            "Residual Autocorrelation (ACF)",
        ],
        vertical_spacing=0.15,
        horizontal_spacing=0.1,
    )

    # (1, 1): Line plot
    fig.add_trace(
        go.Scatter(
            x=std_res.index,
            y=std_res.values,
            mode="lines",
            line=dict(color="#2563eb", width=1.5),
            name="Std Residual",
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    fig.add_hline(y=0, line=dict(color="#64748b", dash="dash"), row=1, col=1)

    # (1, 2): Histogram + Standard normal
    fig.add_trace(
        go.Histogram(
            x=std_res.values,
            histnorm="probability density",
            marker_color="#93c5fd",
            name="Empirical Density",
            showlegend=False,
        ),
        row=1,
        col=2,
    )
    x_span = np.linspace(-3.5, 3.5, 100)
    fig.add_trace(
        go.Scatter(
            x=x_span,
            y=norm.pdf(x_span, 0, 1),
            mode="lines",
            line=dict(color="#1d4ed8", width=2),
            name="N(0,1)",
            showlegend=False,
        ),
        row=1,
        col=2,
    )

    # (2, 1): Q-Q Plot
    osm, osr = probplot(std_res.values, dist="norm")[0]
    slope, intercept, _ = probplot(std_res.values, dist="norm")[1]
    fig.add_trace(
        go.Scatter(
            x=osm,
            y=osr,
            mode="markers",
            marker=dict(color="#7c3aed", size=5),
            name="Quantiles",
            showlegend=False,
        ),
        row=2,
        col=1,
    )
    qq_x = np.array([min(osm), max(osm)])
    fig.add_trace(
        go.Scatter(
            x=qq_x,
            y=slope * qq_x + intercept,
            mode="lines",
            line=dict(color="#ef4444", width=2),
            showlegend=False,
        ),
        row=2,
        col=1,
    )

    # (2, 2): ACF of residuals
    nlags = min(20, max(2, n // 2 - 1))
    acf_vals = acf(clean_res, nlags=nlags, fft=True)
    lags = np.arange(len(acf_vals))
    conf_bound = 1.96 / np.sqrt(n)

    fig.add_trace(
        go.Bar(
            x=lags,
            y=acf_vals,
            width=0.2,
            marker_color="#059669",
            name="Residual ACF",
            showlegend=False,
        ),
        row=2,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=lags,
            y=acf_vals,
            mode="markers",
            marker=dict(size=6, color="#059669"),
            showlegend=False,
        ),
        row=2,
        col=2,
    )
    fig.add_hline(y=conf_bound, line=dict(color="#ef4444", dash="dash"), row=2, col=2)
    fig.add_hline(y=-conf_bound, line=dict(color="#ef4444", dash="dash"), row=2, col=2)
    fig.add_hline(y=0, line=dict(color="#64748b"), row=2, col=2)

    fig.update_layout(
        template="plotly_white",
        height=650,
        title="Residual Diagnostic Panel",
        margin=dict(l=40, r=40, t=60, b=40),
    )

    return fig


def plot_fit_vs_actual(
    actual: pd.Series,
    fitted: pd.Series,
    forecast_df: pd.DataFrame,
    title: Optional[str] = None,
    show_train_window: Optional[int] = None,
) -> go.Figure:
    """
    Build unified fit versus actual plot with in-sample fit, out-of-sample forecast, and 95% interval.
    """
    fig = go.Figure()

    actual_series = actual.dropna()
    if show_train_window and len(actual_series) > show_train_window:
        actual_plot = actual_series.iloc[-show_train_window:]
    else:
        actual_plot = actual_series

    # 1. Actual series (full or windowed)
    fig.add_trace(
        go.Scatter(
            x=actual_plot.index,
            y=actual_plot.values,
            mode="lines",
            name="Actual",
            line=dict(color="#94a3b8", width=2),
            hovertemplate="<b>Actual</b>: %{y:.3f}<br>Date: %{x}<extra></extra>",
        )
    )

    # 2. Fitted values (train portion)
    fitted_clean = fitted.dropna()
    if show_train_window and len(fitted_clean) > show_train_window:
        fitted_plot = fitted_clean.iloc[-show_train_window:]
    else:
        fitted_plot = fitted_clean

    if not fitted_plot.empty:
        fig.add_trace(
            go.Scatter(
                x=fitted_plot.index,
                y=fitted_plot.values,
                mode="lines",
                name="Fitted (In-Sample)",
                line=dict(color="#2563eb", width=2),
                hovertemplate="<b>Fitted</b>: %{y:.3f}<br>Date: %{x}<extra></extra>",
            )
        )

    # 3. Forecast and Confidence Interval
    if not forecast_df.empty:
        test_idx = forecast_df.index

        if "upper" in forecast_df.columns and "lower" in forecast_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=test_idx,
                    y=forecast_df["upper"].values,
                    mode="lines",
                    line=dict(width=0),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=test_idx,
                    y=forecast_df["lower"].values,
                    mode="lines",
                    line=dict(width=0),
                    fill="tonexty",
                    fillcolor="rgba(234, 88, 12, 0.2)",
                    name="95% Interval",
                    hoverinfo="skip",
                )
            )

        if "mean" in forecast_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=test_idx,
                    y=forecast_df["mean"].values,
                    mode="lines",
                    name="Forecast (Holdout)",
                    line=dict(color="#ea580c", width=2.5),
                    hovertemplate="<b>Forecast</b>: %{y:.3f}<br>Date: %{x}<extra></extra>",
                )
            )

        # Train/Test Split Vertical Line (shape-based for datetime safety)
        split_dt = test_idx[0]
        split_str = split_dt.isoformat() if hasattr(split_dt, "isoformat") else str(split_dt)
        fig.add_shape(
            type="line",
            x0=split_str,
            x1=split_str,
            y0=0,
            y1=1,
            yref="paper",
            line=dict(color="#dc2626", width=1.5, dash="dash"),
        )
        fig.add_annotation(
            x=split_str,
            y=1.02,
            yref="paper",
            text="Train / Test Split",
            showarrow=False,
            xanchor="left",
            font=dict(size=11, color="#dc2626"),
        )

    fig.update_layout(
        template="plotly_white",
        title=title or "Model Fit & Forecast vs. Actual",
        xaxis_title="Date / Time",
        yaxis_title="Value (Original Scale)",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=70, b=40),
    )

    return fig


def plot_holdout_zoom(
    actual: pd.Series,
    forecast_df: pd.DataFrame,
    train_tail: Optional[int] = None,
) -> go.Figure:
    """
    Build zoomed view focusing exclusively on the holdout window and recent history.
    """
    fig = go.Figure()

    if forecast_df.empty:
        return fig

    test_len = len(forecast_df)
    tail_n = train_tail or max(10, 2 * test_len)

    # Slice actual series around train tail + test
    test_idx = forecast_df.index
    train_actual = actual.loc[:test_idx[0]].iloc[:-1]
    zoom_train = train_actual.iloc[-tail_n:] if len(train_actual) > tail_n else train_actual
    zoom_test = actual.reindex(test_idx)

    # Historical context line + markers
    if not zoom_train.empty:
        fig.add_trace(
            go.Scatter(
                x=zoom_train.index,
                y=zoom_train.values,
                mode="lines+markers",
                name="Actual (Train)",
                line=dict(color="#94a3b8", width=1.5),
                marker=dict(size=5),
            )
        )

    # Test actual
    if not zoom_test.empty:
        fig.add_trace(
            go.Scatter(
                x=zoom_test.index,
                y=zoom_test.values,
                mode="lines+markers",
                name="Actual (Holdout)",
                line=dict(color="#0f172a", width=2),
                marker=dict(size=6, symbol="circle"),
            )
        )

    # Forecast band
    if "upper" in forecast_df.columns and "lower" in forecast_df.columns:
        fig.add_trace(
            go.Scatter(
                x=test_idx,
                y=forecast_df["upper"].values,
                mode="lines",
                line=dict(width=0),
                showlegend=False,
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=test_idx,
                y=forecast_df["lower"].values,
                mode="lines",
                line=dict(width=0),
                fill="tonexty",
                fillcolor="rgba(234, 88, 12, 0.2)",
                name="95% Interval",
                hoverinfo="skip",
            )
        )

    # Forecast mean
    if "mean" in forecast_df.columns:
        fig.add_trace(
            go.Scatter(
                x=test_idx,
                y=forecast_df["mean"].values,
                mode="lines+markers",
                name="Forecast Mean",
                line=dict(color="#ea580c", width=2.5),
                marker=dict(size=6, symbol="diamond"),
            )
        )

    fig.update_layout(
        template="plotly_white",
        title="Holdout Evaluation Zoom (Actual vs. Forecast)",
        xaxis_title="Date / Time",
        yaxis_title="Value (Original Scale)",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=60, b=40),
    )

    return fig


def plot_inverse_roots(
    ar_roots: Optional[Union[np.ndarray, list]],
    ma_roots: Optional[Union[np.ndarray, list]],
    title: str = "Inverse Roots & Stability / Invertibility",
) -> go.Figure:
    """
    Plot complex inverse AR and MA roots against the unit circle.
    Roots strictly inside unit circle (modulus < 1) signify stability / invertibility.
    """
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=[
            "AR Inverse Roots (Stability: |1/r| < 1)",
            "MA Inverse Roots (Invertibility: |1/r| < 1)",
        ],
    )

    theta = np.linspace(0, 2 * np.pi, 200)
    circle_x = np.cos(theta)
    circle_y = np.sin(theta)

    # Add unit circle to both subplots
    for col in [1, 2]:
        fig.add_trace(
            go.Scatter(
                x=circle_x,
                y=circle_y,
                mode="lines",
                line=dict(color="#94a3b8", dash="dot", width=1.5),
                name="Unit Circle",
                showlegend=(col == 1),
                hoverinfo="skip",
            ),
            row=1,
            col=col,
        )

    # Subplot 1: AR Inverse Roots
    ar_arr = np.asarray(ar_roots) if ar_roots is not None else np.array([])
    if len(ar_arr) == 0:
        fig.add_annotation(
            text="No AR terms",
            xref="x1",
            yref="y1",
            x=0,
            y=0,
            showarrow=False,
            font=dict(size=14, color="#64748b"),
        )
    else:
        valid_ar = ar_arr[ar_arr != 0]
        inv_ar = 1.0 / valid_ar
        moduli = np.abs(inv_ar)
        colors = ["#ef4444" if m >= 1.0 else "#2563eb" for m in moduli]
        fig.add_trace(
            go.Scatter(
                x=np.real(inv_ar),
                y=np.imag(inv_ar),
                mode="markers",
                marker=dict(size=10, color=colors, line=dict(color="black", width=1)),
                name="AR Inverse Roots",
                hovertemplate="Real: %{x:.3f}<br>Imag: %{y:.3f}<extra></extra>",
                showlegend=True,
            ),
            row=1,
            col=1,
        )

    # Subplot 2: MA Inverse Roots
    ma_arr = np.asarray(ma_roots) if ma_roots is not None else np.array([])
    if len(ma_arr) == 0:
        fig.add_annotation(
            text="No MA terms",
            xref="x2",
            yref="y2",
            x=0,
            y=0,
            showarrow=False,
            font=dict(size=14, color="#64748b"),
        )
    else:
        valid_ma = ma_arr[ma_arr != 0]
        inv_ma = 1.0 / valid_ma
        moduli_ma = np.abs(inv_ma)
        colors_ma = ["#ef4444" if m >= 1.0 else "#059669" for m in moduli_ma]
        fig.add_trace(
            go.Scatter(
                x=np.real(inv_ma),
                y=np.imag(inv_ma),
                mode="markers",
                marker=dict(size=10, color=colors_ma, line=dict(color="black", width=1)),
                name="MA Inverse Roots",
                hovertemplate="Real: %{x:.3f}<br>Imag: %{y:.3f}<extra></extra>",
                showlegend=True,
            ),
            row=1,
            col=2,
        )

    for col in [1, 2]:
        fig.update_xaxes(
            range=[-1.4, 1.4],
            zeroline=True,
            zerolinecolor="#cbd5e1",
            title="Real",
            scaleanchor=f"y{col}",
            scaleratio=1,
            row=1,
            col=col,
        )
        fig.update_yaxes(
            range=[-1.4, 1.4],
            zeroline=True,
            zerolinecolor="#cbd5e1",
            title="Imaginary",
            row=1,
            col=col,
        )

    fig.update_layout(
        template="plotly_white",
        title=title,
        height=450,
        margin=dict(l=40, r=40, t=60, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="right", x=1),
    )

    return fig


def plot_theoretical_vs_empirical(
    lags: np.ndarray,
    empirical_acf: np.ndarray,
    empirical_pacf: np.ndarray,
    theo_acf: Optional[np.ndarray],
    theo_pacf: Optional[np.ndarray],
    conf_bound: float,
) -> go.Figure:
    """
    Overlay empirical ACF/PACF with theoretical ARMA model ACF/PACF.
    """
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=["ACF: Empirical vs. Model Theoretical", "PACF: Empirical vs. Model Theoretical"],
    )

    n_lags = len(lags)

    # 1. ACF
    fig.add_trace(
        go.Bar(
            x=lags,
            y=empirical_acf[:n_lags],
            width=0.25,
            marker_color="#94a3b8",
            name="Empirical ACF",
        ),
        row=1,
        col=1,
    )
    if theo_acf is not None:
        theo_len = min(n_lags, len(theo_acf))
        fig.add_trace(
            go.Scatter(
                x=lags[:theo_len],
                y=theo_acf[:theo_len],
                mode="lines+markers",
                line=dict(color="#2563eb", width=2),
                marker=dict(size=6, symbol="circle"),
                name="Theoretical ACF",
            ),
            row=1,
            col=1,
        )

    fig.add_hline(y=conf_bound, line=dict(color="#ef4444", dash="dash"), row=1, col=1)
    fig.add_hline(y=-conf_bound, line=dict(color="#ef4444", dash="dash"), row=1, col=1)
    fig.add_hline(y=0, line=dict(color="#64748b"), row=1, col=1)

    # 2. PACF
    fig.add_trace(
        go.Bar(
            x=lags,
            y=empirical_pacf[:n_lags],
            width=0.25,
            marker_color="#94a3b8",
            name="Empirical PACF",
        ),
        row=1,
        col=2,
    )
    if theo_pacf is not None:
        theo_len = min(n_lags, len(theo_pacf))
        fig.add_trace(
            go.Scatter(
                x=lags[:theo_len],
                y=theo_pacf[:theo_len],
                mode="lines+markers",
                line=dict(color="#7c3aed", width=2),
                marker=dict(size=6, symbol="circle"),
                name="Theoretical PACF",
            ),
            row=1,
            col=2,
        )

    fig.add_hline(y=conf_bound, line=dict(color="#ef4444", dash="dash"), row=1, col=2)
    fig.add_hline(y=-conf_bound, line=dict(color="#ef4444", dash="dash"), row=1, col=2)
    fig.add_hline(y=0, line=dict(color="#64748b"), row=1, col=2)

    fig.update_layout(
        template="plotly_white",
        title="Theoretical vs. Empirical Correlation Structure",
        height=450,
        margin=dict(l=40, r=40, t=60, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="right", x=1),
    )

    return fig


def plot_hw_components(
    extras: Dict[str, Any],
    index: pd.Index,
) -> go.Figure:
    """
    Plot Holt-Winters estimated components (Level, Trend, Seasonality) in the model scale.
    """
    level = extras.get("level")
    trend = extras.get("trend")
    season = extras.get("season")

    plots_to_show = []
    if level is not None:
        plots_to_show.append(("Level", level, "#2563eb"))
    if trend is not None:
        plots_to_show.append(("Trend", trend, "#059669"))
    if season is not None:
        plots_to_show.append(("Seasonality", season, "#7c3aed"))

    if not plots_to_show:
        fig = go.Figure()
        fig.add_annotation(text="No Holt-Winters state components available", showarrow=False)
        fig.update_layout(template="plotly_white", height=300)
        return fig

    n_rows = len(plots_to_show)
    fig = make_subplots(
        rows=n_rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=[name for name, _, _ in plots_to_show],
    )

    for i, (name, comp_s, col_hex) in enumerate(plots_to_show, start=1):
        s_vals = comp_s.values if hasattr(comp_s, "values") else np.asarray(comp_s)
        s_idx = comp_s.index if hasattr(comp_s, "index") else index[-len(s_vals):]
        fig.add_trace(
            go.Scatter(
                x=s_idx,
                y=s_vals,
                mode="lines",
                name=name,
                line=dict(color=col_hex, width=2),
            ),
            row=i,
            col=1,
        )

    fig.update_layout(
        template="plotly_white",
        height=220 * n_rows + 80,
        title="Holt-Winters Estimated State Components (Model Scale)",
        showlegend=False,
        margin=dict(l=40, r=40, t=60, b=40),
    )

    return fig


def plot_grid_results(
    grid_df: pd.DataFrame,
    top_n: int = 15,
) -> go.Figure:
    """
    Horizontal bar chart of top grid search specifications comparing AIC and BIC.
    """
    if grid_df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No grid search results to display", showarrow=False)
        fig.update_layout(template="plotly_white")
        return fig

    top_df = grid_df.head(min(top_n, len(grid_df))).copy()
    # Reverse order so best is at the top of horizontal chart
    top_df = top_df.iloc[::-1].reset_index(drop=True)

    fig = go.Figure()

    # AIC Trace with best model highlighted
    aic_colors = ["#2563eb"] * len(top_df)
    if len(aic_colors) > 0:
        aic_colors[-1] = "#059669"

    fig.add_trace(
        go.Bar(
            y=top_df["spec"],
            x=top_df["aic"],
            orientation="h",
            name="AIC",
            marker=dict(color=aic_colors),
            hovertemplate="<b>%{y}</b><br>AIC: %{x:.2f}<extra></extra>",
        )
    )

    # BIC Trace
    fig.add_trace(
        go.Bar(
            y=top_df["spec"],
            x=top_df["bic"],
            orientation="h",
            name="BIC",
            marker_color="#94a3b8",
            hovertemplate="<b>%{y}</b><br>BIC: %{x:.2f}<extra></extra>",
        )
    )

    fig.update_layout(
        template="plotly_white",
        barmode="group",
        title=f"Top {len(top_df)} Candidate Models (Sorted by AIC)",
        xaxis_title="Information Criterion (Lower is Better)",
        yaxis_title="Specification",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=max(400, 30 * len(top_df) + 120),
        margin=dict(l=100, r=40, t=60, b=40),
    )

    return fig

