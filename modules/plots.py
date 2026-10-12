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
    plot_s = clean_s
    if len(clean_s) > 5000:
        step = max(1, len(clean_s) // 5000)
        plot_s = clean_s.iloc[::step]
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=plot_s.index,
            y=plot_s.values,
            mode="lines+markers" if len(plot_s) <= 100 else "lines",
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
            hovertemplate=f"y(t-{lag}): %{{x:.3f}}<br>y(t): %{{y:.3f}}<extra></extra>",
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


def plot_forecast_overlay(
    actual: pd.Series,
    forecasts: Dict[str, pd.DataFrame],
    tail: Optional[int] = None,
    show_bands_for: Optional[str] = None,
) -> go.Figure:
    """
    Build overlay chart comparing ground-truth actuals against holdout forecasts from candidate models.
    Actuals are styled in dark slate; candidate models are styled with distinct palette colors.
    Optional 95% prediction interval band for one selected model.
    """
    palette = [
        "#2563eb", "#059669", "#d97706", "#7c3aed",
        "#db2777", "#0891b2", "#ea580c", "#4f46e5", "#16a34a", "#9333ea"
    ]

    act_slice = actual.iloc[-tail:] if (tail is not None and tail > 0) else actual
    fig = go.Figure()

    # Actuals
    fig.add_trace(
        go.Scatter(
            x=act_slice.index,
            y=act_slice.values,
            mode="lines+markers",
            marker=dict(size=4),
            line=dict(color="#1e293b", width=2.5),
            name="Actual Holdout",
            hovertemplate="<b>Actual</b>: %{y:.3f}<br>%{x}<extra></extra>",
        )
    )

    # Optional band
    if show_bands_for and show_bands_for in forecasts:
        band_df = forecasts[show_bands_for]
        if "lower" in band_df.columns and "upper" in band_df.columns and band_df["lower"].notna().any():
            fig.add_trace(
                go.Scatter(
                    x=band_df.index,
                    y=band_df["lower"].values,
                    mode="lines",
                    line=dict(width=0),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=band_df.index,
                    y=band_df["upper"].values,
                    mode="lines",
                    line=dict(width=0),
                    fill="tonexty",
                    fillcolor="rgba(37, 99, 235, 0.15)",
                    name=f"{show_bands_for} (95% CI)",
                    hoverinfo="skip",
                )
            )

    # Forecast mean lines
    for idx, (label, fc_df) in enumerate(forecasts.items()):
        if fc_df is None or fc_df.empty or "mean" not in fc_df.columns:
            continue
        c = palette[idx % len(palette)]
        fig.add_trace(
            go.Scatter(
                x=fc_df.index,
                y=fc_df["mean"].values,
                mode="lines+markers",
                marker=dict(size=4),
                line=dict(color=c, width=2),
                name=label,
                hovertemplate=f"<b>{label}</b>: %{{y:.3f}}<br>%{{x}}<extra></extra>",
            )
        )

    fig.update_layout(
        template="plotly_white",
        title="Model Forecasts vs Holdout Ground Truth",
        xaxis_title="Date",
        yaxis_title="Value",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=60, b=40),
    )
    return fig


def plot_metric_bars(table: pd.DataFrame, metric: str = "RMSE") -> go.Figure:
    """
    Build horizontal bar chart comparing models on a specific accuracy metric.
    Bars are sorted ascending, the best model is highlighted, and NaNs are skipped.
    """
    fig = go.Figure()
    if table.empty:
        fig.update_layout(template="plotly_white", title=f"Model Comparison by {metric}")
        return fig

    # Find matching metric column
    target_col = None
    for c in table.columns:
        if c.upper() == metric.upper():
            target_col = c
            break

    if target_col is None:
        fig.update_layout(template="plotly_white", title=f"Metric '{metric}' not found in table")
        return fig

    valid = table.dropna(subset=[target_col]).copy()
    if valid.empty:
        fig.update_layout(template="plotly_white", title=f"No valid data for metric {metric}")
        return fig

    # Sort ascending so lowest error is at the top of horizontal bars
    valid = valid.sort_values(by=target_col, ascending=False).reset_index(drop=True)
    labels = valid["Model"].tolist()
    vals = valid[target_col].tolist()

    colors = ["#94a3b8"] * len(vals)
    if len(colors) > 0:
        colors[-1] = "#2563eb"  # best is the last one (lowest value)

    fig.add_trace(
        go.Bar(
            x=vals,
            y=labels,
            orientation="h",
            marker=dict(color=colors),
            hovertemplate=f"<b>%{{y}}</b><br>{metric}: %{{x:.4f}}<extra></extra>",
        )
    )

    fig.update_layout(
        template="plotly_white",
        title=f"Model Comparison: {metric.upper()} (Lower is Better)",
        xaxis_title=metric.upper(),
        yaxis_title="Model",
        height=max(350, 30 * len(labels) + 100),
        margin=dict(l=120, r=40, t=50, b=40),
    )
    return fig


def plot_ic_bars(table: pd.DataFrame) -> go.Figure:
    """
    Build Information Criteria comparison bar chart grouped and faceted by IC_group.
    Never mixes incompatible model classes (ARIMA vs ETS).
    """
    if table.empty:
        fig = go.Figure()
        fig.update_layout(template="plotly_white", title="Information Criteria Comparison")
        return fig

    # Filter to models with comparable IC groups
    df_ic = table[table["IC_group"].notna() & (table["IC_group"] != "n/a")].copy()
    df_ic = df_ic.dropna(subset=["AIC", "BIC"])

    if df_ic.empty:
        fig = go.Figure()
        fig.update_layout(
            template="plotly_white",
            title="Information Criteria Comparison",
            annotations=[
                dict(
                    text="No fitted models with comparable Information Criteria available.",
                    xref="paper",
                    yref="paper",
                    x=0.5,
                    y=0.5,
                    showarrow=False,
                    font=dict(size=14, color="#64748b"),
                )
            ],
        )
        return fig

    groups = df_ic["IC_group"].unique()
    n_groups = len(groups)

    fig = make_subplots(
        rows=n_groups,
        cols=1,
        shared_xaxes=False,
        vertical_spacing=0.15,
        subplot_titles=[f"IC Group: {g}" for g in groups],
    )

    for row_idx, grp in enumerate(groups, start=1):
        sub_df = df_ic[df_ic["IC_group"] == grp].sort_values(by="AIC", ascending=True)
        fig.add_trace(
            go.Bar(
                x=sub_df["AIC"],
                y=sub_df["Model"],
                orientation="h",
                name="AIC",
                marker_color="#2563eb",
                showlegend=(row_idx == 1),
                hovertemplate="<b>%{y}</b><br>AIC: %{x:.2f}<extra></extra>",
            ),
            row=row_idx,
            col=1,
        )
        fig.add_trace(
            go.Bar(
                x=sub_df["BIC"],
                y=sub_df["Model"],
                orientation="h",
                name="BIC",
                marker_color="#059669",
                showlegend=(row_idx == 1),
                hovertemplate="<b>%{y}</b><br>BIC: %{x:.2f}<extra></extra>",
            ),
            row=row_idx,
            col=1,
        )

    fig.update_layout(
        template="plotly_white",
        barmode="group",
        title="Information Criteria (AIC & BIC) Ranked Within Groups",
        height=max(400, 160 * n_groups + 100),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=120, r=40, t=60, b=40),
    )
    return fig


def plot_error_by_horizon(errors: Dict[str, pd.Series]) -> go.Figure:
    """
    Plot absolute forecast error progression across each step of the holdout horizon (h = 1, 2, ...).
    """
    fig = go.Figure()
    palette = [
        "#2563eb", "#059669", "#d97706", "#7c3aed",
        "#db2777", "#0891b2", "#ea580c", "#4f46e5", "#16a34a", "#9333ea"
    ]

    for idx, (label, err_s) in enumerate(errors.items()):
        if err_s is None or len(err_s) == 0:
            continue
        h_steps = np.arange(1, len(err_s) + 1)
        c = palette[idx % len(palette)]
        fig.add_trace(
            go.Scatter(
                x=h_steps,
                y=np.abs(err_s.values),
                mode="lines+markers",
                marker=dict(size=5),
                line=dict(color=c, width=2),
                name=label,
                hovertemplate=f"<b>{label}</b><br>Step h=%{{x}}: Abs Error=%{{y:.3f}}<extra></extra>",
            )
        )

    fig.update_layout(
        template="plotly_white",
        title="Forecast Error Growth by Horizon Step (h)",
        xaxis_title="Horizon Step Ahead (h)",
        yaxis_title="Absolute Error |y - ŷ|",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=60, b=40),
    )
    return fig


def plot_future_forecast(
    history: pd.Series,
    forecast_df: pd.DataFrame,
    label: str,
    tail: Optional[int] = None,
    holdout_forecast: Optional[pd.DataFrame] = None,
) -> go.Figure:
    """
    Build out-of-sample forward projection plot showing history tail, forecast mean,
    95% prediction intervals, and a vertical demarcation line at the forecast origin.
    Uses add_shape (NOT add_vline with datetime) for compatibility and precision.
    """
    hist_slice = history.iloc[-tail:] if (tail is not None and tail > 0) else history.iloc[-min(len(history), 100):]

    fig = go.Figure()

    # Historical data trace
    fig.add_trace(
        go.Scatter(
            x=hist_slice.index,
            y=hist_slice.values,
            mode="lines+markers",
            marker=dict(size=4),
            line=dict(color="#334155", width=2),
            name="Historical Data",
            hovertemplate="<b>History</b>: %{y:.3f}<br>%{x}<extra></extra>",
        )
    )

    # Optional holdout overlay for context
    if holdout_forecast is not None and not holdout_forecast.empty and "mean" in holdout_forecast.columns:
        fig.add_trace(
            go.Scatter(
                x=holdout_forecast.index,
                y=holdout_forecast["mean"].values,
                mode="lines",
                line=dict(color="#94a3b8", width=1.5, dash="dash"),
                name="Holdout Validation",
                hovertemplate="<b>Holdout</b>: %{y:.3f}<extra></extra>",
            )
        )

    # Prediction Interval Band
    if "lower" in forecast_df.columns and "upper" in forecast_df.columns and forecast_df["lower"].notna().any():
        fig.add_trace(
            go.Scatter(
                x=forecast_df.index,
                y=forecast_df["lower"].values,
                mode="lines",
                line=dict(width=0),
                showlegend=False,
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=forecast_df.index,
                y=forecast_df["upper"].values,
                mode="lines",
                line=dict(width=0),
                fill="tonexty",
                fillcolor="rgba(37, 99, 235, 0.15)",
                name="95% Prediction Interval",
                hoverinfo="skip",
            )
        )

    # Future Mean Forecast
    if "mean" in forecast_df.columns:
        fig.add_trace(
            go.Scatter(
                x=forecast_df.index,
                y=forecast_df["mean"].values,
                mode="lines+markers",
                marker=dict(size=5, color="#2563eb"),
                line=dict(color="#2563eb", width=2.5),
                name=f"{label} Forecast",
                hovertemplate=f"<b>{label}</b>: %{{y:.3f}}<br>%{{x}}<extra></extra>",
            )
        )

    # Vertical demarcation line at forecast origin using add_shape
    origin_x = forecast_df.index[0]
    fig.add_shape(
        type="line",
        x0=origin_x,
        x1=origin_x,
        y0=0,
        y1=1,
        yref="paper",
        line=dict(color="#ef4444", width=2, dash="dash"),
    )

    fig.add_annotation(
        x=origin_x,
        y=1.0,
        yref="paper",
        text="Forecast Origin",
        showarrow=False,
        xanchor="left",
        yanchor="bottom",
        font=dict(color="#ef4444", size=11, family="sans-serif"),
    )

    fig.update_layout(
        template="plotly_white",
        title=f"Out-of-Sample Forward Forecast: {label}",
        xaxis_title="Date",
        yaxis_title="Value",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=60, b=40),
    )
    return fig


def plot_indexed_overlay(
    series_a: pd.Series,
    series_b: pd.Series,
    names: Tuple[str, str] = ("Dataset A", "Dataset B"),
    title: Optional[str] = None,
) -> go.Figure:
    """
    Build an indexed time series overlay plot where both series are rebased to 100 at their start.
    """
    clean_a = series_a.dropna() if series_a is not None else pd.Series(dtype=float)
    clean_b = series_b.dropna() if series_b is not None else pd.Series(dtype=float)

    base_a = float(clean_a.iloc[0]) if len(clean_a) > 0 and clean_a.iloc[0] != 0 else 1.0
    base_b = float(clean_b.iloc[0]) if len(clean_b) > 0 and clean_b.iloc[0] != 0 else 1.0

    idx_a = (clean_a / base_a) * 100.0 if len(clean_a) > 0 else clean_a
    idx_b = (clean_b / base_b) * 100.0 if len(clean_b) > 0 else clean_b

    if len(idx_a) > 5000:
        idx_a = idx_a.iloc[::max(1, len(idx_a) // 5000)]
    if len(idx_b) > 5000:
        idx_b = idx_b.iloc[::max(1, len(idx_b) // 5000)]

    fig = go.Figure()

    if len(idx_a) > 0:
        fig.add_trace(
            go.Scatter(
                x=idx_a.index,
                y=idx_a.values,
                mode="lines",
                name=names[0],
                line=dict(color="#2563eb", width=2),
                hovertemplate=f"<b>{names[0]}</b>: %{{y:.1f}}<extra></extra>",
            )
        )

    if len(idx_b) > 0:
        fig.add_trace(
            go.Scatter(
                x=idx_b.index,
                y=idx_b.values,
                mode="lines",
                name=names[1],
                line=dict(color="#10b981", width=2),
                hovertemplate=f"<b>{names[1]}</b>: %{{y:.1f}}<extra></extra>",
            )
        )

    fig.update_layout(
        template="plotly_white",
        title=title or "Indexed Growth Comparison (Rebased to 100 at Start)",
        xaxis_title="Date / Time",
        yaxis_title="Index Level (Base = 100)",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=60, b=40),
    )

    return fig


def plot_kpi_grouped_bars(
    kpi_df: pd.DataFrame,
    names: Tuple[str, str] = ("Dataset A", "Dataset B"),
) -> go.Figure:
    """
    Build a grouped bar chart comparing key scale-free forecast metrics (MASE, sMAPE, RelRMSE).
    """
    metrics_to_plot = ["Best Model MASE", "Best Model sMAPE", "RelRMSE vs Naive"]
    labels = ["MASE", "sMAPE (%)", "RelRMSE vs Naive"]
    val_a: List[float] = []
    val_b: List[float] = []

    for m in metrics_to_plot:
        row = kpi_df[kpi_df["KPI"] == m] if (kpi_df is not None and "KPI" in kpi_df.columns and not kpi_df.empty) else pd.DataFrame()
        if not row.empty:
            a_str = str(row["A"].iloc[0]).replace("%", "").strip()
            b_str = str(row["B"].iloc[0]).replace("%", "").strip()
            try:
                val_a.append(float(a_str))
            except Exception:
                val_a.append(0.0)
            try:
                val_b.append(float(b_str))
            except Exception:
                val_b.append(0.0)
        else:
            val_a.append(0.0)
            val_b.append(0.0)

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=labels,
            y=val_a,
            name=names[0],
            marker_color="#2563eb",
            text=[f"{v:.2f}" for v in val_a],
            textposition="auto",
        )
    )
    fig.add_trace(
        go.Bar(
            x=labels,
            y=val_b,
            name=names[1],
            marker_color="#10b981",
            text=[f"{v:.2f}" for v in val_b],
            textposition="auto",
        )
    )

    fig.update_layout(
        template="plotly_white",
        title="Comparative Forecast Accuracy (Lower is Better)",
        barmode="group",
        yaxis_title="Metric Value",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=60, b=40),
    )
    return fig


def plot_gauge_or_bar(
    trend_a: float,
    trend_b: float,
    seas_a: float,
    seas_b: float,
    names: Tuple[str, str] = ("Dataset A", "Dataset B"),
) -> go.Figure:
    """
    Build comparative grouped bars for Trend and Seasonal Strength.
    """
    categories = ["Trend Strength (F_t)", "Seasonal Strength (F_s)"]
    val_a = [float(trend_a or 0.0), float(seas_a or 0.0)]
    val_b = [float(trend_b or 0.0), float(seas_b or 0.0)]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=categories,
            y=val_a,
            name=names[0],
            marker_color="#2563eb",
            text=[f"{v:.2f}" for v in val_a],
            textposition="auto",
        )
    )
    fig.add_trace(
        go.Bar(
            x=categories,
            y=val_b,
            name=names[1],
            marker_color="#10b981",
            text=[f"{v:.2f}" for v in val_b],
            textposition="auto",
        )
    )

    fig.update_layout(
        template="plotly_white",
        title="Component Strengths Comparison (Hyndman Metrics)",
        barmode="group",
        yaxis=dict(range=[0, 1.05], title="Strength Index [0, 1]"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=60, b=40),
    )
    return fig


def plot_arima_family_bars(
    bundle_a,
    bundle_b,
    metric: str = "MASE",
    names: Tuple[str, str] = ("Dataset A", "Dataset B"),
) -> go.Figure:
    """
    Build comparative grouped bars for Box-Jenkins ARIMA family models:
    AR, MA, ARMA, ARIMA, and SARIMA.
    """
    families = [
        ("ar", "AR"),
        ("ma", "MA"),
        ("arma", "ARMA"),
        ("arima", "ARIMA"),
        ("sarima", "SARIMA"),
    ]
    categories = [label for _, label in families]

    map_a = {r.family: r for r in getattr(bundle_a, "model_results", [])}
    map_b = {r.family: r for r in getattr(bundle_b, "model_results", [])}

    def _extract_val(res, met):
        if not res:
            return None
        met_lower = met.lower()
        if met_lower in ("mase", "smape", "rmse", "mae"):
            v = res.metrics_test.get(met_lower)
            return float(v) if (v is not None and not np.isnan(v)) else None
        elif met_lower in ("aic", "bic"):
            v = (res.information_criteria or {}).get(met_lower)
            return float(v) if (v is not None and not np.isnan(v)) else None
        return None

    def _format_text(val, met):
        if val is None:
            return "N/A"
        met_lower = met.lower()
        if met_lower == "mase":
            return f"{val:.3f}"
        elif met_lower == "smape":
            return f"{val:.1f}%"
        else:
            return f"{val:.1f}"

    vals_a = [_extract_val(map_a.get(fam_key), metric) for fam_key, _ in families]
    vals_b = [_extract_val(map_b.get(fam_key), metric) for fam_key, _ in families]

    text_a = [_format_text(v, metric) for v in vals_a]
    text_b = [_format_text(v, metric) for v in vals_b]

    plot_y_a = [v if v is not None else 0 for v in vals_a]
    plot_y_b = [v if v is not None else 0 for v in vals_b]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=categories,
            y=plot_y_a,
            name=names[0],
            marker_color="#2563eb",
            text=text_a,
            textposition="auto",
        )
    )
    fig.add_trace(
        go.Bar(
            x=categories,
            y=plot_y_b,
            name=names[1],
            marker_color="#10b981",
            text=text_b,
            textposition="auto",
        )
    )

    metric_label = metric.upper()
    if metric_label == "SMAPE":
        metric_label = "sMAPE (%)"
    elif metric_label == "MASE":
        metric_label = "MASE (Holdout)"

    fig.update_layout(
        template="plotly_white",
        title=f"Box-Jenkins Model Family Comparison ({metric_label})",
        barmode="group",
        xaxis=dict(title="Box-Jenkins Architecture"),
        yaxis=dict(title=metric_label),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=60, b=40),
    )
    return fig


def plot_arima_multi_metric_bars(
    bundle_a,
    bundle_b,
    names: Tuple[str, str] = ("Dataset A", "Dataset B"),
) -> go.Figure:
    """
    Build 3-panel comparative grouped bars for Box-Jenkins ARIMA family models:
    AR, MA, ARMA, ARIMA, and SARIMA across:
    1. Holdout MASE (Lower is Better)
    2. Holdout sMAPE (%) (Lower is Better)
    3. AIC Information Criterion (Lower is Better)
    """
    families = [
        ("ar", "AR"),
        ("ma", "MA"),
        ("arma", "ARMA"),
        ("arima", "ARIMA"),
        ("sarima", "SARIMA"),
    ]
    categories = [label for _, label in families]

    map_a = {r.family: r for r in getattr(bundle_a, "model_results", [])}
    map_b = {r.family: r for r in getattr(bundle_b, "model_results", [])}

    def _extract(res, met):
        if not res:
            return None
        m_low = met.lower()
        if m_low in ("mase", "smape"):
            v = res.metrics_test.get(m_low)
            return float(v) if (v is not None and not np.isnan(v)) else None
        elif m_low == "aic":
            v = (res.information_criteria or {}).get("aic")
            return float(v) if (v is not None and not np.isnan(v)) else None
        return None

    metrics_meta = [
        ("mase", "Holdout MASE", "{:.3f}"),
        ("smape", "Holdout sMAPE (%)", "{:.1f}%"),
        ("aic", "Akaike IC (AIC)", "{:.1f}"),
    ]

    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=[title for _, title, _ in metrics_meta],
        horizontal_spacing=0.08,
    )

    for c_idx, (m_key, title, fmt) in enumerate(metrics_meta, start=1):
        vals_a = [_extract(map_a.get(fk), m_key) for fk, _ in families]
        vals_b = [_extract(map_b.get(fk), m_key) for fk, _ in families]

        txt_a = [fmt.format(v) if v is not None else "N/A" for v in vals_a]
        txt_b = [fmt.format(v) if v is not None else "N/A" for v in vals_b]

        y_a = [v if v is not None else 0 for v in vals_a]
        y_b = [v if v is not None else 0 for v in vals_b]

        fig.add_trace(
            go.Bar(
                x=categories,
                y=y_a,
                name=names[0],
                marker_color="#2563eb",
                text=txt_a,
                textposition="auto",
                showlegend=(c_idx == 1),
            ),
            row=1,
            col=c_idx,
        )
        fig.add_trace(
            go.Bar(
                x=categories,
                y=y_b,
                name=names[1],
                marker_color="#10b981",
                text=txt_b,
                textposition="auto",
                showlegend=(c_idx == 1),
            ),
            row=1,
            col=c_idx,
        )

    fig.update_layout(
        template="plotly_white",
        title="Box-Jenkins Architecture Evaluation: Multi-Metric Comparison (AR, MA, ARMA, ARIMA, SARIMA)",
        barmode="group",
        legend=dict(orientation="h", yanchor="bottom", y=1.08, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=80, b=40),
    )
    return fig


def plot_arima_dual_overlay(
    bundle_a,
    bundle_b,
    names: Tuple[str, str] = ("Dataset A", "Dataset B"),
) -> go.Figure:
    """
    Build stacked 2-row subplot comparing actual holdout against Box-Jenkins forecasts:
    AR, MA, ARMA, ARIMA, and SARIMA for Dataset A (Row 1) and Dataset B (Row 2).
    """
    bj_families = [
        ("ar", "AR", "#2563eb", "dot"),
        ("ma", "MA", "#7c3aed", "dash"),
        ("arma", "ARMA", "#f59e0b", "dashdot"),
        ("arima", "ARIMA", "#059669", "longdash"),
        ("sarima", "SARIMA", "#dc2626", "solid"),
    ]

    fig = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=[
            f"🔹 {names[0]} - Actual Holdout vs Box-Jenkins Models",
            f"🔸 {names[1]} - Actual Holdout vs Box-Jenkins Models",
        ],
        vertical_spacing=0.14,
    )

    for row_idx, bundle in enumerate([bundle_a, bundle_b], start=1):
        if not bundle or bundle.series is None or len(bundle.series) <= bundle.holdout:
            continue

        actual_holdout = bundle.series.iloc[-bundle.holdout:]
        fig.add_trace(
            go.Scatter(
                x=actual_holdout.index,
                y=actual_holdout.values,
                mode="lines+markers",
                marker=dict(size=4),
                line=dict(color="#0f172a", width=2.8),
                name="Actual Holdout" if row_idx == 1 else "Actual (B)",
                showlegend=(row_idx == 1),
                hovertemplate="<b>Actual</b>: %{y:.3f}<br>%{x}<extra></extra>",
            ),
            row=row_idx,
            col=1,
        )

        map_res = {r.family: r for r in getattr(bundle, "model_results", [])}
        for fam_key, label, color, dash_style in bj_families:
            res = map_res.get(fam_key)
            if not res or getattr(res, "test_forecast", None) is None or res.test_forecast.empty:
                continue
            fc_df = res.test_forecast
            if "mean" not in fc_df.columns:
                continue

            fig.add_trace(
                go.Scatter(
                    x=fc_df.index,
                    y=fc_df["mean"].values,
                    mode="lines+markers",
                    marker=dict(size=3),
                    line=dict(color=color, width=2, dash=dash_style),
                    name=label,
                    showlegend=(row_idx == 1),
                    hovertemplate=f"<b>{label}</b>: %{{y:.3f}}<br>%{{x}}<extra></extra>",
                ),
                row=row_idx,
                col=1,
            )

    fig.update_layout(
        template="plotly_white",
        title="Box-Jenkins Holdout Trajectory Comparison (AR, MA, ARMA, ARIMA, SARIMA)",
        legend=dict(orientation="h", yanchor="bottom", y=1.04, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=70, b=40),
        height=700,
    )
    return fig


def plot_arima_radar(
    bundle_a,
    bundle_b,
    names: Tuple[str, str] = ("Dataset A", "Dataset B"),
) -> go.Figure:
    """
    Build comparative radar (spider) plot of Box-Jenkins architectures (AR, MA, ARMA, ARIMA, SARIMA)
    evaluating relative inverse MASE (higher = better accuracy).
    """
    families = [
        ("ar", "AR"),
        ("ma", "MA"),
        ("arma", "ARMA"),
        ("arima", "ARIMA"),
        ("sarima", "SARIMA"),
    ]
    categories = [label for _, label in families]

    map_a = {r.family: r for r in getattr(bundle_a, "model_results", [])}
    map_b = {r.family: r for r in getattr(bundle_b, "model_results", [])}

    def _inv_mase(res):
        if not res:
            return 0.0
        v = res.metrics_test.get("mase")
        if v is not None and not np.isnan(v) and v > 0:
            return round(1.0 / float(v), 3)
        return 0.0

    scores_a = [_inv_mase(map_a.get(fk)) for fk, _ in families]
    scores_b = [_inv_mase(map_b.get(fk)) for fk, _ in families]

    r_a = scores_a + [scores_a[0]]
    r_b = scores_b + [scores_b[0]]
    theta = categories + [categories[0]]

    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(
            r=r_a,
            theta=theta,
            fill="toself",
            name=names[0],
            line=dict(color="#2563eb", width=2),
            fillcolor="rgba(37, 99, 235, 0.2)",
        )
    )
    fig.add_trace(
        go.Scatterpolar(
            r=r_b,
            theta=theta,
            fill="toself",
            name=names[1],
            line=dict(color="#10b981", width=2),
            fillcolor="rgba(16, 185, 129, 0.2)",
        )
    )

    max_rad = max(max(r_a), max(r_b), 1.0) * 1.15

    fig.update_layout(
        template="plotly_white",
        title="Box-Jenkins Relative Accuracy Radar (1 / MASE, Higher = Better)",
        polar=dict(
            radialaxis=dict(visible=True, range=[0, max_rad])
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.04, xanchor="right", x=1),
        margin=dict(l=50, r=50, t=60, b=40),
    )
    return fig


def plot_single_dataset_arima_bars(
    model_results: List[Any],
    metric: str = "MASE",
) -> go.Figure:
    """
    Build bar chart evaluating Box-Jenkins models (AR, MA, ARMA, ARIMA, SARIMA) for a single dataset.
    Highlights the winning Box-Jenkins model.
    """
    families = [
        ("ar", "AR"),
        ("ma", "MA"),
        ("arma", "ARMA"),
        ("arima", "ARIMA"),
        ("sarima", "SARIMA"),
    ]
    categories = [label for _, label in families]
    map_res = {getattr(r, "family", ""): r for r in model_results}

    vals = []
    texts = []
    m_lower = metric.lower()
    for fk, _ in families:
        r = map_res.get(fk)
        if not r:
            vals.append(None)
            texts.append("Not Fitted")
            continue
        if m_lower in ("mase", "smape", "rmse", "mae"):
            v = r.metrics_test.get(m_lower)
        elif m_lower in ("aic", "bic"):
            v = (r.information_criteria or {}).get(m_lower)
        else:
            v = None

        if v is not None and not np.isnan(v):
            vals.append(float(v))
            if m_lower == "smape":
                texts.append(f"{v:.1f}%")
            elif m_lower == "mase":
                texts.append(f"{v:.3f}")
            else:
                texts.append(f"{v:.1f}")
        else:
            vals.append(None)
            texts.append("N/A")

    plot_y = [v if v is not None else 0 for v in vals]
    valid_nums = [v for v in vals if v is not None]
    min_v = min(valid_nums) if valid_nums else None

    colors = []
    for v in vals:
        if v is not None and min_v is not None and v == min_v:
            colors.append("#2563eb")
        else:
            colors.append("#94a3b8")

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=categories,
            y=plot_y,
            text=texts,
            textposition="auto",
            marker=dict(color=colors),
            hovertemplate="<b>%{x}</b><br>Value: %{text}<extra></extra>",
        )
    )

    fig.update_layout(
        template="plotly_white",
        title=f"Box-Jenkins Model Family Comparison ({metric.upper()} - Lowest is Best)",
        xaxis=dict(title="Box-Jenkins Architecture"),
        yaxis=dict(title=metric.upper()),
        margin=dict(l=40, r=40, t=50, b=40),
    )
    return fig





