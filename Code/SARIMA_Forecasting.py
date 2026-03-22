import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pmdarima import auto_arima
import warnings
warnings.filterwarnings('ignore')


# ─────────────────────────────────────────────
#  INDEX HELPERS
# ─────────────────────────────────────────────

def _to_datetime_index(index):
    """Convert any index type to DatetimeIndex. Returns None if impossible."""
    if isinstance(index, pd.DatetimeIndex):
        return index
    if isinstance(index, pd.PeriodIndex):
        return index.to_timestamp()
    try:
        return pd.to_datetime(index)
    except Exception:
        return None


def _get_last_date(series: pd.Series):
    """Robustly extract the last date as a pd.Timestamp from any series."""
    dt_index = _to_datetime_index(series.index)
    if dt_index is not None:
        return dt_index[-1]
    return None


# ─────────────────────────────────────────────
#  FORECAST FROM A FITTED MODEL
# ─────────────────────────────────────────────

def forecast_from_model(
    model,
    inverse_fn,
    n_periods: int = 20,
    alpha: float = 0.05,
    series_name: str = "Series",
    last_date: pd.Timestamp = None,
    freq: str = "QS",
) -> dict:
    """
    Generates a forecast from a pre-fitted pmdarima model.

    Parameters
    ----------
    model : pmdarima.ARIMA
        A fitted pmdarima auto_arima / ARIMA model object.
    inverse_fn : callable
        Function to invert the transformation applied before fitting.
        e.g. np.exp for log, or the inv_boxcox lambda from SARIMA_Fitting.
    n_periods : int
        Number of periods to forecast (default 20 = 5 years of quarterly data).
    alpha : float
        Significance level for confidence intervals (default 0.05 → 95% CI).
    series_name : str
        Label used in output and plots.
    last_date : pd.Timestamp or None
        The last date in the original training series. Used to build a
        date index for the forecast. If None, forecast index is integer-based.
    freq : str
        Pandas frequency string for the forecast index (default "QS").

    Returns
    -------
    dict with keys:
        'forecast_df'   : pd.DataFrame with columns [forecast, lower, upper]
        'model'         : the model used
        'inverse_fn'    : the inverse function used
        'order'         : ARIMA order tuple
        'seasonal_order': seasonal order tuple
        'series_name'   : label
    """

    # --- Point forecast + confidence interval in transformed space ---
    forecast_trans, conf_int_trans = model.predict(
        n_periods=n_periods,
        return_conf_int=True,
        alpha=alpha,
    )

    # --- Invert back to original scale ---
    forecast_orig = inverse_fn(forecast_trans)
    lower_orig    = inverse_fn(conf_int_trans[:, 0])
    upper_orig    = inverse_fn(conf_int_trans[:, 1])

    # --- Build date index if possible ---
    if last_date is not None:
        # Ensure it's a plain Timestamp
        if isinstance(last_date, pd.Period):
            last_date = last_date.to_timestamp()
        elif not isinstance(last_date, pd.Timestamp):
            try:
                last_date = pd.Timestamp(last_date)
            except Exception:
                last_date = None

    if last_date is not None:
        forecast_index = pd.date_range(
            start=last_date + pd.tseries.frequencies.to_offset(freq),
            periods=n_periods,
            freq=freq,
        )
    else:
        forecast_index = range(n_periods)

    forecast_df = pd.DataFrame({
        'forecast': forecast_orig,
        'lower':    lower_orig,
        'upper':    upper_orig,
    }, index=forecast_index)

    return {
        'forecast_df':    forecast_df,
        'model':          model,
        'inverse_fn':     inverse_fn,
        'order':          model.order,
        'seasonal_order': model.seasonal_order,
        'series_name':    series_name,
    }


# ─────────────────────────────────────────────
#  OVERRIDE: REFIT WITH CUSTOM ORDER, THEN FORECAST
# ─────────────────────────────────────────────

def forecast_with_override(
    series: pd.Series,
    transform_fn: callable,
    inverse_fn: callable,
    order: tuple = (1, 1, 1),
    seasonal_order: tuple = (1, 1, 1, 4),
    n_periods: int = 20,
    alpha: float = 0.05,
    series_name: str = "Series",
    freq: str = "QS",
) -> dict:
    """
    Fits a SARIMA model with explicitly specified orders, then forecasts.
    Use this when you want to override auto_arima's selection.

    Parameters
    ----------
    series : pd.Series
        The raw (original-scale) time series.
    transform_fn : callable
        Transformation to apply before fitting.
        Use `lambda x: x` for no transformation.
        Use `np.log` for log, etc.
    inverse_fn : callable
        Corresponding inverse of transform_fn.
        Use `lambda x: x` for no transformation.
        Use `np.exp` for log, etc.
    order : tuple
        (p, d, q) — non-seasonal ARIMA order.
    seasonal_order : tuple
        (P, D, Q, m) — seasonal order.
    n_periods : int
        Forecast horizon (default 20 = 5 years quarterly).
    alpha : float
        Significance level for confidence intervals.
    series_name : str
        Label for output/plots.
    freq : str
        Pandas frequency string.

    Returns
    -------
    Same dict structure as forecast_from_model.
    """
    from pmdarima.arima import ARIMA

    # --- Transform ---
    transformed = transform_fn(series)

    # --- Fit with explicit orders ---
    model = ARIMA(
        order=order,
        seasonal_order=seasonal_order,
        suppress_warnings=True,
    )
    model.fit(transformed)

    print(f"  Fitted override: SARIMA{order}x{seasonal_order}")
    print(f"  AIC={model.aic():.2f}  BIC={model.bic():.2f}")

    # --- Forecast ---
    last_date = _get_last_date(series)

    return forecast_from_model(
        model=model,
        inverse_fn=inverse_fn,
        n_periods=n_periods,
        alpha=alpha,
        series_name=series_name,
        last_date=last_date,
        freq=freq,
    )


# ─────────────────────────────────────────────
#  PLOTTING
# ─────────────────────────────────────────────

def plot_forecast(
    historical: pd.Series,
    forecast_result: dict,
    tail_years: int = 10,
    figsize: tuple = (14, 6),
    save_path: str = None,
):
    """
    Plots historical series + forecast with confidence band.

    Parameters
    ----------
    historical : pd.Series
        The original (untransformed) historical series.
    forecast_result : dict
        Output from forecast_from_model or forecast_with_override.
    tail_years : int
        How many years of history to show before the forecast (default 10).
    figsize : tuple
        Figure size.
    save_path : str or None
        If provided, saves the figure to this path.
    """
    fdf  = forecast_result['forecast_df']
    name = forecast_result['series_name']

    # ── Normalize historical index to DatetimeIndex ──────────────────
    hist = historical.copy()
    dt_idx = _to_datetime_index(hist.index)
    if dt_idx is not None:
        hist.index = dt_idx

    # ── Normalize forecast index ─────────────────────────────────────
    fdf = fdf.copy()
    fdf_dt = _to_datetime_index(fdf.index)
    if fdf_dt is not None:
        fdf.index = fdf_dt

    # ── Trim history for readability ─────────────────────────────────
    if isinstance(hist.index, pd.DatetimeIndex) and tail_years:
        cutoff = hist.index[-1] - pd.DateOffset(years=tail_years)
        hist = hist[hist.index >= cutoff]

    # ── Plot ─────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=figsize, facecolor='white')
    ax.set_facecolor('#fafafa')

    # Historical
    ax.plot(hist.index, hist.values,
            color='#2c3e50', linewidth=1.8, label='Historical', zorder=3)

    # Forecast
    ax.plot(fdf.index, fdf['forecast'],
            color='#e74c3c', linewidth=2.2, label='Forecast', zorder=3)

    # Confidence band
    ax.fill_between(fdf.index, fdf['lower'], fdf['upper'],
                    color='#e74c3c', alpha=0.12, label='95% CI', zorder=2)

    # Connecting line (history → forecast)
    if isinstance(hist.index, pd.DatetimeIndex) and isinstance(fdf.index, pd.DatetimeIndex):
        ax.plot(
            [hist.index[-1], fdf.index[0]],
            [hist.values[-1], fdf['forecast'].iloc[0]],
            color='#e74c3c', linewidth=2, linestyle='--', alpha=0.5, zorder=3,
        )

    # Vertical line at forecast start
    if isinstance(fdf.index, pd.DatetimeIndex):
        ax.axvline(x=fdf.index[0], color='grey', linewidth=0.8,
                   linestyle=':', alpha=0.6, zorder=1)

    # ── Axis formatting ──────────────────────────────────────────────
    order_str = (f"SARIMA{forecast_result['order']}"
                 f"x{forecast_result['seasonal_order']}")
    ax.set_title(f"{name} — 5-Year Forecast\n{order_str}",
                 fontsize=14, fontweight='bold', pad=12)
    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel(name, fontsize=11)

    # Clean date ticks — major every 2 years, minor every year
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_minor_locator(mdates.YearLocator(1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    fig.autofmt_xdate(rotation=0, ha='center')

    # Grid & legend
    ax.grid(True, which='major', alpha=0.25, linewidth=0.8)
    ax.grid(True, which='minor', alpha=0.1, linewidth=0.5)
    ax.legend(loc='best', framealpha=0.9, edgecolor='#cccccc')

    # Clean spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"  Saved → {save_path}")
    plt.show()


# ─────────────────────────────────────────────
#  BATCH FORECAST — ALL COLUMNS
# ─────────────────────────────────────────────

def forecast_all(
    df: pd.DataFrame,
    results: dict,
    n_periods: int = 20,
    freq: str = "QS",
    alpha: float = 0.05,
    plot: bool = True,
    save_dir: str = None,
) -> dict:
    """
    Forecasts every column using the best model stored in `results`
    (the output of run_sarima_pipeline from SARIMA_Fitting.py).

    Parameters
    ----------
    df : pd.DataFrame
        Original data (for historical plot context).
    results : dict
        {col_name: result_dict} from SARIMA_Fitting.run_sarima_pipeline.
    n_periods : int
        Forecast horizon (20 = 5 years quarterly).
    freq : str
        Pandas frequency alias.
    alpha : float
        CI significance level.
    plot : bool
        Whether to auto-plot each forecast.
    save_dir : str or None
        Directory to save plots. None = don't save.

    Returns
    -------
    dict of {col_name: forecast_result_dict}
    """
    import os
    all_forecasts = {}

    for col, res in results.items():
        print(f"\n{'='*60}")
        print(f"  Forecasting: {col}")
        print(f"  Using: [{res['transformation']}] "
              f"SARIMA{res['order']}x{res['seasonal_order']}  "
              f"AIC={res['aic']:.2f}")
        print(f"{'='*60}")

        series    = df[col].dropna()
        last_date = _get_last_date(series)

        fc = forecast_from_model(
            model=res['model'],
            inverse_fn=res['inverse_fn'],
            n_periods=n_periods,
            alpha=alpha,
            series_name=col,
            last_date=last_date,
            freq=freq,
        )

        print(fc['forecast_df'].to_string())
        all_forecasts[col] = fc

        if plot:
            sp = None
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                sp = os.path.join(save_dir, f"{col}_forecast.png")
            plot_forecast(series, fc, save_path=sp)

    return all_forecasts


# ─────────────────────────────────────────────
#  CONVENIENCE: EXPORT FORECASTS TO EXCEL
# ─────────────────────────────────────────────

def export_forecasts(all_forecasts: dict, path: str = "forecasts.xlsx"):
    """Writes all forecast DataFrames to a single Excel file (one sheet per series)."""
    with pd.ExcelWriter(path, engine='openpyxl') as writer:
        for col, fc in all_forecasts.items():
            sheet = col[:31]  # Excel 31-char sheet name limit
            fc['forecast_df'].to_excel(writer, sheet_name=sheet)
    print(f"\n  ✓ Forecasts exported → {path}")


# ─────────────────────────────────────────────
#  EXAMPLE USAGE
# ─────────────────────────────────────────────

if __name__ == "__main__":

    # ── OPTION A: Use the best models from SARIMA_Fitting ────────────
    #
    # from SARIMA_Fitting import run_sarima_pipeline
    # results, comparisons = run_sarima_pipeline(Fred_Data, m=4, n_periods=8)
    #
    # all_forecasts = forecast_all(
    #     df=Fred_Data,
    #     results=results,
    #     n_periods=20,       # 5 years × 4 quarters
    #     freq="QS",
    #     plot=True,
    #     save_dir="./forecast_plots",
    # )
    # export_forecasts(all_forecasts, "sarima_forecasts.xlsx")


    # ── OPTION B: Override with your own model specification ─────────
    #
    # import numpy as np
    # from scipy import stats
    # from scipy.special import inv_boxcox
    #
    # series = Fred_Data['GDP'].dropna()
    #
    # # Example: override with Box-Cox + custom SARIMA order
    # bc_transformed, lam = stats.boxcox(series)
    #
    # fc = forecast_with_override(
    #     series=series,
    #     transform_fn=lambda x: pd.Series(stats.boxcox(x, lmbda=lam), index=x.index),
    #     inverse_fn=lambda x: inv_boxcox(x, lam),
    #     order=(2, 1, 1),
    #     seasonal_order=(1, 1, 1, 4),
    #     n_periods=20,
    #     series_name="GDP",
    #     freq="QS",
    # )
    #
    # plot_forecast(series, fc)


    # ── OPTION C: No transformation override ─────────────────────────
    #
    # fc = forecast_with_override(
    #     series=Fred_Data['GDP'].dropna(),
    #     transform_fn=lambda x: x,       # identity
    #     inverse_fn=lambda x: x,         # identity
    #     order=(1, 1, 0),
    #     seasonal_order=(0, 1, 1, 4),
    #     n_periods=20,
    #     series_name="GDP",
    #     freq="QS",
    # )

    pass
