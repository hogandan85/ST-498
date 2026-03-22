import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pmdarima.arima import ARIMA
from scipy import stats
from scipy.special import inv_boxcox
from sklearn.preprocessing import PowerTransformer
import warnings
warnings.filterwarnings('ignore')


# ─────────────────────────────────────────────
#  INDEX HELPERS (same as forecasting file)
# ─────────────────────────────────────────────

def _to_datetime_index(index):
    if isinstance(index, pd.DatetimeIndex):
        return index
    if isinstance(index, pd.PeriodIndex):
        return index.to_timestamp()
    try:
        return pd.to_datetime(index)
    except Exception:
        return None

def _get_last_date(series):
    dt_index = _to_datetime_index(series.index)
    return dt_index[-1] if dt_index is not None else None


# ─────────────────────────────────────────────
#  TRANSFORMATION REGISTRY
# ─────────────────────────────────────────────

def get_transform_pair(name: str, series: pd.Series):
    """
    Returns (transform_fn, inverse_fn) for a named transformation.
    Matches the transformations used in SARIMA_Fitting.py.
    """
    if name == 'log':
        return np.log, np.exp

    elif name == 'boxcox':
        _, lam = stats.boxcox(series)
        return (
            lambda x, l=lam: pd.Series(stats.boxcox(x, lmbda=l), index=x.index),
            lambda x, l=lam: inv_boxcox(x, l),
        )

    elif name == 'yeo-johnson':
        pt = PowerTransformer(method='yeo-johnson')
        pt.fit((series + 0.0001).values.reshape(-1, 1))
        return (
            lambda x: pd.Series(
                pt.transform((x + 0.0001).values.reshape(-1, 1)).flatten(),
                index=x.index),
            lambda x: pd.Series(
                pt.inverse_transform(np.array(x).reshape(-1, 1)).flatten() - 0.0001),
        )
    else:
        raise ValueError(f"Unknown transformation: {name}")


# ─────────────────────────────────────────────
#  FIT SARIMAX WITH SELECTED EXOGENOUS VARS
# ─────────────────────────────────────────────

def fit_sarimax(
    series: pd.Series,
    exog_df: pd.DataFrame,
    order: tuple,
    seasonal_order: tuple,
    transformation: str = None,
    series_name: str = "Series",
) -> dict:
    """
    Fits a SARIMAX model using the SARIMA order from the univariate fit
    and the exogenous variables from variable selection.

    Parameters
    ----------
    series : pd.Series
        Target variable (original scale, with DatetimeIndex).
    exog_df : pd.DataFrame
        Exogenous variables aligned to the same index as series.
        These should already be stationary.
    order : tuple
        (p, d, q) from the univariate SARIMA fit.
    seasonal_order : tuple
        (P, D, Q, m) from the univariate SARIMA fit.
    transformation : str or None
        'log', 'boxcox', or 'yeo-johnson'. None = no transformation.
    series_name : str
        Label for output.

    Returns
    -------
    dict with keys:
        model, inverse_fn, transform_fn, order, seasonal_order,
        transformation, series_name, exog_cols, aic, bic, resid
    """
    # ── Align series and exog on shared index ────────────────
    shared_idx = series.index.intersection(exog_df.index)
    y = series.loc[shared_idx]
    X = exog_df.loc[shared_idx]

    print(f"\n  Fitting SARIMAX{order}x{seasonal_order}  "
          f"transform=[{transformation or 'none'}]  "
          f"exog={list(X.columns)}")
    print(f"  Observations: {len(y)}  |  Exog vars: {X.shape[1]}")

    # ── Transform target ─────────────────────────────────────
    if transformation:
        transform_fn, inverse_fn = get_transform_pair(transformation, y)
        y_transformed = transform_fn(y)
    else:
        transform_fn = lambda x: x
        inverse_fn   = lambda x: x
        y_transformed = y

    # ── Fit ───────────────────────────────────────────────────
    model = ARIMA(
        order=order,
        seasonal_order=seasonal_order,
        suppress_warnings=True,
    )
    model.fit(y_transformed, X=X)

    resid = pd.Series(model.resid(), index=shared_idx[-len(model.resid()):])

    print(f"  AIC={model.aic():.2f}  BIC={model.bic():.2f}  "
          f"Residual σ={resid.std():.4f}")

    return {
        'model':           model,
        'inverse_fn':      inverse_fn,
        'transform_fn':    transform_fn,
        'order':           order,
        'seasonal_order':  seasonal_order,
        'transformation':  transformation or 'none',
        'series_name':     series_name,
        'exog_cols':       list(X.columns),
        'aic':             model.aic(),
        'bic':             model.bic(),
        'resid':           resid,
        'n_obs':           len(y),
    }


# ─────────────────────────────────────────────
#  FORECAST SARIMAX
# ─────────────────────────────────────────────

def forecast_sarimax(
    sarimax_result: dict,
    future_exog: pd.DataFrame,
    alpha: float = 0.05,
    freq: str = "QS",
    last_date: pd.Timestamp = None,
) -> dict:
    """
    Forecasts from a fitted SARIMAX model.

    Parameters
    ----------
    sarimax_result : dict
        Output from fit_sarimax.
    future_exog : pd.DataFrame
        Exogenous variable values for the forecast horizon.
        Must have exactly len = n_periods rows and same columns as training exog.
        For crisis dummy, set to 0 unless stress-testing.
        For other variables, you need forecasts/assumptions.
    alpha : float
        CI significance level.
    freq : str
        Date frequency.
    last_date : pd.Timestamp or None
        Last historical date. If None, forecast index is integer.

    Returns
    -------
    dict with forecast_df, model, order, seasonal_order, series_name
    """
    model      = sarimax_result['model']
    inverse_fn = sarimax_result['inverse_fn']
    n_periods  = len(future_exog)

    # ── Predict in transformed space ─────────────────────────
    forecast_trans, conf_int_trans = model.predict(
        n_periods=n_periods,
        X=future_exog,
        return_conf_int=True,
        alpha=alpha,
    )

    # ── Invert ───────────────────────────────────────────────
    forecast_orig = inverse_fn(forecast_trans)
    lower_orig    = inverse_fn(conf_int_trans[:, 0])
    upper_orig    = inverse_fn(conf_int_trans[:, 1])

    # ── Date index ───────────────────────────────────────────
    if last_date is not None:
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
        'forecast': np.array(forecast_orig).flatten(),
        'lower':    np.array(lower_orig).flatten(),
        'upper':    np.array(upper_orig).flatten(),
    }, index=forecast_index)

    return {
        'forecast_df':    forecast_df,
        'model':          model,
        'inverse_fn':     inverse_fn,
        'order':          sarimax_result['order'],
        'seasonal_order': sarimax_result['seasonal_order'],
        'series_name':    sarimax_result['series_name'],
        'exog_cols':      sarimax_result['exog_cols'],
    }


# ─────────────────────────────────────────────
#  PLOT (same style as SARIMA_Forecasting)
# ─────────────────────────────────────────────

def plot_sarimax_forecast(
    historical: pd.Series,
    forecast_result: dict,
    tail_years: int = 10,
    figsize: tuple = (14, 6),
    save_path: str = None,
):
    """Plots historical + SARIMAX forecast with CI band."""

    fdf  = forecast_result['forecast_df']
    name = forecast_result['series_name']

    hist = historical.copy()
    dt_idx = _to_datetime_index(hist.index)
    if dt_idx is not None:
        hist.index = dt_idx

    fdf = fdf.copy()
    fdf_dt = _to_datetime_index(fdf.index)
    if fdf_dt is not None:
        fdf.index = fdf_dt

    if isinstance(hist.index, pd.DatetimeIndex) and tail_years:
        cutoff = hist.index[-1] - pd.DateOffset(years=tail_years)
        hist = hist[hist.index >= cutoff]

    fig, ax = plt.subplots(figsize=figsize, facecolor='white')
    ax.set_facecolor('#fafafa')

    ax.plot(hist.index, hist.values,
            color='#2c3e50', linewidth=1.8, label='Historical', zorder=3)
    ax.plot(fdf.index, fdf['forecast'],
            color='#2980b9', linewidth=2.2, label='SARIMAX Forecast', zorder=3)
    ax.fill_between(fdf.index, fdf['lower'], fdf['upper'],
                    color='#2980b9', alpha=0.12, label='95% CI', zorder=2)

    if isinstance(hist.index, pd.DatetimeIndex) and isinstance(fdf.index, pd.DatetimeIndex):
        ax.plot([hist.index[-1], fdf.index[0]],
                [hist.values[-1], fdf['forecast'].iloc[0]],
                color='#2980b9', linewidth=2, linestyle='--', alpha=0.5, zorder=3)
        ax.axvline(x=fdf.index[0], color='grey', linewidth=0.8,
                   linestyle=':', alpha=0.6, zorder=1)

    exog_str = ', '.join(forecast_result.get('exog_cols', []))
    order_str = (f"SARIMAX{forecast_result['order']}"
                 f"x{forecast_result['seasonal_order']}")
    ax.set_title(f"{name} — 5-Year Forecast\n{order_str}  |  exog: [{exog_str}]",
                 fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel(name, fontsize=11)

    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_minor_locator(mdates.YearLocator(1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    fig.autofmt_xdate(rotation=0, ha='center')

    ax.grid(True, which='major', alpha=0.25, linewidth=0.8)
    ax.grid(True, which='minor', alpha=0.1, linewidth=0.5)
    ax.legend(loc='best', framealpha=0.9, edgecolor='#cccccc')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"  Saved → {save_path}")
    plt.show()


# ─────────────────────────────────────────────
#  BUILD FUTURE EXOG (ASSUMPTION-BASED)
# ─────────────────────────────────────────────

def build_future_exog(
    exog_cols: list,
    n_periods: int = 20,
    candidates: dict = None,
    method: str = 'last_mean',
    crisis_periods: list = None,
) -> pd.DataFrame:
    """
    Constructs a future exogenous DataFrame for forecasting.

    Parameters
    ----------
    exog_cols : list
        Column names matching the training exog.
    n_periods : int
        Forecast horizon.
    candidates : dict
        {name: pd.Series} of the stationary candidate variables.
        Used to compute rolling means for 'last_mean' method.
    method : str
        'zero'      → all exog set to 0 (neutral assumption)
        'last_mean' → trailing 8-quarter mean of each variable
        'last_val'  → repeat last observed value
    crisis_periods : list of int or None
        Which forecast periods (0-indexed) to flag as crisis=1.
        Only applies to the 'crisis' column. Default: all 0.

    Returns
    -------
    pd.DataFrame with n_periods rows and exog_cols columns.
    """
    future = pd.DataFrame(index=range(n_periods))

    for col in exog_cols:
        if col == 'crisis':
            future[col] = 0.0
            if crisis_periods:
                for p in crisis_periods:
                    if 0 <= p < n_periods:
                        future.loc[p, col] = 1.0

        elif method == 'zero':
            future[col] = 0.0

        elif method == 'last_mean' and candidates and col in candidates:
            future[col] = candidates[col].tail(8).mean()

        elif method == 'last_val' and candidates and col in candidates:
            future[col] = candidates[col].iloc[-1]

        else:
            future[col] = 0.0

    return future


# ─────────────────────────────────────────────
#  FULL PIPELINE: SELECTION → FIT → FORECAST
# ─────────────────────────────────────────────

def run_sarimax_pipeline(
    df: pd.DataFrame,
    sarima_results: dict,
    selection_results: dict,
    candidates: dict,
    n_periods: int = 20,
    freq: str = "QS",
    future_method: str = 'last_mean',
    plot: bool = True,
    save_dir: str = None,
) -> dict:
    """
    For each target with a non-empty shortlist, fits SARIMAX and forecasts.
    Falls back to univariate SARIMA forecast for targets with empty shortlists.

    Parameters
    ----------
    df : pd.DataFrame
        Original data with all columns.
    sarima_results : dict
        Output from run_sarima_pipeline (results dict).
    selection_results : dict
        Output from the variable selection notebook cells.
        Each entry must have 'shortlist' key.
    candidates : dict
        {name: pd.Series} of stationary candidate variables.
    n_periods : int
        Forecast horizon.
    freq : str
        Date frequency.
    future_method : str
        How to project future exog values ('zero', 'last_mean', 'last_val').
    plot : bool
        Whether to plot each forecast.
    save_dir : str or None
        Directory to save plots.

    Returns
    -------
    dict of {target_name: {sarimax_result, forecast_result, comparison}}
    """
    import os
    all_output = {}

    for target, sel in selection_results.items():
        shortlist = sel.get('shortlist', [])

        if target not in sarima_results:
            continue

        sarima_res = sarima_results[target]
        series = df[target].dropna()
        last_date = _get_last_date(series)

        print(f"\n{'='*70}")
        print(f"  {target}")
        print(f"{'='*70}")

        # ── No exog → use univariate SARIMA ──────────────────
        if not shortlist:
            print(f"  No exog variables selected → univariate SARIMA only")
            from SARIMA_Forecasting import forecast_from_model
            fc = forecast_from_model(
                model=sarima_res['model'],
                inverse_fn=sarima_res['inverse_fn'],
                n_periods=n_periods,
                series_name=target,
                last_date=last_date,
                freq=freq,
            )
            all_output[target] = {
                'type': 'sarima',
                'forecast': fc,
            }
            if plot:
                from SARIMA_Forecasting import plot_forecast
                sp = os.path.join(save_dir, f"{target}_sarima.png") if save_dir else None
                if save_dir:
                    os.makedirs(save_dir, exist_ok=True)
                plot_forecast(series, fc, save_path=sp)
            continue

        # ── Build exog DataFrame ─────────────────────────────
        exog_df = pd.DataFrame({col: candidates[col] for col in shortlist})
        exog_df = exog_df.dropna()

        # ── Fit SARIMAX ──────────────────────────────────────
        sarimax_res = fit_sarimax(
            series=series,
            exog_df=exog_df,
            order=sarima_res['order'],
            seasonal_order=sarima_res['seasonal_order'],
            transformation=sarima_res['transformation'],
            series_name=target,
        )

        # ── Compare AIC: SARIMA vs SARIMAX ───────────────────
        sarima_aic  = sarima_res['aic']
        sarimax_aic = sarimax_res['aic']
        improvement = sarima_aic - sarimax_aic
        better = 'SARIMAX' if improvement > 0 else 'SARIMA'
        print(f"\n  SARIMA  AIC = {sarima_aic:.2f}")
        print(f"  SARIMAX AIC = {sarimax_aic:.2f}")
        print(f"  Δ AIC = {improvement:.2f}  →  {better} wins")

        # ── Build future exog ────────────────────────────────
        future_exog = build_future_exog(
            exog_cols=shortlist,
            n_periods=n_periods,
            candidates=candidates,
            method=future_method,
        )

        # ── Forecast ─────────────────────────────────────────
        fc = forecast_sarimax(
            sarimax_result=sarimax_res,
            future_exog=future_exog,
            last_date=last_date,
            freq=freq,
        )

        print(f"\n{fc['forecast_df'].head(8).to_string()}")

        all_output[target] = {
            'type':          'sarimax',
            'sarimax_result': sarimax_res,
            'forecast':       fc,
            'sarima_aic':     sarima_aic,
            'sarimax_aic':    sarimax_aic,
            'aic_delta':      improvement,
        }

        # ── Plot ─────────────────────────────────────────────
        if plot:
            sp = None
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                sp = os.path.join(save_dir, f"{target}_sarimax.png")
            plot_sarimax_forecast(series, fc, save_path=sp)

    # ── Summary ──────────────────────────────────────────────
    print(f"\n\n{'='*70}")
    print(f"  SARIMAX PIPELINE SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Target':<22s} {'Type':<10s} {'AIC':>10s} {'Exog Vars'}")
    print(f"  {'─'*22} {'─'*10} {'─'*10} {'─'*35}")
    for target, out in all_output.items():
        if out['type'] == 'sarimax':
            exog = ', '.join(out['sarimax_result']['exog_cols'])
            print(f"  {target:<22s} {'SARIMAX':<10s} {out['sarimax_aic']:>10.2f} {exog}")
        else:
            aic = sarima_results[target]['aic'] if target in sarima_results else float('nan')
            print(f"  {target:<22s} {'SARIMA':<10s} {aic:>10.2f} —")

    return all_output
