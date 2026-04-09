import pandas as pd
import numpy as np
import os
from scipy import stats
from scipy.special import inv_boxcox
from sklearn.preprocessing import PowerTransformer
from pmdarima import auto_arima
from pmdarima import ARIMA
from sklearn.model_selection import TimeSeriesSplit
from statsmodels.tsa.stattools import adfuller, kpss, acf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings
warnings.filterwarnings('ignore')


# ─────────────────────────────────────────────
#  AUTO-DETECTION OF SERIES PROPERTIES
# ─────────────────────────────────────────────
#
#  Detects three things per series:
#    1. Stationarity  → controls d, D, and whether transforms are needed
#    2. Returns-like  → refines stationarity (mean≈0, has negatives)
#    3. Seasonality   → controls seasonal terms
#
#  You can still override any column via CONFIG_OVERRIDES.

# Optional: put manual overrides here for any series where the
# auto-detection gets it wrong.  These take priority.
CONFIG_OVERRIDES: dict[str, dict] = {
    # Bond yields are rates, but near-zero episodes make max/min
    # ratio huge → auto-detect sees them as trending levels.
    'us_bond_yield_10y': {'transforms': ['none', 'yeo-johnson'], '_regime': 'persistent-bounded (ovr)'},
    'us_bond_yield_1y':  {'transforms': ['none', 'yeo-johnson'], '_regime': 'persistent-bounded (ovr)'},
}


def _test_stationarity(series: pd.Series, alpha: float = 0.05) -> str:
    """
    Confirmatory approach using both ADF and KPSS.

    ADF  H0: unit root present     → reject = evidence of stationarity
    KPSS H0: series is stationary  → reject = evidence of non-stationarity

    Returns one of: 'stationary', 'non-stationary', 'ambiguous'
    """
    try:
        adf_pval = adfuller(series, autolag='AIC')[1]
    except Exception:
        adf_pval = 1.0

    try:
        kpss_pval = kpss(series, regression='c', nlags='auto')[1]
    except Exception:
        kpss_pval = 0.0

    adf_stationary  = adf_pval < alpha     # reject unit root
    kpss_stationary = kpss_pval >= alpha   # fail to reject stationarity

    if adf_stationary and kpss_stationary:
        return 'stationary'
    elif not adf_stationary and not kpss_stationary:
        return 'non-stationary'
    else:
        return 'ambiguous'


def _is_returns_like(series: pd.Series) -> bool:
    """
    Heuristic: series looks like log returns if it has meaningful
    negative mass and the mean is small relative to the std
    (centered near zero).  Catches: log returns, spreads.
    """
    neg_frac = (series < 0).mean()
    if neg_frac < 0.10:
        return False
    mean_to_std = abs(series.mean()) / series.std() if series.std() > 0 else np.inf
    return mean_to_std < 0.5


def _is_bounded_series(series: pd.Series) -> bool:
    """
    Distinguishes bounded rates / indicators from trending levels.

    Levels (GDP, S&P 500, credit) span orders of magnitude over
    their history: max / min >> 5.  Rates, yields, growth, and
    bounded indicators stay within a narrow band even over decades.

    Series with any negatives are automatically bounded (levels
    don't go negative).  For all-positive series, we check whether
    the historical max/min ratio is small.
    """
    if (series < 0).any():
        return True

    if series.min() <= 0:
        return True   # zeros present → not a typical level

    ratio = series.max() / series.min()
    return ratio < 5


def _detect_seasonality(series: pd.Series, m: int = 4,
                         alpha: float = 0.05) -> bool:
    """
    Checks whether the ACF at the seasonal lag (m) is significantly
    different from zero.  Uses Bartlett's approximation for the
    standard error: SE ≈ 1/√n.

    For robustness we also check lag 2m — a single significant
    seasonal spike could be noise, two is more convincing.
    """
    n = len(series)
    if n < 3 * m:
        return False

    try:
        nlags = min(2 * m + 1, n // 2 - 1)
        acf_vals = acf(series, nlags=nlags, fft=True)
    except Exception:
        return False

    se = 1.96 / np.sqrt(n)

    lag_m_sig  = abs(acf_vals[m]) > se    if len(acf_vals) > m   else False
    lag_2m_sig = abs(acf_vals[2*m]) > se  if len(acf_vals) > 2*m else False

    return lag_m_sig and (lag_2m_sig or abs(acf_vals[m]) > 2 * se)


def detect_series_config(series: pd.Series, col_name: str,
                          m: int = 4) -> dict:
    """
    Auto-detect the appropriate SARIMA config for a single series.

    Decision tree:
      1. Run ADF + KPSS → stationary / non-stationary / ambiguous
      2. Check if returns-like (mean ≈ 0, has negatives)
      3. Check if bounded (rate/yield/indicator, not a trending level)
      4. Check for seasonal ACF spikes

    Key rule: if the tests say "non-stationary" but the series is
    bounded, it's likely just persistent — we still let auto_arima
    pick d, but we DON'T apply log/boxcox (those are for levels).
    """
    series = series.dropna()
    has_negatives = (series < 0).any()
    all_positive  = (series > 0).all()

    stationarity  = _test_stationarity(series)
    is_stationary = stationarity == 'stationary'
    returns_like  = _is_returns_like(series)
    bounded       = _is_bounded_series(series)

    # Seasonality: test on a stationary version
    if is_stationary or bounded:
        test_series = series
    else:
        test_series = series.diff().dropna()
    has_seasonality = _detect_seasonality(test_series, m=m)

    # ── Branch 1: Stationary + returns-like (log returns, spreads) ──
    if is_stationary and returns_like:
        cfg = {
            'd': 0, 'D': 0,
            'seasonal': has_seasonality,
            'max_p': 5, 'max_q': 5,
            'transforms': ['none'],
            '_regime': 'return/growth',
        }

    # ── Branch 2: Stationary + bounded but not returns-like ──
    #    (e.g. unemployment, VIX, consumer confidence)
    elif is_stationary and not returns_like:
        transforms = ['none']
        if all_positive:
            transforms.append('yeo-johnson')
        cfg = {
            'd': 0, 'D': 0,
            'seasonal': has_seasonality,
            'max_p': 3, 'max_q': 3,
            'transforms': transforms,
            '_regime': 'stationary-level',
        }

    # ── Branch 3: Tests say non-stationary, but series is bounded ──
    #    Persistent rate / yield / growth — ADF has low power here.
    #    Let auto_arima decide d (it might still pick 0), but do NOT
    #    apply log/boxcox which only make sense for trending levels.
    elif not is_stationary and bounded:
        transforms = ['none']
        if has_negatives:
            transforms.append('yeo-johnson')
        cfg = {
            'd': None, 'D': 0,
            'seasonal': has_seasonality,
            'max_p': 3, 'max_q': 3,
            'transforms': transforms,
            '_regime': 'persistent-bounded',
        }

    # ── Branch 4: Non-stationary trending level ──
    #    (GDP, S&P 500 close, credit, house prices…)
    else:
        transforms = ['none']
        if all_positive:
            transforms.extend(['log', 'boxcox'])
        elif has_negatives:
            transforms.append('yeo-johnson')
        cfg = {
            'd': None,
            'D': None if has_seasonality else 0,
            'seasonal': has_seasonality,
            'max_p': 3, 'max_q': 3,
            'transforms': transforms,
            '_regime': 'non-stationary',
        }

    # ── Apply manual overrides ──
    overrides = CONFIG_OVERRIDES.get(col_name, {})
    cfg.update(overrides)

    return cfg


def get_series_config(series: pd.Series, col_name: str,
                       m: int = 4) -> dict:
    """
    Main entry point: auto-detect config, with optional manual overrides.
    """
    return detect_series_config(series, col_name, m=m)


# ─────────────────────────────────────────────
#  TRANSFORMATION HELPERS
# ─────────────────────────────────────────────

def apply_transformations(series: pd.Series, requested: list[str] = None) -> dict:
    """
    Returns dict of {name: (transformed_series, inverse_fn)}.
    Only builds transforms listed in `requested`.
    """
    if requested is None:
        requested = ['none', 'log', 'boxcox', 'yeo-johnson']

    transformations = {}
    has_negatives = (series <= 0).any()

    # --- Identity (no transform) ---
    if 'none' in requested:
        transformations['none'] = (series.copy(), lambda x: np.asarray(x))

    # --- Log ---
    if 'log' in requested and not has_negatives:
        log_series = np.log(series)
        transformations['log'] = (log_series, lambda x: np.exp(x))

    # --- Box-Cox ---
    if 'boxcox' in requested and not has_negatives:
        bc_transformed, lam = stats.boxcox(series)
        bc_series = pd.Series(bc_transformed, index=series.index)
        transformations['boxcox'] = (bc_series, lambda x, l=lam: inv_boxcox(x, l))

    # --- Yeo-Johnson (works with negatives) ---
    if 'yeo-johnson' in requested:
        pt = PowerTransformer(method='yeo-johnson')
        yj_vals = pt.fit_transform(series.values.reshape(-1, 1)).flatten()
        yj_series = pd.Series(yj_vals, index=series.index)
        transformations['yeo-johnson'] = (
            yj_series,
            lambda x, _pt=pt: _pt.inverse_transform(
                np.asarray(x).reshape(-1, 1)
            ).flatten()
        )

    return transformations


def extract_diagnostics(model) -> dict:
    """Extracts key diagnostic stats from a fitted auto_arima model."""
    try:
        sarimax_model = model.arima_res_
        diag = {
            'ljung_box_q':    sarimax_model.test_serial_correlation('ljungbox', lags=1)[0][1][0],
            'jarque_bera_jb': sarimax_model.test_normality('jarquebera')[0][1],
            'heterosked_h':   sarimax_model.test_heteroskedasticity('breakvar')[0][1],
        }
        resid = sarimax_model.resid
        from scipy.stats import skew, kurtosis
        return {
            'prob_q':    round(diag['ljung_box_q'], 4),
            'prob_jb':   round(diag['jarque_bera_jb'], 4),
            'prob_h':    round(diag['heterosked_h'], 4),
            'skew':      round(float(skew(resid)), 4),
            'kurtosis':  round(float(kurtosis(resid, fisher=False)), 4),
        }
    except Exception:
        return {'prob_q': None, 'prob_jb': None, 'prob_h': None,
                'skew': None, 'kurtosis': None}


# ─────────────────────────────────────────────
#  FIT & COMPARE MODELS ACROSS TRANSFORMATIONS
# ─────────────────────────────────────────────

def fit_all_transformations(series: pd.Series, col_name: str,
                            m: int = 4, cv_folds: int = 5,
                            cv_horizon: int = None) -> pd.DataFrame:
    """
    Fits auto_arima for each eligible transformation with
    time-series cross-validation.  Uses per-series config for
    differencing, seasonality, lag limits, and transform list.
    CV errors are computed in ORIGINAL scale so that comparisons
    across transforms are meaningful.
    """
    cfg = get_series_config(series, col_name, m=m)
    transformations = apply_transformations(series, requested=cfg['transforms'])

    seasonal = cfg['seasonal']
    d_order  = cfg['d']
    D_order  = cfg['D']
    max_p    = cfg['max_p']
    max_q    = cfg['max_q']

    results = []

    if cv_horizon is None:
        cv_horizon = m if seasonal else 4  # sensible default

    tscv = TimeSeriesSplit(
        n_splits=cv_folds,
        test_size=cv_horizon,
        gap=0,
    )

    for name, (transformed, inverse_fn) in transformations.items():
        try:
            model = auto_arima(
                y=transformed,
                start_p=0, start_q=0,
                max_p=max_p, max_q=max_q,
                start_P=0, start_Q=0,
                max_P=2, max_Q=2,
                m=m if seasonal else 1,
                d=d_order, D=D_order if seasonal else 0,
                seasonal=seasonal,
                information_criterion='aic',
                error_action='ignore',
                suppress_warnings=True,
                stepwise=True,
            )

            # ── Cross-validation (errors in original scale) ──
            cv_maes, cv_rmses = [], []

            for train_idx, test_idx in tscv.split(transformed):
                min_train = 2 * m if seasonal else 20
                if len(train_idx) < min_train:
                    continue

                train_fold = transformed.iloc[train_idx]
                test_fold  = transformed.iloc[test_idx]

                # Corresponding original-scale actuals
                actuals_original = series.iloc[test_idx].values

                try:
                    fold_model = ARIMA(
                        order=model.order,
                        seasonal_order=model.seasonal_order,
                        suppress_warnings=True,
                    )
                    fold_model.fit(train_fold)
                    preds_transformed = fold_model.predict(n_periods=len(test_fold))

                    # Invert to original scale before scoring
                    preds_original = inverse_fn(preds_transformed)

                    errors = actuals_original - preds_original
                    cv_maes.append(np.mean(np.abs(errors)))
                    cv_rmses.append(np.sqrt(np.mean(errors ** 2)))
                except Exception:
                    continue

            cv_mae  = np.mean(cv_maes) if cv_maes else np.nan
            cv_rmse = np.mean(cv_rmses) if cv_rmses else np.nan
            cv_folds_used = len(cv_maes)

            # ── Diagnostics ──
            residuals   = pd.Series(model.resid())
            diagnostics = extract_diagnostics(model)

            results.append({
                'transformation': name,
                'aic':            model.aic(),
                'bic':            model.bic(),
                'cv_rmse':        cv_rmse,
                'cv_mae':         cv_mae,
                'cv_folds_used':  cv_folds_used,
                'order':          model.order,
                'seasonal_order': model.seasonal_order,
                'residual_std':   residuals.std(),
                'abs_skew':       abs(residuals.skew()),
                'abs_kurt':       abs(residuals.kurtosis() - 3),
                'prob_q':         diagnostics['prob_q'],
                'prob_jb':        diagnostics['prob_jb'],
                'prob_h':         diagnostics['prob_h'],
                'skew':           diagnostics['skew'],
                'kurtosis':       diagnostics['kurtosis'],
                '_regime':        cfg.get('_regime', 'unknown'),
                '_model':         model,
                '_inverse_fn':    inverse_fn,
                '_transformed':   transformed,
            })

        except Exception as e:
            print(f"    ⚠ Failed [{name}]: {e}")

    comparison_df = pd.DataFrame(results).sort_values('cv_rmse').reset_index(drop=True)
    return comparison_df


# ─────────────────────────────────────────────
#  FORECAST USING BEST MODEL
# ─────────────────────────────────────────────

def forecast_best_model(comparison_df: pd.DataFrame, n_periods: int = 20,
                        alpha: float = 0.05) -> dict:
    """Selects best model by CV RMSE, returns forecast with CIs in original scale."""
    best = comparison_df.iloc[0]
    model      = best['_model']
    inverse_fn = best['_inverse_fn']

    forecast_transformed, conf_int = model.predict(
        n_periods=n_periods,
        return_conf_int=True,
        alpha=alpha,
    )

    forecast_original = inverse_fn(forecast_transformed)
    ci_lower = inverse_fn(conf_int[:, 0])
    ci_upper = inverse_fn(conf_int[:, 1])

    return {
        'transformation':  best['transformation'],
        'order':           best['order'],
        'seasonal_order':  best['seasonal_order'],
        'aic':             best['aic'],
        'cv_rmse':         best['cv_rmse'],
        'cv_mae':          best['cv_mae'],
        'regime':          best.get('_regime', 'unknown'),
        'forecast':        forecast_original,
        'ci_lower':        ci_lower,
        'ci_upper':        ci_upper,
        'model':           model,
        'inverse_fn':      inverse_fn,
    }


# ─────────────────────────────────────────────
#  PLOTTING — INDIVIDUAL SERIES
# ─────────────────────────────────────────────

def plot_forecast(series: pd.Series, result: dict, col_name: str,
                  n_periods: int = 20, m: int = 4, hist_years: int = 7,
                  ax=None, save_path: str = None):
    """Plots last `hist_years` of history + forecast with confidence interval.
    If save_path is given, saves the figure there (only when creating its own fig)."""
    show_standalone = ax is None
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 5))

    series = series.copy()
    if not isinstance(series.index, pd.DatetimeIndex):
        series.index = pd.to_datetime(series.index)

    cutoff = series.index[-1] - pd.DateOffset(years=hist_years)
    series = series.loc[series.index >= cutoff]

    freq = pd.infer_freq(series.index) or f'{12 // m}MS'
    forecast_idx = pd.date_range(start=series.index[-1],
                                 periods=n_periods + 1, freq=freq)[1:]

    forecast_vals = result['forecast']
    ci_lower      = result['ci_lower']
    ci_upper      = result['ci_upper']

    ax.plot(series.index, series.values, color='#2563EB',
            linewidth=1.5, label='Historical')
    ax.plot(forecast_idx, forecast_vals, color='#DC2626',
            linewidth=1.8, linestyle='--', label='Forecast')
    ax.fill_between(forecast_idx, ci_lower, ci_upper,
                    color='#DC2626', alpha=0.10, label='95% CI')
    ax.plot([series.index[-1], forecast_idx[0]],
            [series.values[-1], forecast_vals[0]],
            color='#DC2626', linewidth=1.8, linestyle='--')
    ax.axvline(series.index[-1], color='#9CA3AF', linewidth=0.8,
               linestyle=':', alpha=0.7)

    regime_tag = result.get('regime', 'unknown')

    model_label = (f"SARIMA{result['order']}x{result['seasonal_order']}  "
                   f"[{result['transformation']}]  ({regime_tag})")
    metrics_label = (f"AIC={result['aic']:.1f}   "
                     f"CV-RMSE={result['cv_rmse']:.4f}   "
                     f"CV-MAE={result['cv_mae']:.4f}")

    ax.set_title(f"{col_name}    {model_label}",
                 fontsize=12, fontweight='bold', loc='left')
    ax.text(1.0, -0.12, metrics_label, transform=ax.transAxes,
            ha='right', va='top', fontsize=8, color='#6B7280')

    ax.legend(loc='upper left', fontsize=8, frameon=False)
    ax.grid(axis='y', alpha=0.25, linewidth=0.6)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.5)
    ax.spines['bottom'].set_linewidth(0.5)

    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.tick_params(axis='both', which='major', labelsize=9, length=4, width=0.5)
    ax.tick_params(axis='x', which='minor', length=0)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=0, ha='center')

    ymin = min(series.values.min(), np.nanmin(ci_lower))
    ymax = max(series.values.max(), np.nanmax(ci_upper))
    margin = (ymax - ymin) * 0.08
    ax.set_ylim(ymin - margin, ymax + margin)

    if show_standalone:
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
        plt.show()
        plt.close(fig)


# ─────────────────────────────────────────────
#  PLOTTING — ALL SERIES (GRID)
# ─────────────────────────────────────────────

def plot_all_forecasts(df: pd.DataFrame, all_results: dict,
                       n_periods: int = 20, m: int = 4, hist_years: int = 7,
                       save_dir: str = None):
    """Plots each column's forecast as a separate figure.
    If save_dir is given, saves each as its own PNG there."""
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    for col, result in all_results.items():
        series = df[col].dropna()
        fpath = os.path.join(save_dir, f"{col}.png") if save_dir else None
        plot_forecast(series, result, col, n_periods=n_periods,
                      m=m, hist_years=hist_years, save_path=fpath)

    if save_dir:
        print(f"\n  📊 Saved {len(all_results)} plots → {save_dir}/")


# ─────────────────────────────────────────────
#  CONFIG SUMMARY — shows what each series gets
# ─────────────────────────────────────────────

def print_config_summary(df: pd.DataFrame, m: int = 4):
    """Pretty-prints the auto-detected config for each column."""
    print(f"\n{'─'*90}")
    print(f"  AUTO-DETECTED SERIES CONFIGURATION")
    print(f"{'─'*90}")
    print(f"  {'Column':<30s} {'Regime':<20s} {'d':>3s} {'D':>3s} {'Seas':>5s} {'max_p':>5s} {'Transforms'}")
    print(f"  {'─'*85}")
    for col in df.columns:
        series = df[col].dropna()
        cfg = get_series_config(series, col, m=m)
        d_str = '?' if cfg['d'] is None else str(cfg['d'])
        D_str = '?' if cfg['D'] is None else str(cfg['D'])
        seas  = '✓' if cfg['seasonal'] else '✗'
        tx    = ', '.join(cfg['transforms'])
        regime = cfg.get('_regime', 'unknown')
        print(f"  {col:<30s} {regime:<20s} {d_str:>3s} {D_str:>3s} {seas:>5s} {cfg['max_p']:>5d}   {tx}")
    print()


# ─────────────────────────────────────────────
#  MAIN PIPELINE — RUNS FOR ALL COLUMNS
# ─────────────────────────────────────────────

def run_sarima_pipeline(df: pd.DataFrame, m: int = 4,
                        n_periods: int = 20, plot: bool = True,
                        save_dir: str = None) -> dict:
    """
    Runs the full SARIMA pipeline for every column in df.
    Auto-detects series type, stationarity, and seasonality.
    If save_dir is given, saves each forecast plot as a separate PNG.
    Returns a dict of {column_name: result_dict}.
    """
    print_config_summary(df, m=m)

    all_results     = {}
    all_comparisons = {}

    for col in df.columns:
        series = df[col].dropna()
        cfg = get_series_config(series, col, m=m)

        print(f"\n{'='*60}")
        print(f"  Processing: {col}")
        print(f"  Regime: {cfg.get('_regime', '?')}  |  d={cfg['d']}  D={cfg['D']}  "
              f"seasonal={cfg['seasonal']}  max_p={cfg['max_p']}")
        print(f"  Transforms: {cfg['transforms']}")
        print(f"{'='*60}")

        # 1. Fit all eligible transformations
        comparison_df = fit_all_transformations(series, col_name=col, m=m)

        if comparison_df.empty:
            print(f"    ⚠ No valid models for {col}, skipping.")
            continue

        # 2. Print comparison table
        display_cols = ['transformation', 'aic', 'bic', 'cv_rmse', 'cv_mae',
                        'cv_folds_used', 'order', 'seasonal_order',
                        'residual_std', 'prob_q', 'prob_jb', 'prob_h',
                        'skew', 'kurtosis']
        print(comparison_df[display_cols].to_string(index=False))

        # 3. Forecast with best model
        result = forecast_best_model(comparison_df, n_periods=n_periods)

        print(f"\n  ✓ Best: [{result['transformation']}]  "
              f"SARIMA{result['order']}x{result['seasonal_order']}  "
              f"AIC={result['aic']:.2f}  "
              f"CV-RMSE={result['cv_rmse']:.4f}")

        all_results[col]     = result
        all_comparisons[col] = comparison_df

    # 4. Plot all forecasts
    if plot:
        plot_all_forecasts(df, all_results, n_periods=n_periods, m=m,
                           hist_years=7, save_dir=save_dir)

    return all_results, all_comparisons


# ─────────────────────────────────────────────
#  RUN IT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    # Assuming your data is in a DataFrame called `Fred_Data`
    # n_periods=20 with m=4 (quarterly) = 5 years
    results, comparisons = run_sarima_pipeline(
        Fred_Data,
        m=4,
        n_periods=20,
        plot=True,
        save_dir='sarima_plots',
    )