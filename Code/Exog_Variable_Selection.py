"""
Exog_Variable_Selection_v2.py
─────────────────────────────
Improved exogenous-variable screening for SARIMAX forecasting.

Key changes from v1:
  1. Replaced Granger-on-residuals with direct SARIMA-vs-SARIMAX model
     comparison (AIC / out-of-sample RMSE).
  2. Added rolling time-series cross-validation so every recommendation
     is backed by out-of-sample evidence.
  3. Uses LAG-ONLY exogenous variables so no future exog forecast is needed.
  4. CCF significance bands now use Bartlett's formula (accounts for
     autocorrelation in residuals) instead of naïve 1/√n.
  5. Added an incremental forward-selection step: adds exog one at a time,
     keeping only those that improve out-of-sample error.

Author: Improved pipeline
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from statsmodels.tsa.stattools import acf
from statsmodels.stats.outliers_influence import variance_inflation_factor
from pmdarima.arima import ARIMA
import warnings
warnings.filterwarnings('ignore')


# ═════════════════════════════════════════════════════════════════════════
#  STEP 1: EXTRACT RESIDUALS  (unchanged, kept for compatibility)
# ═════════════════════════════════════════════════════════════════════════

def get_residuals(result: dict) -> pd.Series:
    """
    Pulls residuals from a fitted SARIMA result dict.

    Parameters
    ----------
    result : dict
        Single-column entry from `results`, e.g. results['PD']

    Returns
    -------
    pd.Series of residuals (transformed space), with the SAME
    DatetimeIndex as the training data used to fit the model.
    """
    model = result['model']
    resid = model.resid()

    # Safely recover the original training series index
    train_series = result.get('train_series', None)
    if train_series is not None and len(train_series) >= len(resid):
        # Model may drop initial observations due to differencing
        idx = train_series.index[-len(resid):]
        return pd.Series(resid, index=idx, name='residuals')

    return pd.Series(resid, name='residuals')


# ═════════════════════════════════════════════════════════════════════════
#  STEP 2: IMPROVED CCF — BARTLETT SIGNIFICANCE BANDS
# ═════════════════════════════════════════════════════════════════════════

def _bartlett_se(x: np.ndarray, max_lag: int) -> np.ndarray:
    """
    Bartlett's formula for the standard error of cross-correlations.

    Under H0 (no cross-correlation), the SE at lag k accounts for
    the autocorrelation structure of each series individually:

        Var(r_xy(k)) ≈ (1/n) * Σ_{j} [ ρ_xx(j)·ρ_yy(j) ]

    This is more conservative (wider bands) than the naïve 1/√n when
    the series have autocorrelation — which SARIMA residuals often do
    at short lags, especially if the model is slightly misspecified.
    """
    n = len(x)
    # Autocorrelation of x up to max_lag
    acf_x = acf(x, nlags=max_lag, fft=True)
    return acf_x


def compute_ccf(residuals: pd.Series,
                exog_series: pd.Series,
                max_lags: int = 12,
                alpha: float = 0.05) -> pd.DataFrame:
    """
    Cross-correlation between SARIMA residuals and a candidate
    exogenous variable, with Bartlett-corrected significance bands.

    Positive lags → exog LEADS residuals (predictive — what you want)
    Negative lags → exog LAGS residuals  (reverse causality)
    """
    shared = residuals.index.intersection(exog_series.index)
    r = residuals.loc[shared].values.astype(float)
    x = exog_series.loc[shared].values.astype(float)

    # Standardize
    r = (r - r.mean()) / (r.std() + 1e-12)
    x = (x - x.mean()) / (x.std() + 1e-12)

    n = len(r)

    # Bartlett SE: use autocorrelations of each series
    acf_r = acf(r, nlags=max_lags, fft=True)
    acf_x = acf(x, nlags=max_lags, fft=True)

    z_crit = norm.ppf(1 - alpha / 2)

    rows = []
    for k in range(-max_lags, max_lags + 1):
        # Cross-correlation at lag k
        if k >= 0:
            corr = np.dot(r[k:], x[:n - k]) / n if k < n else 0.0
        else:
            corr = np.dot(r[:n + k], x[-k:]) / n if -k < n else 0.0

        # Bartlett variance for this lag
        bartlett_var = (1.0 / n) * (1.0 + 2.0 * np.sum(acf_r[1:] * acf_x[1:]))
        threshold = z_crit * np.sqrt(max(bartlett_var, 1.0 / n))

        rows.append({
            'lag': k,
            'ccf': round(corr, 4),
            'threshold': round(threshold, 4),
            'significant': abs(corr) > threshold,
        })

    return pd.DataFrame(rows)


# ═════════════════════════════════════════════════════════════════════════
#  STEP 3: SCAN CANDIDATES  (uses improved CCF)
# ═════════════════════════════════════════════════════════════════════════

def scan_exog_candidates(residuals: pd.Series,
                         candidates: dict,
                         max_lags: int = 8) -> pd.DataFrame:
    """
    Runs CCF for every candidate and returns the strongest
    POSITIVE-LAG signal (exog leads residuals → predictive).
    """
    summary = []

    for name, series in candidates.items():
        ccf_df = compute_ccf(residuals, series, max_lags=max_lags)

        positive = ccf_df[ccf_df['lag'] > 0].copy()
        if positive.empty:
            continue

        best_row = positive.loc[positive['ccf'].abs().idxmax()]

        summary.append({
            'variable':    name,
            'best_lag':    int(best_row['lag']),
            'best_ccf':    best_row['ccf'],
            'abs_ccf':     abs(best_row['ccf']),
            'significant': best_row['significant'],
            'direction':   'positive' if best_row['ccf'] > 0 else 'negative',
        })

    return (pd.DataFrame(summary)
              .sort_values('abs_ccf', ascending=False)
              .reset_index(drop=True))


# ═════════════════════════════════════════════════════════════════════════
#  STEP 4: DIRECT MODEL COMPARISON  (replaces Granger-on-residuals)
# ═════════════════════════════════════════════════════════════════════════
#
#  WHY GRANGER ON RESIDUALS IS PROBLEMATIC
#  ────────────────────────────────────────
#  The standard Granger test asks: "do lagged values of X improve a
#  VAR-style regression of Y on its own lags?"  Internally it fits:
#
#     Y_t = c + Σ a_i·Y_{t-i} + Σ b_j·X_{t-j} + ε_t
#
#  and tests H0: all b_j = 0.
#
#  When Y = SARIMA residuals (approximately white noise), the a_i terms
#  are useless, and you're testing:
#
#     WN_t = c + Σ b_j·X_{t-j} + ε_t
#
#  Problems:
#   1. LOW POWER — the denominator (RSS of restricted model ≈ variance
#      of white noise) is already near its floor, so the F-stat is small.
#      Genuinely helpful exog variables get filtered OUT.
#
#   2. WRONG NULL MODEL — SARIMA has a richer lag structure (seasonal
#      terms, MA terms) than the simple AR(p) that Granger uses
#      internally.  The restricted model inside the Granger test is
#      NOT equivalent to your SARIMA, so "significant" may just mean
#      "X compensates for the difference between AR(p) and SARIMA."
#
#   3. DOUBLE-DIPPING — you selected candidates via CCF on these same
#      residuals, then Granger-test on the same residuals.  The error
#      rates are no longer controlled.
#
#  WHAT TO DO INSTEAD
#  ──────────────────
#  Directly compare SARIMA vs SARIMAX with each candidate:
#   • In-sample:  compare AIC (penalizes extra parameters)
#   • Out-of-sample: rolling-window RMSE (the ultimate test)
#  This uses the ACTUAL model structure and answers the real question:
#  "does adding X to my existing SARIMA improve forecasting?"
#

def compare_sarima_vs_sarimax(
    series: pd.Series,
    exog_series: pd.Series,
    lag: int,
    order: tuple,
    seasonal_order: tuple,
    transform_fn: callable = None,
    inverse_fn: callable = None,
) -> dict:
    """
    Fits SARIMA and SARIMAX (with lagged exog) on the same sample
    and compares AIC / BIC.

    Parameters
    ----------
    series : pd.Series
        Target variable (original scale).
    exog_series : pd.Series
        Candidate exogenous variable (already stationary).
    lag : int
        How many periods to lag the exog (from CCF best_lag).
    order : tuple
        (p, d, q) from your fitted SARIMA.
    seasonal_order : tuple
        (P, D, Q, m) from your fitted SARIMA.
    transform_fn : callable or None
        Transformation applied before fitting (e.g. np.log).
        None → identity.
    inverse_fn : callable or None
        Inverse of transform_fn.  None → identity.

    Returns
    -------
    dict with keys: sarima_aic, sarimax_aic, sarima_bic, sarimax_bic,
                    aic_improvement, bic_improvement, exog_name, lag
    """
    if transform_fn is None:
        transform_fn = lambda x: x
    if inverse_fn is None:
        inverse_fn = lambda x: x

    # Build lagged exog and align with target
    exog_lagged = exog_series.shift(lag).dropna()
    shared = series.index.intersection(exog_lagged.index)
    y = series.loc[shared]
    X = exog_lagged.loc[shared].values.reshape(-1, 1)

    y_trans = transform_fn(y)

    # Fit SARIMA (no exog)
    sarima = ARIMA(order=order, seasonal_order=seasonal_order,
                   suppress_warnings=True)
    sarima.fit(y_trans)

    # Fit SARIMAX (with lagged exog)
    sarimax = ARIMA(order=order, seasonal_order=seasonal_order,
                    suppress_warnings=True)
    sarimax.fit(y_trans, X=X)

    return {
        'sarima_aic':      round(sarima.aic(), 2),
        'sarimax_aic':     round(sarimax.aic(), 2),
        'sarima_bic':      round(sarima.bic(), 2),
        'sarimax_bic':     round(sarimax.bic(), 2),
        'aic_improvement': round(sarima.aic() - sarimax.aic(), 2),
        'bic_improvement': round(sarima.bic() - sarimax.bic(), 2),
        'n_obs':           len(y),
    }


# ═════════════════════════════════════════════════════════════════════════
#  STEP 5: ROLLING TIME-SERIES CROSS-VALIDATION
# ═════════════════════════════════════════════════════════════════════════

def ts_crossval(
    series: pd.Series,
    order: tuple,
    seasonal_order: tuple,
    transform_fn: callable = None,
    inverse_fn: callable = None,
    exog_df: pd.DataFrame = None,
    n_test: int = 8,
    min_train: int = 40,
    step: int = 1,
) -> pd.DataFrame:
    """
    Expanding-window time-series cross-validation.

    Trains on t[0:train_end], forecasts t[train_end : train_end+step],
    slides forward, and collects forecast errors.

    Parameters
    ----------
    series : pd.Series
        Target (original scale).
    order, seasonal_order : tuples
        SARIMA specification.
    transform_fn, inverse_fn : callables or None
        Pre-/post-fitting transformations.
    exog_df : pd.DataFrame or None
        Lagged exogenous variables aligned to series index.
        If None, fits univariate SARIMA.
    n_test : int
        Number of periods reserved for testing (walked through one
        at a time via expanding window).
    min_train : int
        Minimum training window size.
    step : int
        Forecast horizon at each fold (1 = one-step-ahead).

    Returns
    -------
    pd.DataFrame with columns: [date, actual, forecast, error, abs_error, sq_error]
    """
    if transform_fn is None:
        transform_fn = lambda x: x
    if inverse_fn is None:
        inverse_fn = lambda x: x

    n = len(series)
    if n < min_train + n_test:
        raise ValueError(
            f"Series length ({n}) < min_train ({min_train}) + n_test ({n_test}). "
            f"Reduce n_test or min_train."
        )

    start = n - n_test
    results = []

    for t in range(start, n - step + 1):
        y_train = series.iloc[:t]
        y_test  = series.iloc[t:t + step]

        y_trans = transform_fn(y_train)

        try:
            model = ARIMA(order=order, seasonal_order=seasonal_order,
                          suppress_warnings=True)

            if exog_df is not None:
                X_train = exog_df.iloc[:t].values
                X_test  = exog_df.iloc[t:t + step].values
                model.fit(y_trans, X=X_train)
                fc_trans, _ = model.predict(n_periods=step,
                                            X=X_test,
                                            return_conf_int=True)
            else:
                model.fit(y_trans)
                fc_trans, _ = model.predict(n_periods=step,
                                            return_conf_int=True)

            fc = inverse_fn(fc_trans)

            for i in range(step):
                if t + i < n:
                    actual = y_test.iloc[i]
                    pred   = fc[i]
                    results.append({
                        'date':      y_test.index[i],
                        'actual':    actual,
                        'forecast':  pred,
                        'error':     pred - actual,
                        'abs_error': abs(pred - actual),
                        'sq_error':  (pred - actual) ** 2,
                    })
        except Exception as e:
            # If a single fold fails, skip it and continue
            print(f"    CV fold t={t} failed: {e}")
            continue

    return pd.DataFrame(results)


def compare_cv_sarima_vs_sarimax(
    series: pd.Series,
    order: tuple,
    seasonal_order: tuple,
    transform_fn: callable = None,
    inverse_fn: callable = None,
    exog_df: pd.DataFrame = None,
    n_test: int = 8,
    min_train: int = 40,
) -> dict:
    """
    Runs cross-validation for both SARIMA and SARIMAX and returns
    a head-to-head comparison.

    Parameters
    ----------
    series, order, seasonal_order, transform_fn, inverse_fn :
        Same as ts_crossval.
    exog_df : pd.DataFrame
        Lagged exog columns. If None, only SARIMA is evaluated.
    n_test : int
        How many recent periods to use as test folds.
    min_train : int
        Minimum training size.

    Returns
    -------
    dict with RMSE, MAE for SARIMA and SARIMAX, plus improvement %.
    """
    # SARIMA baseline
    cv_sarima = ts_crossval(
        series, order, seasonal_order, transform_fn, inverse_fn,
        exog_df=None, n_test=n_test, min_train=min_train,
    )

    out = {
        'sarima_rmse': np.sqrt(cv_sarima['sq_error'].mean()),
        'sarima_mae':  cv_sarima['abs_error'].mean(),
        'n_folds':     len(cv_sarima),
    }

    if exog_df is not None:
        cv_sarimax = ts_crossval(
            series, order, seasonal_order, transform_fn, inverse_fn,
            exog_df=exog_df, n_test=n_test, min_train=min_train,
        )

        sarimax_rmse = np.sqrt(cv_sarimax['sq_error'].mean())
        sarimax_mae  = cv_sarimax['abs_error'].mean()

        out['sarimax_rmse']      = sarimax_rmse
        out['sarimax_mae']       = sarimax_mae
        out['rmse_improvement%'] = round(
            100 * (out['sarima_rmse'] - sarimax_rmse) / out['sarima_rmse'], 2
        )
        out['mae_improvement%']  = round(
            100 * (out['sarima_mae'] - sarimax_mae) / out['sarima_mae'], 2
        )
        out['cv_sarima']  = cv_sarima
        out['cv_sarimax'] = cv_sarimax

    return out


# ═════════════════════════════════════════════════════════════════════════
#  STEP 6: BUILD LAGGED EXOG  (avoids future-exog problem)
# ═════════════════════════════════════════════════════════════════════════

def build_lagged_exog(candidates: dict,
                      lags: dict,
                      target_index: pd.Index) -> pd.DataFrame:
    """
    Creates a DataFrame of lagged exogenous variables aligned to
    the target series index.

    Because exog at lag k means we use X_{t-k} to predict Y_t,
    at forecast time we only need X values up to the CURRENT period
    (not the future). This eliminates the need to forecast exog.

    Parameters
    ----------
    candidates : dict
        {name: pd.Series} of stationary exog variables.
    lags : dict
        {name: int} — how many periods to lag each variable.
    target_index : pd.Index
        Index of the target series (to align on).

    Returns
    -------
    pd.DataFrame with lagged exog, aligned to target_index.
    """
    frames = {}
    for name, lag in lags.items():
        if name in candidates:
            shifted = candidates[name].shift(lag)
            frames[f"{name}_lag{lag}"] = shifted

    exog_df = pd.DataFrame(frames)
    exog_df = exog_df.reindex(target_index)

    return exog_df


# ═════════════════════════════════════════════════════════════════════════
#  STEP 7: VIF  (unchanged)
# ═════════════════════════════════════════════════════════════════════════

def compute_vif(exog_df: pd.DataFrame) -> pd.DataFrame:
    """Variance Inflation Factor for each column in exog_df."""
    X = exog_df.dropna().values
    if X.shape[1] < 2:
        return pd.DataFrame({'variable': exog_df.columns, 'vif': [1.0] * X.shape[1]})

    vifs = []
    for i in range(X.shape[1]):
        vifs.append({
            'variable': exog_df.columns[i],
            'vif': round(variance_inflation_factor(X, i), 2),
        })
    return pd.DataFrame(vifs).sort_values('vif', ascending=False).reset_index(drop=True)


# ═════════════════════════════════════════════════════════════════════════
#  STEP 8: FORWARD SELECTION — ADD EXOG ONE AT A TIME
# ═════════════════════════════════════════════════════════════════════════

def forward_select_exog(
    series: pd.Series,
    order: tuple,
    seasonal_order: tuple,
    candidates: dict,
    ccf_summary: pd.DataFrame,
    transform_fn: callable = None,
    inverse_fn: callable = None,
    n_test: int = 8,
    min_train: int = 40,
    vif_threshold: float = 5.0,
    max_exog: int = 3,
    verbose: bool = True,
) -> dict:
    """
    Greedy forward selection: starting from the SARIMA baseline,
    adds one exogenous variable at a time (in order of CCF strength),
    keeping it ONLY if out-of-sample RMSE improves.

    Parameters
    ----------
    series : pd.Series
        Target variable (original scale).
    order : tuple
        (p, d, q) from your fitted SARIMA.
    seasonal_order : tuple
        (P, D, Q, m) from your fitted SARIMA.
    candidates : dict
        {name: pd.Series} of stationary exog candidates.
    ccf_summary : pd.DataFrame
        Output of scan_exog_candidates — ranked by |CCF|.
    transform_fn, inverse_fn : callables or None
    n_test, min_train : int
        Cross-validation parameters.
    vif_threshold : float
        Maximum VIF allowed (drop if exceeded).
    max_exog : int
        Maximum number of exog variables to include.
    verbose : bool

    Returns
    -------
    dict with keys:
        'selected'     : list of (variable_name, lag) tuples
        'exog_df'      : pd.DataFrame of lagged exog for final model
        'baseline_rmse': float — SARIMA-only out-of-sample RMSE
        'final_rmse'   : float — best SARIMAX out-of-sample RMSE
        'history'       : list of dicts showing each step
    """
    if transform_fn is None:
        transform_fn = lambda x: x
    if inverse_fn is None:
        inverse_fn = lambda x: x

    # Only consider significant CCF variables
    sig_candidates = ccf_summary[ccf_summary['significant']].copy()

    if sig_candidates.empty:
        if verbose:
            print("  No significant CCF candidates. Staying with SARIMA.")
        return {
            'selected': [],
            'exog_df': None,
            'baseline_rmse': None,
            'final_rmse': None,
            'history': [],
        }

    # SARIMA baseline CV
    if verbose:
        print("  Running SARIMA baseline cross-validation...")

    cv_base = ts_crossval(
        series, order, seasonal_order, transform_fn, inverse_fn,
        exog_df=None, n_test=n_test, min_train=min_train,
    )
    best_rmse = np.sqrt(cv_base['sq_error'].mean())
    baseline_rmse = best_rmse

    if verbose:
        print(f"  Baseline SARIMA RMSE: {best_rmse:.4f}")

    selected = []   # list of (name, lag)
    history = [{'step': 0, 'added': '(baseline)', 'rmse': best_rmse,
                'improvement%': 0.0, 'kept': True}]

    for _, row in sig_candidates.iterrows():
        if len(selected) >= max_exog:
            break

        var_name = row['variable']
        lag      = int(row['best_lag'])

        # Build current exog set + this candidate
        trial_lags = {n: l for n, l in selected}
        trial_lags[var_name] = lag

        exog_df = build_lagged_exog(candidates, trial_lags, series.index)

        # Check VIF
        clean_exog = exog_df.dropna()
        if clean_exog.shape[1] >= 2:
            vif_df = compute_vif(clean_exog)
            max_vif = vif_df['vif'].max()
            if max_vif > vif_threshold:
                if verbose:
                    print(f"  SKIP {var_name} (lag {lag}): VIF={max_vif:.1f} > {vif_threshold}")
                history.append({'step': len(history), 'added': var_name,
                                'rmse': None, 'improvement%': None,
                                'kept': False, 'reason': f'VIF={max_vif:.1f}'})
                continue

        # Cross-validate with this exog set
        # Align series and exog_df to drop leading NaNs
        valid_mask = exog_df.notna().all(axis=1)
        valid_idx  = series.index[valid_mask.reindex(series.index, fill_value=False)]
        y_aligned  = series.loc[valid_idx]
        X_aligned  = exog_df.loc[valid_idx]

        try:
            cv_trial = ts_crossval(
                y_aligned, order, seasonal_order, transform_fn, inverse_fn,
                exog_df=X_aligned, n_test=n_test, min_train=min_train,
            )
            trial_rmse = np.sqrt(cv_trial['sq_error'].mean())
        except Exception as e:
            if verbose:
                print(f"  SKIP {var_name} (lag {lag}): CV failed — {e}")
            history.append({'step': len(history), 'added': var_name,
                            'rmse': None, 'improvement%': None,
                            'kept': False, 'reason': str(e)})
            continue

        improvement = 100 * (best_rmse - trial_rmse) / best_rmse

        if trial_rmse < best_rmse:
            selected.append((var_name, lag))
            best_rmse = trial_rmse
            kept = True
            if verbose:
                print(f"  ADD  {var_name} (lag {lag}): "
                      f"RMSE {trial_rmse:.4f} ({improvement:+.2f}%) ✓")
        else:
            kept = False
            if verbose:
                print(f"  SKIP {var_name} (lag {lag}): "
                      f"RMSE {trial_rmse:.4f} ({improvement:+.2f}%) — no improvement")

        history.append({
            'step': len(history),
            'added': var_name,
            'lag': lag,
            'rmse': trial_rmse,
            'improvement%': round(improvement, 2),
            'kept': kept,
        })

    # Build final exog df
    final_lags = {n: l for n, l in selected}
    final_exog = build_lagged_exog(candidates, final_lags, series.index) if selected else None

    return {
        'selected':      selected,
        'exog_df':       final_exog,
        'baseline_rmse': baseline_rmse,
        'final_rmse':    best_rmse,
        'history':       history,
    }


# ═════════════════════════════════════════════════════════════════════════
#  FULL PIPELINE: SCREEN → SELECT → VALIDATE  (one call per target)
# ═════════════════════════════════════════════════════════════════════════

def select_exog_for_target(
    target_col: str,
    df: pd.DataFrame,
    sarima_result: dict,
    candidates: dict,
    skip_own: str = None,
    max_lags: int = 8,
    n_test: int = 8,
    min_train: int = 40,
    vif_threshold: float = 5.0,
    max_exog: int = 3,
    verbose: bool = True,
) -> dict:
    """
    End-to-end exog selection for a single target column.

    1. Extract residuals → CCF scan (screening)
    2. Forward selection with CV (validation)
    3. AIC comparison (in-sample sanity check)

    Parameters
    ----------
    target_col : str
        Column name in df.
    df : pd.DataFrame
        Full dataset.
    sarima_result : dict
        Entry from results dict for this target.
    candidates : dict
        {name: pd.Series} of stationary exog candidates.
    skip_own : str or None
        Name in candidates to exclude (the target's own derivative).
    max_lags : int
        CCF max lag.
    n_test, min_train, vif_threshold, max_exog :
        Passed to forward_select_exog.
    verbose : bool

    Returns
    -------
    dict with keys: ccf_summary, selection, aic_comparison
    """
    series = df[target_col].dropna()
    order = sarima_result['order']
    seasonal_order = sarima_result['seasonal_order']
    transform_fn = sarima_result.get('transform_fn', None)
    inverse_fn   = sarima_result.get('inverse_fn', None)

    # ── 1. CCF screening ──────────────────────────────────
    resid = get_residuals(sarima_result)
    # Align residual index safely
    if not isinstance(resid.index, pd.DatetimeIndex):
        resid.index = series.index[-len(resid):]

    cands = {k: v for k, v in candidates.items() if k != skip_own}

    if verbose:
        print(f"\n{'='*65}")
        print(f"  {target_col}  —  SARIMA{order}x{seasonal_order}")
        print(f"{'='*65}")

    ccf_summary = scan_exog_candidates(resid, cands, max_lags=max_lags)

    if verbose:
        print("\n  CCF Screening:")
        print(ccf_summary.to_string(index=False))

    # ── 2. Forward selection with CV ──────────────────────
    if verbose:
        print(f"\n  Forward Selection (CV holdout = {n_test} periods):")

    selection = forward_select_exog(
        series=series,
        order=order,
        seasonal_order=seasonal_order,
        candidates=cands,
        ccf_summary=ccf_summary,
        transform_fn=transform_fn,
        inverse_fn=inverse_fn,
        n_test=n_test,
        min_train=min_train,
        vif_threshold=vif_threshold,
        max_exog=max_exog,
        verbose=verbose,
    )

    # ── 3. AIC comparison (sanity check) ──────────────────
    aic_results = []
    for var_name, lag in selection['selected']:
        try:
            aic = compare_sarima_vs_sarimax(
                series, cands[var_name], lag, order, seasonal_order,
                transform_fn, inverse_fn,
            )
            aic['variable'] = var_name
            aic['lag'] = lag
            aic_results.append(aic)
        except Exception:
            pass

    if verbose and aic_results:
        print("\n  AIC Confirmation:")
        for a in aic_results:
            print(f"    {a['variable']} (lag {a['lag']}): "
                  f"ΔAIC={a['aic_improvement']:+.1f}  "
                  f"ΔBIC={a['bic_improvement']:+.1f}")

    if verbose:
        sel_str = ', '.join(f"{n}(lag {l})" for n, l in selection['selected'])
        print(f"\n  → FINAL: {sel_str or '— stay with SARIMA'}")
        if selection['baseline_rmse'] and selection['final_rmse']:
            pct = 100 * (selection['baseline_rmse'] - selection['final_rmse']) / selection['baseline_rmse']
            print(f"    RMSE: {selection['baseline_rmse']:.4f} → {selection['final_rmse']:.4f} ({pct:+.2f}%)")

    return {
        'ccf_summary':    ccf_summary,
        'selection':      selection,
        'aic_comparison': aic_results,
    }


# ═════════════════════════════════════════════════════════════════════════
#  PLOTTING  (improved)
# ═════════════════════════════════════════════════════════════════════════

def plot_ccf(residuals: pd.Series,
             exog_series: pd.Series,
             exog_name: str = "Exog",
             max_lags: int = 12,
             figsize: tuple = (12, 4)):
    """Plots cross-correlation with Bartlett significance bands."""

    ccf_df = compute_ccf(residuals, exog_series, max_lags=max_lags)

    fig, ax = plt.subplots(figsize=figsize)

    colors = ['#e74c3c' if row['significant'] else '#95a5a6'
              for _, row in ccf_df.iterrows()]

    ax.bar(ccf_df['lag'], ccf_df['ccf'], color=colors, width=0.6)

    # Bartlett bands (they vary by lag, so plot the envelope)
    ax.fill_between(ccf_df['lag'], ccf_df['threshold'], -ccf_df['threshold'],
                    color='blue', alpha=0.08, label='95% Bartlett CI')
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.axvline(x=0, color='grey', linewidth=0.5, linestyle=':')

    ax.set_xlabel('Lag (positive = exog leads residuals)')
    ax.set_ylabel('Cross-correlation')
    ax.set_title(f'CCF: SARIMA Residuals × {exog_name}')
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_cv_comparison(cv_result: dict, series_name: str = "",
                       figsize: tuple = (12, 5)):
    """
    Plots actual vs forecast for both SARIMA and SARIMAX
    from the cross-validation results.
    """
    if 'cv_sarima' not in cv_result or 'cv_sarimax' not in cv_result:
        print("  Need both cv_sarima and cv_sarimax in the result dict.")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    for ax, key, label in [(ax1, 'cv_sarima', 'SARIMA'),
                            (ax2, 'cv_sarimax', 'SARIMAX')]:
        cv = cv_result[key]
        ax.plot(cv['date'], cv['actual'], 'ko-', ms=4, label='Actual')
        ax.plot(cv['date'], cv['forecast'], 's--', color='#e74c3c',
                ms=4, label='Forecast')
        rmse = np.sqrt(cv['sq_error'].mean())
        ax.set_title(f'{label}  (RMSE={rmse:.4f})')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f'{series_name} — Cross-Validation Comparison',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.show()


def plot_all_ccfs(residuals: pd.Series,
                  candidates: dict,
                  max_lags: int = 8,
                  figsize: tuple = (14, 3)):
    """Plots CCF for every candidate in a vertical stack."""

    n_vars = len(candidates)
    fig, axes = plt.subplots(n_vars, 1, figsize=(figsize[0], figsize[1] * n_vars),
                             sharex=True)
    if n_vars == 1:
        axes = [axes]

    for ax, (name, series) in zip(axes, candidates.items()):
        ccf_df = compute_ccf(residuals, series, max_lags=max_lags)
        colors = ['#e74c3c' if r['significant'] else '#bdc3c7'
                  for _, r in ccf_df.iterrows()]
        ax.bar(ccf_df['lag'], ccf_df['ccf'], color=colors, width=0.6)
        ax.fill_between(ccf_df['lag'], ccf_df['threshold'], -ccf_df['threshold'],
                        color='blue', alpha=0.08)
        ax.axhline(y=0, color='black', linewidth=0.5)
        ax.set_ylabel(name, fontsize=10)
        ax.set_ylim(-0.5, 0.5)

    axes[-1].set_xlabel('Lag (positive = exog leads residuals)')
    fig.suptitle('Cross-Correlations with Residuals (Bartlett bands)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.show()


# ═════════════════════════════════════════════════════════════════════════
#  EXAMPLE USAGE
# ═════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    # ── Assume you already ran SARIMA_Fitting ────────────────────────
    #
    # from SARIMA_Fitting import run_sarima_pipeline
    # results, comparisons = run_sarima_pipeline(df_US, m=4)
    #
    # ── Prepare candidates (same as before) ──────────────────────────
    #
    # candidates = {
    #     'house_prices_pct':  df_US['US_house_prices'].pct_change().dropna(),
    #     'gdp_growth':        df_US['US_rGDP'].pct_change().dropna(),
    #     ...
    # }
    #
    # ── Run the full pipeline for one target ─────────────────────────
    #
    # result = select_exog_for_target(
    #     target_col='US_Credit',
    #     df=df_US,
    #     sarima_result=results['US_Credit'],
    #     candidates=candidates,
    #     skip_own='credit_diff',
    #     max_lags=8,
    #     n_test=8,        # hold out last 8 quarters (2 years)
    #     min_train=40,    # need at least 40 quarters to fit
    #     max_exog=3,
    #     verbose=True,
    # )
    #
    # ── The result tells you exactly what to include ─────────────────
    #
    # selected = result['selection']['selected']
    # # e.g. [('oil_pct_chg', 2), ('crisis', 1)]
    #
    # # Build final exog for SARIMAX fitting:
    # final_exog = result['selection']['exog_df']
    #
    # ── Loop over all targets ────────────────────────────────────────
    #
    # skip_map = {
    #     'US_house_prices': 'house_prices_pct',
    #     'US_Credit':       'credit_diff',
    #     ...
    # }
    #
    # all_selections = {}
    # for target in results:
    #     all_selections[target] = select_exog_for_target(
    #         target_col=target,
    #         df=df_US,
    #         sarima_result=results[target],
    #         candidates=candidates,
    #         skip_own=skip_map.get(target),
    #     )

    pass
