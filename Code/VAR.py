import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.tsa.api import VAR
from statsmodels.tsa.vector_ar.vecm import VECM, coint_johansen, select_coint_rank
from statsmodels.tsa.stattools import adfuller, kpss, grangercausalitytests
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.diagnostic import acorr_ljungbox
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings
warnings.filterwarnings('ignore')


# ═════════════════════════════════════════════════════════════
#  SECTION 0 — DATA VALIDATION & CLEANING
# ═════════════════════════════════════════════════════════════

def validate_and_clean(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Prepares a DataFrame for VAR/VECM by:
      1. Dropping non-numeric columns
      2. Aligning all columns to a common date range (no NaN rows)
      3. Forward-filling small interior gaps (≤2 periods)
      4. Dropping columns that are constant or near-constant
      5. Checking for multicollinearity and warning
    Returns cleaned DataFrame.
    """
    df = df.copy()

    # 1. Numeric only
    non_numeric = df.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric and verbose:
        print(f"  ⚠ Dropping non-numeric columns: {non_numeric}")
    df = df.select_dtypes(include=[np.number])

    if verbose:
        print(f"\n  Data shape before cleaning: {df.shape}")
        na_counts = df.isna().sum()
        if na_counts.any():
            print(f"  NaN counts per column:")
            for col in df.columns:
                if na_counts[col] > 0:
                    print(f"    {col}: {na_counts[col]} NaNs "
                          f"(first valid: {df[col].first_valid_index()}, "
                          f"last valid: {df[col].last_valid_index()})")

    # 2. Forward-fill small gaps (≤2 consecutive NaNs), then drop remaining
    df = df.ffill(limit=2)

    # 3. Drop rows where ANY column is still NaN → aligned rectangular dataset
    df = df.dropna()

    if verbose:
        print(f"  Data shape after cleaning: {df.shape}")

    if len(df) < 20:
        raise ValueError(f"Only {len(df)} complete observations after aligning columns. "
                         f"Need at least 20. Check for columns with very different date ranges.")

    # 4. Drop constant / near-constant columns
    drop_cols = []
    for col in df.columns:
        if df[col].std() < 1e-10:
            drop_cols.append(col)
    if drop_cols:
        if verbose:
            print(f"  ⚠ Dropping constant columns: {drop_cols}")
        df = df.drop(columns=drop_cols)

    # 5. Multicollinearity check (correlation > 0.99)
    corr = df.corr().abs()
    flagged = set()
    for i in range(len(corr.columns)):
        for j in range(i + 1, len(corr.columns)):
            if corr.iloc[i, j] > 0.99:
                flagged.add((corr.columns[i], corr.columns[j], round(corr.iloc[i, j], 4)))
    if flagged and verbose:
        print(f"\n  ⚠ Near-perfect correlations detected (may cause singular matrix):")
        for c1, c2, r in flagged:
            print(f"    {c1} ↔ {c2}: r = {r}")
        print(f"    Consider dropping one from each pair.")

    return df


# ═════════════════════════════════════════════════════════════
#  SECTION 0B — AUTOMATIC VARIABLE SELECTION
# ═════════════════════════════════════════════════════════════

# Patterns that indicate a column is derived from another
_DERIVED_SUFFIXES = [
    '_log_ret', '_log_return', '_logret',
    '_qoq_growth', '_yoy_growth', '_mom_growth',
    '_qoq', '_yoy', '_mom',
    '_pct_change', '_pctchg', '_return', '_ret',
    '_diff', '_delta',
]


def _detect_derived_columns(columns: list) -> dict:
    """
    Detects columns that are mechanically derived from a base column.
    Returns {derived_col: base_col} mapping.
    """
    derived_map = {}
    cols_lower = {c: c.lower() for c in columns}

    for col in columns:
        col_low = col.lower()

        for suffix in _DERIVED_SUFFIXES:
            if col_low.endswith(suffix):
                base_stem = col_low[:-len(suffix)]
                # Find the base column (exact match or with common level suffixes)
                for other_col, other_low in cols_lower.items():
                    if other_col == col:
                        continue
                    if other_low == base_stem or other_low in [
                        base_stem + '_close', base_stem + '_price',
                        base_stem + '_index', base_stem + '_idx',
                        base_stem + '_level',
                    ]:
                        derived_map[col] = other_col
                        break
                break  # matched a suffix, stop checking others

    return derived_map


def _detect_collinear_pairs(df: pd.DataFrame, threshold: float = 0.95) -> list:
    """Returns list of (col1, col2, correlation) for pairs above threshold."""
    corr = df.corr().abs()
    pairs = []
    for i in range(len(corr.columns)):
        for j in range(i + 1, len(corr.columns)):
            if corr.iloc[i, j] > threshold:
                pairs.append((corr.columns[i], corr.columns[j],
                              round(corr.iloc[i, j], 4)))
    return sorted(pairs, key=lambda x: -x[2])


def _compute_vif(df: pd.DataFrame) -> pd.Series:
    """Variance Inflation Factor for each column."""
    cols = df.columns
    vifs = {}
    X = df.values
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-10)

    for i, col in enumerate(cols):
        y = X[:, i]
        X_other = np.delete(X, i, axis=1)
        try:
            beta = np.linalg.lstsq(X_other, y, rcond=None)[0]
            y_hat = X_other @ beta
            ss_res = np.sum((y - y_hat) ** 2)
            ss_tot = np.sum((y - y.mean()) ** 2)
            r2 = 1 - ss_res / (ss_tot + 1e-10)
            vifs[col] = 1.0 / (1.0 - r2 + 1e-10)
        except np.linalg.LinAlgError:
            vifs[col] = np.inf

    return pd.Series(vifs).sort_values(ascending=False)


def auto_select_variables(df: pd.DataFrame, max_vars: int = None,
                          min_obs_per_param: int = 10,
                          max_lags_expected: int = 4,
                          collinear_threshold: float = 0.95,
                          vif_threshold: float = 10.0,
                          prefer_levels: bool = True,
                          verbose: bool = True) -> pd.DataFrame:
    """
    Automatically selects variables suitable for VAR/VECM:

      Phase 1: Removes derived columns (log returns, growth rates) when the
               base level column is present. VAR captures these dynamics
               internally through differencing and lag structure.

      Phase 2: Removes one column from highly correlated pairs (r > threshold),
               keeping the one with fewer NaNs / longer history.

      Phase 3: Iteratively drops the highest-VIF column until all are below
               vif_threshold, eliminating multicollinearity.

      Phase 4: Enforces a max variable count based on the observation-to-parameter
               ratio: max_vars <= T / (min_obs_per_param * max_lags_expected).
               When trimming, ranks variables by PCA loading scores to keep
               the ones that explain the most variance in the system.

    Parameters
    ----------
    df : cleaned DataFrame (output of validate_and_clean)
    max_vars : hard cap on variables (None = computed from data)
    min_obs_per_param : minimum observations per estimated parameter
    max_lags_expected : assumed max lag for ratio calculation
    collinear_threshold : correlation above which to drop one column
    vif_threshold : VIF above which to iteratively drop columns
    prefer_levels : when choosing between level and derivative, keep level
    verbose : print selection log

    Returns
    -------
    DataFrame with selected columns only
    """
    T = len(df)

    if verbose:
        print(f"\n  Starting with {len(df.columns)} variables, {T} observations")

    # ── Phase 1: Drop derived columns ──
    derived = _detect_derived_columns(list(df.columns))
    drop_derived = []

    if derived:
        if verbose:
            print(f"\n  Phase 1 — Derived column detection:")
        for deriv_col, base_col in derived.items():
            if prefer_levels and base_col in df.columns and deriv_col in df.columns:
                drop_derived.append(deriv_col)
                if verbose:
                    print(f"    Dropping {deriv_col} (derived from {base_col})")
            elif not prefer_levels and deriv_col in df.columns and base_col in df.columns:
                drop_derived.append(base_col)
                if verbose:
                    print(f"    Dropping {base_col} (keeping derivative {deriv_col})")

    df = df.drop(columns=[c for c in drop_derived if c in df.columns])

    if verbose:
        print(f"  → {len(df.columns)} variables after Phase 1")

    # ── Phase 2: Drop collinear pairs ──
    pairs = _detect_collinear_pairs(df, threshold=collinear_threshold)
    dropped_collinear = set()

    if pairs:
        if verbose:
            print(f"\n  Phase 2 — Collinearity removal (|r| > {collinear_threshold}):")
        for c1, c2, r in pairs:
            if c1 in dropped_collinear or c2 in dropped_collinear:
                continue
            if c1 not in df.columns or c2 not in df.columns:
                continue
            # Keep the one with more non-NaN data
            na1 = df[c1].isna().sum()
            na2 = df[c2].isna().sum()
            drop = c2 if na1 <= na2 else c1
            keep = c1 if drop == c2 else c2
            dropped_collinear.add(drop)
            if verbose:
                print(f"    Dropping {drop} (r={r} with {keep})")

    df = df.drop(columns=[c for c in dropped_collinear if c in df.columns])

    if verbose:
        print(f"  → {len(df.columns)} variables after Phase 2")

    # ── Phase 3: Iterative VIF reduction ──
    if len(df.columns) > 3:
        if verbose:
            print(f"\n  Phase 3 — VIF reduction (threshold={vif_threshold}):")

        max_iters = len(df.columns) - 3  # keep at least 3 variables
        dropped_any = False
        for _ in range(max_iters):
            vifs = _compute_vif(df)
            worst = vifs.idxmax()
            worst_vif = vifs[worst]

            if worst_vif <= vif_threshold:
                break

            dropped_any = True
            if verbose:
                print(f"    Dropping {worst} (VIF={worst_vif:.1f})")
            df = df.drop(columns=[worst])

        if not dropped_any and verbose:
            print(f"    All VIFs ≤ {vif_threshold} ✓")

        if verbose:
            print(f"  → {len(df.columns)} variables after Phase 3")

    # ── Phase 4: Enforce obs-to-parameter ratio ──
    if max_vars is None:
        max_vars = max(3, T // (min_obs_per_param * max_lags_expected))

    if len(df.columns) > max_vars:
        if verbose:
            print(f"\n  Phase 4 — Enforcing max {max_vars} variables "
                  f"(T={T}, {min_obs_per_param}:1 per param, "
                  f"~{max_lags_expected} expected lags)")

        # Rank by PCA loading importance
        from sklearn.decomposition import PCA

        X = df.values
        X_std = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-10)

        n_components = min(max_vars, len(df.columns), T)
        pca = PCA(n_components=n_components)
        pca.fit(X_std)

        # Score = sum of absolute loadings weighted by explained variance
        loadings = np.abs(pca.components_)
        weights = pca.explained_variance_ratio_[:n_components]
        scores = (loadings * weights[:, None]).sum(axis=0)

        col_scores = pd.Series(scores, index=df.columns).sort_values(ascending=False)
        keep = col_scores.head(max_vars).index.tolist()
        dropped = [c for c in df.columns if c not in keep]

        if verbose:
            print(f"    Keeping (by PCA loading): {keep}")
            print(f"    Dropping: {dropped}")

        df = df[keep]
    else:
        if verbose:
            print(f"\n  Phase 4 — Variable count OK "
                  f"({len(df.columns)} ≤ {max_vars} max)")

    if verbose:
        ratio = T / (len(df.columns) * max_lags_expected)
        print(f"\n  ✓ Final selection: {list(df.columns)}")
        print(f"    {len(df.columns)} vars × {T} obs "
              f"(≈{ratio:.1f} obs per parameter at {max_lags_expected} lags)")

    return df

def adf_test(series: pd.Series, name: str = '', significance: float = 0.05) -> dict:
    """Augmented Dickey-Fuller test for unit root."""
    result = adfuller(series.dropna(), autolag='AIC')
    return {
        'variable':   name,
        'test':       'ADF',
        'statistic':  round(result[0], 4),
        'p_value':    round(result[1], 4),
        'lags_used':  result[2],
        'stationary': result[1] < significance,
    }


def kpss_test(series: pd.Series, name: str = '', significance: float = 0.05) -> dict:
    """KPSS test for stationarity (null = stationary)."""
    result = kpss(series.dropna(), regression='c', nlags='auto')
    return {
        'variable':   name,
        'test':       'KPSS',
        'statistic':  round(result[0], 4),
        'p_value':    round(result[1], 4),
        'stationary': result[1] > significance,  # KPSS: null = stationary
    }


def stationarity_report(df: pd.DataFrame, significance: float = 0.05) -> pd.DataFrame:
    """
    Runs ADF + KPSS on every column (levels and first differences).
    Returns a summary DataFrame with recommended differencing order.
    """
    records = []

    for col in df.columns:
        series = df[col].dropna()
        diff1 = series.diff().dropna()

        adf_level = adf_test(series, col)
        kpss_level = kpss_test(series, col)
        adf_diff = adf_test(diff1, col)
        kpss_diff = kpss_test(diff1, col)

        # Consensus: both agree on stationarity at level?
        level_stationary = adf_level['stationary'] and kpss_level['stationary']
        diff_stationary = adf_diff['stationary'] and kpss_diff['stationary']

        if level_stationary:
            d = 0
        elif diff_stationary:
            d = 1
        else:
            d = 2  # rare — flag for manual review

        records.append({
            'variable':          col,
            'adf_level_p':       adf_level['p_value'],
            'kpss_level_p':      kpss_level['p_value'],
            'level_stationary':  level_stationary,
            'adf_diff1_p':       adf_diff['p_value'],
            'kpss_diff1_p':      kpss_diff['p_value'],
            'diff1_stationary':  diff_stationary,
            'recommended_d':     d,
        })

    report = pd.DataFrame(records)
    return report


def make_stationary(df: pd.DataFrame, stationarity_df: pd.DataFrame) -> tuple:
    """
    Differences each column according to recommended_d from stationarity_report.
    Returns (stationary_df, d_orders_dict).
    """
    d_orders = dict(zip(stationarity_df['variable'], stationarity_df['recommended_d']))
    df_stationary = pd.DataFrame(index=df.index)

    for col in df.columns:
        d = d_orders.get(col, 1)
        s = df[col]
        for _ in range(d):
            s = s.diff()
        df_stationary[col] = s

    df_stationary = df_stationary.dropna()
    return df_stationary, d_orders


# ═════════════════════════════════════════════════════════════
#  SECTION 2 — COINTEGRATION TESTING
# ═════════════════════════════════════════════════════════════

def johansen_test(df: pd.DataFrame, det_order: int = 0, k_ar_diff: int = 1,
                  significance: str = '5%') -> dict:
    """
    Johansen cointegration test with fallback handling for singular matrices.
    det_order: -1 = no const, 0 = restricted const, 1 = unrestricted const
    Returns rank, trace stats, and critical values.
    """
    sig_map = {'10%': 0, '5%': 1, '1%': 2}
    sig_idx = sig_map.get(significance, 1)

    data = df.dropna()

    # Try multiple configurations if the default fails
    configs = [
        (det_order, k_ar_diff),
        (det_order, max(1, k_ar_diff - 1)),
        (det_order, k_ar_diff + 1),
        (-1, k_ar_diff),           # no deterministic term
        (1, k_ar_diff),            # unrestricted constant
    ]

    result = None
    used_config = None
    for d_ord, k_diff in configs:
        try:
            result = coint_johansen(data, det_order=d_ord, k_ar_diff=k_diff)
            used_config = (d_ord, k_diff)
            break
        except (np.linalg.LinAlgError, ValueError):
            continue

    if result is None:
        print(f"  ⚠ Johansen test failed for all configurations.")
        print(f"    This typically means near-singular covariance matrix.")
        print(f"    Common causes: multicollinear variables, too many NAs,")
        print(f"    or too few observations relative to variables.")
        print(f"    → Falling back to VAR (assuming no cointegration).")
        return {
            'rank':         0,
            'n_variables':  df.shape[1],
            'significance': significance,
            'det_order':    det_order,
            'k_ar_diff':    k_ar_diff,
            'details':      pd.DataFrame(),
            'eigenvectors': None,
            'eigenvalues':  None,
            'failed':       True,
        }

    if used_config != (det_order, k_ar_diff):
        print(f"  ⚠ Default config failed; succeeded with "
              f"det_order={used_config[0]}, k_ar_diff={used_config[1]}")

    # Determine cointegration rank via trace statistic
    trace_stats = result.lr1
    crit_vals = result.cvt[:, sig_idx]
    eigen_stats = result.lr2
    eigen_crit = result.cvm[:, sig_idx]

    rank = 0
    for i in range(len(trace_stats)):
        if trace_stats[i] > crit_vals[i]:
            rank += 1
        else:
            break

    records = []
    for i in range(len(trace_stats)):
        records.append({
            'H0: rank ≤':        i,
            'trace_stat':        round(trace_stats[i], 4),
            'trace_crit':        round(crit_vals[i], 4),
            'trace_reject':      trace_stats[i] > crit_vals[i],
            'eigen_stat':        round(eigen_stats[i], 4),
            'eigen_crit':        round(eigen_crit[i], 4),
            'eigen_reject':      eigen_stats[i] > eigen_crit[i],
        })

    return {
        'rank':          rank,
        'n_variables':   df.shape[1],
        'significance':  significance,
        'det_order':     used_config[0],
        'k_ar_diff':     used_config[1],
        'details':       pd.DataFrame(records),
        'eigenvectors':  result.evec,
        'eigenvalues':   result.eig,
        'failed':        False,
    }


# ═════════════════════════════════════════════════════════════
#  SECTION 3 — GRANGER CAUSALITY
# ═════════════════════════════════════════════════════════════

def granger_causality_matrix(df: pd.DataFrame, max_lag: int = 4,
                             significance: float = 0.05) -> pd.DataFrame:
    """
    Pairwise Granger causality test.
    Returns matrix of p-values: row → caused by column.
    """
    cols = df.columns
    n = len(cols)
    pval_matrix = pd.DataFrame(np.zeros((n, n)), columns=cols, index=cols)

    for target in cols:
        for cause in cols:
            if target == cause:
                pval_matrix.loc[target, cause] = np.nan
                continue
            try:
                test_data = df[[target, cause]].dropna()
                result = grangercausalitytests(test_data, maxlag=max_lag, verbose=False)
                # Take minimum p-value across all lags (F-test)
                min_p = min(result[lag][0]['ssr_ftest'][1] for lag in range(1, max_lag + 1))
                pval_matrix.loc[target, cause] = round(min_p, 4)
            except Exception:
                pval_matrix.loc[target, cause] = np.nan

    return pval_matrix


# ═════════════════════════════════════════════════════════════
#  SECTION 4 — VAR MODEL FITTING
# ═════════════════════════════════════════════════════════════

def fit_var(df: pd.DataFrame, max_lags: int = 12, ic: str = 'aic',
            trend: str = 'c') -> dict:
    """
    Fits a VAR model to stationary data.
    Selects lag order via information criterion, then fits.
    """
    data = df.dropna()
    model = VAR(data)

    # --- Lag selection ---
    lag_order = model.select_order(maxlags=max_lags, trend=trend)
    selected_lag = lag_order.selected_orders[ic]

    # Guard: at least 1 lag
    selected_lag = max(1, selected_lag)

    print(f"\n  Lag selection results:")
    print(f"    AIC → {lag_order.selected_orders.get('aic', '—')}  "
          f"BIC → {lag_order.selected_orders.get('bic', '—')}  "
          f"HQIC → {lag_order.selected_orders.get('hqic', '—')}  "
          f"FPE → {lag_order.selected_orders.get('fpe', '—')}")
    print(f"    Selected: {selected_lag} lags (by {ic.upper()})")

    # --- Fit ---
    fitted = model.fit(selected_lag, trend=trend)

    # --- Diagnostics ---
    diagnostics = var_diagnostics(fitted)

    return {
        'model':        fitted,
        'lag_order':    selected_lag,
        'ic':           ic,
        'lag_table':    lag_order,
        'diagnostics':  diagnostics,
        'aic':          fitted.aic,
        'bic':          fitted.bic,
        'hqic':         fitted.hqic,
        'fpe':          fitted.fpe,
    }


def var_diagnostics(fitted_model) -> dict:
    """Extracts key diagnostics from a fitted VAR model."""
    resid = fitted_model.resid
    n_vars = resid.shape[1]
    cols = resid.columns if hasattr(resid, 'columns') else [f'var_{i}' for i in range(n_vars)]

    # --- Durbin-Watson per equation ---
    dw_stats = durbin_watson(resid)
    dw_dict = {col: round(dw, 4) for col, dw in zip(cols, dw_stats)}

    # --- Ljung-Box per equation ---
    lb_dict = {}
    for i, col in enumerate(cols):
        try:
            lb = acorr_ljungbox(resid.iloc[:, i], lags=[10], return_df=True)
            lb_dict[col] = round(lb['lb_pvalue'].iloc[0], 4)
        except Exception:
            lb_dict[col] = np.nan

    # --- Normality (Jarque-Bera per equation) ---
    jb_dict = {}
    for i, col in enumerate(cols):
        try:
            jb_stat, jb_p = stats.jarque_bera(resid.iloc[:, i])
            jb_dict[col] = round(jb_p, 4)
        except Exception:
            jb_dict[col] = np.nan

    # --- Stability check ---
    try:
        roots = fitted_model.roots
        is_stable = all(np.abs(r) > 1 for r in roots)
    except Exception:
        is_stable = None

    return {
        'durbin_watson':  dw_dict,
        'ljung_box_p':    lb_dict,
        'jarque_bera_p':  jb_dict,
        'is_stable':      is_stable,
    }


# ═════════════════════════════════════════════════════════════
#  SECTION 5 — VECM MODEL FITTING
# ═════════════════════════════════════════════════════════════

def fit_vecm(df: pd.DataFrame, coint_rank: int, max_lags: int = 12,
             det_order: int = 0, ic: str = 'aic') -> dict:
    """
    Fits a VECM model to levels data when cointegration is present.
    df should be in LEVELS (not differenced).
    """
    data = df.dropna()

    # --- Lag selection via VAR on differenced data ---
    diff_data = data.diff().dropna()
    var_tmp = VAR(diff_data)
    try:
        lag_table = var_tmp.select_order(maxlags=max(1, max_lags - 1))
        selected_lag = max(1, lag_table.selected_orders.get(ic, 1))
    except Exception:
        selected_lag = 2

    print(f"\n  VECM lag selection (on differences): {selected_lag}")
    print(f"  Cointegration rank: {coint_rank}")

    # --- Fit VECM ---
    model = VECM(data, k_ar_diff=selected_lag, coint_rank=coint_rank,
                 deterministic='ci' if det_order == 0 else 'co')
    fitted = model.fit()

    # --- Extract diagnostics ---
    diagnostics = vecm_diagnostics(fitted)

    # --- Cointegrating vectors ---
    beta = fitted.beta  # cointegrating vectors
    alpha = fitted.alpha  # adjustment/loading coefficients

    return {
        'model':           fitted,
        'lag_order':       selected_lag,
        'coint_rank':      coint_rank,
        'beta':            beta,
        'alpha':           alpha,
        'diagnostics':     diagnostics,
        'columns':         list(data.columns),
    }


def vecm_diagnostics(fitted_model) -> dict:
    """Diagnostics for fitted VECM."""
    resid = fitted_model.resid
    n_vars = resid.shape[1]

    dw_stats = durbin_watson(resid)

    lb_dict, jb_dict, dw_dict = {}, {}, {}
    for i in range(n_vars):
        col = f'eq_{i}'
        dw_dict[col] = round(dw_stats[i], 4)
        try:
            lb = acorr_ljungbox(resid[:, i], lags=[10], return_df=True)
            lb_dict[col] = round(lb['lb_pvalue'].iloc[0], 4)
        except Exception:
            lb_dict[col] = np.nan
        try:
            jb_stat, jb_p = stats.jarque_bera(resid[:, i])
            jb_dict[col] = round(jb_p, 4)
        except Exception:
            jb_dict[col] = np.nan

    return {
        'durbin_watson': dw_dict,
        'ljung_box_p':   lb_dict,
        'jarque_bera_p': jb_dict,
    }


# ═════════════════════════════════════════════════════════════
#  SECTION 6 — FORECASTING
# ═════════════════════════════════════════════════════════════

def forecast_var(var_result: dict, df_original: pd.DataFrame,
                 d_orders: dict, n_periods: int = 20,
                 alpha: float = 0.05) -> dict:
    """
    Forecasts from a fitted VAR model on stationary data,
    then undifferences back to original levels.
    """
    fitted = var_result['model']
    lag = var_result['lag_order']

    # Forecast on stationary scale
    forecast_input = fitted.y[-lag:]
    fc = fitted.forecast(forecast_input, steps=n_periods)
    fc_lower, fc_upper = _var_confidence_bands(fitted, n_periods, alpha)

    # --- Build forecast index ---
    last_date = df_original.index[-1]
    if not isinstance(df_original.index, pd.DatetimeIndex):
        forecast_idx = pd.RangeIndex(len(df_original), len(df_original) + n_periods)
    else:
        freq = pd.infer_freq(df_original.index) or 'QS'
        forecast_idx = pd.date_range(start=last_date, periods=n_periods + 1, freq=freq)[1:]

    cols = df_original.columns
    fc_df = pd.DataFrame(fc, index=forecast_idx, columns=cols)
    fc_lower_df = pd.DataFrame(fc_lower, index=forecast_idx, columns=cols)
    fc_upper_df = pd.DataFrame(fc_upper, index=forecast_idx, columns=cols)

    # --- Undifference forecasts back to levels ---
    fc_levels = pd.DataFrame(index=forecast_idx, columns=cols, dtype=float)
    ci_lower_levels = pd.DataFrame(index=forecast_idx, columns=cols, dtype=float)
    ci_upper_levels = pd.DataFrame(index=forecast_idx, columns=cols, dtype=float)

    for col in cols:
        d = d_orders.get(col, 1)
        fc_levels[col] = _undifference(
            df_original[col].dropna(), fc_df[col].values, d)
        ci_lower_levels[col] = _undifference(
            df_original[col].dropna(), fc_lower_df[col].values, d)
        ci_upper_levels[col] = _undifference(
            df_original[col].dropna(), fc_upper_df[col].values, d)

    return {
        'forecast':   fc_levels,
        'ci_lower':   ci_lower_levels,
        'ci_upper':   ci_upper_levels,
        'fc_stationary': fc_df,
    }


def forecast_vecm(vecm_result: dict, df_original: pd.DataFrame,
                  n_periods: int = 20, alpha: float = 0.05) -> dict:
    """
    Forecasts from a fitted VECM model (already in levels).
    """
    fitted = vecm_result['model']

    # VECM forecast returns levels directly
    fc, fc_lower, fc_upper = fitted.predict(steps=n_periods, alpha=alpha)

    # --- Build forecast index ---
    last_date = df_original.index[-1]
    if not isinstance(df_original.index, pd.DatetimeIndex):
        forecast_idx = pd.RangeIndex(len(df_original), len(df_original) + n_periods)
    else:
        freq = pd.infer_freq(df_original.index) or 'QS'
        forecast_idx = pd.date_range(start=last_date, periods=n_periods + 1, freq=freq)[1:]

    cols = df_original.columns
    fc_df = pd.DataFrame(fc, index=forecast_idx, columns=cols)
    fc_lower_df = pd.DataFrame(fc_lower, index=forecast_idx, columns=cols)
    fc_upper_df = pd.DataFrame(fc_upper, index=forecast_idx, columns=cols)

    return {
        'forecast':  fc_df,
        'ci_lower':  fc_lower_df,
        'ci_upper':  fc_upper_df,
    }


def _var_confidence_bands(fitted, n_periods, alpha):
    """Compute forecast confidence bands via MSE of forecast errors."""
    fc_result = fitted.forecast_interval(fitted.y[-fitted.k_ar:],
                                         steps=n_periods, alpha=alpha)
    return fc_result[1], fc_result[2]  # lower, upper


def _undifference(original_series: pd.Series, forecast_diffs: np.ndarray,
                  d: int) -> np.ndarray:
    """Reconstructs levels from differenced forecasts."""
    if d == 0:
        return forecast_diffs

    result = forecast_diffs.copy().astype(float)

    if d >= 1:
        last_level = original_series.iloc[-1]
        cumulative = np.cumsum(result) + last_level
        result = cumulative

    if d >= 2:
        # Second undifferencing (rare, for d=2)
        last_level = original_series.iloc[-1]
        last_diff = original_series.iloc[-1] - original_series.iloc[-2]
        level = np.zeros_like(result)
        prev_diff = last_diff
        prev_level = last_level
        for i in range(len(result)):
            curr_diff = prev_diff + forecast_diffs[i]
            curr_level = prev_level + curr_diff
            level[i] = curr_level
            prev_diff = curr_diff
            prev_level = curr_level
        result = level

    return result


# ═════════════════════════════════════════════════════════════
#  SECTION 7 — PLOTTING
# ═════════════════════════════════════════════════════════════

def plot_forecasts(df: pd.DataFrame, forecast_result: dict, model_type: str = 'VAR',
                   hist_years: int = 7, cols_per_row: int = 2,
                   save_path: str = None):
    """
    Plots historical + forecast for all variables in a grid.
    Works for both VAR and VECM forecast outputs.
    """
    fc = forecast_result['forecast']
    ci_lo = forecast_result['ci_lower']
    ci_hi = forecast_result['ci_upper']
    cols = fc.columns
    n = len(cols)
    nrows = int(np.ceil(n / cols_per_row))

    fig, axes = plt.subplots(nrows, cols_per_row,
                             figsize=(7 * cols_per_row, 4.5 * nrows),
                             squeeze=False)

    for i, col in enumerate(cols):
        row, c = divmod(i, cols_per_row)
        ax = axes[row][c]

        series = df[col].dropna().copy()

        # Ensure DatetimeIndex
        if not isinstance(series.index, pd.DatetimeIndex):
            series.index = pd.to_datetime(series.index)

        # Trim to last N years
        cutoff = series.index[-1] - pd.DateOffset(years=hist_years)
        series = series.loc[series.index >= cutoff]

        forecast_idx = fc.index
        if not isinstance(forecast_idx, pd.DatetimeIndex):
            forecast_idx = pd.to_datetime(forecast_idx)

        forecast_vals = fc[col].values
        lower = ci_lo[col].values
        upper = ci_hi[col].values

        # Historical
        ax.plot(series.index, series.values, color='#2563EB',
                linewidth=1.5, label='Historical')

        # Forecast
        ax.plot(forecast_idx, forecast_vals, color='#DC2626',
                linewidth=1.8, linestyle='--', label='Forecast')

        # CI band
        ax.fill_between(forecast_idx, lower, upper,
                        color='#DC2626', alpha=0.10, label='95% CI')

        # Connector
        ax.plot([series.index[-1], forecast_idx[0]],
                [series.values[-1], forecast_vals[0]],
                color='#DC2626', linewidth=1.8, linestyle='--')

        # Divider
        ax.axvline(series.index[-1], color='#9CA3AF', linewidth=0.8,
                   linestyle=':', alpha=0.7)

        # Title
        ax.set_title(f"{col}    [{model_type}]", fontsize=12,
                     fontweight='bold', loc='left')

        # Clean axes
        ax.legend(loc='upper left', fontsize=8, frameon=False)
        ax.grid(axis='y', alpha=0.25, linewidth=0.6)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(0.5)
        ax.spines['bottom'].set_linewidth(0.5)

        # Year-only x-axis
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.tick_params(axis='both', which='major', labelsize=9, length=4, width=0.5)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=0, ha='center')

        # Y breathing room
        ymin = min(series.values.min(), lower.min())
        ymax = max(series.values.max(), upper.max())
        margin = (ymax - ymin) * 0.08
        ax.set_ylim(ymin - margin, ymax + margin)

    # Hide empties
    for j in range(n, nrows * cols_per_row):
        row, c = divmod(j, cols_per_row)
        axes[row][c].set_visible(False)

    plt.tight_layout(h_pad=3.0, w_pad=2.0)

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
        print(f"\n  📊 Saved → {save_path}")

    plt.show()


def plot_irf(var_result: dict, periods: int = 20, orth: bool = True,
             save_path: str = None):
    """Plots impulse response functions from a fitted VAR."""
    fitted = var_result['model']
    irf = fitted.irf(periods=periods)
    fig = irf.plot(orth=orth, signif=0.05)
    fig.suptitle('Impulse Response Functions', fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.show()


def plot_fevd(var_result: dict, periods: int = 20, save_path: str = None):
    """Plots forecast error variance decomposition from a fitted VAR."""
    fitted = var_result['model']
    fevd = fitted.fevd(periods=periods)
    fig = fevd.plot()
    fig.suptitle('Forecast Error Variance Decomposition',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.show()


# ═════════════════════════════════════════════════════════════
#  SECTION 8 — DIAGNOSTICS DISPLAY
# ═════════════════════════════════════════════════════════════

def print_diagnostics(diagnostics: dict, columns: list = None):
    """Pretty-prints model diagnostics."""
    labels = columns or list(diagnostics['durbin_watson'].keys())

    print(f"\n  {'─'*55}")
    print(f"  {'Variable':<20} {'DW':>8} {'LB (p)':>8} {'JB (p)':>8}")
    print(f"  {'─'*55}")

    for key in diagnostics['durbin_watson']:
        label = key
        dw = diagnostics['durbin_watson'].get(key, '—')
        lb = diagnostics['ljung_box_p'].get(key, '—')
        jb = diagnostics['jarque_bera_p'].get(key, '—')

        dw_flag = ' ✓' if isinstance(dw, float) and 1.5 < dw < 2.5 else ' ✗'
        lb_flag = ' ✓' if isinstance(lb, float) and lb > 0.05 else ' ✗'
        jb_flag = ' ✓' if isinstance(jb, float) and jb > 0.05 else ' ✗'

        print(f"  {label:<20} {dw:>7}{dw_flag} {lb:>7}{lb_flag} {jb:>7}{jb_flag}")

    if 'is_stable' in diagnostics:
        stable = diagnostics['is_stable']
        print(f"\n  Stability: {'✓ All roots outside unit circle' if stable else '✗ UNSTABLE — roots inside unit circle'}")

    print(f"  {'─'*55}")
    print(f"  DW ∈ [1.5, 2.5] = no autocorrelation")
    print(f"  LB p > 0.05 = no serial correlation")
    print(f"  JB p > 0.05 = normally distributed residuals")


# ═════════════════════════════════════════════════════════════
#  SECTION 9 — MAIN PIPELINE
# ═════════════════════════════════════════════════════════════

def run_var_vecm_pipeline(df: pd.DataFrame, n_periods: int = 20,
                          max_lags: int = 12, ic: str = 'aic',
                          max_vars: int = None,
                          significance: float = 0.05,
                          hist_years: int = 7,
                          plot: bool = True,
                          save_dir: str = None) -> dict:
    """
    Full VAR / VECM pipeline:
      1. Stationarity testing
      2. Cointegration testing (Johansen)
      3. Granger causality
      4. If cointegrated → VECM on levels
         If not → VAR on stationary data
      5. Diagnostics
      6. Forecasting + IRF + FEVD
      7. Plotting

    Parameters
    ----------
    df : DataFrame with DatetimeIndex, all numeric columns
    n_periods : forecast horizon (e.g. 20 quarters = 5 years)
    max_lags : max lags for model selection
    ic : information criterion ('aic', 'bic', 'hqic', 'fpe')
    max_vars : hard cap on variables (None = computed automatically from T)
    significance : threshold for all statistical tests
    hist_years : years of history to display in plots
    plot : whether to generate plots
    save_dir : directory to save plots (None = don't save)

    Returns
    -------
    dict with keys: 'model_type', 'model_result', 'forecast',
                    'stationarity', 'cointegration', 'granger'
    """
    # Ensure DatetimeIndex
    df = df.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    print(f"{'═'*60}")
    print(f"  VAR / VECM PIPELINE")
    print(f"  Variables: {list(df.columns)}")
    print(f"  Observations: {len(df)}  |  Forecast horizon: {n_periods}")
    print(f"{'═'*60}")

    # ── STEP 0: Data Validation & Cleaning ──
    print(f"\n{'─'*60}")
    print(f"  STEP 0 — Data Validation & Cleaning")
    print(f"{'─'*60}")
    df = validate_and_clean(df, verbose=True)
    print(f"  ✓ Clean dataset: {df.shape[0]} obs × {df.shape[1]} variables")

    # ── STEP 0B: Automatic Variable Selection ──
    print(f"\n{'─'*60}")
    print(f"  STEP 0B — Automatic Variable Selection")
    print(f"{'─'*60}")
    df = auto_select_variables(
        df,
        max_vars=max_vars,
        min_obs_per_param=10,
        max_lags_expected=max(2, max_lags // 2),
        collinear_threshold=0.95,
        vif_threshold=10.0,
        verbose=True,
    )

    # ── STEP 1: Stationarity ──
    print(f"\n{'─'*60}")
    print(f"  STEP 1 — Stationarity Testing (ADF + KPSS)")
    print(f"{'─'*60}")
    stationarity = stationarity_report(df, significance=significance)
    print(stationarity.to_string(index=False))

    # ── STEP 2: Cointegration ──
    print(f"\n{'─'*60}")
    print(f"  STEP 2 — Johansen Cointegration Test")
    print(f"{'─'*60}")
    coint = johansen_test(df, det_order=0, k_ar_diff=2, significance='5%')

    if not coint.get('failed', False):
        print(f"\n  Cointegration rank: {coint['rank']} / {coint['n_variables']}")
        print(coint['details'].to_string(index=False))
    else:
        print(f"\n  Cointegration rank: 0 (test failed — defaulting to VAR)")

    # ── STEP 3: Granger Causality ──
    print(f"\n{'─'*60}")
    print(f"  STEP 3 — Granger Causality (p-values, min across lags)")
    print(f"{'─'*60}")

    # Run Granger on stationary data
    df_stat, d_orders = make_stationary(df, stationarity)
    granger = granger_causality_matrix(df_stat, max_lag=min(4, max_lags))
    print(f"\n  (row is caused by column, p < 0.05 = significant)")
    print(granger.to_string(float_format='{:.4f}'.format))

    # ── STEP 4: Model Selection & Fitting ──
    print(f"\n{'─'*60}")
    print(f"  STEP 4 — Model Fitting")
    print(f"{'─'*60}")

    if coint['rank'] > 0 and not coint.get('failed', False):
        model_type = 'VECM'
        print(f"\n  ✓ Cointegration detected (rank={coint['rank']}) → fitting VECM on levels")
        model_result = fit_vecm(df, coint_rank=coint['rank'],
                                max_lags=max_lags, det_order=0, ic=ic)
    else:
        model_type = 'VAR'
        print(f"\n  ✗ No cointegration → fitting VAR on stationary data")
        model_result = fit_var(df_stat, max_lags=max_lags, ic=ic)

    # ── STEP 5: Diagnostics ──
    print(f"\n{'─'*60}")
    print(f"  STEP 5 — {model_type} Diagnostics")
    print(f"{'─'*60}")
    print_diagnostics(model_result['diagnostics'],
                      columns=list(df.columns))

    # ── STEP 6: Forecasting ──
    print(f"\n{'─'*60}")
    print(f"  STEP 6 — Forecasting ({n_periods} periods)")
    print(f"{'─'*60}")

    if model_type == 'VECM':
        forecast_result = forecast_vecm(model_result, df, n_periods=n_periods)
    else:
        forecast_result = forecast_var(model_result, df, d_orders,
                                       n_periods=n_periods)

    print(f"\n  Forecast (levels):")
    print(forecast_result['forecast'].round(4).to_string())

    # ── STEP 7: Plotting ──
    if plot:
        print(f"\n{'─'*60}")
        print(f"  STEP 7 — Plots")
        print(f"{'─'*60}")

        save_fc = f"{save_dir}/{model_type.lower()}_forecasts.png" if save_dir else None
        plot_forecasts(df, forecast_result, model_type=model_type,
                       hist_years=hist_years, save_path=save_fc)

        if model_type == 'VAR':
            save_irf = f"{save_dir}/irf.png" if save_dir else None
            save_fevd = f"{save_dir}/fevd.png" if save_dir else None
            plot_irf(model_result, periods=n_periods, save_path=save_irf)
            plot_fevd(model_result, periods=n_periods, save_path=save_fevd)

    # ── Summary ──
    print(f"\n{'═'*60}")
    print(f"  ✓ COMPLETE — {model_type} model fitted and forecasted")
    print(f"{'═'*60}")

    return {
        'model_type':         model_type,
        'model_result':       model_result,
        'forecast':           forecast_result,
        'stationarity':       stationarity,
        'cointegration':      coint,
        'granger':            granger,
        'd_orders':           d_orders,
        'selected_variables': list(df.columns),
    }


# ═════════════════════════════════════════════════════════════
#  RUN IT
# ═════════════════════════════════════════════════════════════
if __name__ == "__main__":
    # Assuming your data is in a DataFrame called `Fred_Data`
    # with DatetimeIndex and columns like:
    #   UK_house_prices, UK_cpi, UK_unemployment, UK_BondYield, etc.

    pipeline = run_var_vecm_pipeline(
        Fred_Data,
        n_periods=20,        # 20 quarters = 5 years
        max_lags=8,
        ic='aic',
        hist_years=7,
        plot=True,
        save_dir='.',        # saves PNGs to current directory
    )

    # ── Access results ──
    # pipeline['model_type']                    → 'VAR' or 'VECM'
    # pipeline['forecast']['forecast']          → DataFrame of forecasted levels
    # pipeline['forecast']['ci_lower']          → lower 95% CI
    # pipeline['forecast']['ci_upper']          → upper 95% CI
    # pipeline['cointegration']['rank']         → cointegration rank
    # pipeline['cointegration']['details']      → Johansen trace/eigen table
    # pipeline['granger']                       → pairwise Granger p-value matrix
    # pipeline['stationarity']                  → ADF/KPSS results per variable
    # pipeline['model_result']['model']         → fitted statsmodels VAR or VECM
    # pipeline['model_result']['diagnostics']   → DW, Ljung-Box, Jarque-Bera

    # ── IRF/FEVD (VAR only) ──
    # pipeline['model_result']['model'].irf(20).plot()
    # pipeline['model_result']['model'].fevd(20).summary()