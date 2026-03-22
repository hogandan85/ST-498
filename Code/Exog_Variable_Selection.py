import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import ccf, grangercausalitytests
from statsmodels.stats.outliers_influence import variance_inflation_factor
import warnings
warnings.filterwarnings('ignore')


# ─────────────────────────────────────────────
#  STEP 1: EXTRACT RESIDUALS FROM FITTED MODEL
# ─────────────────────────────────────────────

def get_residuals(result: dict) -> pd.Series:
    """
    Pulls residuals from a fitted SARIMA result dict
    (as returned by SARIMA_Fitting.run_sarima_pipeline).

    Parameters
    ----------
    result : dict
        Single-column entry from the `results` dict, e.g. results['PD']

    Returns
    -------
    pd.Series of residuals in transformed space
    """
    model = result['model']
    return pd.Series(model.resid(), name='residuals')


# ─────────────────────────────────────────────
#  STEP 2: CROSS-CORRELATION (CCF)
# ─────────────────────────────────────────────

def compute_ccf(residuals: pd.Series,
                exog_series: pd.Series,
                max_lags: int = 12,
                alpha: float = 0.05) -> pd.DataFrame:
    """
    Computes cross-correlation between SARIMA residuals and a candidate
    exogenous variable at lags -max_lags to +max_lags.

    Positive lags  → exog LEADS residuals  (exog at t-k vs resid at t)
                     This is what you want for forecasting — the exog
                     variable has predictive power.
    Negative lags  → exog LAGS residuals   (resid leads exog)
                     Less useful for forecasting; means PD predicts the exog.

    Parameters
    ----------
    residuals : pd.Series
        Residuals from fitted SARIMA model.
    exog_series : pd.Series
        Candidate exogenous variable (should be stationary — difference first
        if needed).
    max_lags : int
        Maximum number of lags in each direction.
    alpha : float
        Significance level for the confidence band.

    Returns
    -------
    pd.DataFrame with columns: [lag, ccf, significant]
    """
    # Align on shared index
    shared = residuals.index.intersection(exog_series.index)
    r = residuals.loc[shared].values
    x = exog_series.loc[shared].values

    n = len(r)
    se = 1.0 / np.sqrt(n)                       # standard error under H0
    z_crit = abs(np.percentile([np.random.randn() for _ in range(100000)],
                               [100 * alpha / 2, 100 * (1 - alpha / 2)])[1])
    # Simpler: use scipy
    from scipy.stats import norm
    z_crit = norm.ppf(1 - alpha / 2)
    threshold = z_crit * se

    # Compute CCF at each lag
    rows = []
    for k in range(-max_lags, max_lags + 1):
        if k >= 0:
            # positive lag: x leads r  →  corr(r[k:], x[:n-k])
            corr = np.corrcoef(r[k:], x[:n - k])[0, 1] if k < n else 0.0
        else:
            # negative lag: r leads x  →  corr(r[:n+k], x[-k:])
            corr = np.corrcoef(r[:n + k], x[-k:])[0, 1] if -k < n else 0.0

        rows.append({
            'lag': k,
            'ccf': round(corr, 4),
            'significant': abs(corr) > threshold,
        })

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────
#  STEP 3: SCAN MULTIPLE CANDIDATES AT ONCE
# ─────────────────────────────────────────────

def scan_exog_candidates(residuals: pd.Series,
                         candidates: dict,
                         max_lags: int = 8) -> pd.DataFrame:
    """
    Runs CCF for every candidate exogenous variable and returns a
    summary of the strongest signal found for each.

    Parameters
    ----------
    residuals : pd.Series
        SARIMA residuals (with DatetimeIndex).
    candidates : dict
        {name: pd.Series} of candidate exogenous variables.
        These should already be stationary (differenced / pct-change).
    max_lags : int
        Maximum lag to test.

    Returns
    -------
    pd.DataFrame — one row per candidate, sorted by signal strength:
        variable, best_lag, best_ccf, abs_ccf, significant, direction
    """
    summary = []

    for name, series in candidates.items():
        ccf_df = compute_ccf(residuals, series, max_lags=max_lags)

        # Only look at positive lags (exog leads residuals = predictive)
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


# ─────────────────────────────────────────────
#  STEP 4: GRANGER CAUSALITY (OPTIONAL)
# ─────────────────────────────────────────────

def granger_test(residuals: pd.Series,
                 exog_series: pd.Series,
                 max_lag: int = 4) -> pd.DataFrame:
    """
    Tests whether exog_series Granger-causes the residuals.

    Parameters
    ----------
    residuals : pd.Series
    exog_series : pd.Series  (stationary)
    max_lag : int

    Returns
    -------
    pd.DataFrame with columns: [lag, f_stat, p_value, significant]
    """
    shared = residuals.index.intersection(exog_series.index)
    data = pd.DataFrame({
        'residuals': residuals.loc[shared].values,
        'exog':      exog_series.loc[shared].values,
    })

    results = grangercausalitytests(data[['residuals', 'exog']],
                                    maxlag=max_lag, verbose=False)
    rows = []
    for lag, res in results.items():
        f_test = res[0]['ssr_ftest']
        rows.append({
            'lag':         lag,
            'f_stat':      round(f_test[0], 4),
            'p_value':     round(f_test[1], 4),
            'significant': f_test[1] < 0.05,
        })

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────
#  STEP 5: VIF (STANDALONE — NO FAKE Y NEEDED)
# ─────────────────────────────────────────────

def compute_vif(exog_df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes Variance Inflation Factor for each column in exog_df.
    No Y variable needed.

    Parameters
    ----------
    exog_df : pd.DataFrame
        Candidate exogenous variables (each column = one variable).
        Should be stationary and have no NaNs.

    Returns
    -------
    pd.DataFrame with columns: [variable, vif]
    """
    X = exog_df.dropna().values
    vifs = []
    for i in range(X.shape[1]):
        vifs.append({
            'variable': exog_df.columns[i],
            'vif': round(variance_inflation_factor(X, i), 2),
        })
    return pd.DataFrame(vifs).sort_values('vif', ascending=False).reset_index(drop=True)


# ─────────────────────────────────────────────
#  PLOTTING
# ─────────────────────────────────────────────

def plot_ccf(residuals: pd.Series,
             exog_series: pd.Series,
             exog_name: str = "Exog",
             max_lags: int = 12,
             figsize: tuple = (12, 4)):
    """Plots cross-correlation as a bar chart with significance bands."""

    ccf_df = compute_ccf(residuals, exog_series, max_lags=max_lags)
    n = len(residuals.index.intersection(exog_series.index))
    se = 1.96 / np.sqrt(n)

    fig, ax = plt.subplots(figsize=figsize)

    colors = ['#e74c3c' if row['significant'] else '#95a5a6'
              for _, row in ccf_df.iterrows()]

    ax.bar(ccf_df['lag'], ccf_df['ccf'], color=colors, width=0.6)
    ax.axhline(y=se, color='blue', linestyle='--', alpha=0.5, label='95% CI')
    ax.axhline(y=-se, color='blue', linestyle='--', alpha=0.5)
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.axvline(x=0, color='grey', linewidth=0.5, linestyle=':')

    ax.set_xlabel('Lag (positive = exog leads residuals)')
    ax.set_ylabel('Cross-correlation')
    ax.set_title(f'CCF: SARIMA Residuals × {exog_name}')
    ax.legend()
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

    n = len(residuals)
    se = 1.96 / np.sqrt(n)

    for ax, (name, series) in zip(axes, candidates.items()):
        ccf_df = compute_ccf(residuals, series, max_lags=max_lags)
        colors = ['#e74c3c' if r['significant'] else '#bdc3c7'
                  for _, r in ccf_df.iterrows()]
        ax.bar(ccf_df['lag'], ccf_df['ccf'], color=colors, width=0.6)
        ax.axhline(y=se, color='blue', linestyle='--', alpha=0.4)
        ax.axhline(y=-se, color='blue', linestyle='--', alpha=0.4)
        ax.axhline(y=0, color='black', linewidth=0.5)
        ax.set_ylabel(name, fontsize=10)
        ax.set_ylim(-0.5, 0.5)

    axes[-1].set_xlabel('Lag (positive = exog leads residuals)')
    fig.suptitle('Cross-Correlations with PD Residuals', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────
#  EXAMPLE USAGE
# ─────────────────────────────────────────────

if __name__ == "__main__":

    # ── Assume you already ran SARIMA_Fitting ────────────────────────
    #
    # from SARIMA_Fitting import run_sarima_pipeline
    # results, comparisons = run_sarima_pipeline(Fred_Data, m=4)

    # ── 1. Get residuals for PD ──────────────────────────────────────
    #
    # resid = get_residuals(results['PD'])

    # ── 2. Prepare candidate exogenous variables (make stationary!) ──
    #
    # candidates = {
    #     'oil_pct_chg':    Fred_Data['Oil_Price'].pct_change().dropna(),
    #     'gdp_growth':     Fred_Data['GDP'].pct_change().dropna(),
    #     'fed_funds_diff':  Fred_Data['Fed_Funds'].diff().dropna(),
    #     'crisis_dummy':   crisis_dummy_series,  # already 0/1
    # }

    # ── 3. Scan all candidates ───────────────────────────────────────
    #
    # summary = scan_exog_candidates(resid, candidates, max_lags=8)
    # print(summary)
    #
    # Output: table ranked by |CCF|, showing best lag and significance
    # ┌──────────────────┬──────────┬──────────┬─────────┬─────────────┐
    # │ variable         │ best_lag │ best_ccf │ abs_ccf │ significant │
    # ├──────────────────┼──────────┼──────────┼─────────┼─────────────┤
    # │ oil_pct_chg      │ 2        │ -0.31    │ 0.31    │ True        │
    # │ fed_funds_diff   │ 1        │ 0.24     │ 0.24    │ True        │
    # │ crisis_dummy     │ 1        │ 0.19     │ 0.19    │ True        │
    # │ gdp_growth       │ 3        │ 0.08     │ 0.08    │ False       │
    # └──────────────────┴──────────┴──────────┴─────────┴─────────────┘
    #
    # Interpretation:
    #   - oil_pct_chg at lag 2 → oil shocks 2 quarters ago predict PD residuals
    #   - gdp_growth not significant → SARIMA already captured the GDP cycle
    #   - crisis_dummy significant → worth including

    # ── 4. Visual inspection ─────────────────────────────────────────
    #
    # plot_all_ccfs(resid, candidates, max_lags=8)

    # ── 5. Granger causality for the significant ones ────────────────
    #
    # print(granger_test(resid, candidates['oil_pct_chg'], max_lag=4))

    # ── 6. VIF to check redundancy among your final picks ────────────
    #
    # final_exog = pd.DataFrame({
    #     'oil_pct_chg':   candidates['oil_pct_chg'],
    #     'fed_funds_diff': candidates['fed_funds_diff'],
    #     'crisis_dummy':  candidates['crisis_dummy'],
    # }).dropna()
    #
    # print(compute_vif(final_exog))

    pass
