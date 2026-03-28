import pandas as pd
import numpy as np
from scipy import stats
from scipy.special import inv_boxcox
from sklearn.preprocessing import PowerTransformer
from pmdarima import auto_arima
import warnings
from pmdarima import ARIMA
warnings.filterwarnings('ignore')


# ─────────────────────────────────────────────
#  TRANSFORMATION HELPERS 
# ─────────────────────────────────────────────

def apply_transformations(series: pd.Series) -> dict:
    """Returns dict of {name: (transformed_series, inverse_fn)}"""
    transformations = {}
    has_negatives = (series <= 0).any()

    # --- Log ---
    if not has_negatives:
        log_series = np.log(series)
        transformations['log'] = (log_series, lambda x: np.exp(x))

    # --- Box-Cox ---
    if not has_negatives:
        bc_transformed, lam = stats.boxcox(series)
        bc_series = pd.Series(bc_transformed, index=series.index)
        transformations['boxcox'] = (bc_series, lambda x, l=lam: inv_boxcox(x, l))

    # --- Yeo-Johnson (works with negatives) ---
    pt = PowerTransformer(method='yeo-johnson')
    yj_vals = pt.fit_transform((series + 0.0001).values.reshape(-1, 1)).flatten()
    yj_series = pd.Series(yj_vals, index=series.index)
    transformations['yeo-johnson'] = (yj_series, lambda x: pt.inverse_transform(
        np.array(x).reshape(-1, 1)).flatten() - 0.0001)

    return transformations


def extract_diagnostics(model) -> dict:
    """Extracts key diagnostic stats from a fitted auto_arima model."""
    try:
        # Refit with statsmodels to get full diagnostics
        sarimax_model = model.arima_res_
        diag = {
            'ljung_box_q':   sarimax_model.test_serial_correlation('ljungbox', lags=1)[0][1][0],  # Prob(Q)
            'jarque_bera_jb': sarimax_model.test_normality('jarquebera')[0][1],                   # Prob(JB)
            'heterosked_h':   sarimax_model.test_heteroskedasticity('breakvar')[0][1],            # Prob(H)
            'skew':           sarimax_model.filter_results.standardized_forecasts_error.flatten()[~np.isnan(
                              sarimax_model.filter_results.standardized_forecasts_error.flatten())],
        }
        resid = sarimax_model.resid
        from scipy.stats import skew, kurtosis
        return {
            'prob_q':  round(diag['ljung_box_q'], 4),
            'prob_jb': round(diag['jarque_bera_jb'], 4),
            'prob_h':  round(diag['heterosked_h'], 4),
            'skew':    round(float(skew(resid)), 4),
            'kurtosis': round(float(kurtosis(resid, fisher=False)), 4),  # matches statsmodels (non-excess)
        }
    except Exception as e:
        return {'prob_q': None, 'prob_jb': None, 'prob_h': None, 'skew': None, 'kurtosis': None}

# ─────────────────────────────────────────────
#  FIT & COMPARE MODELS ACROSS TRANSFORMATIONS
# ─────────────────────────────────────────────

from sklearn.model_selection import TimeSeriesSplit

def fit_all_transformations(series: pd.Series, m: int = 4, cv_folds: int = 5,
                            cv_horizon: int = None) -> pd.DataFrame:
    """Fits auto_arima for each transformation with time-series cross-validation."""
    transformations = apply_transformations(series)
    results = []

    if cv_horizon is None:
        cv_horizon = m

    tscv = TimeSeriesSplit(
        n_splits=cv_folds,
        test_size=cv_horizon,
        gap=0  # no gap between train/test; set >0 if you want a buffer
    )

    for name, (transformed, inverse_fn) in transformations.items():
        try:
            model = auto_arima(
                y=transformed,
                start_p=0, start_q=0,
                max_p=3, max_q=3,
                start_P=0, start_Q=0,
                max_P=2, max_Q=2,
                m=m,
                d=None, D=None,
                seasonal=True,
                information_criterion='aic',
                error_action='ignore',
                suppress_warnings=True,
                stepwise=True
            )

            # --- Cross-validation via sklearn TimeSeriesSplit ---
            cv_maes, cv_rmses = [], []

            for train_idx, test_idx in tscv.split(transformed):
                if len(train_idx) < 2 * m:
                    continue  # skip folds without enough seasonal history

                train_fold = transformed.iloc[train_idx]
                test_fold = transformed.iloc[test_idx]

                try:
                    fold_model = ARIMA(
                        order=model.order,
                        seasonal_order=model.seasonal_order,
                        suppress_warnings=True
                    )
                    fold_model.fit(train_fold)
                    preds = fold_model.predict(n_periods=len(test_fold))

                    errors = test_fold.values - preds
                    cv_maes.append(np.mean(np.abs(errors)))
                    cv_rmses.append(np.sqrt(np.mean(errors ** 2)))
                except Exception:
                    continue

            cv_mae = np.mean(cv_maes) if cv_maes else np.nan
            cv_rmse = np.mean(cv_rmses) if cv_rmses else np.nan
            cv_folds_used = len(cv_maes)

            # --- Diagnostics ---
            residuals = pd.Series(model.resid())
            diagnostics = extract_diagnostics(model)

            results.append({
                'transformation':   name,
                'aic':              model.aic(),
                'bic':              model.bic(),
                'cv_rmse':          cv_rmse,
                'cv_mae':           cv_mae,
                'cv_folds_used':    cv_folds_used,
                'order':            model.order,
                'seasonal_order':   model.seasonal_order,
                'residual_std':     residuals.std(),
                'abs_skew':         abs(residuals.skew()),
                'abs_kurt':         abs(residuals.kurtosis() - 3),
                'prob_q':           diagnostics['prob_q'],
                'prob_jb':          diagnostics['prob_jb'],
                'prob_h':           diagnostics['prob_h'],
                'skew':             diagnostics['skew'],
                'kurtosis':         diagnostics['kurtosis'],
                '_model':           model,
                '_inverse_fn':      inverse_fn,
                '_transformed':     transformed,
            })

        except Exception as e:
            print(f"    ⚠ Failed [{name}]: {e}")

    comparison_df = pd.DataFrame(results).sort_values('cv_rmse').reset_index(drop=True)
    return comparison_df


# ─────────────────────────────────────────────
#  FORECAST USING BEST MODEL
# ─────────────────────────────────────────────

def forecast_best_model(comparison_df: pd.DataFrame, n_periods: int = 8) -> dict:
    """Takes comparison df, selects best model by AIC, returns forecast."""
    best = comparison_df.iloc[0]
    model      = best['_model']
    inverse_fn = best['_inverse_fn']

    forecast_transformed = model.predict(n_periods=n_periods)
    forecast_original    = inverse_fn(forecast_transformed)

    return {
        'transformation':  best['transformation'],
        'order':           best['order'],
        'seasonal_order':  best['seasonal_order'],
        'aic':             best['aic'],
        'forecast':        forecast_original,
        'model':           model,
        'inverse_fn':      inverse_fn,
    }


# ─────────────────────────────────────────────
#  MAIN PIPELINE — RUNS FOR ALL COLUMNS
# ─────────────────────────────────────────────

def run_sarima_pipeline(df: pd.DataFrame, m: int = 4, n_periods: int = 8) -> dict:
    """
    Runs the full SARIMA pipeline for every column in df.
    Returns a dict of {column_name: result_dict}
    """
    all_results     = {}
    all_comparisons = {}

    for col in df.columns:
        print(f"\n{'='*60}")
        print(f"  Processing: {col}")
        print(f"{'='*60}")

        series = df[col].dropna()

        # 1. Fit all transformations
        comparison_df = fit_all_transformations(series, m=m)

        # 2. Print comparison table (without internal cols)
        display_cols = ['transformation', 'aic', 'bic', 'order', 'seasonal_order',
                'residual_std', 'prob_q', 'prob_jb', 'prob_h', 'skew', 'kurtosis']
        print(comparison_df[display_cols].to_string(index=False))

        # 3. Forecast best model
        result = forecast_best_model(comparison_df, n_periods=n_periods)

        print(f"\n  ✓ Best: [{result['transformation']}]  "
              f"SARIMA{result['order']}x{result['seasonal_order']}  "
              f"AIC={result['aic']:.2f}")

        all_results[col]     = result
        all_comparisons[col] = comparison_df

    return all_results, all_comparisons


# ─────────────────────────────────────────────
#  RUN IT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    # Assuming your data is in a DataFrame called `Fred_Data`
    results, comparisons = run_sarima_pipeline(Fred_Data, m=4, n_periods=8)

# Access any column's forecast:
# results['GDP']['forecast']      → forecast array (original scale)
# results['GDP']['model']         → fitted pmdarima model
# results['GDP']['transformation'] → e.g. 'boxcox'
# comparisons['GDP']              → full comparison DataFrame