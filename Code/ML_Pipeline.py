# ml_forecasting_pipeline.py
# ---------------------------
# ML forecasting pipeline with:
#   - Time series cross-validation (TimeSeriesSplit)
#   - Hyperparameter tuning via GridSearchCV inside each fold
#   - All standard regression models
#   - Full metrics: MAE, RMSE, MAPE, SMAPE, R²
#
# INSTALL:
#   pip install pandas numpy scikit-learn xgboost lightgbm

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor


# ===========================================================================
# SETTINGS
# ===========================================================================

TARGET_COL  = "value"   # column you are forecasting
N_SPLITS    = 5         # outer CV folds (evaluates model performance)
TUNE_SPLITS = 3         # inner CV folds (used only for hyperparameter tuning)
                        # inner < outer is intentional — keeps tuning fast
RANDOM_SEED = 42


# ===========================================================================
# MODELS
# ===========================================================================
# Each entry is:  "name": (model_instance, param_grid)
#
# param_grid is a dictionary of hyperparameters to search over.
# Set param_grid to {} if you don't want to tune a specific model.
#
# To add a model:    add a new line following the same pattern.
# To remove a model: delete its line.
# To skip tuning:    set its param_grid to {}.

MODELS = {

    # Linear models — no meaningful hyperparameters to tune
    "LinearRegression": (
        LinearRegression(),
        {}   # no tuning
    ),

    # Ridge: alpha controls regularisation strength (higher = more regularised)
    "Ridge": (
        Ridge(),
        {"alpha": [0.01, 0.1, 1.0, 10.0, 50.0]}
    ),

    # Lasso: same alpha logic as Ridge, but also performs feature selection
    "Lasso": (
        Lasso(),
        {"alpha": [0.001, 0.01, 0.1, 1.0, 10.0]}
    ),

    # ElasticNet: mix of Ridge and Lasso
    # l1_ratio=0 is pure Ridge, l1_ratio=1 is pure Lasso
    "ElasticNet": (
        ElasticNet(),
        {
            "alpha":    [0.01, 0.1, 1.0, 10.0],
            "l1_ratio": [0.2, 0.5, 0.8],
        }
    ),

    # Random Forest: n_estimators = number of trees, max_depth = how deep each tree grows
    "RandomForest": (
        RandomForestRegressor(random_state=RANDOM_SEED),
        {
            "n_estimators": [50, 100, 200],
            "max_depth":    [None, 5, 10],
        }
    ),

    # XGBoost: n_estimators = boosting rounds, learning_rate = step size,
    #          max_depth = tree depth, subsample = row sampling per tree
    "XGBoost": (
        XGBRegressor(verbosity=0, random_state=RANDOM_SEED),
        {
            "n_estimators":  [50, 100, 200],
            "learning_rate": [0.01, 0.05, 0.1],
            "max_depth":     [3, 5, 7],
            "subsample":     [0.8, 1.0],
        }
    ),

    # LightGBM: same logic as XGBoost, num_leaves controls tree complexity
    "LightGBM": (
        LGBMRegressor(verbose=-1, random_state=RANDOM_SEED),
        {
            "n_estimators":  [50, 100, 200],
            "learning_rate": [0.01, 0.05, 0.1],
            "num_leaves":    [15, 31, 63],
            "subsample":     [0.8, 1.0],
        }
    ),
}


# ===========================================================================
# METRICS
# ===========================================================================

def compute_metrics(y_true, y_pred):
    """Returns MAE, RMSE, MAPE, SMAPE, R² as a dictionary."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    nonzero      = y_true != 0
    denom_smape  = np.where((np.abs(y_true) + np.abs(y_pred)) == 0, 1e-8,
                            (np.abs(y_true) + np.abs(y_pred)) / 2)

    mae   = mean_absolute_error(y_true, y_pred)
    rmse  = np.sqrt(mean_squared_error(y_true, y_pred))
    mape  = np.mean(np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero])) * 100 \
            if nonzero.any() else np.nan
    smape = np.mean(np.abs(y_true - y_pred) / denom_smape) * 100
    r2    = r2_score(y_true, y_pred)

    return {
        "MAE":   round(mae,   4),
        "RMSE":  round(rmse,  4),
        "MAPE":  round(mape,  4),
        "SMAPE": round(smape, 4),
        "R2":    round(r2,    4),
    }


# ===========================================================================
# PIPELINE CLASS
# ===========================================================================

class MLForecastingPipeline:
    """
    ML forecasting pipeline with nested time series cross-validation.

    Outer loop : TimeSeriesSplit(N_SPLITS)  — measures real performance
    Inner loop : GridSearchCV(TUNE_SPLITS)  — finds best hyperparameters
                                              using only the training fold,
                                              never touching the test fold.

    How to use:
        pipeline = MLForecastingPipeline()
        pipeline.load_data(df)
        pipeline.run()
        pipeline.summary()
        pipeline.best_params()   # see which params won in each fold
    """

    def __init__(self, target_col=TARGET_COL, n_splits=N_SPLITS, tune_splits=TUNE_SPLITS):
        self.target_col   = target_col
        self.n_splits     = n_splits
        self.tune_splits  = tune_splits
        self.df           = None
        self.cv_results   = []    # per-fold performance metrics
        self.tune_results = []    # per-fold best hyperparameters
        self.summary_df   = None

    # -----------------------------------------------------------------------
    def load_data(self, df):
        """
        Pass in your DataFrame.
        Should be sorted by date with lag features already present as columns.
        """
        self.df = df.copy()
        print(f"Data loaded: {len(self.df)} rows, {len(self.df.columns)} columns.")
        print(f"Features: {[c for c in self.df.columns if c != self.target_col]}\n")

    # -----------------------------------------------------------------------
    def run(self):
        """
        Runs all models with nested CV:
          - For each outer fold, GridSearchCV tunes hyperparameters on the
            training split only, then the best model is evaluated on the
            held-out test split.
        """
        if self.df is None:
            print("No data loaded. Call load_data(df) first.")
            return

        X = self.df.drop(columns=[self.target_col]).values
        y = self.df[self.target_col].values

        # Outer CV — the loop that measures performance
        outer_cv = TimeSeriesSplit(n_splits=self.n_splits)

        # Inner CV — used only inside GridSearchCV for tuning
        # Must also be TimeSeriesSplit to respect time ordering during tuning
        inner_cv = TimeSeriesSplit(n_splits=self.tune_splits)

        print(f"Running {len(MODELS)} models | "
              f"{self.n_splits} outer folds | "
              f"{self.tune_splits} inner tuning folds\n")

        for fold_num, (train_idx, test_idx) in enumerate(outer_cv.split(X), start=1):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            for model_name, (model, param_grid) in MODELS.items():

                if param_grid:
                    # --- Tuned path ---
                    # GridSearchCV tries every combination in param_grid,
                    # using inner_cv to score each combination.
                    # It never sees X_test — only X_train is passed in.
                    grid_search = GridSearchCV(
                        estimator  = model,
                        param_grid = param_grid,
                        cv         = inner_cv,
                        scoring    = "neg_root_mean_squared_error",
                        refit      = True,   # refit best model on full X_train after search
                        n_jobs     = -1,     # use all CPU cores
                    )
                    grid_search.fit(X_train, y_train)
                    best_model  = grid_search.best_estimator_
                    best_params = grid_search.best_params_

                else:
                    # --- Untuned path (e.g. LinearRegression) ---
                    model.fit(X_train, y_train)
                    best_model  = model
                    best_params = {}

                # Evaluate the best model on the held-out outer test fold
                y_pred  = best_model.predict(X_test)
                metrics = compute_metrics(y_test, y_pred)

                self.cv_results.append({
                    "fold":  fold_num,
                    "model": model_name,
                    **metrics,
                })

                self.tune_results.append({
                    "fold":        fold_num,
                    "model":       model_name,
                    "best_params": best_params,
                })

            print(f"  Fold {fold_num} complete.")

        print("\nDone.\n")

    # -----------------------------------------------------------------------
    def summary(self):
        """
        Averages metrics across all outer CV folds, ranks by RMSE.
        Returns a DataFrame.
        """
        if not self.cv_results:
            print("No results yet. Call run() first.")
            return

        results_df = pd.DataFrame(self.cv_results)

        self.summary_df = (
            results_df
            .groupby("model")[["MAE", "RMSE", "MAPE", "SMAPE", "R2"]]
            .mean()
            .round(4)
            .sort_values("RMSE")
            .reset_index()
        )

        print("=" * 65)
        print(f"CV SUMMARY  ({self.n_splits}-fold, averaged, tuned per fold)")
        print("=" * 65)
        print(self.summary_df.to_string(index=False))
        print()
        print(f"Best model by RMSE : {self.summary_df.iloc[0]['model']}")
        print(f"Best model by R²   : {self.summary_df.sort_values('R2', ascending=False).iloc[0]['model']}")

        return self.summary_df

    # -----------------------------------------------------------------------
    def best_params(self):
        """
        Shows the best hyperparameters chosen in each fold for each model.
        Useful for spotting whether the tuner is consistent across folds
        or jumping around (which would suggest instability).
        """
        if not self.tune_results:
            print("No tuning results yet. Call run() first.")
            return

        rows = []
        for entry in self.tune_results:
            rows.append({
                "fold":  entry["fold"],
                "model": entry["model"],
                **entry["best_params"],
            })

        params_df = pd.DataFrame(rows).sort_values(["model", "fold"])
        print(params_df.to_string(index=False))
        return params_df

    # -----------------------------------------------------------------------
    def fold_detail(self):
        """Full per-fold metric breakdown (not averaged)."""
        if not self.cv_results:
            print("No results yet. Call run() first.")
            return
        df = pd.DataFrame(self.cv_results).sort_values(["model", "fold"])
        print(df.to_string(index=False))
        return df

    # -----------------------------------------------------------------------
    def add_model(self, name, model, param_grid=None):
        """
        Add a model at any point.
        Example:
            pipeline.add_model("Ridge_wide", Ridge(), {"alpha": [1, 10, 100, 500]})
        """
        MODELS[name] = (model, param_grid or {})
        print(f"Model added: {name}")

    # -----------------------------------------------------------------------
    def remove_model(self, name):
        """Remove a model by name."""
        if name in MODELS:
            del MODELS[name]
            print(f"Model removed: {name}")
        else:
            print(f"'{name}' not found. Available: {list(MODELS.keys())}")

    # -----------------------------------------------------------------------
    def export(self, path="cv_results.csv"):
        """Save fold-by-fold results to CSV."""
        if not self.cv_results:
            print("No results to export.")
            return
        pd.DataFrame(self.cv_results).to_csv(path, index=False)
        print(f"Saved to {path}")


# ===========================================================================
# EXAMPLE USAGE
# ===========================================================================

if __name__ == "__main__":

    # --- Replace with: df = pd.read_csv("your_file.csv") -------------------


    df = pd.read_csv("your_file.csv")
    # -------------------------------------------------------------------------

    pipeline = MLForecastingPipeline()
    pipeline.load_data(df)
    pipeline.run()
    pipeline.summary()

    # See which hyperparameters were selected in each fold
    pipeline.best_params()

    # See per-fold metrics without averaging
    # pipeline.fold_detail()

    # Save results
    # pipeline.export("my_results.csv")