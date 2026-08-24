"""
Model Training Component
==========================

Responsibility (and only this):
    1. Load the pre-scaled train/test tables produced by data_transformation.py.
    2. Train a registry of candidate configurations — model families crossed
       with imbalance-handling strategies — mirroring the comparison run in
       notebook/01_EDA.ipynb, sections 6.2 and 6.3.
    3. Score every candidate with the same metric harness EDA used, so
       results here are directly comparable to the notebook's.
    4. Persist every fitted candidate and a consolidated results table.

model_evaluation.py (Phase 7) reads what this file produces, applies the
selection logic from EDA section 7.2, and promotes exactly one candidate to
models/model.pkl. This file does not pick a winner — it only trains and
measures.

Why no k-fold cross-validation, despite the standard checklist calling for
it: the positive class has only 473 examples total (EDA section 3.7).
Splitting further into folds — even with a leakage-safe TimeSeriesSplit —
would leave some folds with a handful of frauds, and PR-AUC computed on a
handful of positives is too noisy to rank candidates by. EDA section 5.1
already validated that the single chronological holdout has stable fraud
rates on both sides (train 0.176%, test 0.130%, 1.35x ratio); we reuse
exactly that split rather than fragmenting it further. If the dataset later
grows (e.g. combined with more months of data), sklearn's TimeSeriesSplit
is the correct upgrade — swap it in where _build_candidates' outputs are
looped over, without touching the registry itself.

Features arrive here already imputed and RobustScaler-scaled (Phase 5).
Candidates therefore don't need their own scaler step — except where a
resampler's k-NN step (SMOTE/ADASYN) needs scaled input, which it already
has for free.
"""

import gc
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, brier_score_loss, precision_recall_curve, roc_auc_score,
)
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import ADASYN, SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler

from src.component.data_transformation import FEATURES, TARGET
from src.exception import CustomException
from src.logger import logging
from src.utils import save_dataframe, save_object

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RANDOM_STATE = 42

# ---------------------------------------------------------------------- #
# Metric harness — identical to EDA notebook section 6.1. Phase 7 imports
# these three functions from here rather than redefining them, so the
# comparison table and the training results always agree on what "PR-AUC"
# means.
# ---------------------------------------------------------------------- #
ALERT_BUDGET_K = 100  # proxy for "transactions a review team can handle per day"


def precision_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int = ALERT_BUDGET_K) -> float:
    idx = np.argsort(y_score)[::-1][:k]
    return float(np.mean(y_true[idx]))


def recall_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int = ALERT_BUDGET_K) -> float:
    idx = np.argsort(y_score)[::-1][:k]
    return float(y_true[idx].sum() / y_true.sum())


def evaluate(y_true: np.ndarray, y_score: np.ndarray, k: int = ALERT_BUDGET_K) -> dict:
    """The one place PR-AUC, ROC-AUC, Brier, and the operating-point metrics
    get computed. If you need a new metric, add it here — not per-caller."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-9, None)
    best_idx = int(np.nanargmax(f1))
    return {
        "pr_auc": average_precision_score(y_true, y_score),
        "roc_auc": roc_auc_score(y_true, y_score),
        "brier": brier_score_loss(y_true, y_score),
        f"precision@{k}": precision_at_k(y_true, y_score, k),
        f"recall@{k}": recall_at_k(y_true, y_score, k),
        "best_f1": float(f1[best_idx]),
        "best_f1_threshold": float(thresholds[best_idx]) if best_idx < len(thresholds) else 1.0,
    }


def _build_candidates() -> dict:
    """Registry of everything to train. Mirrors EDA 6.2 (model families) and
    6.3 (imbalance strategies on the strongest base learner), collapsed into
    one sweep. SMOTE at sampling_strategy=0.50 is deliberately excluded —
    EDA section 6.3 found it strictly worse than 0.01 and 0.05, so re-running
    it here would only cost time without adding information."""

    def hgb(**kw):
        return HistGradientBoostingClassifier(
            max_iter=400, learning_rate=0.06, max_leaf_nodes=31,
            random_state=RANDOM_STATE, **kw,
        )

    return {
        # --- model family comparison (EDA section 6.2) ---
        "logistic_regression_balanced": ImbPipeline([
            ("model", LogisticRegression(max_iter=3000, class_weight="balanced",
                                         random_state=RANDOM_STATE)),
        ]),
        "decision_tree_balanced": ImbPipeline([
            ("model", DecisionTreeClassifier(max_depth=6, min_samples_leaf=30,
                                             class_weight="balanced",
                                             random_state=RANDOM_STATE)),
        ]),
        "random_forest_balanced": ImbPipeline([
            ("model", RandomForestClassifier(n_estimators=100, max_depth=8,
                                             min_samples_leaf=10, n_jobs=1,
                                             class_weight="balanced_subsample",
                                             random_state=RANDOM_STATE)),
        ]),
        "histgb_none": ImbPipeline([("model", hgb())]),

        # --- imbalance strategy comparison (EDA section 6.3) ---
        "histgb_class_weight_balanced": ImbPipeline([
            ("model", hgb(class_weight="balanced")),
        ]),
        "histgb_smote_0.01": ImbPipeline([
            ("sampler", SMOTE(sampling_strategy=0.01, k_neighbors=5, random_state=RANDOM_STATE)),
            ("model", hgb()),
        ]),
        "histgb_smote_0.05": ImbPipeline([
            ("sampler", SMOTE(sampling_strategy=0.05, k_neighbors=5, random_state=RANDOM_STATE)),
            ("model", hgb()),
        ]),
        "histgb_adasyn_0.05": ImbPipeline([
            ("sampler", ADASYN(sampling_strategy=0.05, random_state=RANDOM_STATE)),
            ("model", hgb()),
        ]),
        "histgb_undersample_0.05": ImbPipeline([
            ("sampler", RandomUnderSampler(sampling_strategy=0.05, random_state=RANDOM_STATE)),
            ("model", hgb()),
        ]),
    }


@dataclass
class ModelTrainingConfig:
    transformed_train_path: Path = PROJECT_ROOT / "notebook" / "data" / "processed" / "train_transformed.csv"
    transformed_test_path: Path = PROJECT_ROOT / "notebook" / "data" / "processed" / "test_transformed.csv"
    candidates_dir: Path = PROJECT_ROOT / "models" / "candidates"
    results_path: Path = PROJECT_ROOT / "models" / "training_results.csv"

    def __post_init__(self):
        self.transformed_train_path = Path(self.transformed_train_path)
        self.transformed_test_path = Path(self.transformed_test_path)
        self.candidates_dir = Path(self.candidates_dir)
        self.results_path = Path(self.results_path)


class ModelTraining:
    def __init__(self, config: ModelTrainingConfig = None):
        self.config = config or ModelTrainingConfig()

    def initiate_model_training(self) -> pd.DataFrame:
        """Trains every candidate, saves each fitted pipeline, and returns
        the results table (also saved to disk for Phase 7)."""
        logger.info("=== Model training started ===")
        try:
            X_train, y_train = self._load(self.config.transformed_train_path)
            X_test, y_test = self._load(self.config.transformed_test_path)
            logger.info("X_train %s (frauds=%d) | X_test %s (frauds=%d)",
                       X_train.shape, int(y_train.sum()), X_test.shape, int(y_test.sum()))

            candidates = _build_candidates()
            logger.info("Registry: %d candidates", len(candidates))

            rows = []
            for name, pipe in candidates.items():
                t0 = time.perf_counter()
                pipe.fit(X_train, y_train)
                fit_seconds = time.perf_counter() - t0

                proba = pipe.predict_proba(X_test)[:, 1]
                metrics = evaluate(y_test, proba)
                metrics["candidate"] = name
                metrics["fit_seconds"] = round(fit_seconds, 2)
                rows.append(metrics)

                save_object(self.config.candidates_dir / f"{name}.pkl", pipe)
                logger.info(
                    "%-32s PR-AUC %.4f | ROC %.4f | Brier %.6f | fit %.1fs",
                    name, metrics["pr_auc"], metrics["roc_auc"], metrics["brier"], fit_seconds,
                )
                # Free the fitted estimator once it's saved to disk — with 9
                # candidates in flight, holding every fitted model in memory
                # simultaneously is unnecessary peak-memory pressure.
                candidates[name] = None
                del pipe, proba
                gc.collect()

            results_df = (
                pd.DataFrame(rows)
                .set_index("candidate")
                .sort_values("pr_auc", ascending=False)
            )
            save_dataframe(results_df.reset_index(), self.config.results_path)

            logger.info("=== Model training completed. Best PR-AUC: %s (%.4f) ===",
                       results_df.index[0], results_df.iloc[0]["pr_auc"])
            return results_df

        except CustomException:
            raise
        except Exception as e:
            raise CustomException(e, sys)

    def _load(self, path: Path) -> tuple[np.ndarray, np.ndarray]:
        df = pd.read_csv(path)
        missing = set(FEATURES) - set(df.columns)
        if missing:
            raise ValueError(f"Transformed data at {path} is missing features: {missing}")
        return df[FEATURES].values, df[TARGET].values


if __name__ == "__main__":
    trainer = ModelTraining()
    results = trainer.initiate_model_training()
    pd.set_option("display.width", 160)
    print("\n" + results[["pr_auc", "roc_auc", "brier", f"precision@{ALERT_BUDGET_K}",
                          f"recall@{ALERT_BUDGET_K}", "fit_seconds"]].to_string())