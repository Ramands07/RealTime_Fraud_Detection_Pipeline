"""
Data Transformation Component
==============================

Responsibility (and only this):
    1. Load the train/test splits produced by data_ingestion.py.
    2. Engineer features EXACTLY as specified in notebook/01_EDA.ipynb,
       section 5.2 — hour_sin, hour_cos, log_Amount.
    3. Drop the raw columns that spec supersedes (Time, Amount).
    4. Impute (defensive) and scale with RobustScaler, fit on TRAIN ONLY.
    5. Persist the fitted preprocessor and the transformed feature tables.

Deliberately NOT done here: class-imbalance resampling (SMOTE / ADASYN /
undersampling). See the module-level note "Why resampling is not here"
below — it's a design decision, not an oversight, and worth understanding.
"""

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_dataframe, save_object

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------------- #
# Single source of truth for feature names and order. Phase 6
# (model_training.py), Phase 7 (predict_pipeline.py) and Phase 8's
# serving API all import FEATURES from here rather than redefining it.
# A training/serving mismatch in feature order is silent and catastrophic
# — tree models will still produce a number, just the wrong one — so this
# constant existing in exactly one place is not a style preference.
# ---------------------------------------------------------------------- #
V_COLUMNS = [f"V{i}" for i in range(1, 29)]
ENGINEERED_COLUMNS = ["log_Amount", "hour_sin", "hour_cos"]
FEATURES = V_COLUMNS + ENGINEERED_COLUMNS
TARGET = "Class"


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """The feature spec from EDA section 5.2, as executable code.

    This function is called identically on train and on test — and later,
    unchanged, on a single live transaction in the serving layer (Phase 8).
    That's the whole point of pulling it out as a standalone function
    instead of inlining it in a class method: it has no dependency on
    anything fitted (no scaler, no imputer), so it is safe to reuse
    anywhere without risking leakage.
    """
    out = df.copy()

    if (out["Amount"] < 0).any():
        n_bad = int((out["Amount"] < 0).sum())
        raise ValueError(
            f"{n_bad} rows have negative Amount — log1p is undefined there. "
            f"This means the upstream data differs from the EDA reference "
            f"snapshot; investigate before proceeding."
        )

    hour = (out["Time"] // 3600) % 24
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    out["log_Amount"] = np.log1p(out["Amount"])  # log1p handles Amount == 0

    return out


@dataclass
class DataTransformationConfig:
    train_data_path: Path = PROJECT_ROOT / "notebook" / "data" / "processed" / "train.csv"
    test_data_path: Path = PROJECT_ROOT / "notebook" / "data" / "processed" / "test.csv"

    # Matches the models/ layout from the original project plan
    # (models/model.pkl, preprocessor.pkl, metadata.json).
    preprocessor_obj_path: Path = PROJECT_ROOT / "models" / "preprocessor.pkl"

    transformed_train_path: Path = PROJECT_ROOT / "notebook"  / "data" / "processed" / "train_transformed.csv"
    transformed_test_path: Path = PROJECT_ROOT / "notebook" / "data" / "processed" / "test_transformed.csv"

    def __post_init__(self):
        self.train_data_path = Path(self.train_data_path)
        self.test_data_path = Path(self.test_data_path)
        self.preprocessor_obj_path = Path(self.preprocessor_obj_path)
        self.transformed_train_path = Path(self.transformed_train_path)
        self.transformed_test_path = Path(self.transformed_test_path)


class DataTransformation:
    def __init__(self, config: DataTransformationConfig = None):
        self.config = config or DataTransformationConfig()

    # ------------------------------------------------------------------ #
    def initiate_data_transformation(self):
        """Returns (X_train, y_train, X_test, y_test, preprocessor_path).
        Arrays, not DataFrames — Phase 6 feeds these straight into
        scikit-learn/imblearn pipelines, which want numpy."""
        logger.info("=== Data transformation started ===")
        try:
            train_df = pd.read_csv(self.config.train_data_path)
            test_df = pd.read_csv(self.config.test_data_path)
            logger.info("Loaded train=%s test=%s", train_df.shape, test_df.shape)

            train_df = build_features(train_df)
            test_df = build_features(test_df)
            logger.info("Engineered features: %s", ENGINEERED_COLUMNS)

            X_train_raw = train_df[FEATURES]
            y_train = train_df[TARGET].values
            X_test_raw = test_df[FEATURES]
            y_test = test_df[TARGET].values

            preprocessor = self._build_preprocessor()

            # FIT ON TRAIN ONLY. This is the single most important line in
            # this file — see EDA section 5.3, procedural leak #1.
            X_train = preprocessor.fit_transform(X_train_raw)
            X_test = preprocessor.transform(X_test_raw)

            self._validate_finite(X_train, "X_train")
            self._validate_finite(X_test, "X_test")
            self._log_scaler_diagnostics(preprocessor)

            save_object(self.config.preprocessor_obj_path , preprocessor)
            self._save_transformed(X_train, y_train, self.config.transformed_train_path)
            self._save_transformed(X_test, y_test, self.config.transformed_test_path)

            logger.info("=== Data transformation completed successfully ===")
            return X_train, y_train, X_test, y_test, str(self.config.preprocessor_obj_path)

        except CustomException:
            raise
        except Exception as e:
            raise CustomException(e, sys)

    # ------------------------------------------------------------------ #
    def _build_preprocessor(self) -> Pipeline:
        """Impute then scale, both fitted on train only.

        A single Pipeline over all 31 features rather than a ColumnTransformer
        with per-column rules — every feature here is already numeric (the
        V columns are PCA output, the engineered ones are numeric by
        construction; EDA confirmed no categorical columns
        exist), so there is nothing for a ColumnTransformer to route
        differently.

        RobustScaler (median/IQR) rather than StandardScaler, for the same
        reason chosen in EDA section : robust to the extreme Amount tail
        without discarding it. Applying it uniformly to the V columns too is
        harmless for tree models (a monotonic per-feature transform doesn't
        change their split points) and is required for Logistic Regression
        and for SMOTE's k-NN step (Phase 6) — so one shared, scaled table
        serves every candidate model without disadvantaging any of them.

        SimpleImputer is defensive, not required: EDA found
        zero nulls on the reference snapshot. It exists so a differently
        sourced CSV with missing values doesn't crash the pipeline instead
        of degrading gracefully.
        """
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler()),
        ])

    def _validate_finite(self, X: np.ndarray, name: str) -> None:
        if not np.all(np.isfinite(X)):
            n_bad = int((~np.isfinite(X)).sum())
            raise ValueError(
                f"{name} contains {n_bad} non-finite values after transform. "
                f"Likely cause: a zero-IQR column making RobustScaler divide "
                f"by zero. Check for constant columns."
            )

    def _log_scaler_diagnostics(self, preprocessor: Pipeline) -> None:
        """Print a couple of fitted statistics 
        so a reviewer (or you, in
        six months) can spot-check that scaling actually happened and was
        fit on sensible data, without having to unpickle the object."""
        scaler: RobustScaler = preprocessor.named_steps["scaler"]
        idx = FEATURES.index("log_Amount")
        logger.info(
            "Scaler fit check — log_Amount: median=%.4f, IQR-scale=%.4f "
            "(these come from TRAIN only)",
            scaler.center_[idx], scaler.scale_[idx],
        )

    def _save_transformed(self, X: np.ndarray, y: np.ndarray, path: Path) -> None:
        df = pd.DataFrame(X, columns=FEATURES)
        df[TARGET] = y
        save_dataframe(df, path)


if __name__ == "__main__":
    transformer = DataTransformation()
    X_train, y_train, X_test, y_test, preproc_path = transformer.initiate_data_transformation()
    print(f"\nX_train: {X_train.shape}  y_train frauds: {int(y_train.sum())}")
    print(f"X_test : {X_test.shape}  y_test frauds: {int(y_test.sum())}")
    print(f"Preprocessor saved to: {preproc_path}")