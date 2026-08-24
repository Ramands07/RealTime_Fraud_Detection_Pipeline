"""
Prediction Pipeline
====================

Scores incoming transactions using the artifacts the training pipeline
promoted: preprocessor.pkl, model.pkl, and the threshold in metadata.json.

This is the file Phase 9's FastAPI endpoint wraps, which makes it the single
highest-risk file in the project for TRAINING/SERVING SKEW.

Skew means the features computed at prediction time differ, even slightly,
from those computed at training time. It is uniquely nasty because nothing
crashes — the model happily returns a number, just the wrong one. There is
no error to debug, only degraded performance nobody can explain.

The defence here is structural, not disciplinary:

  1. Features are built by importing build_features() from
     data_transformation.py — the SAME function, not a reimplementation.
     A copy-pasted version would drift the first time someone edits one copy.
  2. Column order is taken from the imported FEATURES constant, never
     hardcoded. Tree models silently accept a permuted matrix.
  3. The preprocessor is loaded from disk and only ever .transform()-ed,
     never re-fit. Re-fitting on incoming data is the classic serving leak.
  4. tests/test_predict_pipeline.py asserts batch scoring through this class
     equals the training-time scores on the same rows, to floating-point
     tolerance. That test is what makes claims 1-3 verifiable rather than
     aspirational.

Artifacts are loaded once at construction, not per call — see the note in
__init__.
"""

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.component.data_transformation import FEATURES, build_features
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# The raw fields a caller must supply. Deliberately the same schema the model
# was trained on, minus the target — Time and Amount are needed because
# build_features() derives hour_sin/hour_cos/log_Amount from them.
REQUIRED_INPUT_COLUMNS = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]


@dataclass
class PredictPipelineConfig:
    model_path: Path = PROJECT_ROOT / "models" / "model.pkl"
    preprocessor_path: Path = PROJECT_ROOT / "models" / "preprocessor.pkl"
    metadata_path: Path = PROJECT_ROOT / "models" / "metadata.json"

    def __post_init__(self):
        self.model_path = Path(self.model_path)
        self.preprocessor_path = Path(self.preprocessor_path)
        self.metadata_path = Path(self.metadata_path)


class PredictPipeline:
    def __init__(self, config: PredictPipelineConfig = None):
        """Load artifacts ONCE.

        Phase 9 will construct this at FastAPI startup and reuse it for every
        request. Loading a joblib model per request would add tens of
        milliseconds to every call and dominate the latency budget — the
        model is stateless once loaded, so there is no reason to reload it.
        """
        self.config = config or PredictPipelineConfig()
        try:
            self.model = load_object(self.config.model_path)
            self.preprocessor = load_object(self.config.preprocessor_path)

            with open(self.config.metadata_path) as f:
                self.metadata = json.load(f)

            self.threshold = float(self.metadata["operating_point"]["threshold"])
            self.model_name = self.metadata["model_name"]

            self._verify_artifacts_agree()

            logger.info(
                "PredictPipeline ready — model=%s threshold=%.6f features=%d",
                self.model_name, self.threshold, len(FEATURES),
            )
        except Exception as e:
            raise CustomException(e, sys)

    # ------------------------------------------------------------------ #
    def _verify_artifacts_agree(self) -> None:
        """Fail at startup if model.pkl, metadata.json and the code's
        FEATURES list disagree.

        This catches the most likely real-world breakage: someone adds a
        feature to data_transformation.py, re-runs training, but the
        deployed model.pkl is from the previous run. Without this check the
        API starts fine and quietly scores every transaction wrong.
        """
        meta_features = self.metadata.get("features")
        if meta_features and list(meta_features) != list(FEATURES):
            raise ValueError(
                "Feature mismatch between metadata.json and the current "
                "data_transformation.FEATURES. The deployed model was trained "
                "on a different feature set — re-run the training pipeline.\n"
                f"  metadata: {meta_features}\n  code    : {FEATURES}"
            )

        expected = getattr(self.model.named_steps["model"], "n_features_in_", None)
        if expected is not None and expected != len(FEATURES):
            raise ValueError(
                f"Model expects {expected} features but FEATURES defines "
                f"{len(FEATURES)}. Artifacts are out of sync."
            )

    def _validate_input(self, df: pd.DataFrame) -> None:
        missing = set(REQUIRED_INPUT_COLUMNS) - set(df.columns)
        if missing:
            raise ValueError(
                f"Input is missing required columns: {sorted(missing)}. "
                f"Expected the raw transaction schema: Time, V1..V28, Amount."
            )
        if len(df) == 0:
            raise ValueError("Input contains no rows.")

    # ------------------------------------------------------------------ #
    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Fraud probability for each row of a raw transaction frame."""
        try:
            self._validate_input(df)

            # SAME function used at training time — not a reimplementation.
            engineered = build_features(df)

            # Column order from the imported constant, never hardcoded:
            # a permuted matrix produces wrong scores with no error.
            X_raw = engineered[FEATURES]

            # transform() only. Never fit() — that would leak and would also
            # rescale using statistics from whatever happened to arrive.
            X = self.preprocessor.transform(X_raw)

            return self.model.predict_proba(X)[:, 1]
        except CustomException:
            raise
        except Exception as e:
            raise CustomException(e, sys)

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Score a batch and attach decisions.

        Three tiers rather than a binary flag, because a fraud system does
        not have one action available. 'review' sends a transaction to a
        human; 'decline' blocks it outright. The decline band is set well
        above the cost-optimal threshold — the threshold from Phase 7
        optimises total expected cost assuming a review, so blocking at that
        same level would decline far more legitimate customers than the cost
        model ever accounted for.
        """
        proba = self.predict_proba(df)
        decision = np.where(
            proba >= self.threshold * 10, "decline",
            np.where(proba >= self.threshold, "review", "approve"),
        )
        return pd.DataFrame({
            "fraud_probability": proba,
            "decision": decision,
            "threshold": self.threshold,
            "model_name": self.model_name,
        }, index=df.index)

    def predict_one(self, transaction: dict) -> dict:
        """Score a single transaction supplied as a dict.

        This is the exact call Phase 9's POST /predict endpoint makes.
        """
        result = self.predict(pd.DataFrame([transaction]))
        row = result.iloc[0]
        return {
            "fraud_probability": float(row["fraud_probability"]),
            "decision": str(row["decision"]),
            "threshold": float(row["threshold"]),
            "model_name": str(row["model_name"]),
        }


if __name__ == "__main__":
    import time

    pipeline = PredictPipeline()

    # Score a slice of the held-out test set as a smoke test.
    test_df = pd.read_csv(PROJECT_ROOT / "notebook"  / "data" / "processed" / "test.csv")
    sample = test_df.head(1000)

    t0 = time.perf_counter()
    results = pipeline.predict(sample.drop(columns=["Class"]))
    batch_ms = (time.perf_counter() - t0) * 1000

    print(f"\nModel     : {pipeline.model_name}")
    print(f"Threshold : {pipeline.threshold:.6f}")
    print(f"\nScored {len(sample)} transactions in {batch_ms:.1f}ms "
          f"({batch_ms/len(sample):.3f}ms per transaction)")
    print(f"\nDecision breakdown:")
    print(results["decision"].value_counts().to_string())

    # Single-transaction latency — the number that matters for the API.
    one = sample.drop(columns=["Class"]).iloc[0].to_dict()
    pipeline.predict_one(one)  # warm up
    t0 = time.perf_counter()
    for _ in range(100):
        pipeline.predict_one(one)
    single_ms = (time.perf_counter() - t0) * 1000 / 100
    print(f"\nSingle-transaction latency: {single_ms:.2f}ms (mean of 100)")