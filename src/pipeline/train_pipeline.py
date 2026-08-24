"""
Training Pipeline
==================

Wires the four components into a single reproducible command:

    data_ingestion -> data_transformation -> model_training -> model_evaluation

Run with:
    python -m src.pipeline.train_pipeline

This is an orchestrator, not a place for logic. Every decision lives in the
component it belongs to; this file only sequences them, times them, and
reports what came out. If you find yourself adding an `if` here about how a
model is trained or a feature is built, it belongs in a component instead.

Why orchestration is worth a separate file at all: running four scripts by
hand in the right order is a step you WILL get wrong eventually — usually by
editing a feature and forgetting to retrain, so the model in models/ no
longer matches the preprocessor beside it. One entrypoint makes that
mismatch impossible.
"""

import sys
import time
from pathlib import Path

from src.component.data_ignation import DataIngestion
from src.component.data_transformation import DataTransformation
from src.component.model_evaluation import ModelEvaluation
from src.component.model_training import ModelTraining
from src.exception import CustomException
from src.logger import logging

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class TrainPipeline:
    def __init__(self):
        self.stage_times: dict[str, float] = {}

    def run(self) -> dict:
        """Execute all four stages in order. Returns the winner's metadata."""
        logger.info("#" * 70)
        logger.info("TRAINING PIPELINE STARTED")
        logger.info("#" * 70)
        pipeline_start = time.perf_counter()

        try:
            # -------- Stage 1: ingestion --------
            train_path, test_path = self._stage(
                "data_ingestion",
                lambda: DataIngestion().initiate_data_ingestion(),
            )

            # -------- Stage 2: transformation --------
            # Returns arrays, but the components downstream read the parquet
            # files it writes — we ignore the in-memory arrays here so that
            # each stage stays independently runnable and restartable.
            self._stage(
                "data_transformation",
                lambda: DataTransformation().initiate_data_transformation(),
            )

            # -------- Stage 3: training --------
            results = self._stage(
                "model_training",
                lambda: ModelTraining().initiate_model_training(),
            )

            # -------- Stage 4: evaluation & promotion --------
            metadata = self._stage(
                "model_evaluation",
                lambda: ModelEvaluation().initiate_model_evaluation(),
            )

            total = time.perf_counter() - pipeline_start
            self._report(results, metadata, total)
            return metadata

        except CustomException:
            raise
        except Exception as e:
            raise CustomException(e, sys)

    def _stage(self, name: str, fn):
        """Run one stage, time it, and log a clear boundary.

        The boundary logs matter more than they look: when this fails at
        3am in Phase 10's scheduled retrain, the last '--- STAGE ... ---'
        line in the log tells you instantly which component broke, without
        reading a traceback.
        """
        logger.info("--- STAGE START: %s ---", name)
        t0 = time.perf_counter()
        result = fn()
        elapsed = time.perf_counter() - t0
        self.stage_times[name] = elapsed
        logger.info("--- STAGE DONE: %s (%.1fs) ---", name, elapsed)
        return result

    def _report(self, results, metadata, total_seconds: float) -> None:
        logger.info("#" * 70)
        logger.info("TRAINING PIPELINE COMPLETED in %.1fs", total_seconds)
        for stage, secs in self.stage_times.items():
            logger.info("  %-22s %6.1fs  (%4.1f%%)",
                       stage, secs, 100 * secs / total_seconds)
        logger.info("  Candidates trained : %d", len(results))
        logger.info("  Promoted model     : %s", metadata["model_name"])
        logger.info("  PR-AUC             : %.4f", metadata["test_metrics"]["pr_auc"])
        logger.info("  Threshold          : %.6f", metadata["operating_point"]["threshold"])
        logger.info("#" * 70)


if __name__ == "__main__":
    pipeline = TrainPipeline()
    meta = pipeline.run()

    print(f"\n{'='*62}")
    print("TRAINING PIPELINE COMPLETE")
    print(f"{'='*62}")
    print(f"  Promoted model : {meta['model_name']}")
    print(f"  PR-AUC         : {meta['test_metrics']['pr_auc']:.4f}")
    print(f"  Threshold      : {meta['operating_point']['threshold']:.6f}")
    print(f"  Loss reduction : {meta['business_impact']['pct_loss_reduction']:.1f}%")
    print(f"\n  Artifacts:")
    print(f"    models/model.pkl")
    print(f"    models/preprocessor.pkl")
    print(f"    models/metadata.json")
    print(f"    reports/model_comparison.csv")
    print(f"    reports/threshold_analysis.csv")