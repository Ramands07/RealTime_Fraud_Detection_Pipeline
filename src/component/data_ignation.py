"""
Data Ingestion Component
=========================

Responsibility (and only this):
    1. Load the raw CSV.
    2. Validate it has the schema the rest of the pipeline expects.
    3. Drop duplicate rows (BEFORE splitting — see EDA notebook, section 3.3).
    4. Split chronologically into train/test (see EDA notebook, section 5.1).
    5. Persist both splits and return their paths.

Everything below implements decisions made in notebook/fraud_EDA.ipynb. If you
change the split strategy or the dedup logic, update the EDA notebook's
observation cells too — the two must never disagree.

This component does NOT scale, encode, engineer features, or resample.
That is data_transformation.py's job . Keeping ingestion "dumb" is
deliberate: it is the one component every other component depends on, so it
should do the least possible and be the easiest to trust.
"""

import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.exception import CustomException
from src.logger import logging
from src.utils import save_dataframe

logger = logging.getLogger(__name__)

# src/component/data_ingestion.py -> parents[2] is the project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# The exact columns EDA confirmed exist, in the order the source file has them.
# A schema check against this list is what catches "someone handed me a
# different CSV" before it becomes a confusing error three components later.
EXPECTED_COLUMNS = (
    ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount", "Class"]
)


@dataclass
class DataIngestionConfig:
    """All ingestion paths and parameters in one place, so nothing is a
    magic string buried in a method body."""

    raw_data_path: Path = PROJECT_ROOT / "notebook" / "data" / "creditcard.csv"
    train_data_path: Path = PROJECT_ROOT / "notebook" / "data" / "processed" / "train.csv"
    test_data_path: Path = PROJECT_ROOT / "notebook" / "data" / "processed" / "test.csv"

    # From EDA section 5.1: chronological split, most recent 20% held out.
    # Verified on this dataset to give train/test fraud rates within 1.35x
    # of each other (0.176% vs 0.130%) — stable enough to trust.
    test_size: float = 0.20
    time_col: str = "Time"
    target_col: str = "Class"

    def __post_init__(self):
        # Accept plain strings for any path field too — without this, passing
        # DataIngestionConfig(raw_data_path="some/str.csv") fails later with a
        # confusing AttributeError instead of a clear FileNotFoundError.
        self.raw_data_path = Path(self.raw_data_path)
        self.train_data_path = Path(self.train_data_path)
        self.test_data_path = Path(self.test_data_path)


class DataIngestion:
    def __init__(self, config: DataIngestionConfig = None):
        self.config = config or DataIngestionConfig()

    # ------------------------------------------------------------------ #
    # Public entrypoint
    # ------------------------------------------------------------------ #
    def initiate_data_ingestion(self) -> tuple[str, str]:
        """Run the full ingestion flow. Returns (train_path, test_path) as
        strings, which is what train_pipeline.py will pass to the next
        component in Phase 6."""
        logger.info("=== Data ingestion started ===")
        try:
            df = self._load_raw()
            self._validate_schema(df)
            df = self._drop_duplicates(df)
            train_df, test_df = self._chronological_split(df)
            self._log_split_health(train_df, test_df)

            save_dataframe(train_df, self.config.train_data_path)
            save_dataframe(test_df, self.config.test_data_path)

            logger.info("=== Data ingestion completed successfully ===")
            return str(self.config.train_data_path), str(self.config.test_data_path)

        except CustomException:
            raise
        except Exception as e:
            raise CustomException(e, sys)

    # ------------------------------------------------------------------ #
    # Steps — each one does exactly one thing, so a failure log line tells
    # you immediately which stage broke.
    # ------------------------------------------------------------------ #
    def _load_raw(self) -> pd.DataFrame:
        path = self.config.raw_data_path
        if not path.exists():
            raise FileNotFoundError(
                f"Raw data not found at {path}. "
                f"Expected the dataset at notebook/data/creditcard.csv — "
                f"see EDA notebook section 2.2 for the source."
            )
        df = pd.read_csv(path)
        logger.info("Loaded raw data: shape=%s from %s", df.shape, path)
        return df

    def _validate_schema(self, df: pd.DataFrame) -> None:
        """Fail loudly here rather than three components downstream with a
        cryptic KeyError. This is cheap insurance."""
        missing = set(EXPECTED_COLUMNS) - set(df.columns)
        if missing:
            raise ValueError(f"Missing expected columns: {sorted(missing)}")

        unexpected = set(df.columns) - set(EXPECTED_COLUMNS)
        if unexpected:
            logger.info(
                "Note: columns present beyond the expected schema: %s",
                sorted(unexpected),
            )

        null_counts = df[list(EXPECTED_COLUMNS)].isnull().sum()
        if null_counts.sum() > 0:
            logger.info(
                "Nulls found (EDA section 3.2 found none on the reference "
                "snapshot — this is a different run of the raw file):\n%s",
                null_counts[null_counts > 0],
            )
        # Deliberately not raising on nulls: data_transformation.py carries a
        # defensive imputer for exactly this reason (EDA section 3.2).

        target_values = set(df[self.config.target_col].unique())
        if not target_values <= {0, 1}:
            raise ValueError(
                f"Target column '{self.config.target_col}' has unexpected "
                f"values: {target_values}"
            )

        logger.info("Schema validation passed. %d columns confirmed.", len(EXPECTED_COLUMNS))

    def _drop_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Must happen here, before the split — dropping duplicates after
        splitting does not undo the leak (EDA section 3.3)."""
        dup_mask = df.duplicated(keep="first")
        n_dup = int(dup_mask.sum())
        n_dup_fraud = int(df.loc[dup_mask, self.config.target_col].sum())

        if n_dup:
            logger.info(
                "Dropping %d duplicate rows (%d were fraud, %.1f%% of the "
                "positive class) — see EDA section 3.3",
                n_dup, n_dup_fraud,
                100 * n_dup_fraud / max(df[self.config.target_col].sum(), 1),
            )

        deduped = df.drop_duplicates(keep="first").reset_index(drop=True)
        logger.info("Shape after dedup: %s (was %s)", deduped.shape, df.shape)
        return deduped

    def _chronological_split(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Sort by Time, cut at (1 - test_size). This is a temporal split,
        NOT a random one — chosen in EDA section 5.1 because Time is
        monotonic in this dataset and the fraud rate stays stable across
        the cut point (0.176% train vs 0.130% test, a 1.35x ratio)."""
        sorted_df = df.sort_values(self.config.time_col).reset_index(drop=True)
        cut = int(len(sorted_df) * (1 - self.config.test_size))

        train_df = sorted_df.iloc[:cut].copy()
        test_df = sorted_df.iloc[cut:].copy()

        split_time = pd.to_numeric(sorted_df.loc[cut, self.config.time_col])
        logger.info(
            "Chronological split at row %d (hour %.2f): train=%s test=%s",
            cut, float(split_time) / 3600,
            train_df.shape, test_df.shape,
        )
        return train_df, test_df

    def _log_split_health(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
        """A guardrail, not just a log line. If someone swaps the raw CSV
        for a differently-shaped dataset, this is what catches a split gone
        wrong (e.g. all fraud landing on one side) before it silently
        corrupts every downstream metric."""
        train_rate = train_df[self.config.target_col].mean()
        test_rate = test_df[self.config.target_col].mean()
        ratio = train_rate / test_rate if test_rate > 0 else float("inf")

        logger.info(
            "Fraud rate — train: %.4f%% (%d frauds) | test: %.4f%% (%d frauds) | ratio: %.2fx",
            train_rate * 100, int(train_df[self.config.target_col].sum()),
            test_rate * 100, int(test_df[self.config.target_col].sum()),
            ratio,
        )

        if test_df[self.config.target_col].sum() == 0:
            raise ValueError(
                "Test split contains zero frauds — cannot evaluate PR-AUC. "
                "Check test_size or the raw data's time ordering."
            )
        if ratio > 5 or ratio < 0.2:
            logger.info(
                "WARNING: train/test fraud rates differ by more than 5x "
                "(ratio=%.2f). EDA section 5.1 found 1.35x on the reference "
                "snapshot — a large deviation means this run's data differs "
                "materially and the chronological split should be revisited.",
                ratio,
            )


if __name__ == "__main__":
    ingestion = DataIngestion()
    train_path, test_path = ingestion.initiate_data_ingestion()
    print(f"\nTrain data: {train_path}")
    print(f"Test data : {test_path}")