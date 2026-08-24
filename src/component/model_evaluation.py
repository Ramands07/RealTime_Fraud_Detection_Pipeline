"""
Model Evaluation Component
===========================

Responsibility (and only this):
    1. Load every fitted candidate produced by model_training.py.
    2. Quantify uncertainty on the ranking metric via bootstrap resampling,
       so "best PR-AUC" can be distinguished from "best by noise".
    3. Apply the selection policy from notebook/01_EDA.ipynb section 7.2.
    4. Choose an operating threshold from a business cost model, not 0.5.
    5. Promote exactly one winner to models/model.pkl + models/metadata.json,
       and write human-readable reports to reports/.

Why bootstrap confidence intervals are the core of this file:
EDA section 7.2 flagged that Random Forest led HistGB+SMOTE by 0.0064 PR-AUC
on a test set containing only 74 frauds, and warned that gap was "well within
noise". That was an assertion. This file turns it into a measurement — we
resample the test set with replacement, recompute PR-AUC each time, and read
off the interval. If the leader's interval overlaps the runner-up's, we do
NOT claim a winner on PR-AUC alone; we fall through to explicit tiebreakers.

Selecting a model on a point estimate from 74 positives, with no uncertainty
quantification, is the most common way a portfolio project overclaims. A
reviewer will ask "is that difference significant?" — this file is the answer.
"""

import json
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_curve

from src.component.data_transformation import FEATURES, TARGET
from src.component.model_training import ALERT_BUDGET_K, evaluate
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object, save_dataframe, save_object

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RANDOM_STATE = 42


@dataclass
class ModelEvaluationConfig:
    transformed_test_path: Path = PROJECT_ROOT / "notebook" / "data" / "processed" / "test_transformed.csv"
    raw_test_path: Path = PROJECT_ROOT / "notebook" / "data" / "processed" / "test.csv"
    candidates_dir: Path = PROJECT_ROOT / "models" / "candidates"

    final_model_path: Path = PROJECT_ROOT / "models" / "model.pkl"
    metadata_path: Path = PROJECT_ROOT / "models" / "metadata.json"
    evaluation_report_path: Path = PROJECT_ROOT / "reports" / "model_comparison.csv"
    threshold_report_path: Path = PROJECT_ROOT / "reports" / "threshold_analysis.csv"

    # --- Business cost model (EDA section 1, "asymmetric costs") ---
    # A missed fraud costs the full transaction amount (the issuer eats the
    # chargeback). A false alarm costs a fixed analyst review fee. These two
    # numbers are the ONLY reason a threshold other than 0.5 is defensible,
    # so they belong in config where a business stakeholder can change them.
    cost_per_review: float = 3.0
    fn_loss_fraction: float = 1.0

    n_bootstrap: int = 1000

    def __post_init__(self):
        for f in ("transformed_test_path", "raw_test_path", "candidates_dir",
                  "final_model_path", "metadata_path",
                  "evaluation_report_path", "threshold_report_path"):
            setattr(self, f, Path(getattr(self, f)))


def paired_bootstrap(y_true, score_dict, n_boot=1000, seed=RANDOM_STATE):
    """Marginal CIs for every candidate AND paired CIs on the difference
    versus the leader — computed from ONE set of shared resample indices.

    Why paired, and why this matters more than it sounds:
    every candidate is scored on the SAME test set, so their errors are
    correlated — a bootstrap draw containing easy frauds lifts every model
    at once. Comparing two candidates' marginal CIs for overlap ignores that
    correlation and is far too permissive: it will call a 0.65 model "tied"
    with a 0.81 model simply because both intervals are wide.

    Testing the DIFFERENCE on shared draws cancels the shared noise. If the
    95% CI of (leader - challenger) excludes zero, the leader is genuinely
    ahead. This is the standard fix for the well-documented "overlapping
    error bars" fallacy, and on a 74-positive test set it is not a
    technicality — it changes which model gets shipped.
    """
    from sklearn.metrics import average_precision_score

    rng = np.random.default_rng(seed)
    n = len(y_true)
    names = list(score_dict)

    draws = {name: [] for name in names}
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        y_b = y_true[idx]
        if y_b.sum() == 0:
            continue  # PR-AUC undefined; skipping avoids biasing the CI down
        for name in names:
            draws[name].append(average_precision_score(y_b, score_dict[name][idx]))

    draws = {k: np.array(v) for k, v in draws.items()}

    marginal = {}
    for name, vals in draws.items():
        marginal[name] = {
            "pr_auc_ci_low": float(np.percentile(vals, 2.5)),
            "pr_auc_ci_high": float(np.percentile(vals, 97.5)),
            "n_valid_draws": len(vals),
        }

    leader = max(names, key=lambda k: draws[k].mean())
    for name in names:
        delta = draws[leader] - draws[name]
        lo, hi = np.percentile(delta, [2.5, 97.5])
        marginal[name]["delta_vs_leader_low"] = float(lo)
        marginal[name]["delta_vs_leader_high"] = float(hi)
        # Tied == the difference interval contains zero.
        marginal[name]["tied_with_leader"] = bool(lo <= 0 <= hi)

    return marginal, leader


def expected_cost(y_true, y_score, amounts, threshold, cost_per_review, fn_loss_fraction):
    """Total expected cost at a given threshold.

    FP -> pay an analyst to review a legitimate transaction.
    FN -> absorb the full amount of a fraud we let through.
    TP/TN cost nothing here (the TP review cost is a rounding error against
    the fraud it prevents; model it explicitly if your review team disagrees).
    """
    flagged = y_score >= threshold
    fp = flagged & (y_true == 0)
    fn = (~flagged) & (y_true == 1)
    return float(cost_per_review * fp.sum() + fn_loss_fraction * amounts[fn].sum())


class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig = None):
        self.config = config or ModelEvaluationConfig()

    # ------------------------------------------------------------------ #
    def initiate_model_evaluation(self) -> dict:
        logger.info("=== Model evaluation started ===")
        try:
            X_test, y_test, amounts = self._load_test_data()
            candidates = self._load_candidates()
            logger.info("Loaded %d candidates for evaluation", len(candidates))

            comparison = self._compare(candidates, X_test, y_test)
            save_dataframe(comparison.reset_index(), self.config.evaluation_report_path)

            winner_name, rationale = self._select(comparison)
            logger.info("SELECTED: %s", winner_name)
            for line in rationale:
                logger.info("  reason: %s", line)

            winner = candidates[winner_name]
            proba = winner.predict_proba(X_test)[:, 1]

            threshold_df = self._threshold_analysis(y_test, proba, amounts)
            save_dataframe(threshold_df, self.config.threshold_report_path)

            best_row = threshold_df.loc[threshold_df["expected_cost"].idxmin()]
            chosen_threshold = float(best_row["threshold"])
            logger.info(
                "Cost-optimal threshold: %.6f (cost %.2f vs %.2f doing nothing)",
                chosen_threshold, best_row["expected_cost"],
                amounts[y_test == 1].sum(),
            )

            save_object(self.config.final_model_path , winner)
            metadata = self._build_metadata(
                winner_name, comparison, chosen_threshold, best_row,
                y_test, proba, amounts, rationale,
            )
            self._save_metadata(metadata)

            logger.info("=== Model evaluation completed ===")
            return metadata

        except CustomException:
            raise
        except Exception as e:
            raise CustomException(e, sys)

    # ------------------------------------------------------------------ #
    def _load_test_data(self):
        df = pd.read_csv(self.config.transformed_test_path)
        X_test = df[FEATURES].values
        y_test = df[TARGET].values

        # Raw amounts are needed for the cost model — the transformed table
        # holds log_Amount AFTER RobustScaler, which is not a currency value.
        raw = pd.read_csv(self.config.raw_test_path)
        if len(raw) != len(df):
            raise ValueError(
                f"Raw test ({len(raw)}) and transformed test ({len(df)}) row "
                f"counts disagree — the two are out of sync. Re-run Phase 5."
            )
        amounts = raw["Amount"].values
        return X_test, y_test, amounts

    def _load_candidates(self) -> dict:
        paths = sorted(self.config.candidates_dir.glob("*.pkl"))
        if not paths:
            raise FileNotFoundError(
                f"No candidates in {self.config.candidates_dir}. "
                f"Run `python -m src.component.model_training` first."
            )
        return {p.stem: load_object(p) for p in paths}

    def _compare(self, candidates: dict, X_test, y_test) -> pd.DataFrame:
        # Score every candidate once, then bootstrap them all together on
        # shared resample indices so the paired comparison is valid.
        score_dict, base_rows = {}, {}
        for name, model in candidates.items():
            proba = model.predict_proba(X_test)[:, 1]
            score_dict[name] = proba
            base_rows[name] = evaluate(y_test, proba)

        logger.info("Running paired bootstrap (%d draws, %d candidates)...",
                   self.config.n_bootstrap, len(candidates))
        boot, leader = paired_bootstrap(y_test, score_dict, self.config.n_bootstrap)
        logger.info("Bootstrap leader: %s", leader)

        rows = []
        for name, metrics in base_rows.items():
            metrics.update(boot[name])
            metrics["candidate"] = name
            rows.append(metrics)
            logger.info(
                "%-32s PR-AUC %.4f [%.4f-%.4f] | delta vs leader [%+.4f, %+.4f] %s | Brier %.6f",
                name, metrics["pr_auc"], metrics["pr_auc_ci_low"], metrics["pr_auc_ci_high"],
                metrics["delta_vs_leader_low"], metrics["delta_vs_leader_high"],
                "TIED" if metrics["tied_with_leader"] else "    ",
                metrics["brier"],
            )
        return pd.DataFrame(rows).set_index("candidate").sort_values("pr_auc", ascending=False)

    def _select(self, comparison: pd.DataFrame) -> tuple[str, list[str]]:
        """Selection policy from EDA section 7.2, made explicit.

        Step 1: rank by PR-AUC.
        Step 2: find candidates statistically tied with the leader — using
                the PAIRED difference CI, not overlapping marginal CIs.
        Step 3: if the leader stands alone, take it. Otherwise break the tie
                on calibration (Brier), because the threshold in this file is
                chosen from a cost model, and a cost model needs
                probabilities that mean what they say.
        """
        rationale = []
        leader = comparison.index[0]
        leader_pr = comparison.loc[leader, "pr_auc"]

        tied = comparison[comparison["tied_with_leader"]]
        rationale.append(
            f"PR-AUC leader is {leader} at {leader_pr:.4f} "
            f"(95% CI {comparison.loc[leader, 'pr_auc_ci_low']:.4f}-"
            f"{comparison.loc[leader, 'pr_auc_ci_high']:.4f})"
        )

        if len(tied) <= 1:
            rationale.append(
                "Paired bootstrap: no other candidate's difference-CI "
                "contains zero — clear winner on PR-AUC."
            )
            return leader, rationale

        rationale.append(
            f"Paired bootstrap: {len(tied)} candidates are statistically "
            f"indistinguishable from the leader (difference CI contains "
            f"zero): {', '.join(tied.index)}"
        )
        rationale.append(
            "Tie broken on calibration (Brier), because the operating "
            "threshold is chosen from a cost model that requires trustworthy "
            "probabilities."
        )
        winner = tied["brier"].idxmin()
        rationale.append(
            f"{winner} has the lowest Brier at {tied.loc[winner, 'brier']:.6f} "
            f"(vs {tied.loc[leader, 'brier']:.6f} for the PR-AUC leader) — "
            f"a {tied.loc[leader, 'brier'] / tied.loc[winner, 'brier']:.1f}x "
            f"improvement in probability quality at no significant PR-AUC cost."
        )
        return winner, rationale

    def _threshold_analysis(self, y_test, proba, amounts) -> pd.DataFrame:
        """Sweep thresholds and compute the business cost of each.

        The grid is drawn from the score quantiles rather than linspace(0,1)
        — with scores this skewed, a uniform grid would put almost every
        candidate threshold in a region where nothing is flagged.
        """
        grid = np.unique(np.quantile(proba, np.linspace(0.90, 0.99999, 500)))
        rows = []
        for t in grid:
            flagged = proba >= t
            tn, fp, fn, tp = confusion_matrix(y_test, flagged.astype(int)).ravel()
            rows.append({
                "threshold": float(t),
                "alerts": int(flagged.sum()),
                "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
                "precision": float(tp / max(tp + fp, 1)),
                "recall": float(tp / max(tp + fn, 1)),
                "expected_cost": expected_cost(
                    y_test, proba, amounts, t,
                    self.config.cost_per_review, self.config.fn_loss_fraction,
                ),
            })
        return pd.DataFrame(rows)

    def _build_metadata(self, winner_name, comparison, threshold, best_row,
                        y_test, proba, amounts, rationale) -> dict:
        do_nothing_cost = float(amounts[y_test == 1].sum())
        chosen_cost = float(best_row["expected_cost"])
        row = comparison.loc[winner_name]

        return {
            "model_name": winner_name,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "n_features": len(FEATURES),
            "features": FEATURES,
            "target": TARGET,
            "selection_rationale": rationale,
            "test_metrics": {
                "pr_auc": float(row["pr_auc"]),
                "pr_auc_ci_95": [float(row["pr_auc_ci_low"]), float(row["pr_auc_ci_high"])],
                "roc_auc": float(row["roc_auc"]),
                "brier": float(row["brier"]),
                f"precision@{ALERT_BUDGET_K}": float(row[f"precision@{ALERT_BUDGET_K}"]),
                f"recall@{ALERT_BUDGET_K}": float(row[f"recall@{ALERT_BUDGET_K}"]),
                "base_rate": float(y_test.mean()),
            },
            "operating_point": {
                "threshold": threshold,
                "alerts": int(best_row["alerts"]),
                "precision": float(best_row["precision"]),
                "recall": float(best_row["recall"]),
                "true_positives": int(best_row["tp"]),
                "false_positives": int(best_row["fp"]),
                "false_negatives": int(best_row["fn"]),
            },
            "business_impact": {
                "cost_per_review": self.config.cost_per_review,
                "cost_doing_nothing": do_nothing_cost,
                "cost_at_chosen_threshold": chosen_cost,
                "absolute_saving": do_nothing_cost - chosen_cost,
                "pct_loss_reduction": 100 * (do_nothing_cost - chosen_cost) / do_nothing_cost,
            },
        }

    def _save_metadata(self, metadata: dict) -> None:
        path = self.config.metadata_path
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(metadata, f, indent=2)
        logger.info("Saved metadata to %s", path)


if __name__ == "__main__":
    evaluator = ModelEvaluation()
    meta = evaluator.initiate_model_evaluation()

    print(f"\n{'='*62}")
    print(f"SELECTED MODEL: {meta['model_name']}")
    print(f"{'='*62}")
    m = meta["test_metrics"]
    print(f"  PR-AUC     : {m['pr_auc']:.4f}  95% CI [{m['pr_auc_ci_95'][0]:.4f}, {m['pr_auc_ci_95'][1]:.4f}]")
    print(f"  Base rate  : {m['base_rate']:.6f}  (PR-AUC floor)")
    print(f"  Brier      : {m['brier']:.6f}")
    op = meta["operating_point"]
    print(f"\n  Threshold  : {op['threshold']:.6f}")
    print(f"  Alerts     : {op['alerts']}  ->  TP {op['true_positives']} | FP {op['false_positives']} | FN {op['false_negatives']}")
    print(f"  Precision  : {op['precision']:.3f}   Recall: {op['recall']:.3f}")
    b = meta["business_impact"]
    print(f"\n  Cost doing nothing : {b['cost_doing_nothing']:,.2f}")
    print(f"  Cost with model    : {b['cost_at_chosen_threshold']:,.2f}")
    print(f"  Loss reduction     : {b['pct_loss_reduction']:.1f}%")