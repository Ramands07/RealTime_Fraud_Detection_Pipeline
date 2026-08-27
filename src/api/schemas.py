"""
API Schemas
============

Pydantic models defining the contract for POST /predict.

Why validation belongs here rather than inside PredictPipeline: FastAPI
rejects a malformed request with a structured 422 before any of our code
runs, and auto-generates the OpenAPI docs at /docs from these classes. The
pipeline's own _validate_input() stays as a second line of defence for
non-HTTP callers (batch jobs, notebooks, tests).

The V1..V28 fields are declared explicitly rather than as a dict. It is more
verbose, but it means /docs shows every required field with its type, and a
caller who omits V17 gets told exactly that instead of a generic error.
"""

from pydantic import BaseModel, ConfigDict, Field


class TransactionRequest(BaseModel):
    """One raw transaction, matching the schema the model was trained on."""

    model_config = ConfigDict(
        # Reject unknown fields rather than silently ignoring them. If a
        # caller sends "amount" instead of "Amount", they should hear about
        # it — not receive a confident score computed without it.
        extra="forbid",
        json_schema_extra={
            "example": {
                "Time": 145000.0, "Amount": 149.62,
                **{f"V{i}": 0.0 for i in range(1, 29)},
            }
        },
    )

    Time: float = Field(..., ge=0, description="Seconds since first transaction")
    Amount: float = Field(..., ge=0, description="Transaction amount")

    V1: float; V2: float; V3: float; V4: float; V5: float; V6: float; V7: float
    V8: float; V9: float; V10: float; V11: float; V12: float; V13: float; V14: float
    V15: float; V16: float; V17: float; V18: float; V19: float; V20: float; V21: float
    V22: float; V23: float; V24: float; V25: float; V26: float; V27: float; V28: float


class PredictionResponse(BaseModel):
    fraud_probability: float = Field(..., description="P(fraud) in [0, 1]")
    decision: str = Field(..., description="approve | review | decline")
    threshold: float = Field(..., description="Cost-optimal threshold from Phase 7")
    model_name: str
    latency_ms: float


class BatchPredictionRequest(BaseModel):
    transactions: list[TransactionRequest] = Field(..., min_length=1, max_length=1000)


class BatchPredictionResponse(BaseModel):
    predictions: list[PredictionResponse]
    count: int
    total_latency_ms: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_name: str | None = None
    threshold: float | None = None
    n_features: int | None = None