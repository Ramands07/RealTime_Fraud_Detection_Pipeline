"""
FastAPI Serving Layer
======================

Exposes the promoted model over HTTP.

    GET  /            -> HTML form (templates/index.html)
    POST /predict-form-> HTML result page (templates/home.html)
    GET  /health      -> liveness + which model is loaded
    POST /predict     -> score one transaction (JSON)
    POST /predict/batch -> score up to 1000 transactions
    GET  /model-info  -> full metadata.json for the deployed model
    GET  /docs        -> auto-generated Swagger UI

Run:
    uvicorn src.api.main:app --reload --port 8000

Design note — why the model loads in a lifespan handler:
PredictPipeline reads three files from disk and validates them against each
other. Doing that per request would add ~30ms to every call and dominate the
latency budget (the model itself scores in ~7ms). The lifespan handler runs
it exactly once at startup, so a misconfigured deployment fails immediately
and visibly at boot rather than on the first customer transaction.

The trade-off: the process must be restarted to pick up a retrained model.
That is the correct default — a model swapping under live traffic without an
explicit deploy is harder to reason about, not easier. Phase 10 can add an
authenticated /reload endpoint if hot-swapping is genuinely needed.
"""

import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from src.api.schemas import (
    BatchPredictionRequest, BatchPredictionResponse, HealthResponse,
    PredictionResponse, TransactionRequest,
)
from src.component.data_transformation import FEATURES
from src.logger import logging
from src.pipeline.predict_pipeline import PredictPipeline

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TEMPLATES_DIR = PROJECT_ROOT / "src" / "templates"

# Module-level state, populated at startup by the lifespan handler.
STATE: dict = {}

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load artifacts once at startup, release at shutdown."""
    logger.info("API starting — loading model artifacts...")
    try:
        STATE["pipeline"] = PredictPipeline()
        logger.info("Model loaded: %s", STATE["pipeline"].model_name)
    except Exception as e:
        # Deliberately not swallowed. A server that starts without a model
        # would return 503 on every request; failing loudly at boot is the
        # behaviour you want in a deploy pipeline.
        logger.info("FATAL: could not load model at startup: %s", e)
        raise
    yield
    STATE.clear()
    logger.info("API shutdown complete")


app = FastAPI(
    title="Real-Time Fraud Detection API",
    description=(
        "Scores credit card transactions for fraud risk. "
        "Threshold is cost-optimised, not 0.5 — see /model-info."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


def _get_pipeline() -> PredictPipeline:
    pipeline = STATE.get("pipeline")
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return pipeline


# ---------------------------------------------------------------------- #
# JSON API
# ---------------------------------------------------------------------- #
@app.get("/health", response_model=HealthResponse, tags=["ops"])
def health():
    """Liveness probe. Reports WHICH model is live, not just that something is.

    A health check that only returns {"status": "ok"} cannot tell you that a
    deploy silently rolled back to an older model — including the model name
    and threshold makes that visible.
    """
    pipeline = STATE.get("pipeline")
    if pipeline is None:
        return HealthResponse(status="degraded", model_loaded=False)
    return HealthResponse(
        status="ok",
        model_loaded=True,
        model_name=pipeline.model_name,
        threshold=pipeline.threshold,
        n_features=len(FEATURES),
    )


@app.get("/model-info", tags=["ops"])
def model_info():
    """Full metadata for the deployed model, including the selection
    rationale from Phase 7 and the business impact numbers."""
    return _get_pipeline().metadata


@app.post("/predict", response_model=PredictionResponse, tags=["scoring"])
def predict(transaction: TransactionRequest):
    """Score a single transaction."""
    pipeline = _get_pipeline()
    t0 = time.perf_counter()
    try:
        result = pipeline.predict_one(transaction.model_dump())
    except Exception as e:
        logger.info("Prediction failed: %s", e)
        raise HTTPException(status_code=400, detail=str(e))

    latency_ms = (time.perf_counter() - t0) * 1000
    logger.info(
        "scored p=%.6f decision=%s latency=%.2fms",
        result["fraud_probability"], result["decision"], latency_ms,
    )
    return PredictionResponse(**result, latency_ms=latency_ms)


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["scoring"])
def predict_batch(request: BatchPredictionRequest):
    """Score many transactions in one call.

    Uses the vectorised path — scoring 1000 rows as a batch is roughly 200x
    cheaper per row than 1000 single calls, because DataFrame construction
    overhead is paid once instead of per row.
    """
    import pandas as pd

    pipeline = _get_pipeline()
    t0 = time.perf_counter()
    df = pd.DataFrame([t.model_dump() for t in request.transactions])
    try:
        scored = pipeline.predict(df)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    total_ms = (time.perf_counter() - t0) * 1000
    per_row_ms = total_ms / len(scored)

    return BatchPredictionResponse(
        predictions=[
            PredictionResponse(
                fraud_probability=float(r.fraud_probability),
                decision=str(r.decision),
                threshold=float(r.threshold),
                model_name=str(r.model_name),
                latency_ms=per_row_ms,
            )
            for r in scored.itertuples()
        ],
        count=len(scored),
        total_latency_ms=total_ms,
    )


# ---------------------------------------------------------------------- #
# HTML UI (templates/index.html + templates/home.html)
# ---------------------------------------------------------------------- #
@app.get("/", response_class=HTMLResponse, tags=["ui"])
def index(request: Request):
    pipeline = STATE.get("pipeline")
    # Keyword form works on both old and new Starlette. The positional form
    # TemplateResponse(name, {...}) was reordered in newer versions and now
    # raises "unhashable type: 'dict'" — an unhelpful error for a common call.
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "model_name": pipeline.model_name if pipeline else "not loaded",
            "threshold": f"{pipeline.threshold:.6f}" if pipeline else "-",
        },
    )


@app.post("/predict-form", response_class=HTMLResponse, tags=["ui"])
async def predict_form(request: Request):
    """Handle the browser form.

    Any V-field left blank defaults to 0.0 — filling 28 anonymised PCA
    components by hand is not a realistic workflow, and the point of this
    page is to demonstrate the scoring path, not to be a data entry tool.
    """
    pipeline = _get_pipeline()
    form = await request.form()

    try:
        payload = {
            "Time": float(form.get("Time") or 0.0),
            "Amount": float(form.get("Amount") or 0.0),
        }
        for i in range(1, 29):
            payload[f"V{i}"] = float(form.get(f"V{i}") or 0.0)

        result = pipeline.predict_one(payload)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    return templates.TemplateResponse(
        request=request,
        name="home.html",
        context={
            "probability": f"{result['fraud_probability']:.6f}",
            "probability_pct": f"{result['fraud_probability'] * 100:.4f}",
            "decision": result["decision"],
            "threshold": f"{result['threshold']:.6f}",
            "model_name": result["model_name"],
            "amount": payload["Amount"],
        },
    )