"""PoC API routes for vLLM server.

Endpoints:
    POST /api/v1/pow/init/generate - Start continuous generation loop
    POST /api/v1/pow/generate      - Generate artifacts for specific nonces
    GET  /api/v1/pow/generate/{id} - Poll async request result
    GET  /api/v1/pow/status        - Get generation status
    POST /api/v1/pow/stop          - Stop generation
"""
import asyncio
import time
import uuid
from typing import List, Optional, Dict, Any

from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel, ConfigDict

from vllm.logger import init_logger
from .config import PoCState

# Blocking execution: backoff time when chat is active
POC_CHAT_BUSY_BACKOFF_SEC = 0.05
from .data import (
    Artifact,
    DEFAULT_DIST_THRESHOLD,
    DEFAULT_P_MISMATCH,
    DEFAULT_FRAUD_THRESHOLD,
)
from .callbacks import CallbackSender
from .poc_params import PoCParams

logger = init_logger(__name__)

router = APIRouter(prefix="/api/v1/pow", tags=["PoC"])

_poc_tasks: Dict[int, Dict[str, Any]] = {}


# =============================================================================
# Request/Response Models (aligned with reference)
# =============================================================================

class PoCParamsModel(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: str
    seq_len: int
    k_dim: int = 12
    max_tokens: int = 0   # decode steps after prefill (0 = prefill-only)


class ArtifactModel(BaseModel):
    nonce: int
    vector_b64: str
    hidden_state_b64: Optional[str] = None
    reduced_hidden_state_b64: Optional[str] = None
    sphere_k: int = -1
    sphere_k_steps: List[int] = []   # k at each step: [prefill, decode1, …]
    # Validation mode only: steps where locally computed k differed from
    # the inference reference.  -1 for inference requests.
    n_sphere_mismatches: int = -1


class ValidationModel(BaseModel):
    artifacts: List[ArtifactModel]


class StatTestModel(BaseModel):
    dist_threshold: float = DEFAULT_DIST_THRESHOLD
    p_mismatch: float = DEFAULT_P_MISMATCH
    fraud_threshold: float = DEFAULT_FRAUD_THRESHOLD


class PoCInitGenerateRequest(BaseModel):
    block_hash: str
    block_height: int
    public_key: str
    node_id: int
    node_count: int
    group_id: int = 0
    n_groups: int = 1
    batch_size: int = 32
    params: PoCParamsModel
    url: Optional[str] = None
    blocking: bool = False


class PoCGenerateRequest(BaseModel):
    block_hash: str
    block_height: int
    public_key: str
    node_id: int
    node_count: int
    nonces: List[int]
    params: PoCParamsModel
    batch_size: int = 32
    wait: bool = False
    url: Optional[str] = None
    validation: Optional[ValidationModel] = None
    stat_test: Optional[StatTestModel] = None
    blocking: bool = False
    # Validation-mode sphere k tracking.
    # Maps nonce → sphere_k_steps from the reference inference run.
    # When set for a nonce the server runs in validation mode: it computes
    # its own k-ids but uses the reference k-ids to seed each subsequent
    # decode step, and returns the total mismatch count per nonce.
    inference_k_steps: Optional[Dict[int, List[int]]] = None


# =============================================================================
# Helpers
# =============================================================================

async def get_engine_client(request: Request):
    engine_client = getattr(request.app.state, 'engine_client', None)
    if engine_client is None:
        raise HTTPException(status_code=503, detail="Engine not available")
    return engine_client


async def check_poc_enabled(request: Request):
    if not getattr(request.app.state, 'poc_enabled', False):
        raise HTTPException(status_code=503, detail="PoC not enabled")


def check_params_match(request: Request, params: PoCParamsModel):
    """Check params match deployed model. Raises 409 on mismatch."""
    serving_models = getattr(request.app.state, 'openai_serving_models', None)
    if serving_models and hasattr(serving_models, 'base_model_paths'):
        base_paths = serving_models.base_model_paths
        if base_paths:
            model_path = base_paths[0].model_path
            served_names = [p.name for p in base_paths]
            valid_models = {model_path} | set(served_names)
            if params.model not in valid_models:
                raise HTTPException(
                    status_code=409,
                    detail={
                        "error": "params mismatch",
                        "requested": {"model": params.model},
                        "deployed": {"model": list(valid_models)},
                    }
                )


def _is_generation_active(app_id: int) -> bool:
    tasks = _poc_tasks.get(app_id)
    if not tasks:
        return False
    gen_task = tasks.get("gen_task")
    return gen_task is not None and not gen_task.done()


def _get_api_status(app_id: int) -> dict:
    tasks = _poc_tasks.get(app_id)
    if not tasks or not _is_generation_active(app_id):
        return {"status": PoCState.IDLE.value, "config": None, "stats": None}

    config = tasks.get("config", {})
    stats = tasks.get("stats", {})
    start_time = stats.get("start_time", 0)
    total_processed = stats.get("total_processed", 0)
    elapsed = time.time() - start_time if start_time > 0 else 0
    nonces_per_second = total_processed / elapsed if elapsed > 0 else 0

    return {
        "status": PoCState.GENERATING.value,
        "config": {
            "block_hash": config.get("block_hash"),
            "block_height": config.get("block_height"),
            "public_key": config.get("public_key"),
            "node_id": config.get("node_id"),
            "node_count": config.get("node_count"),
            "seq_len": config.get("seq_len"),
            "k_dim": config.get("k_dim"),
        },
        "stats": {
            "total_processed": total_processed,
            "nonces_per_second": nonces_per_second,
        },
    }


async def _cancel_poc_tasks(app_id: int):
    tasks = _poc_tasks.pop(app_id, None)
    if tasks:
        if tasks.get("stop_event"):
            tasks["stop_event"].set()
        if tasks.get("gen_task"):
            tasks["gen_task"].cancel()
            try:
                await tasks["gen_task"]
            except asyncio.CancelledError:
                pass
        if tasks.get("callback_sender"):
            tasks["callback_sender"].clear()


# =============================================================================
# Core: compute artifacts for a list of nonces
# =============================================================================

async def _compute_nonce_artifacts(
    engine_client,
    nonces: List[int],
    block_hash: str,
    public_key: str,
    block_height: int,
    seq_len: int,
    k_dim: int,
    poc_decode: bool = False,
    max_tokens: int = 0,
    app=None,
    blocking: bool = False,
    inference_k_steps: Optional[Dict[int, List[int]]] = None,
) -> List[dict]:
    """Compute artifacts for nonces via the scheduler."""
    exclusive_mode_set = False
    if blocking and app:
        # Step 1: Wait for in-flight requests to drain
        load = getattr(app.state, 'server_load_metrics', 0)
        if load > 0:
            logger.info(f"PoC blocking: waiting for {load} in-flight request(s)")
        while load > 0:
            await asyncio.sleep(POC_CHAT_BUSY_BACKOFF_SEC)
            load = getattr(app.state, 'server_load_metrics', 0)
        # Step 2: Set exclusive mode to reject new chat requests
        app.state.poc_exclusive_mode = True
        exclusive_mode_set = True
        logger.info("PoC exclusive mode: rejecting new chat requests")

    async def compute_one(nonce: int) -> Optional[dict]:
        inf_steps = inference_k_steps.get(nonce) if inference_k_steps else None
        poc_params = PoCParams(
            block_hash=block_hash,
            public_key=public_key,
            block_height=block_height,
            nonce=nonce,
            seq_len=seq_len,
            k_dim=k_dim,
            poc_decode=poc_decode,
            max_tokens=max_tokens,
            inference_sphere_k_steps=inf_steps,
        )
        request_id = f"poc-{uuid.uuid4()}"

        try:
            async for output in engine_client.generate(
                poc_params=poc_params,
                request_id=request_id,
                priority=10,
            ):
                if output.finished:
                    poc_out = output.poc_output
                    if poc_out is None:
                        return None
                    if isinstance(poc_out, dict):
                        return {
                            "nonce": poc_out.get("nonce", nonce),
                            "vector_b64": poc_out.get("vector_b64", ""),
                            "hidden_state_b64": poc_out.get("hidden_state_b64"),
                            "reduced_hidden_state_b64": poc_out.get("reduced_hidden_state_b64"),
                            "sphere_k": poc_out.get("sphere_k", -1),
                            "sphere_k_steps": poc_out.get("sphere_k_steps", []),
                            "n_sphere_mismatches": poc_out.get("n_sphere_mismatches", -1),
                        }
                    return {
                        "nonce": poc_out.nonce,
                        "vector_b64": poc_out.vector_b64,
                        "hidden_state_b64": poc_out.hidden_state_b64,
                        "reduced_hidden_state_b64": poc_out.reduced_hidden_state_b64,
                        "sphere_k": getattr(poc_out, "sphere_k", -1),
                        "sphere_k_steps": getattr(poc_out, "sphere_k_steps", []),
                        "n_sphere_mismatches": getattr(poc_out, "n_sphere_mismatches", -1),
                    }
        except Exception as e:
            logger.error(f"Error computing nonce {nonce}: {e}")
        return None

    try:
        tasks = [compute_one(n) for n in nonces]
        results = await asyncio.gather(*tasks)
        return [r for r in results if r is not None]
    finally:
        if exclusive_mode_set and app:
            app.state.poc_exclusive_mode = False
            logger.info("PoC exclusive mode: ended, accepting chat requests")


# =============================================================================
# Endpoints
# =============================================================================

@router.post("/init/generate")
async def init_generate(request: Request, body: PoCInitGenerateRequest) -> dict:
    await check_poc_enabled(request)
    check_params_match(request, body.params)
    engine_client = await get_engine_client(request)

    app_id = id(request.app)
    if _is_generation_active(app_id):
        raise HTTPException(status_code=409, detail="Already generating")

    await _cancel_poc_tasks(app_id)

    config = {
        "block_hash": body.block_hash,
        "block_height": body.block_height,
        "public_key": body.public_key,
        "node_id": body.node_id,
        "node_count": body.node_count,
        "batch_size": body.batch_size,
        "seq_len": body.params.seq_len,
        "k_dim": body.params.k_dim,
        "poc_decode": getattr(request.app.state, "poc_decode", False),
        "max_tokens": body.params.max_tokens,
    }

    stats = {"start_time": 0, "total_processed": 0}
    stop_event = asyncio.Event()

    callback_sender = None
    callback_task = None
    if body.url:
        callback_sender = CallbackSender(body.url, stop_event, body.params.k_dim)
        callback_task = asyncio.create_task(callback_sender.run())

    gen_task = asyncio.create_task(
        _generation_loop(engine_client, stop_event, callback_sender, config, stats,
                         app=request.app, blocking=body.blocking)
    )

    _poc_tasks[app_id] = {
        "gen_task": gen_task,
        "callback_task": callback_task,
        "callback_sender": callback_sender,
        "stop_event": stop_event,
        "config": config,
        "stats": stats,
    }

    return {"status": "OK", "pow_status": {"status": "GENERATING"}}


async def _generation_loop(
    engine_client,
    stop_event: asyncio.Event,
    callback_sender: Optional[CallbackSender],
    config: dict,
    stats: dict,
    app=None,
    blocking: bool = False,
):
    """Continuous generation loop for /init/generate."""
    from .data import pad_nonces, filter_artifacts

    node_id = config["node_id"]
    node_count = config["node_count"]
    batch_size = config["batch_size"]
    nonce_counter = 0

    start_time = time.time()
    stats["start_time"] = start_time
    stats["total_processed"] = 0
    last_report_time = start_time

    logger.info(f"PoC generation started (node {node_id}/{node_count})")

    try:
        while not stop_event.is_set():
            nonces = [node_id + (nonce_counter + i) * node_count
                      for i in range(batch_size)]
            nonce_counter += batch_size

            artifacts = await _compute_nonce_artifacts(
                engine_client, nonces,
                config["block_hash"], config["public_key"],
                config.get("block_height", 0),
                config["seq_len"], config["k_dim"],
                poc_decode=config.get("poc_decode", False),
                max_tokens=config.get("max_tokens", 0),
                app=app, blocking=blocking,
            )
            
            if artifacts and callback_sender:
                artifact_objs = [Artifact(nonce=a["nonce"], vector_b64=a["vector_b64"]) for a in artifacts]
                callback_sender.add_artifacts(artifact_objs, {
                    "public_key": config["public_key"],
                    "block_hash": config["block_hash"],
                    "block_height": config["block_height"],
                    "node_id": config["node_id"],
                })

            stats["total_processed"] += len(nonces)

            current_time = time.time()
            if current_time - last_report_time >= 5.0:
                elapsed_min = (current_time - start_time) / 60
                rate = stats["total_processed"] / elapsed_min if elapsed_min > 0 else 0
                logger.info(f"Generated: {stats['total_processed']} nonces ({rate:.0f}/min)")
                last_report_time = current_time

    except asyncio.CancelledError:
        elapsed_min = (time.time() - start_time) / 60
        logger.info(f"PoC stopped: {stats['total_processed']} nonces in {elapsed_min:.2f}min")


@router.post("/generate")
async def generate(request: Request, body: PoCGenerateRequest) -> dict:
    await check_poc_enabled(request)
    check_params_match(request, body.params)
    engine_client = await get_engine_client(request)

    app_id = id(request.app)

    poc_decode = getattr(request.app.state, "poc_decode", False)

    if body.wait:
        # Sync path: compute all nonces and return
        artifacts = await _compute_nonce_artifacts(
            engine_client, body.nonces,
            body.block_hash, body.public_key, body.block_height,
            body.params.seq_len, body.params.k_dim,
            poc_decode=poc_decode,
            max_tokens=body.params.max_tokens,
            app=request.app, blocking=body.blocking,
            inference_k_steps=body.inference_k_steps,
        )

        response = {
            "status": "completed",
            "request_id": str(uuid.uuid4()),
            "artifacts": artifacts,
            "encoding": {
                "dtype": "f16",
                "k_dim": body.params.k_dim,
                "endian": "le",
            },
        }

        # Inline validation if provided
        if body.validation:
            from .data import decode_vector, is_mismatch, fraud_test
            import numpy as np

            validation_map = {
                a.nonce: a.vector_b64 for a in body.validation.artifacts
            }
            st = body.stat_test or StatTestModel()

            n_mismatch = 0
            mismatch_nonces = []
            for art in artifacts:
                expected_b64 = validation_map.get(art["nonce"])
                if expected_b64 is None:
                    continue
                computed_vec = decode_vector(art["vector_b64"])
                if is_mismatch(computed_vec, expected_b64, st.dist_threshold):
                    n_mismatch += 1
                    mismatch_nonces.append(art["nonce"])

            p_value, fraud_detected = fraud_test(
                n_mismatch, len(body.nonces),
                st.p_mismatch, st.fraud_threshold,
            )
            response.update({
                "n_total": len(body.nonces),
                "n_mismatch": n_mismatch,
                "mismatch_nonces": mismatch_nonces,
                "p_value": p_value,
                "fraud_detected": fraud_detected,
            })

        return response

    # Async path: queue and return request_id
    request_id = str(uuid.uuid4())

    if app_id not in _poc_tasks:
        _poc_tasks[app_id] = {}

    _poc_tasks[app_id][request_id] = {
        "status": "running",
        "started_at": time.time(),
        "total_nonces": len(body.nonces),
    }

    app = request.app

    async def process_in_background():
        task_state = _poc_tasks[app_id][request_id]
        try:
            artifacts = await _compute_nonce_artifacts(
                engine_client, body.nonces,
                body.block_hash, body.public_key, body.block_height,
                body.params.seq_len, body.params.k_dim,
                poc_decode=poc_decode,
                max_tokens=body.params.max_tokens,
                app=app, blocking=body.blocking,
                inference_k_steps=body.inference_k_steps,
            )
            task_state.update({
                "status": "completed",
                "completed_at": time.time(),
                "result": {
                    "artifacts": artifacts,
                    "encoding": {
                        "dtype": "f16",
                        "k_dim": body.params.k_dim,
                        "endian": "le",
                    },
                },
            })
        except Exception as e:
            task_state.update({
                "status": "failed",
                "error": str(e),
                "completed_at": time.time(),
            })

    asyncio.create_task(process_in_background())

    return {
        "status": "queued",
        "request_id": request_id,
        "queued_count": len(body.nonces),
    }


@router.get("/generate/{request_id}")
async def get_generate_result(request: Request, request_id: str) -> dict:
    await check_poc_enabled(request)
    app_id = id(request.app)

    app_tasks = _poc_tasks.get(app_id, {})
    record = app_tasks.get(request_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Request {request_id} not found")

    response = {"status": record["status"], "request_id": request_id}

    if record["status"] == "completed" and record.get("result"):
        response.update(record["result"])
    elif record["status"] == "failed" and record.get("error"):
        response["error"] = record["error"]

    return response


@router.get("/status")
async def get_status(request: Request) -> dict:
    await check_poc_enabled(request)
    return _get_api_status(id(request.app))


@router.post("/stop")
async def stop_round(request: Request) -> dict:
    await check_poc_enabled(request)
    await _cancel_poc_tasks(id(request.app))
    return {"status": "OK", "pow_status": {"status": "STOPPED"}}
