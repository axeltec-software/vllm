"""PoC (Proof of Compute) API routes for vLLM server.

API Endpoints:
    POST /api/v1/pow/generate - Generate PoC distances for nonces
    POST /api/v1/pow/compute  - Compute single nonce via scheduler (mixed batch)
    POST /api/v1/pow/validate - Validate nonces against expected distances
    GET  /api/v1/pow/status   - Get generation status

Request Modes:
    The /generate endpoint supports two modes via the `wait` parameter:

    1. Async mode (wait=false, default):
       - Returns immediately with {"status": "queued", "request_id": "...", "poll_url": "..."}
       - Poll GET /api/v1/pow/status/{request_id} to check progress
       - Optional: provide callback_url for result notification

    2. Sync mode (wait=true):
       - Blocks until all nonces are processed
       - Returns {"status": "completed", "valid_nonces": [...], ...}

Example:
    curl -X POST http://localhost:8000/api/v1/pow/generate \\
        -H "Content-Type: application/json" \\
        -d '{
            "block_hash": "0xabc",
            "block_height": 100,
            "public_key": "0xdef",
            "r_target": 1.5,
            "nonces": [1, 2, 3],
            "wait": true
        }'
"""
import asyncio
import time
import uuid
from typing import List, Optional, Dict, Any

from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel

from vllm.logger import init_logger
from .config import PoCState, PoCConfig
from .poc_params import PoCParams

logger = init_logger(__name__)

router = APIRouter(prefix="/api/v1/pow", tags=["PoC"])

# Module-level state for PoC tasks (per-app, keyed by id(app))
_poc_tasks: Dict[int, Dict[str, Any]] = {}


class PoCInitRequest(BaseModel):
    block_hash: str
    block_height: int
    public_key: str
    r_target: float
    fraud_threshold: float = 0.01
    node_id: int = -1
    node_count: int = -1
    batch_size: int = 32
    seq_len: int = 256
    callback_url: Optional[str] = None


class PoCStatusResponse(BaseModel):
    state: str
    valid_nonces: List[int]
    valid_distances: List[float]
    total_checked: int
    total_valid: int
    elapsed_seconds: float
    rate_per_second: float


class PoCValidateRequest(BaseModel):
    """Request to validate nonces - accepts full ProofBatch format."""
    public_key: str
    block_hash: str
    block_height: int
    nonces: List[int]
    dist: List[float]
    node_id: int


class PoCGenerateRequest(BaseModel):
    """Request to generate distances for specific nonces.

    Args:
        block_hash: Hex string of block hash (e.g., "0xabc123")
        block_height: Block height for deterministic seed
        public_key: Hex string of node's public key
        r_target: Distance threshold - nonces with distance < r_target are valid
        nonces: List of nonces to check
        node_id: Node identifier (default 0)
        seq_len: Sequence length for embeddings (default 256)
        batch_size: Processing batch size (default 32)
        callback_url: Optional URL for async result callback
        wait: If True, block until complete (recommended for V1 engine)
        return_vectors: If True, include output vectors in response (requires wait=True)
    """
    block_hash: str
    block_height: int
    public_key: str
    r_target: float
    nonces: List[int]
    node_id: int = 0
    seq_len: int = 256
    batch_size: int = 32
    callback_url: Optional[str] = None
    wait: bool = False
    return_vectors: bool = False


class PoCComputeRequest(BaseModel):
    """Request to compute distance for a single nonce via scheduler."""
    block_hash: str
    block_height: int
    public_key: str
    nonce: int
    r_target: float = 1.5
    seq_len: int = 256
    return_vectors: bool = False


async def get_engine_client(request: Request):
    """Get engine client from request app state."""
    engine_client = getattr(request.app.state, 'engine_client', None)
    if engine_client is None:
        raise HTTPException(status_code=503, detail="Engine not available")
    return engine_client


async def check_poc_enabled(request: Request):
    """Check if PoC is enabled."""
    poc_enabled = getattr(request.app.state, 'poc_enabled', False)
    if not poc_enabled:
        raise HTTPException(status_code=503, detail="PoC not enabled")


async def _cancel_poc_tasks(app_id: int):
    """Cancel running PoC tasks for an app."""
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
        if tasks.get("send_task"):
            tasks["send_task"].cancel()
            try:
                await tasks["send_task"]
            except asyncio.CancelledError:
                pass


async def _generation_loop(
    engine_client,
    batch_queue: asyncio.Queue,
    r_target: float,
):
    """Runs batches continuously, puts valid results in queue for callback sender.
    
    Note: In V1, this is a simplified version. The full state machine from V0
    is not available, but we simulate continuous generation.
    """
    total_checked = 0
    total_valid = 0
    batch_count = 0
    start_time = time.time()
    last_report_time = start_time
    
    logger.info(f"PoC generation started (r_target={r_target})")
    
    try:
        while True:
            # In V0, this called engine_client.poc_request("run_batch_with_state", {})
            # In V1, we don't have the old state machine, so we return empty
            # This is a placeholder for V1 compatibility
            result = {
                "should_continue": False,  # No continuous generation in V1 by default
                "nonces": [],
                "valid_nonces": [],
                "valid_distances": [],
            }
            
            if not result.get("should_continue", False):
                logger.warning("Continuous generation not fully supported in V1 - use /generate endpoint")
                break
            
            batch_count += 1
            batch_nonces = len(result.get("nonces", []))
            batch_valid = len(result.get("valid_nonces", []))
            total_checked += batch_nonces
            total_valid += batch_valid
            
            # Log progress every 5 seconds
            current_time = time.time()
            if current_time - last_report_time >= 5.0:
                elapsed_min = (current_time - start_time) / 60
                valid_pct = 100 * total_valid / total_checked if total_checked > 0 else 0
                valid_rate = total_valid / elapsed_min if elapsed_min > 0 else 0
                raw_rate = total_checked / elapsed_min if elapsed_min > 0 else 0
                logger.info(f"Generated: {total_valid} / {total_checked} "
                           f"({valid_pct:.1f} from 100) Time: {elapsed_min:.2f}min "
                           f"({valid_rate:.1f} valid/min, {raw_rate:.0f} raw/min)")
                last_report_time = current_time
            
            # Put valid batch in queue for sender (non-blocking)
            if result.get("valid_nonces"):
                await batch_queue.put({
                    "public_key": result["public_key"],
                    "block_hash": result["block_hash"],
                    "block_height": result["block_height"],
                    "nonces": result["valid_nonces"],
                    "dist": result["valid_distances"],
                    "node_id": result["node_id"],
                    "r_target": r_target,
                })
    except asyncio.CancelledError:
        elapsed_min = (time.time() - start_time) / 60
        valid_pct = 100 * total_valid / total_checked if total_checked > 0 else 0
        valid_rate = total_valid / elapsed_min if elapsed_min > 0 else 0
        logger.info(f"PoC stopped: {total_valid} / {total_checked} ({valid_pct:.1f} from 100) "
                   f"in {elapsed_min:.2f}min ({valid_rate:.1f} valid/min)")


async def _callback_sender_loop(
    batch_queue: asyncio.Queue,
    callback_url: str,
    stop_event: asyncio.Event,
):
    """Sends batches from queue to callback URL."""
    import aiohttp
    
    async with aiohttp.ClientSession() as session:
        while not stop_event.is_set():
            try:
                # Wait for batch with timeout to check stop_event
                batch = await asyncio.wait_for(
                    batch_queue.get(),
                    timeout=1.0
                )
                try:
                    await session.post(
                        f"{callback_url}/generated",
                        json=batch,
                        timeout=aiohttp.ClientTimeout(total=10)
                    )
                    logger.debug(f"Callback sent to {callback_url}/generated")
                except Exception as e:
                    logger.warning(f"Callback failed: {e}")
            except asyncio.TimeoutError:
                continue  # Check stop_event
            except asyncio.CancelledError:
                break


async def _start_generation_tasks(
    request: Request,
    engine_client,
    callback_url: Optional[str],
    r_target: float,
):
    """Start generation loop and optional callback sender tasks."""
    app_id = id(request.app)
    
    # Cancel existing tasks
    await _cancel_poc_tasks(app_id)
    
    # Create queue and stop event
    batch_queue: asyncio.Queue = asyncio.Queue()
    stop_event = asyncio.Event()
    
    # Start generation loop
    gen_task = asyncio.create_task(
        _generation_loop(engine_client, batch_queue, r_target)
    )
    
    # Start callback sender if URL provided
    send_task = None
    if callback_url:
        send_task = asyncio.create_task(
            _callback_sender_loop(batch_queue, callback_url, stop_event)
        )
    
    # Store for cleanup
    _poc_tasks[app_id] = {
        "gen_task": gen_task,
        "send_task": send_task,
        "stop_event": stop_event,
        "queue": batch_queue,
    }


@router.post("/init")
async def init_round(request: Request, body: PoCInitRequest) -> dict:
    """Initialize PoC round without starting generation.
    
    Note: V1 doesn't have the stateful PoC manager from V0.
    This endpoint is kept for API compatibility but has limited functionality.
    """
    await check_poc_enabled(request)
    engine_client = await get_engine_client(request)
    
    # In V0, this called: engine_client.poc_request("init", body.model_dump())
    # In V1, we don't have this, so we just acknowledge
    logger.info(f"PoC init called (V1 - limited functionality): {body.model_dump()}")
    
    return {"status": "OK", "pow_status": {"state": "IDLE"}}


@router.post("/init/generate")
async def init_generate(request: Request, body: PoCInitRequest) -> dict:
    """Initialize PoC round and start generating.
    
    Note: V1 doesn't have continuous generation like V0.
    Use /generate endpoint with specific nonces instead.
    """
    await check_poc_enabled(request)
    engine_client = await get_engine_client(request)
    
    if body.node_id == -1 or body.node_count == -1:
        raise HTTPException(
            status_code=400,
            detail="Node ID and node count must be set"
        )
    
    # Start background tasks (will exit immediately in V1)
    await _start_generation_tasks(request, engine_client, body.callback_url, body.r_target)
    
    logger.warning("Continuous generation limited in V1 - use /generate endpoint for specific nonces")
    
    return {"status": "OK", "pow_status": {"state": "GENERATING"}}


@router.post("/init/validate")
async def init_validate(request: Request, body: PoCInitRequest) -> dict:
    """Initialize PoC round and start validating.
    
    Note: V1 doesn't have the stateful validation mode from V0.
    Use /validate endpoint directly.
    """
    await check_poc_enabled(request)
    engine_client = await get_engine_client(request)
    
    app_id = id(request.app)
    
    # Cancel any generation tasks
    await _cancel_poc_tasks(app_id)
    
    # Store callback URL for validation results (per-app)
    _poc_tasks[app_id] = {
        "callback_url": body.callback_url,
        "stop_event": asyncio.Event(),
        "queue": asyncio.Queue(),
    }
    
    logger.info("Validation mode initialized (V1)")
    
    return {"status": "OK", "pow_status": {"state": "VALIDATING"}}


@router.post("/phase/generate")
async def start_generate(request: Request) -> dict:
    """Switch to generate mode.
    
    Note: V1 doesn't have phase switching. Use /generate endpoint.
    """
    await check_poc_enabled(request)
    engine_client = await get_engine_client(request)
    
    logger.warning("Phase switching not supported in V1 - use /generate endpoint")
    
    return {"status": "OK", "pow_status": {"state": "GENERATING"}}


@router.post("/phase/validate")
async def start_validate(request: Request) -> dict:
    """Switch to validate mode.
    
    Note: V1 doesn't have phase switching. Use /validate endpoint.
    """
    await check_poc_enabled(request)
    engine_client = await get_engine_client(request)
    
    # Cancel generation tasks
    await _cancel_poc_tasks(id(request.app))
    
    logger.info("Validate mode (V1)")
    
    return {"status": "OK", "pow_status": {"state": "VALIDATING"}}


@router.post("/stop")
async def stop_round(request: Request) -> dict:
    """Stop current PoC round."""
    await check_poc_enabled(request)
    engine_client = await get_engine_client(request)
    
    # Cancel all PoC tasks
    await _cancel_poc_tasks(id(request.app))
    
    return {"status": "OK", "pow_status": {"state": "STOPPED"}}


@router.post("/batch")
async def run_one_batch(request: Request) -> dict:
    """Run one batch of PoC generation.
    
    Note: V1 doesn't have the batch API. Use /generate endpoint.
    """
    await check_poc_enabled(request)
    engine_client = await get_engine_client(request)

    logger.warning("/batch endpoint not supported in V1 - use /generate")
    raise HTTPException(
        status_code=501,
        detail="Batch API not available in V1. Use /generate endpoint."
    )


@router.get("/status", response_model=PoCStatusResponse)
async def get_status(request: Request) -> PoCStatusResponse:
    """Get current PoC status.

    Note: V1 doesn't have stateful PoC tracking. Returns empty status.
    """
    await check_poc_enabled(request)
    engine_client = await get_engine_client(request)

    # In V0: engine_client.poc_request("status", {})
    # In V1: No state to query
    return PoCStatusResponse(
        state="IDLE",
        valid_nonces=[],
        valid_distances=[],
        total_checked=0,
        total_valid=0,
        elapsed_seconds=0.0,
        rate_per_second=0.0
    )


@router.get("/status/{request_id}")
async def get_request_status(request: Request, request_id: str) -> dict:
    """Get status of a specific async PoC request.

    Poll this endpoint after submitting a request with wait=false.

    Returns:
        - status: "running", "completed", or "failed"
        - When completed: valid_nonces, valid_distances, total_valid, etc.
    """
    app_id = id(request.app)

    if app_id not in _poc_tasks or request_id not in _poc_tasks[app_id]:
        raise HTTPException(status_code=404, detail=f"Request {request_id} not found")

    task_state = _poc_tasks[app_id][request_id]

    response = {
        "request_id": request_id,
        "status": task_state["status"],
        "total_nonces": task_state.get("total_nonces", 0),
        "completed": task_state.get("completed", 0),
    }

    if task_state["status"] == "completed":
        response.update({
            "total_valid": task_state.get("total_valid", 0),
            "valid_nonces": task_state.get("valid_nonces", []),
            "valid_distances": task_state.get("valid_distances", []),
            "elapsed_seconds": task_state.get("completed_at", 0) - task_state.get("started_at", 0),
        })
    elif task_state["status"] == "failed":
        response["error"] = task_state.get("error", "Unknown error")

    return response


@router.post("/validate")
async def validate_nonces(request: Request, body: PoCValidateRequest) -> dict:
    """Validate submitted nonces by recomputing distances.
    
    Accepts full ProofBatch format (matching original API).
    Results sent to callback_url/validated if configured.
    
    Note: V1 implementation recomputes nonces and compares.
    """
    await check_poc_enabled(request)
    engine_client = await get_engine_client(request)
    
    # Recompute distances for validation
    computed_distances = []
    fraud_detected = False
    
    for i, nonce in enumerate(body.nonces):
        poc_params = PoCParams(
            block_hash=body.block_hash,
            public_key=body.public_key,
            block_height=body.block_height,
            nonce=nonce,
            r_target=1.5,  # Not used for validation
            seq_len=256,
            return_vectors=False,
        )
        request_id = f"poc-validate-{uuid.uuid4()}"
        
        try:
            async for output in engine_client.generate(
                poc_params=poc_params,
                request_id=request_id,
                priority=10,
            ):
                if output.finished:
                    if hasattr(output, 'poc_output') and output.poc_output:
                        poc_out = output.poc_output
                        dist = poc_out['distance'] if isinstance(poc_out, dict) else poc_out.distance
                        computed_distances.append(dist)
                        # Check if distance matches (within tolerance)
                        if i < len(body.dist):
                            expected = body.dist[i]
                            computed = dist
                            if abs(computed - expected) > 0.01:  # 1% tolerance
                                fraud_detected = True
        except Exception as e:
            logger.error(f"Validation error for nonce {nonce}: {e}")
            computed_distances.append(None)
    
    result = {
        "fraud_detected": fraud_detected,
        "computed_distances": computed_distances,
    }
    
    # Send callback if URL configured (per-app)
    app_id = id(request.app)
    callback_url = _poc_tasks.get(app_id, {}).get("callback_url")
    if callback_url:
        import aiohttp
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{callback_url}/validated",
                    json=result,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as resp:
                    if resp.status != 200:
                        logger.warning(f"Validation callback failed: {resp.status}")
        except Exception as e:
            logger.warning(f"Validation callback error: {e}")
    
    return {
        "status": "OK", 
        "fraud_detected": result.get("fraud_detected", False),
        "computed_distances": result.get("computed_distances", []),
    }


@router.post("/generate")
async def generate_nonces(request: Request, body: PoCGenerateRequest) -> dict:
    """Generate distances for specific nonces using vLLM scheduler.
    
    Each nonce is submitted as an individual request to the scheduler, which
    automatically batches them based on token budget. This enables:
    - Dynamic batching with chat requests
    - Priority-based scheduling (PoC yields to chat)
    - No explicit batch_size needed
    
    If wait=True, blocks until all nonces are processed and returns results directly.
    If return_vectors=True (requires wait=True), also returns the output vectors.
    """
    await check_poc_enabled(request)
    engine_client = await get_engine_client(request)
    
    if body.return_vectors and not body.wait:
        raise HTTPException(
            status_code=400,
            detail="return_vectors requires wait=True"
        )
    
    # Submit each nonce as individual request to scheduler
    async def compute_single_nonce(nonce: int) -> dict:
        poc_params = PoCParams(
            block_hash=body.block_hash,
            public_key=body.public_key,
            block_height=body.block_height,
            nonce=nonce,
            r_target=body.r_target,
            seq_len=body.seq_len,
            return_vectors=body.return_vectors,
        )
        request_id = f"poc-{uuid.uuid4()}"
        
        try:
            # Use generate() in V1 (poc_compute() was V0-specific)
            async for output in engine_client.generate(
                poc_params=poc_params,
                request_id=request_id,
                priority=10,  # Low priority - chat has priority 0
            ):
                if output.finished:
                    # In V1, output has .poc_output (not .outputs like V0)
                    if hasattr(output, 'poc_output') and output.poc_output:
                        poc_out = output.poc_output
                        # Handle both dict and object access
                        if isinstance(poc_out, dict):
                            result = {
                                "nonce": poc_out.get('nonce', nonce),
                                "distance": poc_out.get('distance'),
                            }
                            if body.return_vectors and poc_out.get('vector'):
                                result["vector"] = poc_out['vector']
                        else:
                            result = {
                                "nonce": poc_out.nonce,
                                "distance": poc_out.distance,
                            }
                            if body.return_vectors and poc_out.vector:
                                result["vector"] = poc_out.vector
                        return result
            return {"nonce": nonce, "distance": None, "error": "No output"}
        except Exception as e:
            import traceback
            logger.error(f"Error computing nonce {nonce}: {e}\n{traceback.format_exc()}")
            return {"nonce": nonce, "distance": None, "error": str(e)}
    
    if body.wait:
        # Process all nonces concurrently through scheduler
        tasks = [compute_single_nonce(nonce) for nonce in body.nonces]
        results = await asyncio.gather(*tasks)
        
        # Filter valid results
        valid_results = [r for r in results if r.get("distance") is not None]
        valid_under_target = [r for r in valid_results if r["distance"] < body.r_target]
        
        response = {
            "status": "completed",
            "request_id": str(uuid.uuid4()),
            "total_checked": len(results),
            "total_valid": len(valid_under_target),
            "valid_nonces": [r["nonce"] for r in valid_under_target],
            "valid_distances": [r["distance"] for r in valid_under_target],
        }
        
        if body.return_vectors:
            response["all_results"] = results
        
        return response
    
    # Non-blocking: process in background
    request_id = str(uuid.uuid4())
    app_id = id(request.app)

    if app_id not in _poc_tasks:
        _poc_tasks[app_id] = {}

    # Initialize task state
    _poc_tasks[app_id][request_id] = {
        "status": "running",
        "started_at": time.time(),
        "total_nonces": len(body.nonces),
        "completed": 0,
        "results": [],
    }

    async def process_in_background():
        """Background task to process all nonces."""
        task_state = _poc_tasks[app_id][request_id]
        try:
            tasks = [compute_single_nonce(nonce) for nonce in body.nonces]
            results = await asyncio.gather(*tasks)

            valid_results = [r for r in results if r.get("distance") is not None]
            valid_under_target = [r for r in valid_results if r["distance"] < body.r_target]

            task_state.update({
                "status": "completed",
                "completed_at": time.time(),
                "completed": len(results),
                "results": results,
                "total_valid": len(valid_under_target),
                "valid_nonces": [r["nonce"] for r in valid_under_target],
                "valid_distances": [r["distance"] for r in valid_under_target],
            })

            # Call callback if provided
            if body.callback_url:
                try:
                    import httpx
                    async with httpx.AsyncClient() as client:
                        await client.post(body.callback_url, json=task_state, timeout=10)
                except Exception as e:
                    logger.warning(f"Callback failed: {e}")

        except Exception as e:
            task_state.update({
                "status": "failed",
                "error": str(e),
                "completed_at": time.time(),
            })

    # Start background task
    asyncio.create_task(process_in_background())

    return {
        "status": "queued",
        "request_id": request_id,
        "nonce_count": len(body.nonces),
        "poll_url": f"/api/v1/pow/status/{request_id}",
    }


@router.post("/compute")
async def compute_nonce(request: Request, body: PoCComputeRequest) -> dict:
    """Compute distance for a single nonce using scheduler integration.
    
    This endpoint uses vLLM's scheduler for PoC computation:
    - Request is scheduled alongside regular inference
    - Embeddings are generated on GPU
    - Results are returned when computation completes
    """
    await check_poc_enabled(request)
    engine_client = await get_engine_client(request)
    
    # Create PoCParams for this request
    poc_params = PoCParams(
        block_hash=body.block_hash,
        public_key=body.public_key,
        block_height=body.block_height,
        nonce=body.nonce,
        r_target=body.r_target,
        seq_len=body.seq_len,
        return_vectors=body.return_vectors,
    )
    
    request_id = f"poc-{uuid.uuid4()}"
    
    # In V0: engine_client.poc_compute() returned PoCRequestOutput with .outputs
    # In V1: engine_client.generate() returns output with .poc_output
    final_output = None
    async for output in engine_client.generate(
        poc_params=poc_params,
        request_id=request_id,
        priority=10,  # Low priority - chat has priority 0
    ):
        if output.finished:
            final_output = output
    
    if final_output is None:
        raise HTTPException(
            status_code=500,
            detail="No output received from PoC computation"
        )
    
    # Check for poc_output (V1 structure)
    if not hasattr(final_output, 'poc_output') or final_output.poc_output is None:
        raise HTTPException(
            status_code=500,
            detail="No PoC output in response"
        )
    
    poc_out = final_output.poc_output
    if isinstance(poc_out, dict):
        nonce_val = poc_out.get('nonce', body.nonce)
        distance_val = poc_out.get('distance')
        vector_val = poc_out.get('vector')
    else:
        nonce_val = poc_out.nonce
        distance_val = poc_out.distance
        vector_val = poc_out.vector

    response = {
        "nonce": nonce_val,
        "distance": distance_val,
        "valid": distance_val < body.r_target if distance_val is not None else False,
    }

    if body.return_vectors and vector_val:
        response["vector"] = vector_val
    
    return response
