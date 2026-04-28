# Apply PoC engine patch for vLLM 0.15.1 V1 engine
from . import engine_patch
from .config import PoCConfig, PoCState
from .data import (
    PoCParams,
    Artifact,
    Encoding,
    ArtifactBatch,
    ValidationResult,
    encode_vector,
    decode_vector,
    is_mismatch,
    fraud_test,
    compare_artifacts,
    pad_nonces,
    filter_artifacts,
)
from .manager import PoCManager, PoCStats
from .routes import router as poc_router
from .layer_hooks import LayerHouseholderHook
from .poc_params import PoCParams

__all__ = [
    "PoCConfig",
    "PoCState",
    "PoCParams",
    "PoCDataParams",
    "Artifact",
    "Encoding",
    "ArtifactBatch",
    "ValidationResult",
    "encode_vector",
    "decode_vector",
    "is_mismatch",
    "fraud_test",
    "compare_artifacts",
    "pad_nonces",
    "filter_artifacts",
    "PoCManager",
    "PoCStats",
    "LayerHouseholderHook",
    "poc_router",
]
