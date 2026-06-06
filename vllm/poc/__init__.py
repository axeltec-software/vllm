# PoC runs through the engine scheduler via generate(poc_params=...).
# No collective_rpc monkeypatch.
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
)
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
    "LayerHouseholderHook",
    "poc_router",
]
