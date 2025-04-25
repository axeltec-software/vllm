# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: SIM117
from typing import Dict, Optional

from kubernetes import client, config
import torch
from torch import nn

from vllm.config import LoadConfig, ModelConfig, VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.model_loader.base_loader import BaseModelLoader
from vllm.model_executor.model_loader.infiniband import InfinibandModelLoader
from vllm.model_executor.model_loader.utils import (device_loading_context,
                                                    set_default_torch_dtype)
from vllm.utils import get_ip, get_hostname
from vllm.model_executor.model_loader.utils import initialize_model
    
logger = init_logger(__name__)

class IBModelLoader(BaseModelLoader):
    """Model loader that will set model weights to random values."""

    def __init__(self, load_config: LoadConfig, rank: int, weights_source: str):
        super().__init__(load_config)
        self._ib_loader = InfinibandModelLoader(rank, weights_source, get_ip())

    def download_model(self, model_config: ModelConfig) -> None:
        pass  # Nothing to download

    def load_model(self, vllm_config: VllmConfig, model_config: ModelConfig) -> nn.Module:
        device_config = vllm_config.device_config
        target_device = torch.device(device_config.device)
        with set_default_torch_dtype(model_config.dtype):
            with target_device:
                model = initialize_model(vllm_config=vllm_config)

            for _, module in model.named_modules():
                quant_method = getattr(module, "quant_method", None)
                if quant_method is not None:
                    # When quant methods need to process weights after loading
                    # (for repacking, quantizing, etc), they expect parameters
                    # to be on the global target device. This scope is for the
                    # case where cpu offloading is used, where we will move the
                    # parameters onto device for processing and back off after.
                    with device_loading_context(
                            module, target_device):
                        quant_method.process_weights_after_loading(module)
        # Loading weights after quantization because it was performed on a different machine
        # Also quantization "process_weights_after_loading" can change tensor shape
        # with get_lock(model_config.model):
        self._ib_loader.fetch_model_weights(model)
        return model.eval()
    
    def load_weights(self, model: nn.Module,
                     model_config: ModelConfig) -> None:
        raise ValueError("Function load_weights should not be used for IBModelLoader.")

    @staticmethod
    def discover_weights_source(loader_extra_config: Dict[str, str]) -> Optional[str]:
        # Check if there's a direct override for the weight source URL
        if "weights_source_override" in loader_extra_config:
            weight_source = loader_extra_config["weights_source_override"]
            logger.info(f"Using weight source override: {weight_source}")
            return weight_source
            
        if "service" not in loader_extra_config:
            raise ValueError("Service configuration is missing.")
        service = loader_extra_config["service"]
        if "namespace" not in loader_extra_config:
            raise ValueError("Namespace configuration is missing.")
        namespace = loader_extra_config["namespace"]

        try:
            try:
                config.load_incluster_config()
            except config.config_exception.ConfigException:
                # Fallback to kubeconfig if not in a cluster
                config.load_kube_config()
            
            api = client.CoreV1Api()

            try:
                pod_name = get_hostname()
                if not pod_name:
                    logger.warning("Could not determine pod name")
                    return None
                
                current_pod = api.read_namespaced_pod(name=pod_name, namespace=namespace)
                
                replicaset_name = None
                for owner_ref in current_pod.metadata.owner_references:
                    if owner_ref.kind == "ReplicaSet":
                        replicaset_name = owner_ref.name
                        break
                
                if not replicaset_name:
                    logger.warning("Could not find ReplicaSet owner reference for current pod")
                    return None
                
                # Check if there are any endpoints for the service
                try:
                    endpoints = api.read_namespaced_endpoints(name=service, namespace=namespace)
                    
                    # Filter endpoints to only include pods from the same replicaset
                    valid_addresses = []
                    for subset in endpoints.subsets:
                        # Addresses can be None - that's ok, it means that there are no ready pods to pull weights from
                        for address in (subset.addresses or []):
                            if address.target_ref and address.target_ref.kind == "Pod":
                                try:
                                    pod = api.read_namespaced_pod(
                                        name=address.target_ref.name,
                                        namespace=namespace
                                    )
                                    for owner_ref in pod.metadata.owner_references:
                                        if owner_ref.kind == "ReplicaSet" and owner_ref.name == replicaset_name:
                                            valid_addresses.append(address)
                                            break
                                except client.ApiException:
                                    continue
                    
                    if not valid_addresses:
                        logger.info(f"No active endpoints found for service {service} in namespace {namespace} from the same replicaset")
                        return None
                    
                    # Use the first valid address
                    address = valid_addresses[0]
                    url = f"http://{address.ip}:8000"
                    
                    logger.info(f"Discovered weights source at: {url}")
                    return url
                    
                except client.ApiException as e:
                    logger.warning(f"Failed to get endpoints for service {service}: {e}")
                    return None
                    
            except client.ApiException as e:
                logger.warning(f"Failed to get current pod metadata: {e}")
                return None
                
        except ImportError:
            logger.warning("Kubernetes client is not installed. Please install with 'pip install kubernetes'")
            return None
        except Exception as e:
            logger.warning(f"Error discovering weights source: {e}")
            return None
