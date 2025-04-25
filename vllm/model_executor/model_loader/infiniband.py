from typing import Tuple, Generator

import requests
import torch

from vllm.config import KVTransferConfig
from vllm.distributed.kv_transfer.kv_pipe.pynccl_pipe import PyNcclPipe
from vllm.logger import init_logger

logger = init_logger(__name__)


class InfinibandModelLoader:
    def __init__(self, rank: int, weights_source: str, my_ip: str):
        self._rank = rank
        self._weights_source = weights_source
        self._my_ip = my_ip

    def _send_tensor(self, pipe: PyNcclPipe, name: str, tensor: torch.Tensor):
        while True:
            torch.cuda.synchronize()
            check_sum = torch.sum(tensor.to(dtype=torch.bfloat16)).to(device="cpu")
            torch.cuda.synchronize()
            logger.debug(f"Sending tensor {name}, {tensor.shape}, {tensor.dtype}, {check_sum.dtype}, {check_sum}, {tensor.flatten()[:5]}")
            meta = pipe.send_tensor_with_response(tensor, metadata={
                "finished": torch.zeros((1,), dtype=torch.bool, device='cpu'),
                "name": torch.tensor(list(name.encode('u8')), dtype=torch.uint8, device="cpu"),
                "check_sum": check_sum
            })
            if meta["success"].numpy():
                break

    def send_stream(self, dst_ip: str, dst_port: int, stream: Generator[Tuple[str, torch.Tensor], None, None]):
        logger.debug("Starting sending tensors")
        config = KVTransferConfig(
            kv_connector='PyNcclConnector',
            kv_buffer_device='cuda',
            kv_buffer_size=1e9,
            kv_rank=1,
            kv_role="kv_both",  # this arg doesn't matter in this test
            kv_parallel_size=2,
            kv_ip=dst_ip,
            kv_port=dst_port,
        )
        pipe = PyNcclPipe(
            local_rank=self._rank,
            config=config,
            device="cuda",
            port_offset=self._rank * 2,
        )

        for name, tensor in stream:
            self._send_tensor(pipe, name, tensor)

        pipe.send_metadata_only({"finished": torch.ones((1,), dtype=torch.bool, device='cpu')})
        pipe.group.barrier()
        pipe.close()

    def load_tensors(self) -> Generator[Tuple[str, torch.Tensor], None, None]:
        logger.debug("Starting load tensors")
        config = KVTransferConfig(
            kv_connector='PyNcclConnector',
            kv_buffer_device='cuda',
            kv_buffer_size=1e9,
            kv_rank=0,
            kv_role="kv_both",  # this arg doesn't matter in this test
            kv_parallel_size=2,
            kv_ip=self._my_ip,
            kv_port=29503,
        )
        # TODO : make this request only for master process
        if self._rank == 0:
            resp = requests.post(f'{self._weights_source}/infiniband_load',
                          json={
                              "dst_ip": self._my_ip,
                              "dst_port": 29503,
                          })
            # Fail fast in case connection setup failed
            resp.raise_for_status()

        # TODO : here we can hang on TCPStore creation in case when something happened to sender
        # Need to address this hangup
        pipe = PyNcclPipe(
            local_rank=self._rank,
            config=config,
            port_offset=self._rank * 2,
            # device=device,
        )
        while True:
            tensor, metadata = pipe.recv_tensor()
            done = metadata['finished']
            if done.numpy()[0]:
                break
            name_raw = metadata['name']
            name = bytes(name_raw.numpy()).decode('u8')
            check_sum = metadata['check_sum']
            torch.cuda.synchronize()
            real_sum = torch.sum(tensor.to(dtype=torch.bfloat16)).to(device="cpu")
            torch.cuda.synchronize()
            logger.debug(f"Receiving tensor {name}, {tensor.shape}, {tensor.dtype}, {check_sum.dtype}, {check_sum}, {real_sum.dtype}, {real_sum}, {tensor.flatten()[:5]}")
            logger.debug("Check sum difference: {}".format(check_sum - real_sum))
            if abs(check_sum - real_sum) < 1e-6:
                pipe.send_metadata_only({
                    'success': torch.ones((1,), dtype=torch.bool, device='cpu')
                })
                yield name, tensor
            else:
                pipe.send_metadata_only({
                    'success': torch.zeros((1,), dtype=torch.bool, device='cpu')
                })

        logger.debug("Finished loading tensors")
        pipe.group.barrier()
        pipe.close()
        logger.debug("Closed remote pipes")

    def send_model_weights(self, dst_ip: str, dst_port: int, model: torch.nn.Module):
        self.send_stream(dst_ip, dst_port,
                         ((name, param)
                          for name, param in model.state_dict().items()
                          if torch.is_floating_point(param)))

    def fetch_model_weights(self, model: torch.nn.Module):
        state = model.state_dict()
        for name, tensor in self.load_tensors():
            assert (name in state), f"Unexpected tensor {name}"
            param = state[name]
            param.data.copy_(tensor.to(param.data.dtype))
