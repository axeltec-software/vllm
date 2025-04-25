import torch

from vllm.config import KVTransferConfig
from vllm.distributed.kv_transfer.kv_pipe.pynccl_pipe import PyNcclPipe

device = torch.device("cuda:0")

config = KVTransferConfig(
    kv_connector='PyNcclConnector',
    kv_buffer_device='cuda',
    kv_buffer_size=1e9,
    kv_rank=0,
    kv_role="kv_both",  # this arg doesn't matter in this test
    kv_parallel_size=2,
    kv_ip="192.168.0.145",
    kv_port=29500,
)

pipe = PyNcclPipe(
    local_rank=device.index,
    config=config,
    device="cuda",
)

for T in range(40):
    while True:
        tensor, metadata = pipe.recv_tensor()

        # Check if the sender has finished sending tensors
        if 'finished' in metadata and metadata['finished'].item():
            print("Receiver: Finished receiving tensors.")
            break

        name = metadata['name']
        check_sum = metadata['check_sum']
        torch.cuda.synchronize()
        real_sum = torch.sum(tensor).to(device="cpu")
        torch.cuda.synchronize()

        print(
            f"Receiving tensor {bytes(name.numpy()).decode('u8')}, {tensor.shape}, {tensor.dtype}, "
            f"{check_sum.dtype}, {check_sum}, {real_sum.dtype}, {real_sum}"
        )
        print("Check sum difference: {}".format(check_sum - real_sum))

pipe.close()
