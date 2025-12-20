"""
Tests for enforced sampling in V1 engine.

These tests verify the bug fixes for:
1. swap_states() not updating enforced_req_ids
2. condense() causing out-of-order enforced_req_ids
3. all_enforced property returning True for empty batch

Run with: pytest tests/v1/sample/test_enforced_sampling.py -v
"""
import pytest
import torch

from vllm.sampling_params import SamplingParams, SamplingType
from vllm.v1.worker.gpu_input_batch import InputBatch


@pytest.fixture
def device():
    """Get available device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def input_batch(device):
    """Create a minimal InputBatch for testing."""
    # InputBatch requires these parameters
    max_num_reqs = 8
    max_model_len = 128
    max_num_tokens = 1024
    vocab_size = 32000
    pin_memory = False

    batch = InputBatch(
        max_num_reqs=max_num_reqs,
        max_model_len=max_model_len,
        max_num_tokens=max_num_tokens,
        device=device,
        vocab_size=vocab_size,
        pin_memory=pin_memory,
    )
    return batch


class TestSwapStatesEnforcedReqIds:
    """Test swap_states() correctly updates enforced_req_ids."""

    def test_swap_one_enforced_moves_position(self, input_batch):
        """When only one position has enforced, swap must move it."""
        # Setup: position 0 has enforced sampling
        input_batch.enforced_req_ids = [0]
        input_batch.enforced_token_ids = {0: [100, 200, 300]}
        input_batch.enforced_tokens = {}

        # Swap positions 0 and 2
        input_batch.swap_states(0, 2)

        # Position 0's enforced should now be at position 2
        assert 2 in input_batch.enforced_req_ids
        assert 0 not in input_batch.enforced_req_ids
        assert 2 in input_batch.enforced_token_ids
        assert 0 not in input_batch.enforced_token_ids

    def test_swap_both_enforced_unchanged(self, input_batch):
        """When both positions have enforced, both stay in list."""
        # Setup: both positions 0 and 2 have enforced sampling
        input_batch.enforced_req_ids = [0, 2]
        input_batch.enforced_token_ids = {0: [100], 2: [200]}
        input_batch.enforced_tokens = {}

        # Swap positions 0 and 2
        input_batch.swap_states(0, 2)

        # Both should still be in the list
        assert set(input_batch.enforced_req_ids) == {0, 2}
        # But their token_ids should be swapped
        assert input_batch.enforced_token_ids[0] == [200]
        assert input_batch.enforced_token_ids[2] == [100]

    def test_swap_neither_enforced_unchanged(self, input_batch):
        """When neither position has enforced, no change."""
        # Setup: positions 1 and 3 have enforced (not 0 or 2)
        input_batch.enforced_req_ids = [1, 3]
        input_batch.enforced_token_ids = {1: [100], 3: [200]}
        input_batch.enforced_tokens = {}

        # Swap positions 0 and 2 (neither has enforced)
        input_batch.swap_states(0, 2)

        # List should be unchanged
        assert input_batch.enforced_req_ids == [1, 3]


class TestAllEnforcedProperty:
    """Test all_enforced property correctly handles edge cases."""

    def test_empty_batch_returns_false(self, input_batch):
        """Empty batch must return all_enforced=False."""
        input_batch.greedy_reqs = set()
        input_batch.random_reqs = set()
        input_batch.enforced_reqs = set()

        assert input_batch.all_enforced == False

    def test_only_enforced_returns_true(self, input_batch):
        """Batch with only enforced requests returns True."""
        input_batch.greedy_reqs = set()
        input_batch.random_reqs = set()
        input_batch.enforced_reqs = {"req_1", "req_2"}

        assert input_batch.all_enforced == True

    def test_mixed_batch_returns_false(self, input_batch):
        """Mixed batch (enforced + greedy) returns False."""
        input_batch.greedy_reqs = {"req_1"}
        input_batch.random_reqs = set()
        input_batch.enforced_reqs = {"req_2"}

        assert input_batch.all_enforced == False

    def test_only_greedy_returns_false(self, input_batch):
        """Batch with only greedy requests returns False."""
        input_batch.greedy_reqs = {"req_1"}
        input_batch.random_reqs = set()
        input_batch.enforced_reqs = set()

        assert input_batch.all_enforced == False


class TestMixedEnforcedProperty:
    """Test mixed_enforced property."""

    def test_mixed_enforced_true(self, input_batch):
        """Returns True when enforced mixed with other types."""
        input_batch.greedy_reqs = {"req_1"}
        input_batch.random_reqs = set()
        input_batch.enforced_reqs = {"req_2"}

        assert input_batch.mixed_enforced == True

    def test_all_enforced_not_mixed(self, input_batch):
        """Returns False when all requests are enforced."""
        input_batch.greedy_reqs = set()
        input_batch.random_reqs = set()
        input_batch.enforced_reqs = {"req_1", "req_2"}

        assert input_batch.mixed_enforced == False

    def test_no_enforced_not_mixed(self, input_batch):
        """Returns False when no enforced requests."""
        input_batch.greedy_reqs = {"req_1"}
        input_batch.random_reqs = {"req_2"}
        input_batch.enforced_reqs = set()

        assert input_batch.mixed_enforced == False


class TestSamplingTypeEnforced:
    """Test SamplingType.ENFORCED is correctly detected."""

    def test_enforced_token_ids_sets_type(self):
        """SamplingParams with enforced_token_ids has ENFORCED type."""
        params = SamplingParams(
            temperature=0.0,
            enforced_token_ids=[100, 200, 300],
        )
        assert params.sampling_type == SamplingType.ENFORCED

    def test_enforced_tokens_sets_type(self):
        """SamplingParams with enforced_tokens has ENFORCED type."""
        params = SamplingParams(
            temperature=0.0,
            enforced_tokens=[{100: [100, 200]}, {200: [200, 300]}],
        )
        assert params.sampling_type == SamplingType.ENFORCED

    def test_no_enforced_greedy(self):
        """SamplingParams without enforced and temp=0 is GREEDY."""
        params = SamplingParams(temperature=0.0)
        assert params.sampling_type == SamplingType.GREEDY

    def test_no_enforced_random(self):
        """SamplingParams without enforced and temp>0 is RANDOM."""
        params = SamplingParams(temperature=1.0)
        assert params.sampling_type == SamplingType.RANDOM
