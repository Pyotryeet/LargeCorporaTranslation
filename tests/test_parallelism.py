"""Tests for tensor parallelism configuration.

NOTE: These tests validate the TensorParallelConfig logic at the Python
level. They intentionally use hardcoded tp_size values (1, 2) rather than
probing the host's actual GPU count. This ensures the tests are deterministic
and runnable on any machine — including single-GPU developer laptops and CI
runners. Multi-GPU correctness is validated via logit-identity end-to-end
tests on dedicated multi-GPU hardware, not in this file.

The layer partition arithmetic, global attention layer selection, and KV-cache
size estimation are independent of the number of physical GPUs available.
"""

from benchmark.hardware.parallelism import (
    TensorParallelConfig,
    get_tensor_parallel_config,
    GEMMA_3_12B_NUM_LAYERS,
    GEMMA_3_12B_NUM_KV_HEADS,
    GEMMA_3_12B_LOCAL_GLOBAL_RATIO,
)


class TestTensorParallelConfig:
    def test_default_single_gpu(self):
        cfg = TensorParallelConfig(tp_size=1)
        assert cfg.tp_size == 1
        assert cfg.layers_per_gpu == [48]  # all 48 layers on GPU 0

    def test_two_gpu_partition(self):
        cfg = get_tensor_parallel_config(2)
        assert cfg.tp_size == 2
        assert len(cfg.layer_ranges) == 2
        start0, end0 = cfg.layer_ranges[0]
        start1, end1 = cfg.layer_ranges[1]
        assert start0 == 0
        assert end0 == GEMMA_3_12B_NUM_LAYERS // 2  # 24
        assert start1 == GEMMA_3_12B_NUM_LAYERS // 2  # 24
        assert end1 == GEMMA_3_12B_NUM_LAYERS  # 48

    def test_single_gpu_config(self):
        cfg = get_tensor_parallel_config(1)
        assert cfg.tp_size == 1

    def test_global_attention_layers(self):
        cfg = TensorParallelConfig(tp_size=2)
        global_layers = cfg.get_global_attention_layers()
        assert len(global_layers) == 8
        assert global_layers[0] == 5
        assert global_layers[-1] == 47
        assert cfg.is_global_attention_layer(5) is True
        assert cfg.is_global_attention_layer(0) is False
        assert cfg.is_global_attention_layer(11) is True

    def test_layer_to_device_mapping(self):
        cfg = get_tensor_parallel_config(2)
        assert cfg.get_device_for_layer(0) == 0
        assert cfg.get_device_for_layer(23) == 0
        assert cfg.get_device_for_layer(24) == 1
        assert cfg.get_device_for_layer(47) == 1

    def test_kv_cache_estimation(self):
        cfg = TensorParallelConfig(tp_size=2)
        mb = cfg.estimate_kv_cache_mb(batch_size=64, sequence_length=8192, bytes_per_element=2)
        assert mb > 100  # Should be several GB

    def test_architecture_constants(self):
        assert GEMMA_3_12B_NUM_LAYERS == 48
        assert GEMMA_3_12B_NUM_KV_HEADS == 8
        assert GEMMA_3_12B_LOCAL_GLOBAL_RATIO == 5
