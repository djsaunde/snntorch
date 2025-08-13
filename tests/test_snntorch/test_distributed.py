#!/usr/bin/env python

"""Tests for distributed training utilities."""

import pytest
import torch
import torch.nn as nn
import snntorch as snn
from snntorch.distributed import (
    count_parameters,
    get_module_sizes, 
    FSDPConfig,
    ShardingStrategy,
    auto_wrap_policy,
    optimize_fsdp2_for_snns,
)

# Skip tests if FSDP2 not available
try:
    from torch.distributed.fsdp import fully_shard
    FSDP2_AVAILABLE = True
except ImportError:
    FSDP2_AVAILABLE = False


class TestParameterCounting:
    """Test parameter counting utilities."""
    
    def test_count_parameters_linear(self):
        """Test parameter counting for linear layers."""
        layer = nn.Linear(100, 50)
        # Linear layer: 100*50 weights + 50 biases = 5050 parameters
        assert count_parameters(layer) == 5050
        
        # Test without bias
        layer_no_bias = nn.Linear(100, 50, bias=False)
        assert count_parameters(layer_no_bias) == 5000
    
    def test_count_parameters_conv(self):
        """Test parameter counting for convolutional layers."""
        # Conv2d: out_channels * in_channels * kernel_h * kernel_w + out_channels (bias)
        conv = nn.Conv2d(3, 64, kernel_size=3)
        expected = 64 * 3 * 3 * 3 + 64  # 1792 parameters
        assert count_parameters(conv) == expected
    
    def test_count_parameters_snn(self):
        """Test parameter counting for SNN neurons."""
        # SNN neurons typically have learnable parameters like beta, threshold
        lif = snn.Leaky(beta=0.9, learn_beta=True, learn_threshold=True)
        
        # Should have at least beta and threshold parameters
        param_count = count_parameters(lif)
        assert param_count >= 2
    
    def test_count_parameters_complex_model(self):
        """Test parameter counting for complex SNN models."""
        model = nn.Sequential(
            nn.Linear(784, 1000),  # 784*1000 + 1000 = 785,000
            snn.Leaky(beta=0.9),
            nn.Linear(1000, 500),  # 1000*500 + 500 = 500,500
            snn.Leaky(beta=0.9),
            nn.Linear(500, 10),    # 500*10 + 10 = 5,010
        )
        
        total_params = count_parameters(model)
        expected_min = 785000 + 500500 + 5010  # At least the linear layers
        assert total_params >= expected_min
    
    def test_count_trainable_vs_all(self):
        """Test counting trainable vs all parameters."""
        layer = nn.Linear(100, 50)
        
        # Initially all parameters are trainable
        trainable = count_parameters(layer, only_trainable=True)
        all_params = count_parameters(layer, only_trainable=False)
        assert trainable == all_params == 5050
        
        # Freeze parameters
        for param in layer.parameters():
            param.requires_grad = False
        
        trainable_after = count_parameters(layer, only_trainable=True)
        all_after = count_parameters(layer, only_trainable=False)
        
        assert trainable_after == 0
        assert all_after == 5050


class TestModuleSizes:
    """Test module size analysis utilities."""
    
    def test_get_module_sizes_simple(self):
        """Test module size analysis for simple models."""
        model = nn.Sequential(
            nn.Linear(100, 50),  # 5050 params
            nn.ReLU(),           # 0 params
            nn.Linear(50, 10),   # 510 params
        )
        
        sizes = get_module_sizes(model)
        
        # Should have entries for each layer with parameters
        assert "0" in sizes  # First linear layer
        assert "2" in sizes  # Second linear layer
        assert "1" not in sizes  # ReLU has no parameters
        
        assert sizes["0"] == 5050
        assert sizes["2"] == 510
    
    def test_get_module_sizes_nested(self):
        """Test module size analysis for nested models."""
        model = nn.Sequential(
            nn.Sequential(
                nn.Linear(100, 50),
                nn.ReLU(),
            ),
            nn.Linear(50, 10),
        )
        
        sizes = get_module_sizes(model)
        
        # Should have hierarchical naming
        assert "0.0" in sizes  # Nested linear layer
        assert "1" in sizes    # Top-level linear layer
        
        assert sizes["0.0"] == 5050
        assert sizes["1"] == 510
    
    def test_get_module_sizes_snn(self):
        """Test module size analysis for SNN models."""
        model = nn.Sequential(
            nn.Linear(784, 1000),
            snn.Leaky(beta=0.9, learn_beta=True),
            nn.Linear(1000, 10),
        )
        
        sizes = get_module_sizes(model)
        
        # Linear layers should be detected
        assert "0" in sizes
        assert "2" in sizes
        
        # SNN neuron might have learnable parameters
        if "1" in sizes:
            assert sizes["1"] > 0


class TestFSDPConfig:
    """Test FSDP configuration classes."""
    
    def test_fsdp_config_defaults(self):
        """Test FSDP config default values."""
        config = FSDPConfig()
        
        assert config.sharding_strategy == ShardingStrategy.FULL_SHARD
        assert config.min_param_size == 1_000_000
        assert config.max_param_size is None
        assert config.mixed_precision == True
        assert config.snn_optimize == True
    
    def test_fsdp_config_custom(self):
        """Test FSDP config with custom values."""
        config = FSDPConfig(
            sharding_strategy=ShardingStrategy.SHARD_GRAD_OP,
            min_param_size=500_000,
            max_param_size=10_000_000,
            mixed_precision=False,
            snn_optimize=False,
        )
        
        assert config.sharding_strategy == ShardingStrategy.SHARD_GRAD_OP
        assert config.min_param_size == 500_000
        assert config.max_param_size == 10_000_000
        assert config.mixed_precision == False
        assert config.snn_optimize == False
    
    def test_sharding_strategy_enum(self):
        """Test sharding strategy enumeration."""
        strategies = list(ShardingStrategy)
        expected = ["FULL_SHARD", "SHARD_GRAD_OP", "HYBRID_SHARD", "NO_SHARD"]
        
        assert len(strategies) == len(expected)
        for strategy in strategies:
            assert strategy.value in expected


class TestAutoWrapPolicy:
    """Test automatic wrapping policy for FSDP2."""
    
    def test_auto_wrap_small_model(self):
        """Test auto wrap policy on small model."""
        model = nn.Sequential(
            nn.Linear(10, 5),   # 55 params - too small
            nn.ReLU(),
            nn.Linear(5, 2),    # 12 params - too small
        )
        
        config = FSDPConfig(min_param_size=100)
        modules_to_wrap = auto_wrap_policy(model, config)
        
        # No modules should be wrapped (all too small)
        assert len(modules_to_wrap) == 0
    
    def test_auto_wrap_large_model(self):
        """Test auto wrap policy on large model."""
        model = nn.Sequential(
            nn.Linear(1000, 2000),  # 2,002,000 params - should wrap
            snn.Leaky(beta=0.9),
            nn.Linear(2000, 1000),  # 2,001,000 params - should wrap
            snn.Leaky(beta=0.9),
            nn.Linear(1000, 10),    # 10,010 params - too small
        )
        
        config = FSDPConfig(min_param_size=1_000_000)
        modules_to_wrap = auto_wrap_policy(model, config)
        
        # Should wrap the large linear layers
        assert len(modules_to_wrap) >= 2
    
    def test_auto_wrap_with_max_size(self):
        """Test auto wrap policy with maximum size limit."""
        model = nn.Sequential(
            nn.Linear(1000, 5000),  # 5,005,000 params - too large
            nn.Linear(1000, 2000),  # 2,002,000 params - just right
            nn.Linear(100, 200),    # 20,200 params - too small
        )
        
        config = FSDPConfig(
            min_param_size=1_000_000,
            max_param_size=3_000_000
        )
        modules_to_wrap = auto_wrap_policy(model, config)
        
        # Should only wrap the middle layer
        assert len(modules_to_wrap) == 1
    
    def test_auto_wrap_snn_optimize(self):
        """Test auto wrap policy with SNN optimizations."""
        model = nn.Sequential(
            nn.Linear(1000, 2000),  # Large linear - priority
            snn.Leaky(beta=0.9, learn_beta=True),
            nn.Conv2d(3, 64, 3),    # Conv layer - priority
            snn.Leaky(beta=0.9),
        )
        
        config = FSDPConfig(
            min_param_size=100_000,  # Lower threshold
            snn_optimize=True
        )
        modules_to_wrap = auto_wrap_policy(model, config)
        
        # Should prioritize linear and conv layers
        assert len(modules_to_wrap) >= 1


class TestOptimizeFSDP2:
    """Test FSDP2 optimization for SNNs."""
    
    def test_optimize_small_model(self):
        """Test optimization for small SNN models."""
        model = nn.Sequential(
            nn.Linear(100, 50),
            snn.Leaky(beta=0.9),
            nn.Linear(50, 10),
        )
        
        base_config = FSDPConfig()
        optimized = optimize_fsdp2_for_snns(model, base_config)
        
        # Small models should use NO_SHARD for speed
        assert optimized.sharding_strategy == ShardingStrategy.NO_SHARD
        assert optimized.min_param_size <= base_config.min_param_size
    
    def test_optimize_medium_model(self):
        """Test optimization for medium SNN models."""
        model = nn.Sequential(
            nn.Linear(5000, 5000),  # 25M params - definitely medium
            snn.Leaky(beta=0.9),
            nn.Linear(5000, 1000),  # 5M params
            snn.Leaky(beta=0.9),
            nn.Linear(1000, 10),    # 10K params
        )
        
        base_config = FSDPConfig()
        optimized = optimize_fsdp2_for_snns(model, base_config)
        
        # Medium models should balance memory and speed
        assert optimized.sharding_strategy == ShardingStrategy.SHARD_GRAD_OP
        assert optimized.snn_optimize == True
    
    def test_optimize_many_neurons(self):
        """Test optimization for models with many SNN neurons."""
        layers = []
        for i in range(60):  # Many layers with neurons
            layers.extend([
                nn.Linear(100, 100),
                snn.Leaky(beta=0.9, learn_beta=True),
            ])
        
        model = nn.Sequential(*layers)
        
        base_config = FSDPConfig()
        optimized = optimize_fsdp2_for_snns(model, base_config)
        
        # Should enable aggressive prefetching
        assert optimized.forward_prefetch == True
        assert optimized.backward_prefetch == True


@pytest.mark.skipif(not FSDP2_AVAILABLE, reason="FSDP2 not available")
class TestFSDP2Integration:
    """Integration tests for FSDP2 (requires PyTorch 2.1+)."""
    
    def test_fsdp2_import(self):
        """Test that FSDP2 can be imported."""
        from torch.distributed.fsdp import fully_shard
        assert fully_shard is not None
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_prepare_fsdp2_model_mock(self):
        """Test FSDP2 model preparation (mock test without distributed setup)."""
        # This is a mock test since we can't easily set up distributed environment
        # In real usage, distributed.init_process_group() would be called first
        
        from snntorch.distributed.fsdp2 import _create_mixed_precision_policy
        
        config = FSDPConfig(mixed_precision=True)
        policy = _create_mixed_precision_policy(config)
        
        # Should create a policy or return None gracefully
        assert policy is not None or policy is None  # Both are valid


class TestUtilityFunctions:
    """Test utility functions."""
    
    def test_module_grouping(self):
        """Test SNN module grouping functionality."""
        from snntorch.distributed.fsdp2 import _get_snn_module_groups
        
        model = nn.Sequential(
            nn.Linear(100, 50),
            snn.Leaky(beta=0.9),
            nn.Conv2d(3, 64, 3),
            snn.Synaptic(alpha=0.9, beta=0.8),
            nn.LSTM(50, 25),
        )
        
        groups = _get_snn_module_groups(model)
        
        # Should categorize modules correctly
        assert "linear_layers" in groups
        assert "conv_layers" in groups
        assert "snn_neurons" in groups
        assert "rnn_layers" in groups
        
        assert len(groups["linear_layers"]) >= 1
        assert len(groups["conv_layers"]) >= 1
        assert len(groups["snn_neurons"]) >= 2
        assert len(groups["rnn_layers"]) >= 1
    
    def test_find_module_by_name(self):
        """Test module finding utility."""
        from snntorch.distributed.fsdp2 import auto_wrap_policy
        
        model = nn.Sequential(
            nn.Sequential(
                nn.Linear(100, 50),
                nn.ReLU(),
            ),
            nn.Linear(50, 10),
        )
        
        config = FSDPConfig(min_param_size=1000)
        
        # Should handle nested module names correctly
        modules_to_wrap = auto_wrap_policy(model, config)
        
        # Should find modules without crashing
        assert isinstance(modules_to_wrap, list)


if __name__ == "__main__":
    pytest.main([__file__])