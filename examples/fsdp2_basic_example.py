#!/usr/bin/env python

"""
Basic FSDP2 Example with SNNTorch

This example shows how to use FSDP2 utilities to analyze and prepare
SNN models for distributed training, even without a distributed setup.

This is useful for:
- Understanding which modules will be wrapped
- Analyzing model parameter distribution
- Testing FSDP2 configuration before distributed training
- Model optimization and memory planning
"""

import torch.nn as nn
import snntorch as snn
from snntorch.distributed import (
    count_parameters,
    get_module_sizes,
    FSDPConfig,
    auto_wrap_policy,
    optimize_fsdp2_for_snns,
)


def create_example_snn():
    """Create an example SNN model of various sizes."""
    
    # Small model
    small_model = nn.Sequential(
        nn.Linear(784, 256),
        snn.Leaky(beta=0.9, learn_beta=True),
        nn.Linear(256, 128),
        snn.Leaky(beta=0.9, learn_beta=True), 
        nn.Linear(128, 10),
        snn.Leaky(beta=0.9, learn_beta=True),
    )
    
    # Medium model
    medium_model = nn.Sequential(
        nn.Linear(784, 2048),
        snn.Leaky(beta=0.9, learn_beta=True),
        nn.Linear(2048, 2048),
        snn.Leaky(beta=0.9, learn_beta=True),
        nn.Linear(2048, 1024),
        snn.Leaky(beta=0.9, learn_beta=True),
        nn.Linear(1024, 512),
        snn.Leaky(beta=0.9, learn_beta=True),
        nn.Linear(512, 10),
        snn.Leaky(beta=0.9, learn_beta=True),
    )
    
    # Large model with convolutional layers
    large_model = nn.Sequential(
        # Convolutional feature extraction
        nn.Conv2d(1, 64, kernel_size=3, padding=1),
        snn.Leaky(beta=0.9),
        nn.MaxPool2d(2),
        
        nn.Conv2d(64, 128, kernel_size=3, padding=1),
        snn.Leaky(beta=0.9),
        nn.MaxPool2d(2),
        
        nn.Conv2d(128, 256, kernel_size=3, padding=1),
        snn.Leaky(beta=0.9),
        nn.AdaptiveAvgPool2d((4, 4)),
        
        # Flatten and fully connected layers
        nn.Flatten(),
        nn.Linear(256 * 4 * 4, 4096),
        snn.Leaky(beta=0.9, learn_beta=True, learn_threshold=True),
        
        nn.Linear(4096, 4096),
        snn.Leaky(beta=0.9, learn_beta=True, learn_threshold=True),
        
        nn.Linear(4096, 2048),
        snn.Leaky(beta=0.9, learn_beta=True, learn_threshold=True),
        
        nn.Linear(2048, 1024),
        snn.Leaky(beta=0.9, learn_beta=True, learn_threshold=True),
        
        nn.Linear(1024, 10),
        snn.Leaky(beta=0.9, learn_beta=True, learn_threshold=True),
    )
    
    return small_model, medium_model, large_model


def analyze_model(model, model_name):
    """Analyze a model's structure and parameters."""
    print(f"\n{'='*60}")
    print(f"Analyzing {model_name}")
    print(f"{'='*60}")
    
    # Basic parameter statistics
    total_params = count_parameters(model)
    trainable_params = count_parameters(model, only_trainable=True)
    non_trainable_params = count_parameters(model, only_trainable=False) - trainable_params
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {non_trainable_params:,}")
    
    # Module-wise parameter breakdown
    module_sizes = get_module_sizes(model)
    print(f"\nModule-wise parameter breakdown:")
    print(f"{'Module':<20} {'Parameters':<15} {'% of Total':<12}")
    print("-" * 50)
    
    for module_name, param_count in sorted(module_sizes.items(), key=lambda x: x[1], reverse=True):
        percentage = (param_count / total_params) * 100
        print(f"{module_name:<20} {param_count:<15,} {percentage:<12.2f}%")
    
    return total_params, module_sizes


def test_fsdp2_configs(model, model_name, total_params):
    """Test different FSDP2 configurations on the model."""
    print(f"\n{'='*60}")
    print(f"FSDP2 Analysis for {model_name}")
    print(f"{'='*60}")
    
    # Test different parameter size thresholds
    thresholds = [10_000, 100_000, 500_000, 1_000_000, 5_000_000]
    
    for threshold in thresholds:
        if threshold > total_params:
            continue
            
        config = FSDPConfig(
            min_param_size=threshold,
            snn_optimize=True
        )
        
        modules_to_wrap = auto_wrap_policy(model, config)
        
        print(f"\nThreshold: {threshold:,} parameters")
        print(f"Modules to wrap: {len(modules_to_wrap)}")
        
        if modules_to_wrap:
            wrapped_params = sum(count_parameters(module) for module in modules_to_wrap)
            coverage = (wrapped_params / total_params) * 100
            print(f"Parameters in wrapped modules: {wrapped_params:,} ({coverage:.1f}% coverage)")
        
        # Show which modules would be wrapped
        if len(modules_to_wrap) <= 10:  # Only show details for small numbers
            module_sizes = get_module_sizes(model)
            wrapped_module_names = []
            
            for module in modules_to_wrap:
                # Find the module name by comparing objects
                for name, _ in model.named_modules():
                    if model.get_submodule(name) is module:
                        wrapped_module_names.append(name)
                        break
            
            if wrapped_module_names:
                print(f"Wrapped modules: {', '.join(wrapped_module_names)}")


def test_optimization(model, model_name):
    """Test FSDP2 optimization for the model."""
    print(f"\n{'='*60}")
    print(f"FSDP2 Optimization for {model_name}")
    print(f"{'='*60}")
    
    # Start with default config
    base_config = FSDPConfig()
    print(f"Base configuration:")
    print(f"  Sharding strategy: {base_config.sharding_strategy.value}")
    print(f"  Min param size: {base_config.min_param_size:,}")
    print(f"  Mixed precision: {base_config.mixed_precision}")
    print(f"  SNN optimize: {base_config.snn_optimize}")
    
    # Optimize for this specific model
    optimized_config = optimize_fsdp2_for_snns(model, base_config)
    
    print(f"\nOptimized configuration:")
    print(f"  Sharding strategy: {optimized_config.sharding_strategy.value}")
    print(f"  Min param size: {optimized_config.min_param_size:,}")
    print(f"  Mixed precision: {optimized_config.mixed_precision}")
    print(f"  Forward prefetch: {optimized_config.forward_prefetch}")
    print(f"  Backward prefetch: {optimized_config.backward_prefetch}")
    print(f"  Activation checkpointing: {optimized_config.activation_checkpointing}")
    print(f"  CPU offload: {optimized_config.cpu_offload}")
    
    # Test the optimized policy
    modules_to_wrap = auto_wrap_policy(model, optimized_config)
    print(f"\nOptimized wrapping strategy:")
    print(f"  Modules to wrap: {len(modules_to_wrap)}")
    
    if modules_to_wrap:
        wrapped_params = sum(count_parameters(module) for module in modules_to_wrap)
        total_params = count_parameters(model)
        coverage = (wrapped_params / total_params) * 100
        print(f"  Parameter coverage: {coverage:.1f}%")


def demonstrate_usage():
    """Demonstrate various FSDP2 utilities."""
    print("SNNTorch FSDP2 Utilities Demonstration")
    print("=" * 60)
    
    # Create example models
    small_model, medium_model, large_model = create_example_snn()
    models = [
        (small_model, "Small SNN Model"),
        (medium_model, "Medium SNN Model"),
        (large_model, "Large SNN Model"),
    ]
    
    # Analyze each model
    for model, name in models:
        total_params, _ = analyze_model(model, name)
        test_fsdp2_configs(model, name, total_params)
        test_optimization(model, name)
    
    print(f"\n{'='*60}")
    print("Summary and Recommendations")
    print(f"{'='*60}")
    
    print("""
For distributed training with FSDP2:

1. Small models (<10M parameters):
   - Consider using ShardingStrategy.NO_SHARD for speed
   - Set lower min_param_size thresholds (10K-100K)
   - Focus on data parallelism rather than model parallelism

2. Medium models (10M-1B parameters):
   - Use ShardingStrategy.SHARD_GRAD_OP for balanced memory/speed
   - Set moderate min_param_size thresholds (100K-1M)
   - Enable prefetching for better overlap

3. Large models (>1B parameters):
   - Use ShardingStrategy.FULL_SHARD for maximum memory efficiency
   - Set higher min_param_size thresholds (1M+)
   - Enable CPU offloading and activation checkpointing
   - Consider gradient accumulation to maintain effective batch size

SNN-Specific Optimizations:
- Linear and convolutional layers are prioritized for wrapping
- SNN neurons are grouped intelligently to minimize communication
- Mixed precision is configured to work well with spike-based computation
- Temporal dynamics are considered in memory planning
    """)


if __name__ == "__main__":
    demonstrate_usage()