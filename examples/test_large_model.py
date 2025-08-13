#!/usr/bin/env python

"""
Test Large SNN Model - Memory and Performance Analysis

This script tests the large SNN model from fsdp2_large_model_training.py
without requiring distributed setup. It's useful for:

- Analyzing model parameter distribution
- Testing FSDP2 wrapping strategies  
- Memory profiling and optimization
- Performance benchmarking
- Understanding model architecture

Usage:
    python test_large_model.py --model-size large
    python test_large_model.py --model-size xlarge --profile-memory
    python test_large_model.py --analyze-fsdp2
"""

import argparse
import time
import sys
import os

# Add the parent directory to path to import the large model
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
from fsdp2_large_model_training import LargeScaleSNN

# Add snntorch to path
sys.path.insert(0, '/workspace/snntorch')
import snntorch as snn
from snntorch.distributed import (
    count_parameters,
    get_module_sizes,
    FSDPConfig,
    ShardingStrategy,
    auto_wrap_policy,
    optimize_fsdp2_for_snns,
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Test Large SNN Model")
    
    parser.add_argument("--model-size", type=str, default="large",
                       choices=["medium", "large", "xlarge"],
                       help="Model size to test")
    parser.add_argument("--batch-size", type=int, default=4,
                       help="Batch size for testing")
    parser.add_argument("--num-steps", type=int, default=10,
                       help="Number of time steps")
    parser.add_argument("--analyze-fsdp2", action="store_true",
                       help="Analyze FSDP2 wrapping strategies")
    parser.add_argument("--profile-memory", action="store_true",
                       help="Profile memory usage")
    parser.add_argument("--benchmark-speed", action="store_true",
                       help="Benchmark forward/backward pass speed")
    parser.add_argument("--device", type=str, default="auto",
                       choices=["auto", "cpu", "cuda"],
                       help="Device to use")
    
    return parser.parse_args()


def get_device(device_arg):
    """Get the appropriate device."""
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif device_arg == "cuda":
        if not torch.cuda.is_available():
            print("Warning: CUDA not available, falling back to CPU")
            return torch.device("cpu")
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def analyze_model_structure(model, model_name):
    """Analyze the structure and parameters of the model."""
    print(f"\n{'='*80}")
    print(f"Model Analysis: {model_name}")
    print(f"{'='*80}")
    
    # Basic statistics
    total_params = count_parameters(model)
    trainable_params = count_parameters(model, only_trainable=True)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model size: {total_params / 1e6:.1f}M parameters")
    print(f"Estimated memory (FP32): {total_params * 4 / 1e9:.2f} GB")
    print(f"Estimated memory (FP16): {total_params * 2 / 1e9:.2f} GB")
    
    # Module breakdown
    module_sizes = get_module_sizes(model)
    print(f"\nTop 15 Largest Modules:")
    print(f"{'Module':<40} {'Parameters':<15} {'% of Total':<12} {'Size (MB)':<10}")
    print("-" * 80)
    
    sorted_modules = sorted(module_sizes.items(), key=lambda x: x[1], reverse=True)
    for i, (module_name, param_count) in enumerate(sorted_modules[:15]):
        percentage = (param_count / total_params) * 100
        size_mb = param_count * 4 / 1e6  # Assuming FP32
        print(f"{module_name:<40} {param_count:<15,} {percentage:<12.2f}% {size_mb:<10.1f}")
    
    # Architecture breakdown by component
    print(f"\nComponent Analysis:")
    component_stats = {}
    
    for module_name, param_count in module_sizes.items():
        if 'feature_extractor' in module_name:
            component_stats['CNN Feature Extractor'] = component_stats.get('CNN Feature Extractor', 0) + param_count
        elif 'classifier' in module_name:
            component_stats['SNN Classifier'] = component_stats.get('SNN Classifier', 0) + param_count
        elif 'spike_encoder' in module_name:
            component_stats['Spike Encoder'] = component_stats.get('Spike Encoder', 0) + param_count
        else:
            component_stats['Other'] = component_stats.get('Other', 0) + param_count
    
    for component, params in component_stats.items():
        percentage = (params / total_params) * 100
        print(f"  {component}: {params:,} parameters ({percentage:.1f}%)")
    
    return total_params, module_sizes


def analyze_fsdp2_strategies(model, model_name):
    """Analyze different FSDP2 strategies for the model."""
    print(f"\n{'='*80}")
    print(f"FSDP2 Strategy Analysis: {model_name}")
    print(f"{'='*80}")
    
    total_params = count_parameters(model)
    
    # Test different wrapping thresholds
    thresholds = [100_000, 500_000, 1_000_000, 5_000_000, 10_000_000]
    
    print(f"Model has {total_params:,} total parameters\n")
    
    for threshold in thresholds:
        if threshold > total_params:
            continue
        
        config = FSDPConfig(
            min_param_size=threshold,
            snn_optimize=True
        )
        
        modules_to_wrap = auto_wrap_policy(model, config)
        
        if modules_to_wrap:
            wrapped_params = sum(count_parameters(module) for module in modules_to_wrap)
            coverage = (wrapped_params / total_params) * 100
            print(f"Threshold: {threshold:,} parameters")
            print(f"  Modules to wrap: {len(modules_to_wrap)}")
            print(f"  Parameters wrapped: {wrapped_params:,} ({coverage:.1f}%)")
            print(f"  Unwrapped parameters: {total_params - wrapped_params:,}")
            print()
    
    # Test optimization strategies
    print("Optimization Strategy Analysis:")
    base_config = FSDPConfig()
    optimized_config = optimize_fsdp2_for_snns(model, base_config)
    
    print(f"  Recommended strategy: {optimized_config.sharding_strategy.value}")
    print(f"  Recommended min_param_size: {optimized_config.min_param_size:,}")
    print(f"  Mixed precision: {optimized_config.mixed_precision}")
    print(f"  Forward prefetch: {optimized_config.forward_prefetch}")
    print(f"  Backward prefetch: {optimized_config.backward_prefetch}")
    print(f"  CPU offload: {optimized_config.cpu_offload}")
    print(f"  Activation checkpointing: {optimized_config.activation_checkpointing}")
    
    # Test the optimized strategy
    modules_to_wrap = auto_wrap_policy(model, optimized_config)
    if modules_to_wrap:
        wrapped_params = sum(count_parameters(module) for module in modules_to_wrap)
        coverage = (wrapped_params / total_params) * 100
        print(f"\nOptimized Strategy Results:")
        print(f"  Modules to wrap: {len(modules_to_wrap)}")
        print(f"  Parameter coverage: {coverage:.1f}%")


def profile_memory_usage(model, batch_size, num_steps, device):
    """Profile memory usage during forward and backward passes."""
    print(f"\n{'='*80}")
    print(f"Memory Profiling (Batch size: {batch_size}, Steps: {num_steps})")
    print(f"{'='*80}")
    
    if device.type == 'cpu':
        print("Memory profiling only available on CUDA devices")
        return
    
    model = model.to(device)
    model.train()
    
    # Create sample input
    sample_input = torch.randn(batch_size, 1, 28, 28, device=device)
    sample_target = torch.randint(0, 10, (batch_size,), device=device)
    
    # Clear cache and reset peak memory
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    
    def print_memory_stats(stage):
        allocated = torch.cuda.memory_allocated(device) / 1e9
        reserved = torch.cuda.memory_reserved(device) / 1e9
        max_allocated = torch.cuda.max_memory_allocated(device) / 1e9
        print(f"{stage:<20}: Allocated: {allocated:>6.2f}GB, "
              f"Reserved: {reserved:>6.2f}GB, Max: {max_allocated:>6.2f}GB")
    
    print_memory_stats("Initial")
    
    # Forward pass
    torch.cuda.synchronize()
    outputs = model(sample_input)
    torch.cuda.synchronize()
    print_memory_stats("After Forward")
    
    # Compute loss
    mean_outputs = outputs.mean(dim=0)
    loss = nn.functional.cross_entropy(mean_outputs, sample_target)
    print_memory_stats("After Loss")
    
    # Backward pass
    loss.backward()
    torch.cuda.synchronize()
    print_memory_stats("After Backward")
    
    # Calculate approximate memory per sample
    memory_per_sample = torch.cuda.max_memory_allocated(device) / batch_size / 1e9
    print(f"\nEstimated memory per sample: {memory_per_sample:.3f} GB")
    
    # Estimate maximum batch size (assuming 80% memory utilization)
    if torch.cuda.is_available():
        total_memory = torch.cuda.get_device_properties(device).total_memory / 1e9
        max_batch_size = int(total_memory * 0.8 / memory_per_sample)
        print(f"Estimated max batch size: {max_batch_size} (on {total_memory:.0f}GB GPU)")


def benchmark_speed(model, batch_size, num_steps, device, num_iterations=10):
    """Benchmark forward and backward pass speed."""
    print(f"\n{'='*80}")
    print(f"Speed Benchmark (Batch size: {batch_size}, Steps: {num_steps})")
    print(f"{'='*80}")
    
    model = model.to(device)
    model.train()
    
    # Create sample data
    sample_input = torch.randn(batch_size, 1, 28, 28, device=device)
    sample_target = torch.randint(0, 10, (batch_size,), device=device)
    
    # Warmup
    for _ in range(3):
        outputs = model(sample_input)
        mean_outputs = outputs.mean(dim=0)
        loss = nn.functional.cross_entropy(mean_outputs, sample_target)
        loss.backward()
        model.zero_grad()
    
    # Synchronize
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    # Benchmark forward pass
    start_time = time.time()
    for _ in range(num_iterations):
        outputs = model(sample_input)
        if device.type == 'cuda':
            torch.cuda.synchronize()
    forward_time = (time.time() - start_time) / num_iterations
    
    # Benchmark full forward + backward
    start_time = time.time()
    for _ in range(num_iterations):
        outputs = model(sample_input)
        mean_outputs = outputs.mean(dim=0)
        loss = nn.functional.cross_entropy(mean_outputs, sample_target)
        loss.backward()
        model.zero_grad()
        if device.type == 'cuda':
            torch.cuda.synchronize()
    full_time = (time.time() - start_time) / num_iterations
    
    backward_time = full_time - forward_time
    
    print(f"Forward pass:  {forward_time*1000:>7.2f} ms")
    print(f"Backward pass: {backward_time*1000:>7.2f} ms")
    print(f"Full pass:     {full_time*1000:>7.2f} ms")
    print(f"Throughput:    {batch_size/full_time:>7.1f} samples/sec")
    
    # Calculate operations per second (rough estimate)
    total_params = count_parameters(model)
    ops_per_sample = total_params * num_steps * 2  # Forward + backward
    total_ops = ops_per_sample * batch_size / full_time / 1e9  # GOPS
    print(f"Compute:       {total_ops:>7.1f} GOPS")


def test_model_functionality(model, batch_size, num_steps, device):
    """Test basic model functionality."""
    print(f"\n{'='*80}")
    print(f"Functionality Test")
    print(f"{'='*80}")
    
    model = model.to(device)
    
    # Test forward pass
    sample_input = torch.randn(batch_size, 1, 28, 28, device=device)
    
    print(f"Input shape: {sample_input.shape}")
    
    model.eval()
    with torch.no_grad():
        outputs = model(sample_input)
    
    print(f"Output shape: {outputs.shape}")
    print(f"Output range: [{outputs.min():.3f}, {outputs.max():.3f}]")
    print(f"Output mean: {outputs.mean():.3f}")
    print(f"Output std: {outputs.std():.3f}")
    
    # Test gradient flow
    model.train()
    sample_target = torch.randint(0, 10, (batch_size,), device=device)
    
    outputs = model(sample_input)
    mean_outputs = outputs.mean(dim=0)
    loss = nn.functional.cross_entropy(mean_outputs, sample_target)
    loss.backward()
    
    # Check gradients
    has_gradients = 0
    total_params = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            total_params += 1
            if param.grad is not None:
                has_gradients += 1
    
    print(f"Gradient flow: {has_gradients}/{total_params} parameters have gradients")
    print(f"Loss: {loss.item():.4f}")
    
    print("✓ Model functionality test passed!")


def main():
    """Main test function."""
    args = parse_args()
    
    print("Large-Scale SNN Model Testing")
    print("="*80)
    
    # Setup device
    device = get_device(args.device)
    print(f"Using device: {device}")
    
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(device).total_memory / 1e9:.1f} GB")
    
    # Create model
    print(f"\nCreating {args.model_size} model...")
    model = LargeScaleSNN(
        input_channels=1,
        num_classes=10,
        model_size=args.model_size,
        spike_encoding="rate",
        num_steps=args.num_steps,
        beta=0.9,
    )
    
    # Analyze model structure
    total_params, module_sizes = analyze_model_structure(model, f"{args.model_size.title()} SNN")
    
    # Test functionality
    test_model_functionality(model, args.batch_size, args.num_steps, device)
    
    # FSDP2 analysis
    if args.analyze_fsdp2:
        analyze_fsdp2_strategies(model, f"{args.model_size.title()} SNN")
    
    # Memory profiling
    if args.profile_memory and device.type == 'cuda':
        profile_memory_usage(model, args.batch_size, args.num_steps, device)
    
    # Speed benchmarking
    if args.benchmark_speed:
        benchmark_speed(model, args.batch_size, args.num_steps, device)
    
    print(f"\n{'='*80}")
    print("Testing completed!")
    print(f"Model: {args.model_size.title()} SNN ({total_params/1e6:.1f}M parameters)")
    print(f"Device: {device}")
    print("="*80)


if __name__ == "__main__":
    main()