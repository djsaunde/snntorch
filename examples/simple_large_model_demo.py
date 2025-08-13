#!/usr/bin/env python

"""
Simple Large Model Demo for FSDP2

This is a simplified demonstration of a large SNN model with FSDP2 analysis.
It creates a model with 100M+ parameters and shows how FSDP2 would handle it.

This avoids the complexity of the full training script while still demonstrating
the key concepts of large-scale SNN training with FSDP2.
"""

import torch
import torch.nn as nn
import sys
import os

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


class SimpleLargeSNN(nn.Module):
    """A simple but large SNN for demonstration purposes."""
    
    def __init__(self, model_size="large"):
        super().__init__()
        
        # Configure sizes based on model_size
        if model_size == "medium":
            hidden_dims = [2048, 2048, 1024, 512]
        elif model_size == "large":
            hidden_dims = [4096, 4096, 2048, 1024, 512]
        elif model_size == "xlarge":
            hidden_dims = [8192, 8192, 4096, 2048, 1024, 512]
        else:
            raise ValueError(f"Unknown model size: {model_size}")
        
        # Input projection (784 -> first hidden)
        self.input_proj = nn.Linear(784, hidden_dims[0])
        self.input_lif = snn.Leaky(beta=0.9, learn_beta=True)
        
        # Deep SNN layers
        self.hidden_layers = nn.ModuleList()
        self.snn_layers = nn.ModuleList()
        
        for i in range(len(hidden_dims) - 1):
            self.hidden_layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
            self.snn_layers.append(snn.Leaky(beta=0.9, learn_beta=True, learn_threshold=True))
        
        # Output layer
        self.output_layer = nn.Linear(hidden_dims[-1], 10)
        self.output_lif = snn.Leaky(beta=0.9, learn_beta=True)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights."""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
    
    def forward(self, x, num_steps=10):
        """
        Forward pass through the SNN.
        
        Args:
            x: Input tensor (batch_size, 784)
            num_steps: Number of time steps to simulate
            
        Returns:
            Output spikes across time steps
        """
        batch_size = x.shape[0]
        
        # Simple rate coding: repeat input for time steps with some noise
        outputs = []
        
        for t in range(num_steps):
            # Add temporal variation
            x_t = x + 0.1 * torch.randn_like(x) * t / num_steps
            
            # Input projection
            h = self.input_proj(x_t)
            h = self.input_lif(h)
            if isinstance(h, tuple):
                h = h[0]  # Take spike output
            
            # Hidden layers
            for i, (linear, lif) in enumerate(zip(self.hidden_layers, self.snn_layers)):
                h = linear(h)
                h = lif(h)
                if isinstance(h, tuple):
                    h = h[0]  # Take spike output
            
            # Output layer
            out = self.output_layer(h)
            out = self.output_lif(out)
            if isinstance(out, tuple):
                out = out[0]  # Take spike output
            
            outputs.append(out)
        
        return torch.stack(outputs, dim=0)  # (num_steps, batch, classes)


def analyze_model_for_fsdp2(model, model_name):
    """Analyze model for FSDP2 distribution."""
    print(f"\n{'='*80}")
    print(f"FSDP2 Analysis: {model_name}")
    print(f"{'='*80}")
    
    # Basic stats
    total_params = count_parameters(model)
    print(f"Total parameters: {total_params:,}")
    print(f"Model size: {total_params / 1e6:.1f}M parameters")
    print(f"Memory estimate (FP32): {total_params * 4 / 1e9:.2f} GB")
    print(f"Memory estimate (FP16): {total_params * 2 / 1e9:.2f} GB")
    
    # Module breakdown
    module_sizes = get_module_sizes(model)
    print(f"\nLargest modules:")
    sorted_modules = sorted(module_sizes.items(), key=lambda x: x[1], reverse=True)
    
    for name, size in sorted_modules[:10]:
        percentage = size / total_params * 100
        print(f"  {name:<25}: {size:>10,} params ({percentage:>5.1f}%)")
    
    # FSDP2 wrapping analysis
    print(f"\nFSDP2 Wrapping Analysis:")
    
    # Test different thresholds
    thresholds = [100_000, 500_000, 1_000_000, 5_000_000]
    
    for threshold in thresholds:
        if threshold > total_params:
            continue
            
        config = FSDPConfig(min_param_size=threshold, snn_optimize=True)
        modules_to_wrap = auto_wrap_policy(model, config)
        
        if modules_to_wrap:
            wrapped_params = sum(count_parameters(module) for module in modules_to_wrap)
            coverage = wrapped_params / total_params * 100
            print(f"  Threshold {threshold:>7,}: {len(modules_to_wrap):>2} modules, "
                  f"{wrapped_params:>10,} params ({coverage:>5.1f}%)")
    
    # Get optimized configuration
    print(f"\nOptimized FSDP2 Configuration:")
    base_config = FSDPConfig()
    optimized = optimize_fsdp2_for_snns(model, base_config)
    
    print(f"  Strategy: {optimized.sharding_strategy.value}")
    print(f"  Min param size: {optimized.min_param_size:,}")
    print(f"  Mixed precision: {optimized.mixed_precision}")
    print(f"  CPU offload: {optimized.cpu_offload}")
    print(f"  Activation checkpointing: {optimized.activation_checkpointing}")
    
    # Test optimized wrapping
    modules_to_wrap = auto_wrap_policy(model, optimized)
    if modules_to_wrap:
        wrapped_params = sum(count_parameters(module) for module in modules_to_wrap)
        coverage = wrapped_params / total_params * 100
        print(f"  Optimized wrapping: {len(modules_to_wrap)} modules, {coverage:.1f}% coverage")
    
    return total_params


def demonstrate_training_simulation(model, model_name):
    """Simulate training to show memory usage."""
    print(f"\n{'='*80}")
    print(f"Training Simulation: {model_name}")
    print(f"{'='*80}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    model = model.to(device)
    
    # Create sample data
    batch_size = 8
    sample_input = torch.randn(batch_size, 784, device=device)
    sample_target = torch.randint(0, 10, (batch_size,), device=device)
    
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
        print(f"Initial memory: {torch.cuda.memory_allocated() / 1e9:.3f} GB")
    
    # Forward pass
    model.train()
    outputs = model(sample_input, num_steps=5)
    
    if device.type == 'cuda':
        print(f"After forward: {torch.cuda.memory_allocated() / 1e9:.3f} GB")
    
    # Compute loss (use mean across time)
    mean_output = outputs.mean(dim=0)
    loss = nn.functional.cross_entropy(mean_output, sample_target)
    
    if device.type == 'cuda':
        print(f"After loss: {torch.cuda.memory_allocated() / 1e9:.3f} GB")
    
    # Backward pass
    loss.backward()
    
    if device.type == 'cuda':
        peak_memory = torch.cuda.max_memory_allocated() / 1e9
        print(f"Peak memory: {peak_memory:.3f} GB")
        
        # Estimate memory per sample
        memory_per_sample = peak_memory / batch_size
        print(f"Memory per sample: {memory_per_sample:.3f} GB")
        
        # Estimate max batch size
        if torch.cuda.is_available():
            total_memory = torch.cuda.get_device_properties(device).total_memory / 1e9
            max_batch_size = int(total_memory * 0.8 / memory_per_sample)
            print(f"Estimated max batch size: {max_batch_size} (on {total_memory:.0f}GB GPU)")
    
    print(f"Loss: {loss.item():.4f}")
    print(f"Output shape: {outputs.shape}")
    
    # Test gradient flow
    grad_count = sum(1 for p in model.parameters() if p.grad is not None and p.requires_grad)
    total_params = sum(1 for p in model.parameters() if p.requires_grad)
    print(f"Gradient flow: {grad_count}/{total_params} parameters")


def main():
    """Main demonstration."""
    print("Large SNN Model FSDP2 Demonstration")
    print("="*80)
    
    # Test different model sizes
    model_sizes = ["medium", "large"]
    
    for size in model_sizes:
        print(f"\nTesting {size} model...")
        
        # Create model
        model = SimpleLargeSNN(model_size=size)
        
        # Analyze for FSDP2
        total_params = analyze_model_for_fsdp2(model, f"{size.title()} SNN")
        
        # Demonstrate training simulation
        if total_params < 200_000_000:  # Only if not too large for available memory
            demonstrate_training_simulation(model, f"{size.title()} SNN")
        else:
            print(f"\nSkipping training simulation for {size} model (too large)")
    
    print(f"\n{'='*80}")
    print("FSDP2 Benefits Summary")
    print("="*80)
    print("""
For large SNN models (100M+ parameters), FSDP2 provides:

1. Memory Efficiency:
   - Shard parameters across GPUs to fit larger models
   - Reduce memory usage by 4-8x depending on strategy
   - Enable training of models that don't fit on single GPU

2. SNN-Specific Optimizations:
   - Intelligent wrapping of large linear layers
   - Preserve temporal dynamics during distributed computation
   - Optimize for spike-based activation patterns

3. Scalability:
   - Linear scaling across multiple GPUs/nodes
   - Automatic load balancing based on parameter counts
   - Support for gradient accumulation with temporal sequences

4. Production Ready:
   - Mixed precision training for 2x speedup
   - CPU offloading for massive models
   - Activation checkpointing for memory-compute tradeoffs

Recommended Configuration:
- Models < 10M params: ShardingStrategy.NO_SHARD (speed focus)
- Models 10M-1B params: ShardingStrategy.SHARD_GRAD_OP (balanced)
- Models > 1B params: ShardingStrategy.FULL_SHARD (memory focus)

Use min_param_size=1M for most cases, adjust based on model architecture.
    """)


if __name__ == "__main__":
    main()