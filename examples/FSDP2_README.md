# FSDP2 Support for snntorch

This directory contains examples demonstrating how to use PyTorch's Fully Sharded Data Parallel (FSDP2) with spiking neural networks built using snntorch.

## Overview

FSDP2 enables memory-efficient distributed training by sharding model parameters, gradients, and optimizer states across multiple GPUs. For spiking neural networks, this is particularly important because:

1. **Memory Efficiency**: SNNs process data across multiple time steps, which can be memory-intensive
2. **Temporal Dynamics**: Each rank processes its own batch data independently with its own membrane states
3. **Parameter Synchronization**: Only model parameters and gradients need synchronization (handled automatically by FSDP2)

## Key Features

- **Automatic Strategy Selection**: Intelligent wrapping policies based on model size and architecture
- **SNN-Specific Optimizations**: Custom policies for different neuron types and learnable parameters
- **Mixed Precision Support**: Reduced memory usage and faster training
- **Comprehensive Logging**: Detailed performance monitoring and checkpointing
- **Easy Integration**: Drop-in replacement for standard PyTorch training

## Quick Start

### 1. Simple Example

The simplest way to get started:

```python
import snntorch as snn
from snntorch.distributed import quick_fsdp_setup

# Create your SNN model
model = create_your_snn_model()

# Apply FSDP2 with automatic configuration
model = quick_fsdp_setup(model)

# Train as usual!
```

### 2. Run the Examples

```bash
# Simple example (2 GPUs)
torchrun --nproc_per_node=2 simple_fsdp2_example.py

# Full training example with synthetic data
torchrun --nproc_per_node=2 fsdp2_snn_training.py

# Training with MNIST dataset
torchrun --nproc_per_node=2 fsdp2_snn_training.py --dataset mnist

# Multi-node training (4 GPUs per node, 2 nodes)
torchrun --nproc_per_node=4 --nnodes=2 --rdzv_endpoint=master_ip:29500 fsdp2_snn_training.py
```

## Examples

### simple_fsdp2_example.py

A minimal example showing:
- Basic FSDP2 setup
- Simple SNN model creation
- Distributed training loop
- Synthetic spike data generation

Perfect for understanding the core concepts.

### fsdp2_snn_training.py

A comprehensive example featuring:
- Command-line argument parsing
- Multiple FSDP strategies
- Mixed precision training
- Checkpointing and resuming
- Support for synthetic and MNIST data
- Performance monitoring

Use this as a template for your own projects.

## FSDP Strategies

### Available Strategies

1. **minimal**: Only wrap the entire model
   - Lowest communication overhead
   - Minimal memory savings
   - Best for small models

2. **selective**: Strategic wrapping based on layer analysis
   - Balanced approach
   - Good for medium-sized models
   - Considers layer types and parameter counts

3. **aggressive**: Wrap every layer with parameters
   - Maximum memory savings
   - Higher communication overhead
   - Good for memory-constrained scenarios

4. **auto**: Size-based automatic wrapping (recommended)
   - Intelligent parameter threshold
   - Adapts to model characteristics
   - Good default choice

5. **transformer**: Layer-type based wrapping
   - Optimized for transformer-like architectures
   - Can be adapted for specific SNN architectures

### Strategy Selection Guide

```python
# For small models (<1M parameters)
model = FSDPSNNWrapper.apply_fsdp(model, strategy="minimal")

# For medium models (1M-100M parameters) - recommended
model = FSDPSNNWrapper.apply_fsdp(model, strategy="auto", min_num_params=100000)

# For large models (>100M parameters)
model = FSDPSNNWrapper.apply_fsdp(model, strategy="auto", min_num_params=1000000)

# Custom strategy
model = FSDPSNNWrapper.apply_fsdp(model, strategy="selective")
```

## SNN-Specific Considerations

### Membrane State Handling

**Important**: Membrane potential states should NOT be synchronized across ranks. Each rank processes its own batch data independently:

```python
# ✅ Correct: Each rank has its own membrane states
for step in range(num_steps):
    spk, mem = model(data[step])  # Independent per rank
    
# ❌ Incorrect: Don't sync membrane states
# torch.distributed.all_reduce(mem)  # Never do this!
```

### Neuron Parameter Wrapping

The FSDP wrapper automatically detects learnable SNN parameters:

```python
# These will be considered for wrapping if learn_* = True
snn.Leaky(beta=0.9, learn_beta=True)      # Learnable decay
snn.Leaky(threshold=1.0, learn_threshold=True)  # Learnable threshold
```

### Time Step Processing

SNNs process data across multiple time steps. The trainer handles this efficiently:

```python
trainer = DistributedSNNTrainer(
    model=model,
    device=device,
    num_steps=25,  # Number of simulation time steps
    mixed_precision=True  # Optional: use mixed precision
)
```

## Performance Tips

### Memory Optimization

1. **Use Mixed Precision**: Reduces memory usage by ~50%
   ```python
   trainer = DistributedSNNTrainer(mixed_precision=True)
   ```

2. **Gradient Accumulation**: Simulate larger batch sizes
   ```python
   trainer = DistributedSNNTrainer(grad_accumulation_steps=4)
   ```

3. **Appropriate Wrapping**: Don't over-wrap small layers
   ```python
   # Good: Size-based wrapping
   model = FSDPSNNWrapper.apply_fsdp(model, strategy="auto", min_num_params=100000)
   ```

### Communication Optimization

1. **Minimize Small Layer Wrapping**: Reduces communication overhead
2. **Use Appropriate Batch Sizes**: Balance memory and communication
3. **Monitor Network Utilization**: Use tools like `nvidia-smi` and `torch.profiler`

### Monitoring Performance

```python
# The trainer provides detailed timing information
train_stats = trainer.train_epoch(...)
print(f"Forward time: {train_stats['avg_forward_time']:.4f}s")
print(f"Backward time: {train_stats['avg_backward_time']:.4f}s")
print(f"Data loading time: {train_stats['avg_data_time']:.4f}s")
```

## Requirements

- PyTorch 2.0+ (for full FSDP2 support)
- CUDA-capable GPUs
- snntorch
- torchvision (for MNIST example)

## Troubleshooting

### Common Issues

1. **"No module named torch.distributed.fsdp"**
   - Update PyTorch to 2.0+
   - Ensure PyTorch was compiled with distributed support

2. **Memory errors during training**
   - Reduce batch size
   - Enable mixed precision
   - Increase gradient accumulation steps
   - Use more aggressive wrapping strategy

3. **Slow training**
   - Check if wrapping is too aggressive (many small layers)
   - Monitor GPU utilization
   - Verify network bandwidth for multi-node training

4. **Inconsistent results**
   - Ensure proper random seeding
   - Check that membrane states aren't being synchronized
   - Verify data loader sampling is set correctly

### Debugging

Enable detailed logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Or use the strategy guide
from snntorch.distributed import print_fsdp_strategy_guide
print_fsdp_strategy_guide()
```

## Advanced Usage

### Custom Wrapping Policies

```python
from snntorch.distributed import SNNAutoWrapPolicy

# Custom policy for your specific architecture
def my_custom_policy(module, recurse, nonwrapped_numel):
    # Your custom logic here
    return should_wrap_decision

model = FSDPSNNWrapper.apply_fsdp(model, auto_wrap_policy=my_custom_policy)
```

### Integration with Existing Code

FSDP2 support is designed to be minimally invasive:

```python
# Before: Standard training
model = MySnNetwork()
optimizer = torch.optim.Adam(model.parameters())

# After: Add FSDP2 support
from snntorch.distributed import quick_fsdp_setup
model = quick_fsdp_setup(model)  # Only addition needed!
optimizer = torch.optim.Adam(model.parameters())
```

## Contributing

If you encounter issues or have suggestions for improving FSDP2 support in snntorch, please:

1. Check existing issues
2. Create a minimal reproduction case
3. Submit an issue or pull request

## References

- [PyTorch FSDP Documentation](https://pytorch.org/docs/stable/fsdp.html)
- [snntorch Documentation](https://snntorch.readthedocs.io/)
- [Distributed Training Best Practices](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)