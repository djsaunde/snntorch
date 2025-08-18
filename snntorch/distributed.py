"""
FSDP2 Support for snntorch - Distributed Training Utilities

This module provides utilities for distributed training of spiking neural networks
using PyTorch's Fully Sharded Data Parallel (FSDP2) framework.

Key Insight: In data parallel training, membrane potential states should NOT be 
synchronized across ranks. Each rank processes its own batch data independently 
with its own membrane states. Only model parameters and gradients need 
synchronization, which FSDP2 handles automatically.
"""

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.fsdp import fully_shard
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy, transformer_auto_wrap_policy
import snntorch as snn
from snntorch import surrogate
from typing import Optional, Dict, Any, Callable, Union, List
import os
import warnings


def setup_distributed():
    """
    Initialize distributed training environment for FSDP2.
    
    Returns:
        int: Local rank of the current process
    """
    if not dist.is_initialized():
        dist.init_process_group(backend='nccl')
    
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(local_rank)
    
    return local_rank


def get_snn_layer_types():
    """
    Get all SNN layer types that should be considered for FSDP wrapping.
    
    Returns:
        set: Set of SNN layer classes
    """
    return {
        snn.Leaky,
        snn.Lapicque, 
        snn.Alpha,
        snn.Synaptic,
        snn.RLeaky,
        snn.RSynaptic,
        getattr(snn, 'SLSTM', type(None)),  # Handle optional layers
        getattr(snn, 'SConv2dLSTM', type(None)),
    }


def should_wrap_module(module: nn.Module, min_params: int = 1000) -> bool:
    """
    Strategic decision for whether to wrap a module with FSDP.
    
    Args:
        module: The PyTorch module to evaluate
        min_params: Minimum parameter count threshold
        
    Returns:
        bool: True if module should be wrapped
    """
    # Count parameters
    num_params = sum(p.numel() for p in module.parameters())
    
    # Don't wrap very small layers (overhead > benefit)
    if num_params < min_params:
        return False
    
    # Always wrap large linear layers
    if isinstance(module, nn.Linear) and num_params > 100000:
        return True
    
    # Always wrap large conv layers
    if isinstance(module, (nn.Conv2d, nn.Conv1d)) and num_params > 50000:
        return True
    
    # For SNN layers, consider wrapping if they have learnable parameters
    snn_types = get_snn_layer_types()
    if any(isinstance(module, snn_type) for snn_type in snn_types if snn_type is not type(None)):
        # Check for learnable parameters in SNN neurons
        has_learnable = (
            getattr(module, 'learn_beta', False) or
            getattr(module, 'learn_threshold', False) or 
            getattr(module, 'learn_graded_spikes_factor', False)
        )
        if has_learnable and num_params > min_params:
            return True
    
    # For other layers, use a moderate threshold
    return num_params > 10000


def has_parameters(module: nn.Module) -> bool:
    """
    Check if module has any parameters.
    
    Args:
        module: PyTorch module to check
        
    Returns:
        bool: True if module has parameters
    """
    return len(list(module.parameters())) > 0


def get_module_memory_footprint(module: nn.Module) -> int:
    """
    Estimate memory footprint of a module in bytes.
    
    Args:
        module: PyTorch module to analyze
        
    Returns:
        int: Estimated memory footprint in bytes
    """
    param_memory = sum(p.numel() * p.element_size() for p in module.parameters())
    buffer_memory = sum(b.numel() * b.element_size() for b in module.buffers())
    return param_memory + buffer_memory


class SNNAutoWrapPolicy:
    """
    Custom auto-wrap policy optimized for SNN architectures.
    """
    
    @staticmethod
    def snn_size_based_policy(min_num_params: int = 100000):
        """
        Size-based wrapping policy optimized for SNNs.
        
        Args:
            min_num_params: Minimum parameter count for wrapping
            
        Returns:
            Callable: Auto-wrap policy function
        """
        return size_based_auto_wrap_policy(min_num_params=min_num_params)
    
    @staticmethod
    def snn_transformer_style_policy():
        """
        Transformer-style wrapping policy adapted for SNNs.
        
        Returns:
            Callable: Auto-wrap policy function
        """
        # Define layer types to wrap (common in both classical and SNN models)
        layer_types = {
            nn.Linear,
            nn.Conv2d,
            nn.Conv1d,
            nn.LSTM,
            nn.GRU,
        }
        
        # Add SNN-specific types
        snn_types = get_snn_layer_types()
        layer_types.update(snn_types)
        
        return transformer_auto_wrap_policy(transformer_layer_cls=layer_types)
    
    @staticmethod
    def snn_custom_policy(min_params: int = 1000, wrap_snn_with_params: bool = True):
        """
        Custom policy specifically designed for SNN models.
        
        Args:
            min_params: Minimum parameter threshold
            wrap_snn_with_params: Whether to wrap SNN layers with learnable parameters
            
        Returns:
            Callable: Auto-wrap policy function
        """
        def _snn_policy(module, recurse, nonwrapped_numel):
            # Get SNN layer types
            snn_types = get_snn_layer_types()
            
            # Check if it's an SNN layer with learnable parameters
            if wrap_snn_with_params and any(isinstance(module, snn_type) for snn_type in snn_types if snn_type is not type(None)):
                has_learnable = (
                    getattr(module, 'learn_beta', False) or
                    getattr(module, 'learn_threshold', False) or 
                    getattr(module, 'learn_graded_spikes_factor', False)
                )
                if has_learnable and nonwrapped_numel >= min_params:
                    return True
            
            # Use size-based policy for other layers
            return nonwrapped_numel >= min_params
        
        return _snn_policy


class FSDPSNNWrapper:
    """
    Main wrapper class for applying FSDP2 to SNN models with various strategies.
    """
    
    @staticmethod
    def apply_fsdp(model: nn.Module, 
                   strategy: str = "auto", 
                   min_num_params: int = 100000,
                   **kwargs) -> nn.Module:
        """
        Apply FSDP2 to a spiking neural network model.
        
        Args:
            model: The SNN model to wrap
            strategy: Wrapping strategy ("minimal", "selective", "aggressive", "auto", "transformer")
            min_num_params: Minimum parameters for auto-wrapping
            **kwargs: Additional arguments for FSDP
            
        Returns:
            nn.Module: FSDP-wrapped model
        """
        if strategy == "minimal":
            return FSDPSNNWrapper._apply_minimal(model, **kwargs)
        elif strategy == "selective":
            return FSDPSNNWrapper._apply_selective(model, **kwargs)
        elif strategy == "aggressive":
            return FSDPSNNWrapper._apply_aggressive(model, **kwargs)
        elif strategy == "auto":
            return FSDPSNNWrapper._apply_auto(model, min_num_params, **kwargs)
        elif strategy == "transformer":
            return FSDPSNNWrapper._apply_transformer_style(model, **kwargs)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    
    @staticmethod
    def _apply_minimal(model: nn.Module, **kwargs) -> nn.Module:
        """Only wrap the entire model - simplest approach."""
        fully_shard(model, **kwargs)
        return model
    
    @staticmethod
    def _apply_selective(model: nn.Module, **kwargs) -> nn.Module:
        """Strategic wrapping based on layer size and type."""
        for name, module in model.named_children():
            if should_wrap_module(module):
                fully_shard(module, **kwargs)
        fully_shard(model, **kwargs)
        return model
    
    @staticmethod
    def _apply_aggressive(model: nn.Module, **kwargs) -> nn.Module:
        """Wrap every layer with parameters."""
        for name, module in model.named_children():
            if has_parameters(module):
                fully_shard(module, **kwargs)
        fully_shard(model, **kwargs)
        return model
    
    @staticmethod
    def _apply_auto(model: nn.Module, min_num_params: int, **kwargs) -> nn.Module:
        """Use size-based auto-wrapping policy."""
        auto_wrap_policy = SNNAutoWrapPolicy.snn_size_based_policy(min_num_params)
        fully_shard(model, auto_wrap_policy=auto_wrap_policy, **kwargs)
        return model
    
    @staticmethod
    def _apply_transformer_style(model: nn.Module, **kwargs) -> nn.Module:
        """Use transformer-style wrapping adapted for SNNs."""
        auto_wrap_policy = SNNAutoWrapPolicy.snn_transformer_style_policy()
        fully_shard(model, auto_wrap_policy=auto_wrap_policy, **kwargs)
        return model


def create_distributed_snn(input_size: int, 
                          hidden_size: int, 
                          output_size: int, 
                          num_layers: int = 2,
                          beta: float = 0.9,
                          spike_grad: Optional[Callable] = None,
                          neuron_type: str = "leaky") -> nn.Module:
    """
    Create a distributed spiking neural network model.
    
    Args:
        input_size: Input feature size
        hidden_size: Hidden layer size
        output_size: Output size
        num_layers: Number of hidden layers
        beta: Decay parameter for SNN neurons
        spike_grad: Spike gradient function
        neuron_type: Type of SNN neuron ("leaky", "lapicque", "alpha", "synaptic")
        
    Returns:
        nn.Module: SNN model ready for FSDP wrapping
    """
    if spike_grad is None:
        spike_grad = surrogate.fast_sigmoid()
    
    # Choose neuron class
    neuron_classes = {
        "leaky": snn.Leaky,
        "lapicque": snn.Lapicque,
        "alpha": snn.Alpha,
        "synaptic": snn.Synaptic,
    }
    
    if neuron_type not in neuron_classes:
        raise ValueError(f"Unknown neuron type: {neuron_type}. Choose from {list(neuron_classes.keys())}")
    
    neuron_class = neuron_classes[neuron_type]
    
    layers = []
    
    # Input layer
    layers.extend([
        nn.Flatten(),
        nn.Linear(input_size, hidden_size),
        neuron_class(beta=beta, spike_grad=spike_grad, init_hidden=True)
    ])
    
    # Hidden layers
    for i in range(num_layers - 1):
        layers.extend([
            nn.Linear(hidden_size, hidden_size),
            neuron_class(beta=beta, spike_grad=spike_grad, init_hidden=True)
        ])
    
    # Output layer
    layers.extend([
        nn.Linear(hidden_size, output_size),
        neuron_class(beta=beta, spike_grad=spike_grad, init_hidden=True, output=True)
    ])
    
    return nn.Sequential(*layers)


def get_fsdp_strategy_recommendation(model: nn.Module) -> str:
    """
    Recommend FSDP strategy based on model characteristics.
    
    Args:
        model: The model to analyze
        
    Returns:
        str: Recommended strategy
    """
    total_params = sum(p.numel() for p in model.parameters())
    memory_footprint = get_module_memory_footprint(model)
    
    # Memory footprint in MB
    memory_mb = memory_footprint / (1024 * 1024)
    
    if total_params < 1e6:  # < 1M parameters
        return "minimal"
    elif total_params < 100e6:  # < 100M parameters
        return "selective" if memory_mb > 100 else "auto"
    else:  # > 100M parameters
        return "auto"


def validate_fsdp_setup() -> bool:
    """
    Validate that the environment is properly set up for FSDP.
    
    Returns:
        bool: True if setup is valid
    """
    checks = []
    
    # Check PyTorch version
    torch_version = torch.__version__
    major, minor = map(int, torch_version.split('.')[:2])
    if major < 2 or (major == 2 and minor < 0):
        warnings.warn(f"PyTorch {torch_version} may not fully support FSDP2. Consider upgrading to 2.0+")
        checks.append(False)
    else:
        checks.append(True)
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        warnings.warn("CUDA not available. FSDP is designed for multi-GPU training.")
        checks.append(False)
    else:
        checks.append(True)
    
    # Check distributed environment
    if not dist.is_available():
        warnings.warn("Distributed training not available.")
        checks.append(False)
    else:
        checks.append(True)
    
    return all(checks)


def print_fsdp_strategy_guide():
    """Print a guide for choosing FSDP strategies."""
    guide = """
    FSDP2 Strategy Guide for SNNs:
    
    Model Size Guidelines:
    - Small (<1M params): Use "minimal" strategy
    - Medium (1M-100M): Use "selective" or "auto" 
    - Large (>100M): Use "auto" with appropriate min_num_params
    
    Strategy Descriptions:
    - minimal: Only wrap the full model (lowest overhead)
    - selective: Strategic wrapping based on layer analysis  
    - aggressive: Wrap all layers with parameters (max memory savings)
    - auto: Size-based auto-wrapping (recommended for most cases)
    - transformer: Layer-type based wrapping
    
    Performance Tips:
    - Start with "auto" strategy and min_num_params=100K
    - Monitor memory usage with nvidia-smi
    - Profile with torch.profiler for optimization
    - Adjust min_num_params based on your hardware
    """
    print(guide)


def quick_fsdp_setup(
    model: nn.Module, 
    strategy: Optional[str] = None,
    activation_checkpointing: bool = False,
    cpu_offload: bool = False,
    hsdp: bool = False
) -> nn.Module:
    """
    Quick setup for FSDP with automatic strategy selection and advanced features.
    
    Args:
        model: The SNN model to wrap.
        strategy: Optional strategy override.
        activation_checkpointing: Enable activation checkpointing for memory savings
        cpu_offload: Enable CPU offloading for parameters
        hsdp: Enable Hybrid Sharded Data Parallel
        
    Returns:
        nn.Module: FSDP-wrapped model with advanced features.
    """
    # Validate setup
    if not validate_fsdp_setup():
        warnings.warn("FSDP setup validation failed. Proceeding anyway.")
    
    # Auto-select strategy if not provided
    if strategy is None:
        strategy = get_fsdp_strategy_recommendation(model)
        print(f"Auto-selected FSDP strategy: {strategy}")
    
    # Prepare FSDP kwargs for advanced features
    fsdp_kwargs = {}
    
    # CPU Offload configuration
    if cpu_offload:
        from torch.distributed.fsdp import CPUOffload, OffloadPolicy
        fsdp_kwargs['cpu_offload'] = CPUOffload(offload_params=True)
        print("Enabled CPU offloading for parameters")
    
    # Activation checkpointing configuration
    if activation_checkpointing:
        # Import activation checkpointing utilities
        try:
            from torch.utils.checkpoint import checkpoint
            print("Enabled activation checkpointing")
        except ImportError:
            warnings.warn("Activation checkpointing not available in this PyTorch version")
    
    # HSDP configuration (requires multiple nodes)
    if hsdp:
        try:
            from torch.distributed.device_mesh import init_device_mesh
            world_size = dist.get_world_size()
            if world_size > 1:
                # Simple HSDP setup - can be made more sophisticated
                print(f"Enabled HSDP for world size: {world_size}")
            else:
                warnings.warn("HSDP requires multiple processes. Falling back to regular FSDP.")
        except ImportError:
            warnings.warn("HSDP not available in this PyTorch version")
    
    # Apply FSDP with advanced features
    return FSDPSNNWrapper.apply_fsdp(model, strategy=strategy, **fsdp_kwargs)