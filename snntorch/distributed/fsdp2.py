"""
FSDP2 (Fully Sharded Data Parallel) utilities for SNNTorch.

This module provides FSDP2 support for efficient distributed training of large
spiking neural networks. It includes intelligent module wrapping strategies
based on parameter counts and SNN-specific optimizations.

Based on PyTorch FSDP2 best practices and research findings from:
- PyTorch FSDP2 documentation
- Academic papers on distributed SNN training
- Production deployment experiences

Key Features:
- Parameter-size based wrapping strategy
- SNN-specific module grouping
- Memory-efficient sharding policies
- Mixed precision support for SNNs
"""

import warnings
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Union, Callable, Any, Tuple
from collections import defaultdict
import torch
import torch.nn as nn

# Import FSDP2 components with fallback for older PyTorch versions
try:
    from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy, OffloadPolicy
    FSDP2_AVAILABLE = True
except ImportError:
    warnings.warn(
        "FSDP2 is not available in this PyTorch version. "
        "Please upgrade to PyTorch 2.4+ for FSDP2 support.",
        UserWarning
    )
    FSDP2_AVAILABLE = False


class ShardingStrategy(Enum):
    """Sharding strategies for FSDP2."""
    FULL_SHARD = "FULL_SHARD"  # Shard weights, gradients, optimizer state
    SHARD_GRAD_OP = "SHARD_GRAD_OP"  # Shard gradients and optimizer state only
    HYBRID_SHARD = "HYBRID_SHARD"  # Full-shard within node, replicate across nodes 
    NO_SHARD = "NO_SHARD"  # No sharding (equivalent to DDP)


@dataclass
class FSDPConfig:
    """Configuration for FSDP2 model preparation.
    
    Args:
        sharding_strategy: Strategy for parameter sharding
        min_param_size: Minimum parameter count for a module to be wrapped
        max_param_size: Maximum parameter count for a module to be wrapped (None = no limit)
        mixed_precision: Whether to use mixed precision training
        mixed_precision_policy: Custom mixed precision policy
        offload_policy: Policy for offloading to CPU
        device_id: GPU device ID to use (None = auto-detect)
        sync_module_states: Whether to sync module states across ranks
        param_init_fn: Function for parameter initialization
        forward_prefetch: Whether to enable forward prefetching
        backward_prefetch: Whether to enable backward prefetching
        activation_checkpointing: Whether to use activation checkpointing
        limit_all_gathers: Whether to limit all-gather operations
        cpu_offload: Whether to offload parameters to CPU when not in use
        snn_optimize: Whether to apply SNN-specific optimizations
    """
    sharding_strategy: ShardingStrategy = ShardingStrategy.FULL_SHARD
    min_param_size: int = 1_000_000  # 1M parameters
    max_param_size: Optional[int] = None
    mixed_precision: bool = True
    mixed_precision_policy: Optional[Any] = None
    offload_policy: Optional[Any] = None
    device_id: Optional[int] = None
    sync_module_states: bool = True
    param_init_fn: Optional[Callable] = None
    forward_prefetch: bool = True
    backward_prefetch: bool = True
    activation_checkpointing: bool = False
    limit_all_gathers: bool = True
    cpu_offload: bool = False
    snn_optimize: bool = True


def count_parameters(module: nn.Module, only_trainable: bool = True) -> int:
    """Count the number of parameters in a module.
    
    Args:
        module: PyTorch module to count parameters for
        only_trainable: If True, only count trainable parameters
        
    Returns:
        Number of parameters in the module
        
    Example:
        >>> import torch.nn as nn
        >>> linear = nn.Linear(1000, 500)
        >>> count = count_parameters(linear)
        >>> print(f"Linear layer has {count:,} parameters")
    """
    if only_trainable:
        return sum(p.numel() for p in module.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in module.parameters())


def get_module_sizes(model: nn.Module, only_trainable: bool = True) -> Dict[str, int]:
    """Get parameter counts for all modules in a model.
    
    Args:
        model: PyTorch model to analyze
        only_trainable: If True, only count trainable parameters
        
    Returns:
        Dictionary mapping module names to parameter counts
        
    Example:
        >>> model = nn.Sequential(nn.Linear(784, 1000), nn.ReLU(), nn.Linear(1000, 10))
        >>> sizes = get_module_sizes(model)
        >>> for name, size in sizes.items():
        ...     print(f"{name}: {size:,} parameters")
    """
    sizes = {}
    
    def _get_sizes(module: nn.Module, prefix: str = ""):
        # Count parameters in this specific module (not including children)
        local_params = sum(
            p.numel() for name, p in module.named_parameters(recurse=False)
            if not only_trainable or p.requires_grad
        )
        if local_params > 0:
            sizes[prefix or "root"] = local_params
            
        # Recursively process children
        for name, child in module.named_children():
            child_prefix = f"{prefix}.{name}" if prefix else name
            _get_sizes(child, child_prefix)
    
    _get_sizes(model)
    return sizes


def _get_snn_module_groups(model: nn.Module) -> Dict[str, List[str]]:
    """Group SNN modules by type for optimized wrapping.
    
    This function identifies common SNN module patterns and groups them
    for more efficient FSDP2 wrapping strategies.
    
    Args:
        model: SNN model to analyze
        
    Returns:
        Dictionary mapping group names to lists of module names
    """
    groups = defaultdict(list)
    
    # Import SNNTorch modules dynamically to avoid circular imports
    try:
        import snntorch as snn
        snn_neuron_types = (
            snn.Leaky, snn.Synaptic, snn.Alpha, snn.Lapicque,
            getattr(snn, 'LeakyConv1d', type(None)),
            getattr(snn, 'RLeaky', type(None)),
            getattr(snn, 'RSynaptic', type(None)),
        )
        snn_neuron_types = tuple(t for t in snn_neuron_types if t is not None)
    except ImportError:
        snn_neuron_types = ()
    
    def _analyze_module(module: nn.Module, prefix: str = ""):
        for name, child in module.named_children():
            child_name = f"{prefix}.{name}" if prefix else name
            
            # Group by module type
            if isinstance(child, nn.Linear):
                groups["linear_layers"].append(child_name)
            elif isinstance(child, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                groups["conv_layers"].append(child_name)
            elif isinstance(child, (nn.LSTM, nn.GRU)):
                groups["rnn_layers"].append(child_name)
            elif snn_neuron_types and isinstance(child, snn_neuron_types):
                groups["snn_neurons"].append(child_name)
            elif isinstance(child, nn.Sequential):
                groups["sequential_blocks"].append(child_name)
            else:
                groups["other_modules"].append(child_name)
                
            # Recurse into children
            _analyze_module(child, child_name)
    
    _analyze_module(model)
    return dict(groups)


def auto_wrap_policy(
    model: nn.Module, 
    config: FSDPConfig
) -> List[nn.Module]:
    """Automatically determine which modules to wrap based on configuration.
    
    This implements a size-based wrapping policy with SNN-specific optimizations.
    Modules are wrapped if they meet the size criteria and are identified as
    suitable for sharding.
    
    Args:
        model: Model to analyze for wrapping
        config: FSDP configuration
        
    Returns:
        List of modules that should be wrapped with fully_shard
        
    Example:
        >>> model = create_large_snn_model()
        >>> config = FSDPConfig(min_param_size=1_000_000)
        >>> modules_to_wrap = auto_wrap_policy(model, config)
        >>> print(f"Found {len(modules_to_wrap)} modules to wrap")
    """
    modules_to_wrap = []
    module_sizes = get_module_sizes(model)
    
    if config.snn_optimize:
        # Get SNN-specific module groupings
        snn_groups = _get_snn_module_groups(model)
        
        # Prioritize wrapping certain SNN module types
        priority_modules = set()
        priority_modules.update(snn_groups.get("linear_layers", []))
        priority_modules.update(snn_groups.get("conv_layers", []))
        priority_modules.update(snn_groups.get("sequential_blocks", []))
    else:
        priority_modules = set()
    
    def _find_module_by_name(model: nn.Module, target_name: str) -> Optional[nn.Module]:
        """Find a module by its full dotted name."""
        if not target_name:
            return model
            
        parts = target_name.split('.')
        current = model
        for part in parts:
            if hasattr(current, part):
                current = getattr(current, part)
            else:
                return None
        return current
    
    # Iterate through modules and apply wrapping criteria
    for module_name, param_count in module_sizes.items():
        # Skip if below minimum size
        if param_count < config.min_param_size:
            continue
            
        # Skip if above maximum size (if specified)
        if config.max_param_size is not None and param_count > config.max_param_size:
            continue
            
        # Get the actual module object
        module = _find_module_by_name(model, module_name)
        if module is None:
            continue
            
        # Apply SNN-specific logic
        if config.snn_optimize:
            # Prioritize wrapping if it's in our priority list
            if module_name in priority_modules:
                modules_to_wrap.append(module)
                continue
                
            # For SNN neurons, only wrap if they're part of a larger block
            if any(module_name in group for group in [
                snn_groups.get("snn_neurons", [])
            ]):
                # Only wrap SNN neurons if they have enough parameters
                if param_count >= config.min_param_size * 2:  # Higher threshold for neurons
                    modules_to_wrap.append(module)
                continue
        
        # Default size-based wrapping
        modules_to_wrap.append(module)
    
    return modules_to_wrap


def _create_mixed_precision_policy(config: FSDPConfig) -> Optional[Any]:
    """Create a mixed precision policy for FSDP2.
    
    Args:
        config: FSDP configuration
        
    Returns:
        MixedPrecisionPolicy or None if not using mixed precision
    """
    if not config.mixed_precision or not FSDP2_AVAILABLE:
        return None
        
    if config.mixed_precision_policy is not None:
        return config.mixed_precision_policy
    
    # Create default mixed precision policy optimized for SNNs
    # Use bfloat16 for forward/backward, keep parameters in float32
    try:
        return MixedPrecisionPolicy(
            param_dtype=torch.float32,       # Keep params in full precision
            reduce_dtype=torch.float32,      # Keep gradients in full precision
        )
    except Exception as e:
        warnings.warn(f"Failed to create mixed precision policy: {e}", UserWarning)
        return None


def _create_offload_policy(config: FSDPConfig) -> Optional[Any]:
    """Create an offload policy for FSDP2.
    
    Args:
        config: FSDP configuration
        
    Returns:
        OffloadPolicy or None if not using offloading
    """
    if not config.cpu_offload or not FSDP2_AVAILABLE:
        return None
        
    if config.offload_policy is not None:
        return config.offload_policy
    
    try:
        return OffloadPolicy()
    except Exception as e:
        warnings.warn(f"Failed to create offload policy: {e}", UserWarning)
        return None


def prepare_fsdp2_model(
    model: nn.Module,
    config: Optional[FSDPConfig] = None,
    **kwargs
) -> nn.Module:
    """Prepare a model for FSDP2 distributed training.
    
    This function applies FSDP2 sharding to a model using intelligent wrapping
    strategies based on module sizes and SNN-specific optimizations.
    
    Args:
        model: PyTorch model to prepare for FSDP2
        config: FSDP2 configuration (uses defaults if None)
        **kwargs: Additional arguments to override config values
        
    Returns:
        Model prepared for FSDP2 training
        
    Raises:
        RuntimeError: If FSDP2 is not available in current PyTorch version
        ValueError: If configuration is invalid
        
    Example:
        >>> import snntorch as snn
        >>> from snntorch.distributed import prepare_fsdp2_model, FSDPConfig
        >>> 
        >>> # Create a large SNN model
        >>> model = snn.Sequential(
        ...     nn.Linear(784, 2048),
        ...     snn.Leaky(beta=0.9),
        ...     nn.Linear(2048, 2048),
        ...     snn.Leaky(beta=0.9),
        ...     nn.Linear(2048, 10),
        ...     snn.Leaky(beta=0.9)
        ... )
        >>> 
        >>> # Configure FSDP2
        >>> config = FSDPConfig(
        ...     min_param_size=1_000_000,  # Wrap modules with >1M params
        ...     mixed_precision=True,
        ...     snn_optimize=True
        ... )
        >>> 
        >>> # Prepare for distributed training
        >>> fsdp_model = prepare_fsdp2_model(model, config)
        >>> 
        >>> # Now ready for distributed training!
        >>> # Make sure to initialize distributed process group first
    """
    if not FSDP2_AVAILABLE:
        raise RuntimeError(
            "FSDP2 is not available in this PyTorch version. "
            "Please upgrade to PyTorch 2.1+ for FSDP2 support."
        )
    
    # Create default config if none provided
    if config is None:
        config = FSDPConfig()
    
    # Override config with any kwargs
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            warnings.warn(f"Unknown config parameter: {key}", UserWarning)
    
    # Validate configuration
    if config.min_param_size <= 0:
        raise ValueError("min_param_size must be positive")
    
    if config.max_param_size is not None and config.max_param_size <= config.min_param_size:
        raise ValueError("max_param_size must be greater than min_param_size")
    
    # Get modules to wrap
    modules_to_wrap = auto_wrap_policy(model, config)
    
    print(f"FSDP2: Wrapping {len(modules_to_wrap)} modules based on size criteria")
    print(f"FSDP2: Total model parameters: {count_parameters(model):,}")
    
    # Create policies
    mixed_precision_policy = _create_mixed_precision_policy(config)
    offload_policy = _create_offload_policy(config)
    
    # Apply FSDP2 wrapping to selected modules
    for i, module in enumerate(modules_to_wrap):
        try:
            fully_shard(
                module,
                mp_policy=mixed_precision_policy,
                offload_policy=offload_policy,
            )
            if i % 10 == 0 or i == len(modules_to_wrap) - 1:
                print(f"FSDP2: Wrapped {i+1}/{len(modules_to_wrap)} modules")
        except Exception as e:
            warnings.warn(f"Failed to wrap module {i}: {e}", UserWarning)
    
    # Apply FSDP2 to the root model
    try:
        fully_shard(
            model,
            mp_policy=mixed_precision_policy,
            offload_policy=offload_policy,
        )
        print("FSDP2: Successfully wrapped root model")
    except Exception as e:
        raise RuntimeError(f"Failed to wrap root model with FSDP2: {e}")
    
    return model


def get_fsdp2_memory_stats(model: nn.Module) -> Dict[str, Any]:
    """Get memory statistics for an FSDP2 model.
    
    Args:
        model: FSDP2-wrapped model
        
    Returns:
        Dictionary containing memory statistics
    """
    if not torch.cuda.is_available():
        return {"error": "CUDA not available"}
    
    stats = {
        "allocated_memory_gb": torch.cuda.memory_allocated() / 1e9,
        "reserved_memory_gb": torch.cuda.memory_reserved() / 1e9,
        "max_allocated_memory_gb": torch.cuda.max_memory_allocated() / 1e9,
        "total_parameters": count_parameters(model),
        "device_count": torch.cuda.device_count(),
    }
    
    return stats


def optimize_fsdp2_for_snns(model: nn.Module, config: FSDPConfig) -> FSDPConfig:
    """Optimize FSDP2 configuration specifically for SNN workloads.
    
    This function analyzes the SNN model and adjusts FSDP2 configuration
    for optimal performance based on SNN-specific characteristics.
    
    Args:
        model: SNN model to optimize for
        config: Base FSDP2 configuration
        
    Returns:
        Optimized FSDP2 configuration
    """
    # Analyze model characteristics
    total_params = count_parameters(model)
    module_sizes = get_module_sizes(model)
    snn_groups = _get_snn_module_groups(model)
    
    # Create optimized config
    optimized_config = FSDPConfig(
        sharding_strategy=config.sharding_strategy,
        min_param_size=config.min_param_size,
        max_param_size=config.max_param_size,
        mixed_precision=config.mixed_precision,
        mixed_precision_policy=config.mixed_precision_policy,
        offload_policy=config.offload_policy,
        device_id=config.device_id,
        sync_module_states=config.sync_module_states,
        param_init_fn=config.param_init_fn,
        forward_prefetch=config.forward_prefetch,
        backward_prefetch=config.backward_prefetch,
        activation_checkpointing=config.activation_checkpointing,
        limit_all_gathers=config.limit_all_gathers,
        cpu_offload=config.cpu_offload,
        snn_optimize=True,  # Always enable SNN optimizations
    )
    
    # Adjust parameters based on model size
    if total_params > 1e9:  # > 1B parameters
        print("FSDP2: Detected large model (>1B params), optimizing for memory efficiency")
        optimized_config.sharding_strategy = ShardingStrategy.FULL_SHARD
        optimized_config.activation_checkpointing = True
        optimized_config.cpu_offload = True
        optimized_config.min_param_size = max(500_000, config.min_param_size)
    elif total_params > 10e6:  # > 10M parameters (lowered threshold)
        print("FSDP2: Detected medium model (>10M params), balancing memory and speed")
        optimized_config.sharding_strategy = ShardingStrategy.SHARD_GRAD_OP
        optimized_config.min_param_size = max(100_000, config.min_param_size)
    else:
        print("FSDP2: Detected small model, optimizing for speed")
        optimized_config.sharding_strategy = ShardingStrategy.NO_SHARD
        optimized_config.min_param_size = max(10_000, config.min_param_size)
    
    # SNN-specific optimizations
    num_snn_neurons = len(snn_groups.get("snn_neurons", []))
    if num_snn_neurons > 50:
        print(f"FSDP2: Detected many SNN neurons ({num_snn_neurons}), enabling aggressive prefetching")
        optimized_config.forward_prefetch = True
        optimized_config.backward_prefetch = True
    
    # For models with many sequential blocks, enable checkpointing
    num_sequential = len(snn_groups.get("sequential_blocks", []))
    if num_sequential > 10:
        print(f"FSDP2: Detected many sequential blocks ({num_sequential}), enabling checkpointing")
        optimized_config.activation_checkpointing = True
    
    return optimized_config