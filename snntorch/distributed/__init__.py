"""
Distributed training utilities for SNNTorch.

This module provides utilities for distributed training of spiking neural networks,
including FSDP2 (Fully Sharded Data Parallel) support for large-scale model training.

Key features:
- FSDP2 model preparation with smart module wrapping
- Parameter counting and module sizing for optimal sharding
- SNN-specific distributed strategies
- Memory-efficient training utilities

Example:
    >>> import snntorch as snn
    >>> from snntorch.distributed import prepare_fsdp2_model
    >>> 
    >>> model = snn.Sequential(
    ...     snn.Linear(784, 1000),
    ...     snn.Leaky(beta=0.9),
    ...     snn.Linear(1000, 10),
    ...     snn.Leaky(beta=0.9)
    ... )
    >>> 
    >>> # Prepare model for FSDP2 with automatic module wrapping
    >>> fsdp_model = prepare_fsdp2_model(
    ...     model, 
    ...     min_param_size=1e6,  # Wrap modules with >1M parameters
    ...     sharding_strategy="FULL_SHARD",
    ...     mixed_precision=True
    ... )
"""

from .fsdp2 import (
    prepare_fsdp2_model,
    count_parameters,
    get_module_sizes,
    FSDPConfig,
    ShardingStrategy,
    auto_wrap_policy,
    optimize_fsdp2_for_snns,
)

__all__ = [
    "prepare_fsdp2_model",
    "count_parameters", 
    "get_module_sizes",
    "FSDPConfig",
    "ShardingStrategy",
    "auto_wrap_policy",
    "optimize_fsdp2_for_snns",
]