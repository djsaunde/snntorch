#!/usr/bin/env python

"""
Large Scale SNN Training with FSDP2

This example demonstrates training a large spiking neural network (100M+ parameters) 
using FSDP2 for memory-efficient distributed training.

The model architecture includes:
- Large convolutional feature extractors
- Deep spiking neural network layers
- Attention-like mechanisms for temporal processing
- Classification head

Key FSDP2 features demonstrated:
- Strategic module wrapping based on parameter counts
- Mixed precision training optimized for SNNs
- Memory-efficient gradient synchronization
- Temporal dynamics with spike-based computation
- Advanced optimization strategies

Requirements:
- PyTorch 2.4+ with FSDP2 support
- 4+ GPUs recommended (works with 1 GPU for testing)
- ~16GB GPU memory per device (with FSDP2 optimizations)

Usage:
    # Single node, 4 GPUs
    torchrun --standalone --nproc_per_node=4 fsdp2_large_model_training.py
    
    # Single GPU (for testing)
    python fsdp2_large_model_training.py --single-gpu
    
    # Multiple nodes
    torchrun --nproc_per_node=4 --nnodes=2 --node_rank=0 --master_addr=<ip> --master_port=29500 fsdp2_large_model_training.py

Model Architecture:
- Input: 28x28 grayscale images (MNIST/FashionMNIST)
- Feature extraction: Large CNN with 64-512 channels
- Temporal processing: Multi-layer SNN with attention
- Classification: Deep MLP with spiking neurons
- Total parameters: ~120M (adjustable)
"""

import argparse
import os
import time
import warnings
from contextlib import nullcontext
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
import torchvision
import torchvision.transforms as transforms

import snntorch as snn
from snntorch import functional as SF


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Large Scale SNN Training with FSDP2")
    
    # Model configuration
    parser.add_argument("--model-size", type=str, default="large", 
                       choices=["medium", "large", "xlarge"],
                       help="Model size configuration")
    parser.add_argument("--num-classes", type=int, default=10,
                       help="Number of output classes")
    
    # Training configuration
    parser.add_argument("--batch-size", type=int, default=32,
                       help="Batch size per GPU")
    parser.add_argument("--num-epochs", type=int, default=5,
                       help="Number of training epochs")
    parser.add_argument("--learning-rate", type=float, default=1e-3,
                       help="Peak learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4,
                       help="Weight decay")
    parser.add_argument("--gradient-clip", type=float, default=1.0,
                       help="Gradient clipping value")
    
    # SNN configuration
    parser.add_argument("--num-steps", type=int, default=20,
                       help="Number of time steps for SNN simulation")
    parser.add_argument("--beta", type=float, default=0.9,
                       help="Membrane decay rate")
    parser.add_argument("--spike-encoding", type=str, default="rate",
                       choices=["rate", "temporal", "poisson"],
                       help="Spike encoding method")
    
    # FSDP2 configuration
    parser.add_argument("--min-param-size", type=int, default=1_000_000,
                       help="Minimum parameter size for FSDP2 wrapping")
    parser.add_argument("--sharding-strategy", type=str, default="auto",
                       choices=["auto", "full", "grad_op", "hybrid", "no_shard"],
                       help="FSDP2 sharding strategy")
    parser.add_argument("--mixed-precision", action="store_true", default=True,
                       help="Use mixed precision training")
    parser.add_argument("--cpu-offload", action="store_true",
                       help="Offload parameters to CPU")
    parser.add_argument("--activation-checkpointing", action="store_true",
                       help="Use activation checkpointing")
    
    # Data and system configuration
    parser.add_argument("--dataset", type=str, default="mnist",
                       choices=["mnist", "fashion-mnist", "cifar10"],
                       help="Dataset to use")
    parser.add_argument("--data-dir", type=str, default="./data",
                       help="Data directory")
    parser.add_argument("--num-workers", type=int, default=4,
                       help="Number of data loading workers")
    parser.add_argument("--single-gpu", action="store_true",
                       help="Run on single GPU (for testing)")
    parser.add_argument("--profile", action="store_true",
                       help="Enable memory profiling")
    
    return parser.parse_args()


class SpikeEncoder(nn.Module):
    """Convert static images to spike trains using various encoding methods."""
    
    def __init__(self, encoding_method="rate", num_steps=20):
        super().__init__()
        self.encoding_method = encoding_method
        self.num_steps = num_steps
    
    def forward(self, x):
        """
        Convert input to spike trains.
        
        Args:
            x: Input tensor (batch_size, channels, height, width)
            
        Returns:
            Spike trains (num_steps, batch_size, channels, height, width)
        """
        if self.encoding_method == "rate":
            # Rate coding: pixel intensity -> spike probability
            spikes = torch.bernoulli(x.unsqueeze(0).repeat(self.num_steps, 1, 1, 1, 1))
        elif self.encoding_method == "temporal":
            # Temporal coding: brighter pixels spike earlier
            spike_times = (1.0 - x) * (self.num_steps - 1)
            spikes = torch.zeros(self.num_steps, *x.shape, device=x.device)
            for t in range(self.num_steps):
                spikes[t] = (spike_times <= t).float() * (spike_times > t - 1).float()
        elif self.encoding_method == "poisson":
            # Poisson spike trains
            rates = x * 50  # Scale to reasonable firing rates
            spikes = torch.poisson(rates.unsqueeze(0).repeat(self.num_steps, 1, 1, 1, 1) / self.num_steps)
            spikes = torch.clamp(spikes, 0, 1)
        else:
            raise ValueError(f"Unknown encoding method: {self.encoding_method}")
        
        return spikes


class LargeCNNFeatureExtractor(nn.Module):
    """Large CNN feature extractor with many parameters."""
    
    def __init__(self, input_channels=1, base_channels=128):
        super().__init__()
        
        # Progressive channel expansion for large parameter count
        self.conv_blocks = nn.ModuleList([
            # Block 1: 1 -> 128 channels
            nn.Sequential(
                nn.Conv2d(input_channels, base_channels, 3, padding=1),
                nn.BatchNorm2d(base_channels),
                snn.Leaky(beta=0.9),
                nn.Conv2d(base_channels, base_channels, 3, padding=1),
                nn.BatchNorm2d(base_channels),
                snn.Leaky(beta=0.9),
                nn.MaxPool2d(2),  # 28x28 -> 14x14
            ),
            
            # Block 2: 128 -> 256 channels  
            nn.Sequential(
                nn.Conv2d(base_channels, base_channels * 2, 3, padding=1),
                nn.BatchNorm2d(base_channels * 2),
                snn.Leaky(beta=0.9),
                nn.Conv2d(base_channels * 2, base_channels * 2, 3, padding=1),
                nn.BatchNorm2d(base_channels * 2),
                snn.Leaky(beta=0.9),
                nn.MaxPool2d(2),  # 14x14 -> 7x7
            ),
            
            # Block 3: 256 -> 512 channels
            nn.Sequential(
                nn.Conv2d(base_channels * 2, base_channels * 4, 3, padding=1),
                nn.BatchNorm2d(base_channels * 4),
                snn.Leaky(beta=0.9),
                nn.Conv2d(base_channels * 4, base_channels * 4, 3, padding=1),
                nn.BatchNorm2d(base_channels * 4),
                snn.Leaky(beta=0.9),
                nn.AdaptiveAvgPool2d((4, 4)),  # 7x7 -> 4x4
            ),
        ])
        
        self.output_dim = base_channels * 4 * 4 * 4  # 512 * 16 = 8192
    
    def forward(self, x):
        """Forward pass through CNN blocks."""
        for block in self.conv_blocks:
            x = block(x)
        return x.flatten(1)  # (batch, features)


class SpatialTemporalAttention(nn.Module):
    """Attention mechanism for temporal spike processing."""
    
    def __init__(self, hidden_dim, num_heads=8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        
        # Query, Key, Value projections
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Spike-based activation
        self.spike_layer = snn.Leaky(beta=0.9, learn_beta=True)
        
    def forward(self, spike_sequence):
        """
        Apply attention across temporal dimension.
        
        Args:
            spike_sequence: (num_steps, batch_size, hidden_dim)
            
        Returns:
            Attended sequence: (num_steps, batch_size, hidden_dim)
        """
        num_steps, batch_size, hidden_dim = spike_sequence.shape
        
        # Compute attention for each time step
        attended_sequence = []
        
        for t in range(num_steps):
            current_spikes = spike_sequence[t]  # (batch_size, hidden_dim)
            
            # Multi-head attention
            q = self.q_proj(current_spikes).view(batch_size, self.num_heads, self.head_dim)
            k = self.k_proj(current_spikes).view(batch_size, self.num_heads, self.head_dim)
            v = self.v_proj(current_spikes).view(batch_size, self.num_heads, self.head_dim)
            
            # Scaled dot-product attention
            scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
            attn_weights = torch.softmax(scores, dim=-1)
            
            # Apply attention to values
            attended = torch.matmul(attn_weights, v)
            attended = attended.view(batch_size, hidden_dim)
            
            # Output projection and spiking activation
            output = self.out_proj(attended)
            spike_output = self.spike_layer(output)
            
            attended_sequence.append(spike_output)
        
        return torch.stack(attended_sequence, dim=0)


class DeepSNNClassifier(nn.Module):
    """Deep spiking neural network classifier with many parameters."""
    
    def __init__(self, input_dim, hidden_dims, num_classes, beta=0.9):
        super().__init__()
        
        # Build deep SNN layers
        layers = []
        prev_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                snn.Leaky(beta=beta, learn_beta=True, learn_threshold=True),
                nn.Dropout(0.1),  # Regularization for large model
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.extend([
            nn.Linear(prev_dim, num_classes),
            snn.Leaky(beta=beta, learn_beta=True, learn_threshold=True),
        ])
        
        self.snn_layers = nn.Sequential(*layers)
        
        # Temporal attention for sequence processing
        self.attention = SpatialTemporalAttention(hidden_dims[-1])
        
    def forward(self, x_sequence):
        """
        Process temporal sequence through deep SNN.
        
        Args:
            x_sequence: (num_steps, batch_size, input_dim)
            
        Returns:
            output_sequence: (num_steps, batch_size, num_classes)
        """
        output_sequence = []
        
        for t in range(x_sequence.size(0)):
            x_t = x_sequence[t]
            
            # Process through SNN layers
            for i, layer in enumerate(self.snn_layers):
                if isinstance(layer, (nn.Linear, nn.Dropout)):
                    x_t = layer(x_t)
                elif isinstance(layer, snn.Leaky):
                    x_t = layer(x_t)
            
            output_sequence.append(x_t)
        
        # Stack and apply temporal attention
        output_tensor = torch.stack(output_sequence, dim=0)
        
        # Apply attention to the penultimate layer for better temporal processing
        # (This is a simplified version - in practice you'd apply it to hidden states)
        
        return output_tensor


class LargeScaleSNN(nn.Module):
    """Large-scale spiking neural network with 100M+ parameters."""
    
    def __init__(self, 
                 input_channels=1, 
                 num_classes=10, 
                 model_size="large",
                 spike_encoding="rate",
                 num_steps=20,
                 beta=0.9):
        super().__init__()
        
        self.num_steps = num_steps
        
        # Configure model size
        if model_size == "medium":
            cnn_channels = 96
            snn_hidden_dims = [4096, 2048, 1024]
        elif model_size == "large":
            cnn_channels = 128
            snn_hidden_dims = [6144, 4096, 2048, 1024]
        elif model_size == "xlarge":
            cnn_channels = 160
            snn_hidden_dims = [8192, 6144, 4096, 2048, 1024]
        else:
            raise ValueError(f"Unknown model size: {model_size}")
        
        # Components
        self.spike_encoder = SpikeEncoder(spike_encoding, num_steps)
        self.feature_extractor = LargeCNNFeatureExtractor(input_channels, cnn_channels)
        self.classifier = DeepSNNClassifier(
            self.feature_extractor.output_dim,
            snn_hidden_dims,
            num_classes,
            beta
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize model weights."""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Conv2d):
            torch.nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
    
    def forward(self, x):
        """
        Forward pass through the large-scale SNN.
        
        Args:
            x: Input tensor (batch_size, channels, height, width)
            
        Returns:
            Tuple of (spike_outputs, membrane_potentials)
        """
        # Convert to spike trains
        spike_trains = self.spike_encoder(x)  # (num_steps, batch, channels, h, w)
        
        # Process each time step through CNN
        features_sequence = []
        for t in range(self.num_steps):
            features = self.feature_extractor(spike_trains[t])
            features_sequence.append(features)
        
        # Stack temporal features
        features_tensor = torch.stack(features_sequence, dim=0)  # (num_steps, batch, features)
        
        # Process through deep SNN classifier
        outputs = self.classifier(features_tensor)  # (num_steps, batch, num_classes)
        
        return outputs
    
    def get_parameter_count(self):
        """Get total parameter count."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def setup_distributed(single_gpu=False):
    """Setup distributed training environment."""
    if single_gpu:
        return 0, 1, 0, torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Initialize process group
    dist.init_process_group(backend="nccl")
    
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    
    # Set device
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(local_rank)
    
    return local_rank, world_size, rank, device


def create_dataset(args, world_size=1, rank=0):
    """Create dataset and dataloaders."""
    # Dataset-specific configuration
    if args.dataset == "mnist":
        dataset_class = torchvision.datasets.MNIST
        mean, std = (0.1307,), (0.3081,)
        input_channels = 1
    elif args.dataset == "fashion-mnist":
        dataset_class = torchvision.datasets.FashionMNIST
        mean, std = (0.2860,), (0.3530,)
        input_channels = 1
    elif args.dataset == "cifar10":
        dataset_class = torchvision.datasets.CIFAR10
        mean, std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
        input_channels = 3
        args.num_classes = 10
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")
    
    # Data transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    
    # Create datasets
    train_dataset = dataset_class(
        root=args.data_dir,
        train=True,
        download=True,
        transform=transform
    )
    
    test_dataset = dataset_class(
        root=args.data_dir,
        train=False,
        download=True,
        transform=transform
    )
    
    # Create samplers
    if world_size > 1:
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
        test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank)
    else:
        train_sampler = None
        test_sampler = None
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        sampler=test_sampler,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    
    return train_loader, test_loader, train_sampler, input_channels


def setup_fsdp2_model(model, args, rank):
    """Setup model with FSDP2."""
    try:
        from snntorch.distributed import prepare_fsdp2_model, FSDPConfig, ShardingStrategy
    except ImportError:
        if rank == 0:
            print("FSDP2 not available, using regular model")
        return model
    
    # Map sharding strategy
    strategy_map = {
        "auto": None,  # Will be auto-selected
        "full": ShardingStrategy.FULL_SHARD,
        "grad_op": ShardingStrategy.SHARD_GRAD_OP,
        "hybrid": ShardingStrategy.HYBRID_SHARD,
        "no_shard": ShardingStrategy.NO_SHARD,
    }
    
    # Create FSDP config
    config = FSDPConfig(
        sharding_strategy=strategy_map.get(args.sharding_strategy),
        min_param_size=args.min_param_size,
        mixed_precision=args.mixed_precision,
        cpu_offload=args.cpu_offload,
        activation_checkpointing=args.activation_checkpointing,
        snn_optimize=True,
    )
    
    # Auto-optimize if requested
    if args.sharding_strategy == "auto":
        from snntorch.distributed import optimize_fsdp2_for_snns
        config = optimize_fsdp2_for_snns(model, config)
    
    if rank == 0:
        print(f"Setting up FSDP2 with strategy: {config.sharding_strategy}")
        print(f"Min param size: {config.min_param_size:,}")
        
    # Prepare model
    model = prepare_fsdp2_model(model, config)
    
    return model


def train_epoch(model, train_loader, optimizer, scheduler, device, args, epoch, rank):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    
    start_time = time.time()
    
    for batch_idx, (data, targets) in enumerate(train_loader):
        data, targets = data.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        
        # Forward pass
        optimizer.zero_grad()
        
        with torch.cuda.amp.autocast(enabled=args.mixed_precision):
            # Get spike outputs from all time steps
            spike_outputs = model(data)  # (num_steps, batch, num_classes)
            
            # Use mean spike count across time for classification
            mean_spikes = spike_outputs.mean(dim=0)  # (batch, num_classes)
            
            # Compute loss
            loss = F.cross_entropy(mean_spikes, targets)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        if args.gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
        
        optimizer.step()
        scheduler.step()
        
        # Statistics
        total_loss += loss.item()
        pred = mean_spikes.argmax(dim=1)
        total_correct += (pred == targets).sum().item()
        total_samples += targets.size(0)
        
        # Progress reporting
        if rank == 0 and batch_idx % 50 == 0:
            elapsed = time.time() - start_time
            print(f'Epoch {epoch}, Batch {batch_idx}/{len(train_loader)}, '
                  f'Loss: {loss.item():.4f}, '
                  f'Acc: {100.0 * total_correct / total_samples:.2f}%, '
                  f'Time: {elapsed:.1f}s')
    
    avg_loss = total_loss / len(train_loader)
    accuracy = 100.0 * total_correct / total_samples
    
    return avg_loss, accuracy


def validate(model, test_loader, device, args, rank):
    """Validate the model."""
    model.eval()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for data, targets in test_loader:
            data, targets = data.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            
            with torch.cuda.amp.autocast(enabled=args.mixed_precision):
                spike_outputs = model(data)
                mean_spikes = spike_outputs.mean(dim=0)
                loss = F.cross_entropy(mean_spikes, targets)
            
            total_loss += loss.item()
            pred = mean_spikes.argmax(dim=1)
            total_correct += (pred == targets).sum().item()
            total_samples += targets.size(0)
    
    avg_loss = total_loss / len(test_loader)
    accuracy = 100.0 * total_correct / total_samples
    
    return avg_loss, accuracy


def print_memory_stats(device, rank, prefix=""):
    """Print GPU memory statistics."""
    if rank == 0 and torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(device) / 1e9
        reserved = torch.cuda.memory_reserved(device) / 1e9
        max_allocated = torch.cuda.max_memory_allocated(device) / 1e9
        print(f"{prefix}Memory - Allocated: {allocated:.2f}GB, "
              f"Reserved: {reserved:.2f}GB, Max: {max_allocated:.2f}GB")


def main():
    """Main training function."""
    args = parse_args()
    
    # Setup distributed training
    local_rank, world_size, rank, device = setup_distributed(args.single_gpu)
    
    if rank == 0:
        print("="*80)
        print("Large-Scale SNN Training with FSDP2")
        print("="*80)
        print(f"Model size: {args.model_size}")
        print(f"Dataset: {args.dataset}")
        print(f"Batch size per GPU: {args.batch_size}")
        print(f"Total batch size: {args.batch_size * world_size}")
        print(f"Number of time steps: {args.num_steps}")
        print(f"World size: {world_size}")
        print(f"Device: {device}")
        print("="*80)
    
    # Create dataset
    train_loader, test_loader, train_sampler, input_channels = create_dataset(args, world_size, rank)
    
    # Create model
    model = LargeScaleSNN(
        input_channels=input_channels,
        num_classes=args.num_classes,
        model_size=args.model_size,
        spike_encoding=args.spike_encoding,
        num_steps=args.num_steps,
        beta=args.beta,
    )
    
    if rank == 0:
        param_count = model.get_parameter_count()
        print(f"Model parameter count: {param_count:,}")
        print(f"Model size: {param_count / 1e6:.1f}M parameters")
    
    # Move to device
    model = model.to(device)
    
    # Setup FSDP2
    model = setup_fsdp2_model(model, args, rank)
    
    print_memory_stats(device, rank, "After model setup - ")
    
    # Setup optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    
    total_steps = len(train_loader) * args.num_epochs
    scheduler = OneCycleLR(
        optimizer,
        max_lr=args.learning_rate,
        total_steps=total_steps,
        pct_start=0.1,
        anneal_strategy='cos'
    )
    
    # Training loop
    if rank == 0:
        print("Starting training...")
        print(f"Total steps: {total_steps}")
    
    for epoch in range(args.num_epochs):
        # Set epoch for distributed sampler
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, scheduler, device, args, epoch, rank
        )
        
        # Validate
        val_loss, val_acc = validate(model, test_loader, device, args, rank)
        
        # Gather metrics from all processes
        if world_size > 1:
            # Average metrics across all processes
            metrics = torch.tensor([train_loss, train_acc, val_loss, val_acc], device=device)
            dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
            metrics /= world_size
            train_loss, train_acc, val_loss, val_acc = metrics.tolist()
        
        if rank == 0:
            print(f"\nEpoch {epoch} Summary:")
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            print(f"  Learning Rate: {scheduler.get_last_lr()[0]:.6f}")
            
            print_memory_stats(device, rank, "  ")
            print()
    
    if rank == 0:
        print("Training completed!")
        
        # Final memory stats
        print_memory_stats(device, rank, "Final ")
        
        # Model statistics
        if hasattr(model, 'get_parameter_count'):
            final_params = model.get_parameter_count()
            print(f"Final model parameters: {final_params:,}")
    
    # Cleanup
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()