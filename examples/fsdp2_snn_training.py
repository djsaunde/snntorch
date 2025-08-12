"""
FSDP2 Distributed Training Example for SNNs

This script demonstrates how to train spiking neural networks using 
PyTorch's Fully Sharded Data Parallel (FSDP2) for distributed training.

Usage:
    # Single node, multiple GPUs
    torchrun --nproc_per_node=2 fsdp2_snn_training.py

    # Multi-node training
    torchrun --nproc_per_node=4 --nnodes=2 --rdzv_endpoint=master_ip:29500 fsdp2_snn_training.py

Key Features:
    - Automatic FSDP strategy selection based on model size
    - Mixed precision training support
    - Comprehensive logging and checkpointing
    - Memory-efficient spike train processing
"""

import os
import argparse
import time
from typing import Dict, Any

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, TensorDataset
import torchvision
import torchvision.transforms as transforms

import snntorch as snn
from snntorch import surrogate
from snntorch.distributed import (
    setup_distributed, 
    FSDPSNNWrapper, 
    create_distributed_snn,
    quick_fsdp_setup,
    print_fsdp_strategy_guide
)
from snntorch.trainer import (
    DistributedSNNTrainer,
    create_distributed_dataloader,
    get_optimizer_and_scheduler
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='FSDP2 SNN Training Example')
    
    # Model parameters
    parser.add_argument('--input-size', type=int, default=784, help='Input size')
    parser.add_argument('--hidden-size', type=int, default=256, help='Hidden layer size')
    parser.add_argument('--output-size', type=int, default=10, help='Output size')
    parser.add_argument('--num-layers', type=int, default=2, help='Number of hidden layers')
    parser.add_argument('--num-steps', type=int, default=25, help='Number of time steps')
    parser.add_argument('--beta', type=float, default=0.9, help='Neuron decay parameter')
    parser.add_argument('--neuron-type', type=str, default='leaky', 
                       choices=['leaky', 'lapicque', 'alpha', 'synaptic'],
                       help='Type of SNN neuron')
    
    # Training parameters
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size per GPU')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--optimizer', type=str, default='adam', 
                       choices=['adam', 'adamw', 'sgd'], help='Optimizer type')
    parser.add_argument('--scheduler', type=str, default=None,
                       choices=['cosine', 'step', 'exponential'], help='LR scheduler')
    
    # FSDP parameters
    parser.add_argument('--fsdp-strategy', type=str, default='auto',
                       choices=['minimal', 'selective', 'aggressive', 'auto', 'transformer'],
                       help='FSDP wrapping strategy')
    parser.add_argument('--min-num-params', type=int, default=100000,
                       help='Minimum parameters for auto-wrapping')
    parser.add_argument('--mixed-precision', action='store_true',
                       help='Use mixed precision training')
    parser.add_argument('--grad-accumulation-steps', type=int, default=1,
                       help='Gradient accumulation steps')
    
    # Data parameters
    parser.add_argument('--dataset', type=str, default='synthetic',
                       choices=['synthetic', 'mnist'], help='Dataset to use')
    parser.add_argument('--data-path', type=str, default='./data', help='Data directory')
    parser.add_argument('--num-workers', type=int, default=0, help='Number of data workers')
    
    # Logging and checkpointing
    parser.add_argument('--log-interval', type=int, default=100, help='Logging interval')
    parser.add_argument('--save-checkpoint', action='store_true', help='Save checkpoints')
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints',
                       help='Checkpoint directory')
    parser.add_argument('--resume-from', type=str, default=None,
                       help='Resume training from checkpoint')
    
    # Other
    parser.add_argument('--print-strategy-guide', action='store_true',
                       help='Print FSDP strategy guide and exit')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    return parser.parse_args()


def set_random_seed(seed: int):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    import numpy as np
    np.random.seed(seed)


def create_synthetic_spike_data(num_samples: int, 
                               input_size: int, 
                               num_steps: int,
                               output_size: int,
                               spike_prob: float = 0.1) -> tuple:
    """
    Create synthetic spike train data for testing.
    
    Args:
        num_samples: Number of samples
        input_size: Input feature size
        num_steps: Number of time steps
        output_size: Number of output classes
        spike_prob: Probability of spike at each time step
        
    Returns:
        Tuple of (data, targets)
    """
    # Generate random spike trains
    data = torch.rand(num_samples, num_steps, input_size)
    data = (data < spike_prob).float()  # Convert to binary spikes
    
    # Random targets
    targets = torch.randint(0, output_size, (num_samples,))
    
    # Reshape to (num_steps, num_samples, input_size) for SNN processing
    data = data.transpose(0, 1)
    
    return data, targets


def create_mnist_spike_data(data_path: str, 
                           num_steps: int,
                           train: bool = True) -> tuple:
    """
    Create MNIST spike data by converting images to spike trains.
    
    Args:
        data_path: Path to MNIST data
        num_steps: Number of time steps
        train: Whether to load training or test set
        
    Returns:
        Tuple of (data, targets)
    """
    # Load MNIST dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    dataset = torchvision.datasets.MNIST(
        root=data_path, train=train, download=True, transform=transform
    )
    
    # Convert to tensors
    data_list = []
    targets_list = []
    
    for image, target in dataset:
        # Convert image to spike train using rate coding
        spike_train = torch.rand(num_steps, *image.shape) < image.abs()
        data_list.append(spike_train.float())
        targets_list.append(target)
    
    data = torch.stack(data_list, dim=1)  # (num_steps, num_samples, channels, height, width)
    targets = torch.tensor(targets_list)
    
    return data, targets


def create_datasets(args) -> tuple:
    """Create training and validation datasets."""
    if args.dataset == 'synthetic':
        # Create synthetic data
        train_data, train_targets = create_synthetic_spike_data(
            num_samples=5000,
            input_size=args.input_size,
            num_steps=args.num_steps,
            output_size=args.output_size
        )
        
        val_data, val_targets = create_synthetic_spike_data(
            num_samples=1000,
            input_size=args.input_size,
            num_steps=args.num_steps,
            output_size=args.output_size
        )
    
    elif args.dataset == 'mnist':
        # Create MNIST spike data
        train_data, train_targets = create_mnist_spike_data(
            args.data_path, args.num_steps, train=True
        )
        val_data, val_targets = create_mnist_spike_data(
            args.data_path, args.num_steps, train=False
        )
        
        # Flatten spatial dimensions for fully connected network
        if len(train_data.shape) > 3:  # Has spatial dimensions
            batch_size = train_data.shape[1]
            train_data = train_data.view(args.num_steps, batch_size, -1)
            val_data = val_data.view(args.num_steps, val_data.shape[1], -1)
            
            # Update input size
            args.input_size = train_data.shape[2]
    
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    
    # Create datasets
    train_dataset = TensorDataset(train_data.transpose(0, 1), train_targets)
    val_dataset = TensorDataset(val_data.transpose(0, 1), val_targets)
    
    return train_dataset, val_dataset


def main():
    """Main training function."""
    args = parse_args()
    
    # Print strategy guide if requested
    if args.print_strategy_guide:
        print_fsdp_strategy_guide()
        return
    
    # Setup distributed environment
    local_rank = setup_distributed()
    device = torch.device(f'cuda:{local_rank}')
    
    # Set random seed
    set_random_seed(args.seed)
    
    if dist.get_rank() == 0:
        print(f"Starting FSDP2 SNN training with {dist.get_world_size()} GPUs")
        print(f"Arguments: {args}")
    
    # Create datasets
    train_dataset, val_dataset = create_datasets(args)
    
    # Create data loaders
    train_loader = create_distributed_dataloader(
        train_dataset, args.batch_size, args.num_workers, shuffle=True
    )
    val_loader = create_distributed_dataloader(
        val_dataset, args.batch_size, args.num_workers, shuffle=False
    )
    
    if dist.get_rank() == 0:
        print(f"Training samples: {len(train_dataset)}")
        print(f"Validation samples: {len(val_dataset)}")
    
    # Create model
    model = create_distributed_snn(
        input_size=args.input_size,
        hidden_size=args.hidden_size,
        output_size=args.output_size,
        num_layers=args.num_layers,
        beta=args.beta,
        neuron_type=args.neuron_type
    )
    
    # Apply FSDP
    if args.fsdp_strategy == 'auto':
        model = quick_fsdp_setup(model)
    else:
        model = FSDPSNNWrapper.apply_fsdp(
            model, 
            strategy=args.fsdp_strategy,
            min_num_params=args.min_num_params
        )
    
    model = model.to(device)
    
    if dist.get_rank() == 0:
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Model created with {total_params:,} parameters")
    
    # Create optimizer and scheduler
    optimizer, scheduler = get_optimizer_and_scheduler(
        model=model,
        lr=args.lr,
        optimizer_type=args.optimizer,
        scheduler_type=args.scheduler,
        T_max=args.epochs if args.scheduler == 'cosine' else None
    )
    
    # Create loss function
    loss_fn = nn.CrossEntropyLoss()
    
    # Create trainer
    trainer = DistributedSNNTrainer(
        model=model,
        device=device,
        num_steps=args.num_steps,
        grad_accumulation_steps=args.grad_accumulation_steps,
        mixed_precision=args.mixed_precision,
        clip_grad_norm=1.0
    )
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume_from and os.path.exists(args.resume_from):
        trainer.load_checkpoint(args.resume_from, optimizer, scheduler)
        start_epoch = trainer.current_epoch + 1
        if dist.get_rank() == 0:
            print(f"Resumed training from epoch {start_epoch}")
    
    # Training loop
    best_val_acc = 0.0
    
    for epoch in range(start_epoch, args.epochs):
        trainer.current_epoch = epoch
        
        # Set epoch for distributed sampler
        train_loader.sampler.set_epoch(epoch)
        
        # Training
        if dist.get_rank() == 0:
            print(f"\nEpoch {epoch}/{args.epochs}")
        
        train_stats = trainer.train_epoch(
            dataloader=train_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            loss_mode="sum_spikes",
            log_interval=args.log_interval,
            scheduler=scheduler
        )
        
        # Validation
        val_stats = trainer.evaluate(
            dataloader=val_loader,
            loss_fn=loss_fn,
            loss_mode="sum_spikes"
        )
        
        # Logging
        if dist.get_rank() == 0:
            print(f"Epoch {epoch} completed:")
            print(f"  Train Loss: {train_stats['loss']:.6f}")
            print(f"  Val Loss: {val_stats['loss']:.6f}")
            print(f"  Val Accuracy: {val_stats['accuracy']:.4f}")
            print(f"  Forward Time: {train_stats['avg_forward_time']:.4f}s")
            print(f"  Backward Time: {train_stats['avg_backward_time']:.4f}s")
        
        # Save checkpoint
        if args.save_checkpoint and val_stats['accuracy'] > best_val_acc:
            best_val_acc = val_stats['accuracy']
            
            if dist.get_rank() == 0:
                os.makedirs(args.checkpoint_dir, exist_ok=True)
                checkpoint_path = os.path.join(
                    args.checkpoint_dir, 
                    f"best_model_epoch_{epoch}.pt"
                )
                trainer.save_checkpoint(
                    checkpoint_path, 
                    optimizer, 
                    scheduler,
                    additional_info={
                        'val_accuracy': best_val_acc,
                        'args': args.__dict__
                    }
                )
    
    # Final evaluation
    if dist.get_rank() == 0:
        print(f"\nTraining completed!")
        print(f"Best validation accuracy: {best_val_acc:.4f}")
    
    # Cleanup
    dist.destroy_process_group()


if __name__ == "__main__":
    main()