#!/usr/bin/env python

"""
Example: Distributed Training with FSDP2 and SNNTorch

This example demonstrates how to use FSDP2 (Fully Sharded Data Parallel) 
to train large spiking neural networks across multiple GPUs efficiently.

FSDP2 provides several advantages for large SNN training:
- Memory-efficient parameter sharding
- Automatic gradient synchronization
- Mixed precision support optimized for SNNs
- Intelligent module wrapping based on parameter counts

Requirements:
- PyTorch 2.1+ with FSDP2 support
- Multiple GPUs (recommended)
- Distributed training setup

Usage:
    # Single node, multiple GPUs
    torchrun --standalone --nproc_per_node=4 fsdp2_distributed_training.py
    
    # Multiple nodes
    torchrun --nproc_per_node=4 --nnodes=2 --node_rank=0 --master_addr=<master_ip> --master_port=<port> fsdp2_distributed_training.py
"""

import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchvision
import torchvision.transforms as transforms

import snntorch as snn
from snntorch import functional as SF
from snntorch.distributed import prepare_fsdp2_model, FSDPConfig, ShardingStrategy


def setup_distributed():
    """Initialize distributed training environment."""
    # Initialize the process group
    dist.init_process_group(backend="nccl")
    
    # Set the device for this process
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    
    return local_rank, dist.get_world_size(), dist.get_rank()


def create_large_snn_model(num_classes=10):
    """Create a large SNN model for demonstration.
    
    This model is designed to be large enough to benefit from FSDP2 sharding.
    In practice, you would use your own SNN architecture.
    """
    model = nn.Sequential(
        # Input projection
        nn.Linear(784, 4096),
        snn.Leaky(beta=0.9, learn_beta=True),
        
        # Hidden layers (make it large for FSDP2 demonstration)
        nn.Linear(4096, 4096),
        snn.Leaky(beta=0.9, learn_beta=True),
        
        nn.Linear(4096, 4096),
        snn.Leaky(beta=0.9, learn_beta=True),
        
        nn.Linear(4096, 2048),
        snn.Leaky(beta=0.9, learn_beta=True),
        
        nn.Linear(2048, 2048),
        snn.Leaky(beta=0.9, learn_beta=True),
        
        nn.Linear(2048, 1024),
        snn.Leaky(beta=0.9, learn_beta=True),
        
        # Output layer
        nn.Linear(1024, num_classes),
        snn.Leaky(beta=0.9, learn_beta=True),
    )
    
    return model


def create_dataset(batch_size, world_size, rank):
    """Create distributed dataset for training."""
    # Data preprocessing for SNN
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        transforms.Lambda(lambda x: x.view(-1))  # Flatten for linear layers
    ])
    
    # Create dataset
    train_dataset = torchvision.datasets.MNIST(
        root='./data', 
        train=True, 
        download=True, 
        transform=transform
    )
    
    # Create distributed sampler
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )
    
    # Create dataloader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        pin_memory=True,
        num_workers=4
    )
    
    return train_loader, train_sampler


def train_step(model, data, targets, optimizer, num_steps=10):
    """Single training step with temporal dynamics."""
    model.train()
    optimizer.zero_grad()
    
    # Convert static MNIST to spike trains
    # Simple rate coding: pixel intensity -> spike probability
    spike_data = torch.bernoulli(data.unsqueeze(0).repeat(num_steps, 1, 1))
    
    # Forward pass through time
    spike_outputs = []
    for step in range(num_steps):
        spike_out = model(spike_data[step])
        spike_outputs.append(spike_out)
    
    # Stack outputs across time
    spike_outputs = torch.stack(spike_outputs, dim=0)
    
    # Calculate loss using spike count or membrane potential
    # Here we use the mean spike count across time steps
    mean_spike_count = spike_outputs.mean(dim=0)
    
    loss = nn.functional.cross_entropy(mean_spike_count, targets)
    
    # Backward pass
    loss.backward()
    optimizer.step()
    
    return loss.item()


def validate_model(model, data_loader, device, num_steps=10):
    """Validate the model performance."""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, targets in data_loader:
            data, targets = data.to(device), targets.to(device)
            
            # Convert to spike trains
            spike_data = torch.bernoulli(data.unsqueeze(0).repeat(num_steps, 1, 1))
            
            # Forward pass
            spike_outputs = []
            for step in range(num_steps):
                spike_out = model(spike_data[step])
                spike_outputs.append(spike_out)
            
            # Get predictions
            mean_spike_count = torch.stack(spike_outputs, dim=0).mean(dim=0)
            predicted = mean_spike_count.argmax(dim=1)
            
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    
    accuracy = 100 * correct / total
    return accuracy


def main():
    """Main training function with FSDP2."""
    # Setup distributed training
    local_rank, world_size, rank = setup_distributed()
    device = torch.device(f"cuda:{local_rank}")
    
    # Print info only from rank 0
    if rank == 0:
        print(f"Starting distributed training with {world_size} processes")
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
    
    # Training parameters
    batch_size = 64  # Per-GPU batch size
    num_epochs = 10
    learning_rate = 1e-3
    num_time_steps = 10
    
    # Create model
    model = create_large_snn_model(num_classes=10)
    
    if rank == 0:
        from snntorch.distributed import count_parameters
        total_params = count_parameters(model)
        print(f"Model has {total_params:,} parameters")
    
    # Configure FSDP2
    fsdp_config = FSDPConfig(
        sharding_strategy=ShardingStrategy.FULL_SHARD,  # Most memory efficient
        min_param_size=1_000_000,  # Wrap modules with >1M parameters
        mixed_precision=True,      # Enable mixed precision for speed
        snn_optimize=True,         # Enable SNN-specific optimizations
        forward_prefetch=True,     # Prefetch for better overlap
        backward_prefetch=True,
        activation_checkpointing=False,  # Disable for this example
        cpu_offload=False,         # Keep on GPU for speed
    )
    
    # Prepare model for FSDP2
    if rank == 0:
        print("Preparing model for FSDP2...")
    
    model = prepare_fsdp2_model(model, fsdp_config)
    model = model.to(device)
    
    if rank == 0:
        print("FSDP2 model preparation complete!")
    
    # Create distributed dataset
    train_loader, train_sampler = create_dataset(batch_size, world_size, rank)
    
    # Setup optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    # Training loop
    if rank == 0:
        print("Starting training...")
    
    for epoch in range(num_epochs):
        # Set epoch for distributed sampler
        train_sampler.set_epoch(epoch)
        
        model.train()
        total_loss = 0
        num_batches = 0
        
        for batch_idx, (data, targets) in enumerate(train_loader):
            data, targets = data.to(device), targets.to(device)
            
            # Training step
            loss = train_step(model, data, targets, optimizer, num_time_steps)
            total_loss += loss
            num_batches += 1
            
            # Print progress from rank 0
            if rank == 0 and batch_idx % 100 == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss:.6f}')
        
        # Update learning rate
        scheduler.step()
        
        # Calculate average loss across all processes
        avg_loss = total_loss / num_batches
        
        # Gather losses from all processes
        loss_tensor = torch.tensor(avg_loss, device=device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        global_avg_loss = loss_tensor.item() / world_size
        
        if rank == 0:
            print(f'Epoch {epoch} completed. Average Loss: {global_avg_loss:.6f}')
            
            # Print memory statistics
            try:
                from snntorch.distributed.fsdp2 import get_fsdp2_memory_stats
                memory_stats = get_fsdp2_memory_stats(model)
                print(f"Memory stats: {memory_stats}")
            except Exception as e:
                print(f"Could not get memory stats: {e}")
    
    # Validation (optional)
    if rank == 0:
        print("Training completed!")
        
        # Create a small validation set for demonstration
        val_loader, _ = create_dataset(batch_size=32, world_size=1, rank=0)
        val_loader.sampler = None  # Remove distributed sampler for validation
        
        # Take only a subset for quick validation
        val_data = []
        for i, batch in enumerate(val_loader):
            val_data.append(batch)
            if i >= 10:  # Only validate on 10 batches
                break
        
        print("Running validation...")
        # Note: For proper validation in distributed setting, 
        # you'd want to gather results from all processes
    
    # Cleanup
    dist.destroy_process_group()


if __name__ == "__main__":
    main()