"""
Simple FSDP2 Example for SNNs

This is a minimal example showing how to use FSDP2 with snntorch.
Perfect for getting started with distributed SNN training.

Usage:
    torchrun --nproc_per_node=2 simple_fsdp2_example.py
"""

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import TensorDataset, DataLoader

import snntorch as snn
from snntorch import surrogate
from snntorch.distributed import setup_distributed, quick_fsdp_setup
from snntorch.trainer import DistributedSNNTrainer, create_distributed_dataloader


def create_simple_snn():
    """Create a simple SNN model."""
    spike_grad = surrogate.fast_sigmoid()
    
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 256),
        snn.Leaky(beta=0.9, spike_grad=spike_grad, init_hidden=True),
        nn.Linear(256, 128),
        snn.Leaky(beta=0.9, spike_grad=spike_grad, init_hidden=True),
        nn.Linear(128, 10),
        snn.Leaky(beta=0.9, spike_grad=spike_grad, init_hidden=True, output=True)
    )
    
    return model


def create_sample_data(batch_size=64, num_samples=1000, num_steps=25):
    """Create synthetic spike train data."""
    # Random spike trains: (num_samples, num_steps, 784)
    data = torch.rand(num_samples, num_steps, 784)
    data = (data > 0.8).float()  # 20% spike probability
    
    # Random labels
    targets = torch.randint(0, 10, (num_samples,))
    
    return TensorDataset(data, targets)


def main():
    """Main function demonstrating FSDP2 usage."""
    
    # Setup distributed training
    local_rank = setup_distributed()
    device = torch.device(f'cuda:{local_rank}')
    
    print(f"Rank {dist.get_rank()}: Starting simple FSDP2 example")
    
    # Create model
    model = create_simple_snn()
    
    # Apply FSDP2 with automatic strategy selection
    model = quick_fsdp_setup(model)
    model = model.to(device)
    
    if dist.get_rank() == 0:
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Model has {total_params:,} parameters")
    
    # Create data
    dataset = create_sample_data()
    dataloader = create_distributed_dataloader(dataset, batch_size=32)
    
    # Setup training components
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    
    # Create trainer
    trainer = DistributedSNNTrainer(
        model=model,
        device=device,
        num_steps=25  # Number of time steps
    )
    
    # Simple training loop
    num_epochs = 3
    
    for epoch in range(num_epochs):
        dataloader.sampler.set_epoch(epoch)  # Important for distributed training
        
        if dist.get_rank() == 0:
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
        
        # Train for one epoch
        train_stats = trainer.train_epoch(
            dataloader=dataloader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            log_interval=50
        )
        
        # Print results on rank 0
        if dist.get_rank() == 0:
            print(f"Average loss: {train_stats['loss']:.4f}")
            print(f"Average forward time: {train_stats['avg_forward_time']:.4f}s")
    
    if dist.get_rank() == 0:
        print("\nTraining completed successfully!")
    
    # Cleanup
    dist.destroy_process_group()


if __name__ == "__main__":
    main()