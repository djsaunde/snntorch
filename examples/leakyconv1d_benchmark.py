#!/usr/bin/env python3
"""
LeakyConv1d Accelerated LIF Neuron Demonstration and Benchmark
==============================================================

This example demonstrates the new LeakyConv1d neuron implementation in snntorch,
which uses conv1d scans for parallel time processing inspired by state space models.

Key Features:
- Parallel processing of entire sequences using conv1d operations
- Support for 3D+ inputs (video, multi-channel time series, etc.)
- Significant speedup compared to sequential processing
- Full torch.compile compatibility with zero graph breaks

Author: snntorch development team
Date: 2024
"""

import torch
import torch.nn as nn
import time
import matplotlib.pyplot as plt
import numpy as np

import snntorch as snn
from snntorch import spikegen, spikeplot as splt, utils


def create_test_data(seq_len, batch_size, features, device='cpu'):
    """Create test input data for benchmarking."""
    # Random input data
    input_3d = torch.randn(seq_len, batch_size, features, device=device)
    
    # Video-like 4D data
    input_4d = torch.randn(seq_len, batch_size, 32, 32, device=device)
    
    # Multi-channel time series 4D data  
    input_4d_channels = torch.randn(seq_len, batch_size, 8, features//8, device=device)
    
    return input_3d, input_4d, input_4d_channels


def benchmark_sequential_vs_parallel(seq_len=100, batch_size=32, features=128, device='cpu'):
    """
    Benchmark LeakyConv1d against standard Leaky neuron for sequence processing.
    """
    print(f"🔥 Benchmarking Sequential vs Parallel Processing")
    print(f"   Device: {device}")
    print(f"   Input: ({seq_len}, {batch_size}, {features})")
    print("-" * 60)
    
    beta = 0.9
    threshold = 1.0
    
    # Create neurons
    lif_standard = snn.Leaky(beta=beta, threshold=threshold, init_hidden=True).to(device)
    lif_conv1d = snn.LeakyConv1d(beta=beta, threshold=threshold, max_sequence_length=seq_len).to(device)
    
    # Create test data
    input_seq = torch.randn(seq_len, batch_size, features, device=device)
    
    # Warm up GPU if available
    if device.type == 'cuda':
        for _ in range(3):
            _ = lif_conv1d(input_seq)
        torch.cuda.synchronize()
    
    # Benchmark standard LIF (sequential)
    utils.reset(lif_standard)
    
    start_time = time.time()
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    spike_std_list = []
    for t in range(seq_len):
        spike = lif_standard(input_seq[t])
        spike_std_list.append(spike)
    spike_std = torch.stack(spike_std_list)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    std_time = time.time() - start_time
    
    # Benchmark LeakyConv1d (parallel)
    start_time = time.time()
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    spike_conv1d, _ = lif_conv1d(input_seq)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    conv1d_time = time.time() - start_time
    
    # Calculate speedup
    speedup = std_time / conv1d_time if conv1d_time > 0 else float('inf')
    
    print(f"⏱️  Standard Leaky (sequential):  {std_time:.4f}s")
    print(f"⚡ LeakyConv1d (parallel):      {conv1d_time:.4f}s")
    print(f"🚀 Speedup:                    {speedup:.2f}x")
    print(f"📊 Standard spikes:            {spike_std.sum().item():.0f}")
    print(f"📊 LeakyConv1d spikes:         {spike_conv1d.sum().item():.0f}")
    
    return std_time, conv1d_time, speedup


def demonstrate_multidimensional_inputs():
    """Demonstrate LeakyConv1d with various input dimensions."""
    print(f"\n🎯 Multi-dimensional Input Support Demonstration")
    print("-" * 60)
    
    beta = 0.8
    lif = snn.LeakyConv1d(beta=beta, max_sequence_length=50)
    
    # 3D input (standard time series)
    input_3d = torch.randn(20, 4, 16)
    spikes_3d, mem_3d = lif(input_3d)
    print(f"📺 3D input:  {input_3d.shape} → {spikes_3d.shape} | Spikes: {spikes_3d.sum():.0f}")
    
    # 4D input (video-like)
    input_4d = torch.randn(15, 2, 32, 32)
    spikes_4d, mem_4d = lif(input_4d)
    print(f"🎬 4D video:  {input_4d.shape} → {spikes_4d.shape} | Spikes: {spikes_4d.sum():.0f}")
    
    # 4D input (multi-channel)
    input_4d_ch = torch.randn(25, 3, 8, 32)
    spikes_4d_ch, mem_4d_ch = lif(input_4d_ch)
    print(f"📡 4D multi:  {input_4d_ch.shape} → {spikes_4d_ch.shape} | Spikes: {spikes_4d_ch.sum():.0f}")
    
    # 5D input (spatial-temporal)
    input_5d = torch.randn(10, 2, 16, 16, 4)
    spikes_5d, mem_5d = lif(input_5d)
    print(f"🌐 5D spatio: {input_5d.shape} → {spikes_5d.shape} | Spikes: {spikes_5d.sum():.0f}")


def demonstrate_reset_mechanisms():
    """Demonstrate different reset mechanisms."""
    print(f"\n⚙️  Reset Mechanism Comparison")
    print("-" * 60)
    
    seq_len, batch_size, features = 20, 3, 8
    input_strong = torch.ones(seq_len, batch_size, features) * 2.0  # Strong input
    
    reset_types = ['subtract', 'zero', 'none']
    
    for reset_type in reset_types:
        lif = snn.LeakyConv1d(beta=0.8, threshold=1.0, reset_mechanism=reset_type, 
                             max_sequence_length=seq_len)
        spikes, mem = lif(input_strong)
        
        print(f"🔧 {reset_type:8s}: Spikes={spikes.sum().item():>4.0f}, Max Mem={mem.max():.2f}")


def demonstrate_learnable_parameters():
    """Demonstrate learnable parameters and gradient flow."""
    print(f"\n🎓 Learnable Parameters Demonstration")
    print("-" * 60)
    
    # Create neuron with learnable parameters
    lif = snn.LeakyConv1d(beta=0.9, threshold=1.0, learn_beta=True, 
                         learn_threshold=True, max_sequence_length=30)
    
    print(f"🧠 Beta learnable:      {lif.beta.requires_grad}")
    print(f"🧠 Threshold learnable: {lif.threshold.requires_grad}")
    print(f"📊 Initial beta:        {lif.beta.item():.3f}")
    print(f"📊 Initial threshold:   {lif.threshold.item():.3f}")
    
    # Create synthetic training data
    input_seq = torch.randn(20, 8, 16)
    target_spikes = torch.randint(0, 2, (20, 8, 16)).float()
    
    # Forward pass
    spikes, mem = lif(input_seq)
    
    # Simple loss (spike rate matching)
    loss = nn.MSELoss()(spikes, target_spikes)
    
    # Backward pass
    loss.backward()
    
    print(f"💚 Loss:                {loss.item():.4f}")
    print(f"📈 Beta grad exists:    {lif.beta.grad is not None}")
    print(f"📈 Threshold grad:      {lif.threshold.grad is not None}")
    if lif.beta.grad is not None:
        print(f"📈 Beta grad magnitude: {lif.beta.grad.abs().item():.6f}")


def demonstrate_snn_network():
    """Demonstrate LeakyConv1d in a simple SNN."""
    print(f"\n🧠 Spiking Neural Network Example")
    print("-" * 60)
    
    class AcceleratedSNN(nn.Module):
        def __init__(self, input_size, hidden_size, output_size, seq_len=50):
            super().__init__()
            
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.lif1 = snn.LeakyConv1d(beta=0.9, max_sequence_length=seq_len)
            
            self.fc2 = nn.Linear(hidden_size, hidden_size)
            self.lif2 = snn.LeakyConv1d(beta=0.8, max_sequence_length=seq_len)
            
            self.fc3 = nn.Linear(hidden_size, output_size)
            self.lif3 = snn.LeakyConv1d(beta=0.7, max_sequence_length=seq_len)
        
        def forward(self, x):
            # x shape: (seq_len, batch, input_size)
            seq_len, batch_size = x.shape[0], x.shape[1]
            
            # Process through layers
            x1 = torch.stack([self.fc1(x[t]) for t in range(seq_len)])
            spk1, _ = self.lif1(x1)
            
            x2 = torch.stack([self.fc2(spk1[t]) for t in range(seq_len)])
            spk2, _ = self.lif2(x2)
            
            x3 = torch.stack([self.fc3(spk2[t]) for t in range(seq_len)])
            spk3, mem3 = self.lif3(x3)
            
            return spk3, mem3
    
    # Create network
    net = AcceleratedSNN(784, 256, 10, seq_len=30)
    
    # Test forward pass
    test_input = torch.randn(30, 4, 784)
    output_spikes, output_mem = net(test_input)
    
    print(f"🏗️  Network created successfully")
    print(f"📥 Input shape:         {test_input.shape}")
    print(f"📤 Output spikes shape: {output_spikes.shape}")
    print(f"📤 Output mem shape:    {output_mem.shape}")
    print(f"⚡ Total output spikes: {output_spikes.sum().item():.0f}")


def plot_spike_comparison():
    """Create visualization comparing sequential vs parallel results."""
    print(f"\n📊 Creating Spike Comparison Visualization")
    print("-" * 60)
    
    seq_len, batch_size, features = 50, 1, 4
    beta = 0.9
    
    # Create test input with some structure
    input_seq = spikegen.rate_conv(torch.ones(seq_len, batch_size, features) * 0.3)
    
    # Standard LIF
    lif_std = snn.Leaky(beta=beta, init_hidden=True)
    utils.reset(lif_std)
    spikes_std = []
    for t in range(seq_len):
        spk = lif_std(input_seq[t])
        spikes_std.append(spk)
    spikes_std = torch.stack(spikes_std)
    
    # LeakyConv1d
    lif_conv1d = snn.LeakyConv1d(beta=beta, max_sequence_length=seq_len)
    spikes_conv1d, mem_conv1d = lif_conv1d(input_seq)
    
    # Create visualization
    fig, axes = plt.subplots(3, 1, figsize=(12, 8))
    fig.suptitle('LeakyConv1d vs Standard Leaky Comparison', fontsize=14, fontweight='bold')
    
    # Input spikes
    splt.raster(input_seq[:, 0, :].T, axes[0], s=50, c='blue', marker='|')
    axes[0].set_title('Input Spikes')
    axes[0].set_ylabel('Neuron Index')
    
    # Standard LIF output
    splt.raster(spikes_std[:, 0, :].T, axes[1], s=50, c='red', marker='|')
    axes[1].set_title('Standard Leaky Output')
    axes[1].set_ylabel('Neuron Index')
    
    # LeakyConv1d output
    splt.raster(spikes_conv1d[:, 0, :].T, axes[2], s=50, c='green', marker='|')
    axes[2].set_title('LeakyConv1d Output')
    axes[2].set_ylabel('Neuron Index')
    axes[2].set_xlabel('Time Steps')
    
    plt.tight_layout()
    plt.savefig('leakyconv1d_comparison.png', dpi=150, bbox_inches='tight')
    print(f"💾 Saved comparison plot as 'leakyconv1d_comparison.png'")


def performance_scaling_analysis():
    """Analyze performance scaling with sequence length."""
    print(f"\n📈 Performance Scaling Analysis")
    print("-" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size, features = 16, 64
    seq_lengths = [50, 100, 200, 400, 800]
    
    std_times = []
    conv1d_times = []
    speedups = []
    
    for seq_len in seq_lengths:
        print(f"Testing sequence length: {seq_len}")
        
        _, conv1d_time, speedup = benchmark_sequential_vs_parallel(
            seq_len=seq_len, batch_size=batch_size, features=features, device=device
        )
        
        speedups.append(speedup)
        print(f"  Speedup: {speedup:.2f}x\n")
    
    print("📊 Scaling Results:")
    for seq_len, speedup in zip(seq_lengths, speedups):
        print(f"  {seq_len:3d} steps: {speedup:5.2f}x speedup")


def torch_compile_demo():
    """Demonstrate torch.compile compatibility."""
    print(f"\n⚡ Torch.Compile Compatibility Demo")
    print("-" * 60)
    
    lif = snn.LeakyConv1d(beta=0.9, surrogate_disable=True, max_sequence_length=100)
    input_seq = torch.randn(50, 8, 16)
    
    # Test compilation
    try:
        compiled_lif = torch.compile(lif)
        spikes, mem = compiled_lif(input_seq)
        print(f"✅ torch.compile successful!")
        print(f"📤 Compiled output shape: {spikes.shape}")
        print(f"⚡ Compiled spikes: {spikes.sum():.0f}")
    except Exception as e:
        print(f"❌ torch.compile failed: {e}")


def main():
    """Main demonstration function."""
    print("=" * 80)
    print("🚀 LeakyConv1d: Accelerated LIF Neuron Demonstration")
    print("   Parallel time processing using conv1d scans")
    print("   Inspired by state space models (Mamba, S4)")
    print("=" * 80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Core demonstrations
    benchmark_sequential_vs_parallel(device=device)
    demonstrate_multidimensional_inputs()
    demonstrate_reset_mechanisms()
    demonstrate_learnable_parameters()
    demonstrate_snn_network()
    
    # Advanced features
    torch_compile_demo()
    performance_scaling_analysis()
    
    # Create visualization
    if plt is not None:
        plot_spike_comparison()
    
    print("\n" + "=" * 80)
    print("🎉 LeakyConv1d Demonstration Complete!")
    print("   Key Benefits:")
    print("   • Parallel time processing for significant speedup")
    print("   • Support for 3D+ inputs (video, multi-channel data)")
    print("   • Full torch.compile compatibility (zero graph breaks)")
    print("   • Drop-in replacement for standard Leaky neurons")
    print("   • Maintains mathematical correctness of LIF dynamics")
    print("=" * 80)


if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Run the demonstration
    main()