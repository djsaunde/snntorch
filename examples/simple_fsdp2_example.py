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
import matplotlib.pyplot as plt
import matplotlib
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import argparse

matplotlib.use("Agg")

import snntorch as snn
import snntorch.spikeplot as splt
import snntorch.spikegen as spikegen
from snntorch import surrogate
from snntorch.distributed import setup_distributed, quick_fsdp_setup
from snntorch.trainer import (
    DistributedSNNTrainer,
    create_distributed_dataloader,
)


def parse_args():
    """Parse command line arguments for configurable training parameters."""
    parser = argparse.ArgumentParser(
        description="FSDP2 SNN Training with MNIST"
    )

    # Model parameters
    parser.add_argument(
        "--num-inputs",
        type=int,
        default=784,
        help="Number of input features (default: 784)",
    )
    parser.add_argument(
        "--num-hidden",
        type=int,
        default=256,
        help="Number of hidden units (default: 256)",
    )
    parser.add_argument(
        "--num-outputs",
        type=int,
        default=10,
        help="Number of output classes (default: 10)",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.9,
        help="Leaky neuron decay rate (default: 0.9)",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=50,
        help="Number of time steps (default: 50)",
    )

    # Training parameters
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs (default: 3)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size per GPU (default: 32)",
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3, help="Learning rate (default: 1e-3)"
    )

    # Training options
    parser.add_argument(
        "--mixed-precision",
        action="store_true",
        help="Enable mixed precision training",
    )
    parser.add_argument(
        "--clip-grad-norm",
        type=float,
        default=1.0,
        help="Gradient clipping norm (default: 1.0)",
    )

    # Output options
    parser.add_argument(
        "--plot-prefix",
        type=str,
        default="fsdp2",
        help="Prefix for output plot files (default: fsdp2)",
    )
    parser.add_argument(
        "--disable-plots",
        action="store_true",
        help="Disable generation of visualization plots",
    )

    return parser.parse_args()


class SNNModel(nn.Module):
    """Simple SNN model with proper time stepping and state management."""

    def __init__(
        self,
        num_inputs=784,
        num_hidden=256,
        num_outputs=10,
        beta=0.9,
        num_steps=50,
    ):
        super().__init__()

        self.num_steps = num_steps

        # Initialize layers
        # self.fc1 = nn.Linear(num_inputs, num_hidden)
        # self.lif1 = snn.Leaky(
        #     beta=beta,
        #     spike_grad=surrogate.fast_sigmoid(),
        #     reset_mechanism="zero",
        # )
        # self.fc2 = nn.Linear(num_hidden, num_hidden // 2)
        # self.lif2 = snn.Leaky(
        #     beta=beta,
        #     spike_grad=surrogate.fast_sigmoid(),
        #     reset_mechanism="zero",
        # )
        # self.fc3 = nn.Linear(num_hidden // 2, num_outputs)
        self.fc3 = nn.Linear(num_inputs, num_outputs)
        self.lif3 = snn.Leaky(
            beta=beta,
            spike_grad=surrogate.fast_sigmoid(),
            output=True,
            reset_mechanism="zero",
        )

    def forward(self, x):
        """
        Forward pass handling a single time step.
        Called by the trainer for each time step.
        """
        # Layer 1
        # cur1 = self.fc1(x)
        # spk1, mem1 = self.lif1(cur1)

        # # Layer 2
        # cur2 = self.fc2(spk1)
        # spk2, mem2 = self.lif2(cur2)

        # Layer 3 (output)
        # cur3 = self.fc3(spk2)
        cur3 = self.fc3(x)
        spk3, mem3 = self.lif3(cur3)

        return spk3, mem3


def create_simple_snn(args):
    """Create a simple SNN model."""
    return SNNModel(
        num_inputs=args.num_inputs,
        num_hidden=args.num_hidden,
        num_outputs=args.num_outputs,
        beta=args.beta,
        num_steps=args.num_steps,
    )


class MNISTSpikeDataset(Dataset):
    """MNIST dataset that converts images to spike trains on-the-fly."""
    
    def __init__(self, num_steps=50, train=True, root="./data"):
        """
        Args:
            num_steps: Number of time steps for spike encoding
            train: Whether to use training or test set
            root: Root directory for MNIST data
        """
        self.num_steps = num_steps
        
        # Download MNIST dataset - don't normalize for spike encoding
        transform = transforms.Compose(
            [transforms.ToTensor()]  # Keep raw pixel values in [0,1] range
        )
        
        # Only rank 0 downloads, others wait
        if dist.get_rank() == 0:
            self.mnist_dataset = torchvision.datasets.MNIST(
                root=root, train=train, download=True, transform=transform
            )
        
        # Synchronize all processes
        dist.barrier()
        
        # Now all processes can load the dataset
        if dist.get_rank() != 0:
            self.mnist_dataset = torchvision.datasets.MNIST(
                root=root, train=train, download=False, transform=transform
            )
    
    def __len__(self):
        return len(self.mnist_dataset)
    
    def __getitem__(self, idx):
        """Get item and convert to spike train on-the-fly."""
        image, label = self.mnist_dataset[idx]
        
        # Flatten image from (1, 28, 28) to (784,)
        image_flat = image.flatten()
        
        # Convert pixel intensities to spike trains using rate encoding
        # Pixel values are already in [0,1] range from ToTensor()
        pixel_rates = image_flat  # Use raw pixel intensities as spike rates
        
        # Generate spike trains for each pixel across time steps
        spike_train = spikegen.rate(pixel_rates, num_steps=self.num_steps)
        
        # Debug: print spike statistics for first few samples
        if idx < 3 and dist.get_rank() == 0:
            total_spikes = spike_train.sum().item()
            max_rate = pixel_rates.max().item()
            min_rate = pixel_rates.min().item()
            print(
                f"Sample {idx}: {total_spikes} total spikes, pixel rates: [{min_rate:.3f}, {max_rate:.3f}]"
            )
        
        return spike_train, label  # Shape: (num_steps, 784), int


def create_mnist_spike_data(num_steps=50):
    """Create MNIST spike dataset with on-the-fly conversion."""
    return MNISTSpikeDataset(num_steps=num_steps, train=True)


def main():
    """Main function demonstrating FSDP2 usage."""

    # Parse command line arguments
    args = parse_args()

    # Setup distributed training
    local_rank = setup_distributed()
    device = torch.device(f"cuda:{local_rank}")

    if dist.get_rank() == 0:
        print(f"Starting FSDP2 SNN training with arguments:")
        print(
            f"  Model: {args.num_inputs}->{args.num_hidden}->{args.num_outputs}"
        )
        print(f"  Time steps: {args.num_steps}, Beta: {args.beta}")
        print(
            f"  Epochs: {args.epochs}, Batch size: {args.batch_size}, LR: {args.lr}"
        )
        print(
            f"  Mixed precision: {args.mixed_precision}"
        )

    # Create model
    model = create_simple_snn(args)

    # Apply FSDP2 with automatic strategy selection
    model = quick_fsdp_setup(model)
    model = model.to(device)

    if dist.get_rank() == 0:
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Model has {total_params:,} parameters")

    # Create MNIST spike data
    if dist.get_rank() == 0:
        print("Creating MNIST spike dataset...")
    dataset = create_mnist_spike_data(num_steps=args.num_steps)
    dataloader = create_distributed_dataloader(
        dataset, batch_size=args.batch_size
    )

    # Setup training components
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    # Create trainer
    trainer = DistributedSNNTrainer(
        model=model,
        device=device,
        num_steps=args.num_steps,
        mixed_precision=args.mixed_precision,
        clip_grad_norm=args.clip_grad_norm,
    )
    # Simple training loop
    epoch_losses = []  # Track loss for plotting
    step_losses = []  # Track step-wise losses
    step_accuracies = []  # Track step-wise accuracies
    spike_data_recorded = None  # Store spike data for plotting
    mem_data_recorded = None  # Store membrane voltage data for plotting
    input_data_recorded = None  # Store input data for plotting

    for epoch in range(args.epochs):
        dataloader.sampler.set_epoch(
            epoch
        )  # Important for distributed training

        if dist.get_rank() == 0:
            print(f"\nEpoch {epoch + 1}/{args.epochs}")

        # Train for one epoch with custom loop to handle data format
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        num_batches = 0

        for batch_idx, (data, targets) in enumerate(dataloader):
            # Move to device and transpose data to (time, batch, features)
            data = data.to(device).transpose(0, 1)
            targets = targets.to(device)

            # Training step
            step_stats = trainer.train_step(data, targets, loss_fn, loss_mode="sum_spikes")
            
            # Manual optimizer step since trainer doesn't handle it
            optimizer.step()
            optimizer.zero_grad()
            step_loss = step_stats["loss"]
            total_loss += step_loss
            num_batches += 1
            
            # Calculate accuracy for this step
            # Get output spikes and sum over time to get firing rates
            with torch.no_grad():
                # Forward pass to get predictions (reuse trainer's last forward pass)
                if trainer.spike_recordings:
                    output_spikes = torch.stack(trainer.spike_recordings)  # (time, batch, classes)
                    spike_counts = torch.sum(output_spikes, dim=0)  # Sum over time: (batch, classes)
                    predicted = torch.argmax(spike_counts, dim=1)  # Get class with most spikes
                    correct = (predicted == targets).sum().item()
                    step_accuracy = correct / targets.size(0)
                    
                    total_correct += correct
                    total_samples += targets.size(0)
                    
                    # Store step-wise metrics (only on rank 0)
                    if dist.get_rank() == 0:
                        step_losses.append(step_loss)
                        step_accuracies.append(step_accuracy)

            # Record spike data from first batch of first epoch for plotting
            if epoch == 0 and batch_idx == 0 and spike_data_recorded is None:
                # The trainer already has the spike recordings from the last forward pass
                spike_data_recorded = (
                    torch.stack(trainer.spike_recordings)
                    .clone()
                    .detach()
                    .cpu()
                )
                # Record membrane voltage data if available
                if trainer.mem_recordings:
                    mem_data_recorded = (
                        torch.stack(trainer.mem_recordings)
                        .clone()
                        .detach()
                        .cpu()
                    )
                # Also record input data for comparison
                input_data_recorded = data.clone().detach().cpu()

            if batch_idx % 50 == 0 and dist.get_rank() == 0:
                if trainer.spike_recordings:
                    # Debug: check spike saturation
                    output_spikes = torch.stack(trainer.spike_recordings)
                    total_output_spikes = output_spikes.sum().item()
                    max_possible_spikes = output_spikes.numel()
                    spike_saturation = total_output_spikes / max_possible_spikes
                    print(f"Batch {batch_idx}, Loss: {step_loss:.4f}, Accuracy: {step_accuracy:.4f}, Spike saturation: {spike_saturation:.3f}")
                else:
                    print(f"Batch {batch_idx}, Loss: {step_loss:.4f}")

        # Calculate epoch metrics
        epoch_loss = total_loss / num_batches if num_batches > 0 else 0.0
        epoch_accuracy = total_correct / total_samples if total_samples > 0 else 0.0
        
        train_stats = {
            "loss": epoch_loss,
            "accuracy": epoch_accuracy
        }
        epoch_losses.append(train_stats["loss"])

        # Print results on rank 0
        if dist.get_rank() == 0:
            print(f"Average loss: {train_stats['loss']:.4f}, Accuracy: {train_stats['accuracy']:.4f}")

    if dist.get_rank() == 0:
        print("\nTraining completed successfully!")

        if not args.disable_plots:
            # Create and save loss plot
            plt.figure(figsize=(10, 6))
            plt.plot(
                range(1, args.epochs + 1),
                epoch_losses,
                "b-",
                marker="o",
                linewidth=2,
            )
            plt.title("Training Loss Over Epochs", fontsize=16)
            plt.xlabel("Epoch", fontsize=14)
            plt.ylabel("Average Loss", fontsize=14)
            plt.grid(True, alpha=0.3)
            plt.xticks(range(1, args.epochs + 1))

            # Add loss values as text annotations
            for i, loss in enumerate(epoch_losses):
                plt.annotate(
                    f"{loss:.4f}",
                    (i + 1, loss),
                    textcoords="offset points",
                    xytext=(0, 10),
                    ha="center",
                )

            plt.tight_layout()
            loss_plot_name = f"{args.plot_prefix}_training_loss.png"
            plt.savefig(loss_plot_name, dpi=300, bbox_inches="tight")
            print(f"Training loss plot saved as '{loss_plot_name}'")
            plt.close()
            
            # Create step-wise metrics plots if we have step data
            if step_losses and step_accuracies:
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
                
                # Step-wise loss plot
                ax1.plot(range(len(step_losses)), step_losses, 'b-', linewidth=1, alpha=0.7)
                ax1.set_title('Step-wise Training Loss', fontsize=16)
                ax1.set_xlabel('Training Step', fontsize=14)
                ax1.set_ylabel('Loss', fontsize=14)
                ax1.grid(True, alpha=0.3)
                
                # Step-wise accuracy plot
                ax2.plot(range(len(step_accuracies)), step_accuracies, 'g-', linewidth=1, alpha=0.7)
                ax2.set_title('Step-wise Training Accuracy', fontsize=16)
                ax2.set_xlabel('Training Step', fontsize=14)
                ax2.set_ylabel('Accuracy', fontsize=14)
                ax2.set_ylim(0, 1)
                ax2.grid(True, alpha=0.3)
                
                plt.tight_layout()
                step_metrics_name = f"{args.plot_prefix}_step_metrics.png"
                plt.savefig(step_metrics_name, dpi=300, bbox_inches="tight")
                print(f"Step-wise metrics plot saved as '{step_metrics_name}'")
                plt.close()

        # Create spike plots if we have recorded data
        if (
            not args.disable_plots
            and spike_data_recorded is not None
            and input_data_recorded is not None
        ):
            print(f"Spike data shape: {spike_data_recorded.shape}")
            print(f"Input data shape: {input_data_recorded.shape}")
            if mem_data_recorded is not None:
                print(
                    f"Membrane voltage data shape: {mem_data_recorded.shape}"
                )

            # Plot 1: Input vs Output spike comparison
            _, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))

            # Input spikes (first sample)
            input_spikes = input_data_recorded[:, 0]
            splt.raster(input_spikes, ax1, s=5, c="blue")
            ax1.set_title(
                "Input Layer Spike Raster",
                fontsize=14,
            )
            ax1.set_xlabel("Time Step")
            ax1.set_ylabel("Input Neuron Index")
            ax1.grid(True, alpha=0.3)

            # Output layer spikes (last layer) - show all 10 output neurons
            output_spikes = spike_data_recorded[
                :, 0, :
            ]  # (time_steps, all_output_neurons)
            if output_spikes.sum() > 0:  # Only plot if there are spikes
                splt.raster(output_spikes, ax2, s=20, c="red")
            ax2.set_title(
                "Output Layer Spike Raster (All 10 Output Neurons)",
                fontsize=14,
            )
            ax2.set_xlabel("Time Step")
            ax2.set_ylabel("Output Neuron Index")
            ax2.grid(True, alpha=0.3)

            # Plot spike counts over time
            input_spike_counts = torch.sum(
                input_spikes, dim=1
            )  # Sum over input neurons per time step
            output_spike_counts = torch.sum(
                output_spikes, dim=1
            )  # Sum over output neurons per time step

            ax3.plot(
                range(len(input_spike_counts)),
                input_spike_counts.numpy(),
                "b-",
                linewidth=2,
                label="Input Spikes",
            )
            ax3.plot(
                range(len(output_spike_counts)),
                output_spike_counts.numpy(),
                "r-",
                linewidth=2,
                label="Output Spikes",
            )
            ax3.set_title("Spike Count Over Time", fontsize=14)
            ax3.set_xlabel("Time Step")
            ax3.set_ylabel("Total Spike Count")
            ax3.legend()
            ax3.grid(True, alpha=0.3)

            plt.tight_layout()
            spike_analysis_name = f"{args.plot_prefix}_spike_analysis.png"
            plt.savefig(spike_analysis_name, dpi=300, bbox_inches="tight")
            print(f"Spike analysis plot saved as '{spike_analysis_name}'")
            plt.close()

            # Plot 3: Individual neuron spike trains
            fig, axes = plt.subplots(2, 3, figsize=(15, 8))
            axes = axes.flatten()

            for i in range(
                min(6, spike_data_recorded.shape[2])
            ):  # Plot first 6 neurons
                neuron_spikes = spike_data_recorded[
                    :, 0, i
                ]  # Shape: (time_steps,)
                spike_times = torch.where(neuron_spikes)[0].numpy()

                # Create spike train visualization
                if len(spike_times) > 0:
                    axes[i].scatter(
                        spike_times,
                        [1] * len(spike_times),
                        s=30,
                        c="red",
                        marker="|",
                    )
                axes[i].set_xlim(0, len(neuron_spikes))
                axes[i].set_ylim(0.5, 1.5)
                axes[i].set_title(f"Neuron {i} Spike Train")
                axes[i].set_xlabel("Time Step")
                axes[i].set_yticks([])
                axes[i].grid(True, alpha=0.3)

            plt.tight_layout()
            individual_neurons_name = (
                f"{args.plot_prefix}_individual_neurons.png"
            )
            plt.savefig(individual_neurons_name, dpi=300, bbox_inches="tight")
            print(
                f"Individual neuron plots saved as '{individual_neurons_name}'"
            )
            plt.close()

            # Plot 4: Membrane voltage analysis if available
            if mem_data_recorded is not None:
                # Voltage overview plot
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

                # Plot voltage traces for first few output neurons
                for i in range(min(5, mem_data_recorded.shape[2])):
                    voltage_trace = mem_data_recorded[
                        :, 0, i
                    ].numpy()  # (time_steps,)
                    ax1.plot(
                        range(len(voltage_trace)),
                        voltage_trace,
                        linewidth=2,
                        label=f"Neuron {i}",
                    )

                ax1.set_title(
                    "Membrane Voltage Traces (First 5 Output Neurons)",
                    fontsize=14,
                )
                ax1.set_xlabel("Time Step")
                ax1.set_ylabel("Membrane Voltage")
                ax1.legend()
                ax1.grid(True, alpha=0.3)

                # Plot average voltage across all neurons
                avg_voltage = torch.mean(
                    mem_data_recorded[:, 0, :], dim=1
                ).numpy()
                ax2.plot(
                    range(len(avg_voltage)),
                    avg_voltage,
                    "k-",
                    linewidth=2,
                    label="Average",
                )
                ax2.set_title(
                    "Average Membrane Voltage (All Output Neurons)",
                    fontsize=14,
                )
                ax2.set_xlabel("Time Step")
                ax2.set_ylabel("Average Membrane Voltage")
                ax2.grid(True, alpha=0.3)

                plt.tight_layout()
                voltage_analysis_name = (
                    f"{args.plot_prefix}_voltage_analysis.png"
                )
                plt.savefig(
                    voltage_analysis_name, dpi=300, bbox_inches="tight"
                )
                print(
                    f"Voltage analysis plot saved as '{voltage_analysis_name}'"
                )
                plt.close()

                # Plot 5: Combined spike-voltage plots for individual neurons
                fig, axes = plt.subplots(3, 2, figsize=(15, 10))

                for i in range(min(6, mem_data_recorded.shape[2])):
                    row = i // 2
                    col = i % 2
                    ax = axes[row, col]

                    # Plot voltage trace
                    voltage_trace = mem_data_recorded[:, 0, i].numpy()
                    time_steps = range(len(voltage_trace))
                    ax.plot(
                        time_steps,
                        voltage_trace,
                        "b-",
                        linewidth=2,
                        label="Membrane Voltage",
                    )

                    # Overlay spikes
                    neuron_spikes = spike_data_recorded[:, 0, i]
                    spike_times = torch.where(neuron_spikes)[0].numpy()
                    if len(spike_times) > 0:
                        spike_voltages = voltage_trace[spike_times]
                        ax.scatter(
                            spike_times,
                            spike_voltages,
                            color="red",
                            s=50,
                            zorder=5,
                            label="Spikes",
                        )

                    ax.set_title(f"Neuron {i}: Voltage & Spikes")
                    ax.set_xlabel("Time Step")
                    ax.set_ylabel("Membrane Voltage")
                    ax.grid(True, alpha=0.3)
                    ax.legend()

                plt.tight_layout()
                spike_voltage_combined_name = (
                    f"{args.plot_prefix}_spike_voltage_combined.png"
                )
                plt.savefig(
                    spike_voltage_combined_name, dpi=300, bbox_inches="tight"
                )
                print(
                    f"Combined spike-voltage plots saved as '{spike_voltage_combined_name}'"
                )
                plt.close()
            else:
                print(
                    "No membrane voltage data recorded (may need output=True in neuron layers)"
                )

            # Plot 6: Show original MNIST digit vs its spike representation
            if input_data_recorded is not None:
                fig, axes = plt.subplots(2, 3, figsize=(12, 8))

                for i in range(3):
                    # Original MNIST digit (reconstruct from spikes by averaging over time)
                    avg_spikes = torch.mean(
                        input_data_recorded[:, i, :], dim=0
                    )  # Average over time
                    digit_image = avg_spikes.reshape(28, 28).numpy()

                    axes[0, i].imshow(digit_image, cmap="gray")
                    axes[0, i].set_title(
                        f"Sample {i}: Reconstructed from Spikes"
                    )
                    axes[0, i].axis("off")

                    # Spike raster for this digit (show subset of pixels)
                    pixel_subset = input_data_recorded[:, i]
                    splt.raster(
                        pixel_subset.T, axes[1, i], s=2, c="blue"
                    )  # Transpose for raster plot
                    axes[1, i].set_title(f"Sample {i}: Spike Raster")
                    axes[1, i].set_xlabel("Time Step")
                    axes[1, i].set_ylabel("Pixel Index")

                plt.tight_layout()
                mnist_digit_spikes_name = (
                    f"{args.plot_prefix}_mnist_digit_spikes.png"
                )
                plt.savefig(
                    mnist_digit_spikes_name, dpi=300, bbox_inches="tight"
                )
                print(
                    f"MNIST digit vs spike visualization saved as '{mnist_digit_spikes_name}'"
                )
                plt.close()

    # Cleanup
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
