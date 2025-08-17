"""
Distributed Training Module for snntorch

This module provides trainer classes specifically designed for distributed
training of spiking neural networks using FSDP2. It handles the temporal
dynamics of SNNs while leveraging distributed training capabilities.
"""

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from snntorch import utils
from typing import Optional, Dict, Any, Callable, Tuple
import time
from contextlib import nullcontext


class DistributedSNNTrainer:
    """
    Trainer class for distributed spiking neural networks with FSDP2.

    This trainer handles:
    - Temporal dynamics across multiple time steps
    - Distributed training with proper gradient synchronization
    - Memory-efficient processing of spike trains
    - Comprehensive logging and monitoring
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        num_steps: int = 25,
        grad_accumulation_steps: int = 1,
        mixed_precision: bool = False,
        clip_grad_norm: Optional[float] = 1.0,
    ):
        """
        Initialize the distributed SNN trainer.

        Args:
            model: FSDP-wrapped SNN model
            device: Training device
            num_steps: Number of time steps for SNN simulation
            grad_accumulation_steps: Steps for gradient accumulation
            mixed_precision: Whether to use mixed precision training
            clip_grad_norm: Maximum norm for gradient clipping
        """
        self.model = model
        self.device = device
        self.num_steps = num_steps
        self.grad_accumulation_steps = grad_accumulation_steps
        self.clip_grad_norm = clip_grad_norm

        # Setup mixed precision
        self.mixed_precision = mixed_precision
        self.scaler = torch.cuda.amp.GradScaler() if mixed_precision else None

        # Tracking variables
        self.spike_recordings = []
        self.mem_recordings = []
        self.current_epoch = 0
        self.global_step = 0

        # Performance tracking
        self.training_stats = {
            "total_time": 0.0,
            "forward_time": 0.0,
            "backward_time": 0.0,
            "data_time": 0.0,
        }

    def reset_recordings(self):
        """Reset spike and membrane potential recordings."""
        self.spike_recordings = []
        self.mem_recordings = []

    def forward_pass(
        self, data: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform forward pass through time with distributed model.

        Args:
            data: Input data tensor with shape (time_steps, batch_size, ...)

        Returns:
            Tuple of spike recordings and membrane recordings
        """
        # Reset recordings and network states
        self.reset_recordings()
        utils.reset(self.model)

        # Context for mixed precision
        context = (
            torch.cuda.amp.autocast()
            if self.mixed_precision
            else nullcontext()
        )

        with context:
            # Process through time steps
            for step in range(self.num_steps):
                # Get current time step data
                if data.dim() == 5:  # (time, batch, channels, height, width)
                    current_input = data[step]
                elif data.dim() == 4:  # (time, batch, height, width)
                    current_input = data[step]
                else:  # (time, batch, features)
                    current_input = data[step]

                # Forward pass through distributed model
                spk, mem = self.model(current_input)

                # Record outputs
                self.spike_recordings.append(spk)
                if mem is not None:
                    self.mem_recordings.append(mem)

        spike_tensor = (
            torch.stack(self.spike_recordings)
            if self.spike_recordings
            else None
        )
        mem_tensor = (
            torch.stack(self.mem_recordings) if self.mem_recordings else None
        )

        return spike_tensor, mem_tensor

    def compute_loss(
        self,
        spike_data: torch.Tensor,
        targets: torch.Tensor,
        loss_fn: Callable,
        loss_mode: str = "sum_spikes",
    ) -> torch.Tensor:
        """
        Compute loss for SNN output.

        Args:
            spike_data: Spike recordings with shape (time_steps, batch_size, output_size)
            mem_data: Membrane potential recordings (optional)
            targets: Target labels
            loss_fn: Loss function
            loss_mode: How to compute loss ("sum_spikes", "final_step", "mean_spikes")

        Returns:
            torch.Tensor: Computed loss
        """
        if spike_data is None:
            raise ValueError("No spike data available for loss computation")

        if loss_mode == "sum_spikes":
            # Sum spikes over time for classification
            total_spikes = torch.sum(spike_data, dim=0)
            loss = loss_fn(total_spikes, targets)
        elif loss_mode == "final_step":
            # Use final time step output
            loss = loss_fn(spike_data[-1], targets)
        elif loss_mode == "mean_spikes":
            # Mean spikes over time
            mean_spikes = torch.mean(spike_data, dim=0)
            loss = loss_fn(mean_spikes, targets)
        else:
            raise ValueError(f"Unknown loss_mode: {loss_mode}")

        return loss

    def train_step(
        self,
        data: torch.Tensor,
        targets: torch.Tensor,
        loss_fn: Callable,
        loss_mode: str = "sum_spikes",
    ) -> Dict[str, float]:
        """
        Single training step with gradient accumulation across time.

        Args:
            data: Input data
            targets: Target labels
            optimizer: Optimizer
            loss_fn: Loss function
            loss_mode: Loss computation mode

        Returns:
            Dict with loss and timing information
        """
        start_time = time.time()

        # Forward pass
        forward_start = time.time()
        spike_data, _ = self.forward_pass(data)
        forward_time = time.time() - forward_start

        # Calculate loss
        loss = self.compute_loss(
            spike_data, targets, loss_fn, loss_mode
        )
        loss = loss / self.grad_accumulation_steps  # Scale for accumulation

        # Backward pass
        backward_start = time.time()
        if self.mixed_precision and self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        backward_time = time.time() - backward_start

        # Update stats
        total_time = time.time() - start_time

        return {
            "loss": loss.item()
            * self.grad_accumulation_steps,  # Unscale for reporting
            "forward_time": forward_time,
            "backward_time": backward_time,
            "total_time": total_time,
        }

    def optimizer_step(self, optimizer: torch.optim.Optimizer):
        """
        Perform optimizer step with gradient clipping and mixed precision handling.

        Args:
            optimizer: The optimizer to step
        """
        if self.mixed_precision and self.scaler is not None:
            # Gradient clipping for mixed precision
            if self.clip_grad_norm is not None:
                self.scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.clip_grad_norm
                )

            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            # Standard gradient clipping
            if self.clip_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.clip_grad_norm
                )

            optimizer.step()

    def train_epoch(
        self,
        dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
        loss_fn: Callable,
        loss_mode: str = "sum_spikes",
        log_interval: int = 100,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    ) -> Dict[str, float]:
        """
        Train for one epoch.

        Args:
            dataloader: Training data loader
            optimizer: Optimizer
            loss_fn: Loss function
            loss_mode: Loss computation mode
            log_interval: Logging interval
            scheduler: Optional learning rate scheduler

        Returns:
            Dict with epoch statistics
        """
        self.model.train()

        total_loss = 0.0
        num_batches = 0
        epoch_stats = {
            "total_time": 0.0,
            "forward_time": 0.0,
            "backward_time": 0.0,
            "data_time": 0.0,
        }

        data_start_time = time.time()

        for batch_idx, (data, targets) in enumerate(dataloader):
            # Data loading time
            data_time = time.time() - data_start_time
            epoch_stats["data_time"] += data_time

            # Move to device
            data, targets = data.to(self.device), targets.to(self.device)

            # Training step
            step_stats = self.train_step(
                data, targets, loss_fn, loss_mode
            )

            # Accumulate gradients
            if (batch_idx + 1) % self.grad_accumulation_steps == 0:
                self.optimizer_step(optimizer)
                optimizer.zero_grad()

                if scheduler is not None:
                    scheduler.step()

                self.global_step += 1

            # Update statistics
            total_loss += step_stats["loss"]
            num_batches += 1

            for key in ["forward_time", "backward_time", "total_time"]:
                epoch_stats[key] += step_stats[key]

            # Logging
            if batch_idx % log_interval == 0 and dist.get_rank() == 0:
                avg_loss = total_loss / num_batches
                print(
                    f"Epoch: {self.current_epoch}, "
                    f"Batch: {batch_idx}/{len(dataloader)}, "
                    f'Loss: {step_stats["loss"]:.6f}, '
                    f"Avg Loss: {avg_loss:.6f}, "
                    f'LR: {optimizer.param_groups[0]["lr"]:.2e}'
                )

            data_start_time = time.time()

        # Final gradient step if needed
        if num_batches % self.grad_accumulation_steps != 0:
            self.optimizer_step(optimizer)
            optimizer.zero_grad()

        # Calculate averages
        avg_stats = {
            "loss": total_loss / num_batches,
            "avg_forward_time": epoch_stats["forward_time"] / num_batches,
            "avg_backward_time": epoch_stats["backward_time"] / num_batches,
            "avg_data_time": epoch_stats["data_time"] / num_batches,
            "total_epoch_time": epoch_stats["total_time"],
        }

        return avg_stats

    def evaluate(
        self,
        dataloader: DataLoader,
        loss_fn: Callable,
        loss_mode: str = "sum_spikes",
    ) -> Dict[str, float]:
        """
        Evaluate the model on validation/test data.

        Args:
            dataloader: Evaluation data loader
            loss_fn: Loss function
            loss_mode: Loss computation mode

        Returns:
            Dict with evaluation metrics
        """
        self.model.eval()

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for data, targets in dataloader:
                data, targets = data.to(self.device), targets.to(self.device)

                # Forward pass
                spike_data, _ = self.forward_pass(data)

                # Calculate loss
                loss = self.compute_loss(
                    spike_data, targets, loss_fn, loss_mode
                )
                total_loss += loss.item()

                # Calculate accuracy
                if loss_mode == "sum_spikes":
                    predictions = torch.sum(spike_data, dim=0)
                elif loss_mode == "final_step":
                    predictions = spike_data[-1]
                elif loss_mode == "mean_spikes":
                    predictions = torch.mean(spike_data, dim=0)

                predicted_labels = torch.argmax(predictions, dim=1)
                total_correct += (predicted_labels == targets).sum().item()
                total_samples += targets.size(0)

        avg_loss = total_loss / len(dataloader)
        accuracy = total_correct / total_samples

        return {
            "loss": avg_loss,
            "accuracy": accuracy,
            "total_samples": total_samples,
        }

    def save_checkpoint(
        self,
        filepath: str,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        additional_info: Optional[Dict[str, Any]] = None,
    ):
        """
        Save training checkpoint.

        Args:
            filepath: Path to save checkpoint
            optimizer: Optimizer state
            scheduler: Optional scheduler state
            additional_info: Additional information to save
        """
        if dist.get_rank() == 0:  # Only save on rank 0
            checkpoint = {
                "epoch": self.current_epoch,
                "global_step": self.global_step,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "training_stats": self.training_stats,
            }

            if scheduler is not None:
                checkpoint["scheduler_state_dict"] = scheduler.state_dict()

            if additional_info is not None:
                checkpoint.update(additional_info)

            torch.save(checkpoint, filepath)
            print(f"Checkpoint saved to {filepath}")

    def load_checkpoint(
        self,
        filepath: str,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
    ) -> Dict[str, Any]:
        """
        Load training checkpoint.

        Args:
            filepath: Path to checkpoint file
            optimizer: Optimizer to load state into
            scheduler: Optional scheduler to load state into

        Returns:
            Dict with additional checkpoint information
        """
        checkpoint = torch.load(filepath, map_location=self.device)

        self.current_epoch = checkpoint["epoch"]
        self.global_step = checkpoint["global_step"]
        self.model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if "training_stats" in checkpoint:
            self.training_stats = checkpoint["training_stats"]

        if scheduler is not None and "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        print(f"Checkpoint loaded from {filepath}")

        # Return additional info
        additional_info = {
            k: v
            for k, v in checkpoint.items()
            if k
            not in [
                "epoch",
                "global_step",
                "model_state_dict",
                "optimizer_state_dict",
                "training_stats",
                "scheduler_state_dict",
            ]
        }

        return additional_info


def create_distributed_dataloader(
    dataset,
    batch_size: int,
    num_workers: int = 0,
    shuffle: bool = True,
    drop_last: bool = True,
) -> DataLoader:
    """
    Create a distributed data loader for training.

    Args:
        dataset: Dataset to wrap
        batch_size: Batch size per GPU
        num_workers: Number of worker processes
        shuffle: Whether to shuffle data
        drop_last: Whether to drop last incomplete batch

    Returns:
        DataLoader with DistributedSampler
    """
    sampler = DistributedSampler(dataset, shuffle=shuffle, drop_last=drop_last)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
    )


def get_optimizer_and_scheduler(
    model: nn.Module,
    lr: float = 1e-3,
    optimizer_type: str = "adam",
    scheduler_type: Optional[str] = None,
    **kwargs,
) -> Tuple[
    torch.optim.Optimizer, Optional[torch.optim.lr_scheduler._LRScheduler]
]:
    """
    Create optimizer and scheduler for SNN training.

    Args:
        model: The model to optimize
        lr: Learning rate
        optimizer_type: Type of optimizer ("adam", "adamw", "sgd")
        scheduler_type: Type of scheduler ("cosine", "step", "exponential")
        **kwargs: Additional arguments for optimizer/scheduler

    Returns:
        Tuple of optimizer and optional scheduler
    """
    # Create optimizer
    if optimizer_type.lower() == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, **kwargs)
    elif optimizer_type.lower() == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, **kwargs)
    elif optimizer_type.lower() == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, **kwargs)
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")

    # Create scheduler
    scheduler = None
    if scheduler_type is not None:
        if scheduler_type.lower() == "cosine":
            T_max = kwargs.get("T_max", 100)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=T_max
            )
        elif scheduler_type.lower() == "step":
            step_size = kwargs.get("step_size", 30)
            gamma = kwargs.get("gamma", 0.1)
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=step_size, gamma=gamma
            )
        elif scheduler_type.lower() == "exponential":
            gamma = kwargs.get("gamma", 0.95)
            scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer, gamma=gamma
            )
        else:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}")

    return optimizer, scheduler
