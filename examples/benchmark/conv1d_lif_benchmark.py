import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import time
import warnings

warnings.filterwarnings("ignore")


class LIFConv1D(nn.Module):
    """
    Time-parallel LIF neuron with an iterative reset mechanism.

    - Calculates the "potential" voltage trace via convolution, as if no spikes were to
        occur.
    - Iteratively finds the next spike for each neuron in the batch.
    - When a spike is found, its "reset effect" (a decaying exponential) is subtracted
        from the potential voltage for all subsequent time steps.
    - This process repeats until no new spikes are found or a max number of iterations
        is reached.
    """

    def __init__(
        self,
        tau=20.0,
        resistance=1.0,
        threshold=1.0,
        dt=1.0,
        max_time_steps=1000,
    ):
        super().__init__()

        self.tau = tau
        self.resistance = resistance
        self.threshold = threshold
        self.dt = dt

        # Discrete-time LIF parameters
        alpha = torch.exp(torch.tensor(-dt / tau))
        beta = resistance * (1 - alpha)

        # Create discrete impulse response kernel: h[n] = β * α^n
        n = torch.arange(max_time_steps, dtype=torch.float32)
        kernel = beta * (alpha**n)
        self.register_buffer("kernel", kernel.unsqueeze(0).unsqueeze(0))
        self.register_buffer("alpha", alpha)

    def forward(self, input_current, v_init=0.0):
        """
        Args:
            input_current: (batch_size, seq_length)
            v_init: initial membrane potential

        Returns:
            voltage: (batch_size, seq_length) - final membrane potential with resets
            spikes: (batch_size, seq_length) - binary spike train
        """
        seq_length = input_current.size(1)
        device = input_current.device

        # Calculate the potential voltage trace without any resets
        voltage_potential = self._calculate_potential(input_current, v_init, seq_length)

        # Initialize outputs
        voltage = voltage_potential.clone()
        spikes = torch.zeros_like(input_current, device=device)
        time_steps_vec = torch.arange(seq_length, device=device, dtype=torch.float32)

        while True:
            # Check where voltage crosses threshold (and no spike has been recorded yet)
            would_spike = (voltage >= self.threshold) & (spikes < 1)
            if not torch.any(would_spike):
                break

            # Find the *first* time step of a new spike for each batch item
            spike_times = torch.argmax(would_spike.float(), dim=1)

            # Create a mask for only the new spikes found in this iteration
            new_spikes_mask = F.one_hot(spike_times, num_classes=seq_length).bool()
            new_spikes_mask &= would_spike

            # Add new spikes to the output spike train
            spikes += new_spikes_mask.float()

            # Calculate and subtract the reset effect for the new spikes
            time_since_spike = time_steps_vec.view(1, -1) - spike_times.view(-1, 1)
            reset_decay = torch.where(
                time_since_spike >= 0,
                self.threshold * (self.alpha**time_since_spike),
                0.0,
            )

            # Only apply the reset for batches that had a new spike
            reset_effect = reset_decay * new_spikes_mask.any(dim=1).view(-1, 1)

            # Subtract the reset effect from the voltage
            voltage -= reset_effect

        return voltage, spikes

    def _calculate_potential(self, input_current, v_init, seq_length):
        """Computes voltage response to current as if there were no threshold."""
        # Add channel dim for conv1d: (batch, channels, length)
        input_current_conv = input_current.unsqueeze(1)

        # Truncate kernel to sequence length
        kernel = self.kernel[:, :, :seq_length]

        # Apply causal convolution
        padded_input = F.pad(input_current_conv, (seq_length - 1, 0))
        voltage_response = F.conv1d(padded_input, torch.flip(kernel, [2]))
        voltage_response = voltage_response.squeeze(1)[:, :seq_length]

        # Add initial condition contribution: v_init * α^n
        if v_init != 0.0:
            n = torch.arange(
                seq_length, device=input_current.device, dtype=torch.float32
            )
            init_decay = v_init * (self.alpha**n)
            voltage_response += init_decay.unsqueeze(0)

        return voltage_response


class LIFIterative(nn.Module):
    """Iterative LIF neuron implementation with reset by subtraction."""

    def __init__(self, tau=20.0, resistance=1.0, threshold=1.0, dt=1.0):
        super().__init__()
        self.tau = tau
        self.resistance = resistance
        self.threshold = threshold
        self.dt = dt

        # Precompute decay factor
        self.alpha = torch.exp(torch.tensor(-dt / tau))

    def forward(self, input_current, v_init=0.0):
        """
        Args:
            input_current: (batch_size, seq_length) - input current over time
            v_init: initial membrane potential

        Returns:
            voltage: (batch_size, seq_length) - membrane potential over time
            spikes: (batch_size, seq_length) - binary spike train
        """
        batch_size, seq_length = input_current.shape
        device = input_current.device

        # Initialize outputs
        voltage = torch.zeros_like(input_current)
        spikes = torch.zeros_like(input_current)

        # Initialize membrane potential
        v = torch.full((batch_size,), v_init, device=device)

        # Iterative update
        for t in range(seq_length):
            # LIF dynamics: v[t+1] = α*v[t] + R*I[t]*(1-α)
            v = self.alpha * v + self.resistance * input_current[:, t] * (
                1 - self.alpha
            )

            # Generate spikes
            spike_mask = v >= self.threshold
            spikes[:, t] = spike_mask.float()

            # Store voltage *before* reset for analysis
            voltage[:, t] = v

            # Reset by subtraction
            v[spike_mask] -= self.threshold

        return voltage, spikes


def generate_test_inputs(
    batch_size: int, seq_length: int, input_type: str = "step"
) -> torch.Tensor:
    """Generate different types of test inputs with batch variation."""
    if input_type == "step":
        # Step input with varied amplitude and start time
        inputs = torch.zeros(batch_size, seq_length)
        amplitudes = 1.0 + torch.rand(batch_size, 1) * 1.0  # Vary amplitude
        start_times = torch.randint(
            10, seq_length // 4, (batch_size,)
        )  # Vary start time
        for i in range(batch_size):
            inputs[i, start_times[i] :] = amplitudes[i]

    elif input_type == "impulse":
        # Single impulse at a varied location
        inputs = torch.zeros(batch_size, seq_length)
        locations = torch.randint(seq_length // 4, 3 * seq_length // 4, (batch_size,))
        inputs[torch.arange(batch_size), locations] = 5.0

    elif input_type == "ramp":
        # Ramp input with varied slope
        slopes = 1.5 + torch.rand(batch_size, 1) * 1.0  # Vary slope
        base_ramp = torch.linspace(0, 1, seq_length).unsqueeze(0)
        inputs = base_ramp * slopes

    elif input_type == "sinusoidal":
        # Sinusoidal input with varied phase and frequency
        t = torch.linspace(0, 4 * np.pi, seq_length)
        phases = torch.rand(batch_size, 1) * np.pi
        frequencies = 0.8 + torch.rand(batch_size, 1) * 0.4
        inputs = 1.5 + torch.sin(t * frequencies + phases)

    elif input_type == "noise":
        # White noise is already varied by nature
        inputs = torch.randn(batch_size, seq_length) * 0.5 + 1.0

    else:
        raise ValueError(f"Unknown input type: {input_type}")

    return inputs


def accuracy_test(tau: float = 20.0, dt: float = 1.0, seq_length: int = 200):
    """Test accuracy between the two implementations."""
    print(f"\n=== Accuracy Test (tau={tau}, dt={dt}, seq_length={seq_length}) ===")

    # Create models with same parameters
    conv_model = LIFConv1D(tau=tau, dt=dt, max_time_steps=seq_length * 2)
    iter_model = LIFIterative(tau=tau, dt=dt)

    input_types = ["step", "impulse", "ramp", "sinusoidal", "noise"]

    for input_type in input_types:
        print(f"\nTesting {input_type} input:")

        # Generate test input
        inputs = generate_test_inputs(1, seq_length, input_type)

        # Test both models
        with torch.no_grad():
            v_conv, s_conv = conv_model(inputs)
            v_iter, s_iter = iter_model(inputs)

        # Calculate errors
        voltage_mse = F.mse_loss(v_conv, v_iter).item()
        voltage_mae = F.l1_loss(v_conv, v_iter).item()
        spike_accuracy = (s_conv == s_iter).float().mean().item()

        print(f"  Voltage MSE: {voltage_mse:.6f}")
        print(f"  Voltage MAE: {voltage_mae:.6f}")
        print(f"  Spike Accuracy: {spike_accuracy:.4f}")


def performance_benchmark():
    """Benchmark performance across different configurations."""
    print(f"\n=== Performance Benchmark ===")

    # Test configurations
    configs = [
        {"batch_size": 1, "seq_length": 100},
        {"batch_size": 1, "seq_length": 500},
        {"batch_size": 1, "seq_length": 1000},
        {"batch_size": 1, "seq_length": 2000},
        {"batch_size": 1, "seq_length": 4000},
        {"batch_size": 8, "seq_length": 100},
        {"batch_size": 32, "seq_length": 100},
        {"batch_size": 64, "seq_length": 100},
        {"batch_size": 128, "seq_length": 100},
        {"batch_size": 256, "seq_length": 100},
        {"batch_size": 8, "seq_length": 500},
        {"batch_size": 32, "seq_length": 500},
        {"batch_size": 64, "seq_length": 500},
        {"batch_size": 128, "seq_length": 500},
        {"batch_size": 256, "seq_length": 500},
    ]

    # Parameters
    tau, dt = 20.0, 1.0

    results = []

    for config in configs:
        batch_size = config["batch_size"]
        seq_length = config["seq_length"]

        print(f"\nTesting batch_size={batch_size}, seq_length={seq_length}")

        # Create models
        conv_model = LIFConv1D(tau=tau, dt=dt, max_time_steps=seq_length * 2)
        iter_model = LIFIterative(tau=tau, dt=dt)

        # Torch compile
        conv_model = torch.compile(conv_model)
        iter_model = torch.compile(iter_model)

        # Generate test input
        inputs = generate_test_inputs(batch_size, seq_length, "noise")

        # Warmup
        with torch.no_grad():
            conv_model(inputs)
            iter_model(inputs)

        # Benchmark Conv1D model
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start_time = time.time()

        n_runs = 30
        for _ in range(n_runs):
            with torch.no_grad():
                _, _ = conv_model(inputs)

        torch.cuda.synchronize() if torch.cuda.is_available() else None
        conv_time = (time.time() - start_time) / n_runs

        # Benchmark Iterative model
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start_time = time.time()

        for _ in range(n_runs):
            with torch.no_grad():
                _, _ = iter_model(inputs)

        torch.cuda.synchronize() if torch.cuda.is_available() else None
        iter_time = (time.time() - start_time) / n_runs

        speedup = iter_time / conv_time

        print(f"  Conv1D time: {conv_time*1000:.2f}ms")
        print(f"  Iterative time: {iter_time*1000:.2f}ms")
        print(f"  Speedup: {speedup:.2f}x")

        results.append(
            {
                "batch_size": batch_size,
                "seq_length": seq_length,
                "conv_time": conv_time,
                "iter_time": iter_time,
                "speedup": speedup,
            }
        )

    return results


def memory_benchmark():
    """Benchmark memory usage."""
    print(f"\n=== Memory Benchmark ===")

    if not torch.cuda.is_available():
        print("CUDA not available, skipping memory benchmark")
        return

    configs = [
        {"batch_size": 1, "seq_length": 1000},
        {"batch_size": 8, "seq_length": 1000},
        {"batch_size": 32, "seq_length": 1000},
        {"batch_size": 1, "seq_length": 5000},
    ]

    for config in configs:
        batch_size = config["batch_size"]
        seq_length = config["seq_length"]

        print(f"\nTesting batch_size={batch_size}, seq_length={seq_length}")

        # Test Conv1D model
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        conv_model = LIFConv1D(tau=20.0, dt=1.0, max_time_steps=seq_length * 2).cuda()
        # conv_model = torch.compile(conv_model)
        inputs = generate_test_inputs(batch_size, seq_length, "noise").cuda()

        with torch.no_grad():
            _, _ = conv_model(inputs)

        conv_memory = torch.cuda.max_memory_allocated() / 1024**2  # MB

        # Test Iterative model
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        iter_model = LIFIterative(tau=20.0, dt=1.0).cuda()
        # iter_model = torch.compile(iter_model)
        inputs = inputs.cuda()

        with torch.no_grad():
            _, _ = iter_model(inputs)

        iter_memory = torch.cuda.max_memory_allocated() / 1024**2  # MB

        print(f"  Conv1D memory: {conv_memory:.1f}MB")
        print(f"  Iterative memory: {iter_memory:.1f}MB")
        print(f"  Memory ratio: {conv_memory/iter_memory:.2f}x")


def gradient_benchmark():
    """Test gradient computation performance."""
    print(f"\n=== Gradient Benchmark ===")

    batch_size, seq_length = 8, 500
    tau, dt = 20.0, 1.0

    # Create models
    conv_model = LIFConv1D(tau=tau, dt=dt, max_time_steps=seq_length * 2)
    iter_model = LIFIterative(tau=tau, dt=dt)

    # Add a simple linear layer on top to create a loss
    conv_linear = nn.Linear(seq_length, 1)
    iter_linear = nn.Linear(seq_length, 1)

    # Generate inputs and targets
    inputs = generate_test_inputs(batch_size, seq_length, "noise")
    targets = torch.randn(batch_size, 1)

    # Benchmark Conv1D gradients
    start_time = time.time()
    n_runs = 50

    for _ in range(n_runs):
        conv_model.zero_grad()
        conv_linear.zero_grad()

        v_conv, _ = conv_model(inputs)
        output = conv_linear(v_conv)
        loss = F.mse_loss(output, targets)
        loss.backward()

    conv_grad_time = (time.time() - start_time) / n_runs

    # Benchmark Iterative gradients
    start_time = time.time()

    for _ in range(n_runs):
        iter_model.zero_grad()
        iter_linear.zero_grad()

        v_iter, _ = iter_model(inputs)
        output = iter_linear(v_iter)
        loss = F.mse_loss(output, targets)
        loss.backward()

    iter_grad_time = (time.time() - start_time) / n_runs

    print(f"Conv1D gradient time: {conv_grad_time*1000:.2f}ms")
    print(f"Iterative gradient time: {iter_grad_time*1000:.2f}ms")
    print(f"Gradient speedup: {iter_grad_time/conv_grad_time:.2f}x")


def visualize_comparison():
    """Create visualizations comparing the two implementations."""
    # Parameters
    tau, dt = 20.0, 1.0
    seq_length = 200

    # Create models
    conv_model = LIFConv1D(tau=tau, dt=dt, max_time_steps=seq_length * 2)
    iter_model = LIFIterative(tau=tau, dt=dt)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle("LIF Neuron Comparison: Conv1D vs Iterative", fontsize=14)

    input_types = ["step", "impulse", "sinusoidal"]

    for i, input_type in enumerate(input_types):
        inputs = generate_test_inputs(1, seq_length, input_type)

        with torch.no_grad():
            v_conv, s_conv = conv_model(inputs)
            v_iter, s_iter = iter_model(inputs)

        # Plot inputs
        t = np.arange(seq_length) * dt
        axes[0, i].plot(t, inputs[0].numpy(), "k-", label="Input", linewidth=2)
        axes[0, i].set_title(f"{input_type.capitalize()} Input")
        axes[0, i].set_ylabel("Current")
        axes[0, i].grid(True, alpha=0.3)
        axes[0, i].legend()

        # Plot voltages and spikes
        axes[1, i].plot(t, v_conv[0].numpy(), "b-", label="Conv1D", linewidth=2)
        axes[1, i].plot(
            t,
            v_iter[0].numpy(),
            "r--",
            label="Iterative",
            linewidth=2,
            alpha=0.8,
        )

        # Add spike markers
        spike_times_conv = t[s_conv[0].numpy().astype(bool)]
        spike_times_iter = t[s_iter[0].numpy().astype(bool)]

        if len(spike_times_conv) > 0:
            axes[1, i].scatter(
                spike_times_conv,
                [conv_model.threshold] * len(spike_times_conv),
                color="blue",
                marker="o",
                s=50,
                label="Conv1D spikes",
            )
        if len(spike_times_iter) > 0:
            axes[1, i].scatter(
                spike_times_iter,
                [iter_model.threshold] * len(spike_times_iter),
                color="red",
                marker="x",
                s=50,
                label="Iter spikes",
            )

        axes[1, i].axhline(
            y=conv_model.threshold,
            color="gray",
            linestyle=":",
            alpha=0.7,
            label="Threshold",
        )
        axes[1, i].set_xlabel("Time")
        axes[1, i].set_ylabel("Voltage")
        axes[1, i].grid(True, alpha=0.3)
        axes[1, i].legend()

    plt.tight_layout()
    plt.savefig("lif_comparison.png", dpi=150, bbox_inches="tight")
    plt.show()


def main():
    """Run all benchmarks."""
    print("=" * 60)
    print("LIF Neuron Benchmark: Conv1D vs Iterative Implementation")
    print("=" * 60)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Run tests
    accuracy_test(tau=20.0, dt=1.0, seq_length=200)
    accuracy_test(tau=10.0, dt=0.5, seq_length=400)  # Different parameters

    perf_results = performance_benchmark()

    memory_benchmark()
    gradient_benchmark()
    visualize_comparison()

    # Summary
    print(f"\n=== Summary ===")
    avg_speedup = np.mean([r["speedup"] for r in perf_results])
    print(f"Average speedup: {avg_speedup:.2f}x")


if __name__ == "__main__":
    main()
