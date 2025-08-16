import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import time
import warnings
import snntorch as snn
from snntorch import utils

warnings.filterwarnings("ignore")


def generate_test_inputs(
    batch_size: int, seq_length: int, input_type: str = "step"
) -> torch.Tensor:
    """Generate different types of test inputs with batch variation."""
    if input_type == "step":
        # Step input with varied amplitude and start time
        inputs = torch.zeros(seq_length, batch_size)
        amplitudes = 1.0 + torch.rand(1, batch_size) * 1.0  # Vary amplitude
        start_times = torch.randint(
            10, seq_length // 4, (batch_size,)
        )  # Vary start time
        for i in range(batch_size):
            inputs[start_times[i] :, i] = amplitudes[0, i]

    elif input_type == "impulse":
        # Single impulse at a varied location
        inputs = torch.zeros(seq_length, batch_size)
        locations = torch.randint(seq_length // 4, 3 * seq_length // 4, (batch_size,))
        inputs[locations, torch.arange(batch_size)] = 5.0

    elif input_type == "ramp":
        # Ramp input with varied slope
        slopes = 1.5 + torch.rand(1, batch_size) * 1.0  # Vary slope
        base_ramp = torch.linspace(0, 1, seq_length).unsqueeze(1)
        inputs = base_ramp * slopes

    elif input_type == "sinusoidal":
        # Sinusoidal input with varied phase and frequency
        t = torch.linspace(0, 4 * np.pi, seq_length).unsqueeze(1)
        phases = torch.rand(1, batch_size) * np.pi
        frequencies = 0.8 + torch.rand(1, batch_size) * 0.4
        inputs = 1.5 + torch.sin(t * frequencies + phases)

    elif input_type == "noise":
        # White noise
        inputs = torch.randn(seq_length, batch_size) * 0.5 + 1.0
    
    elif input_type == "spiking":
        # Spike train input (0s and 1s) with varied spike patterns
        inputs = torch.zeros(seq_length, batch_size)
        # Create a pattern with spikes at regular intervals, some bursts, and some isolated spikes
        spike_times = [10, 15, 16, 25, 35, 45, 50, 55, 65, 75, 76, 77, 90, 110, 130, 150, 170, 175, 190]
        for spike_time in spike_times:
            if spike_time < seq_length:
                # Use amplitude > 0 for spikes (could be 1.0 or higher for stronger effect)
                inputs[spike_time, :] = 1  # Strong spike input

    else:
        raise ValueError(f"Unknown input type: {input_type}")

    return inputs


def accuracy_test(beta: float = 0.5, seq_length: int = 200):
    """Test accuracy between the two implementations."""
    print(f"\n=== Accuracy Test (beta={beta}, seq_length={seq_length}) ===")

    # Create models with same parameters
    leaky_conv1d = snn.LeakyConv1d(beta=beta)
    leaky = snn.Leaky(beta=beta, reset_delay=False)

    input_types = ["step", "impulse", "ramp", "sinusoidal", "noise", "spiking"]

    for input_type in input_types:
        print(f"\nTesting {input_type} input:")

        # Generate test input (seq_len, batch_size)
        inputs = generate_test_inputs(1, seq_length, input_type)

        # Test both models
        with torch.no_grad():
            # Reset hidden states
            utils.reset(leaky_conv1d)
            
            spk_conv1d, mem_conv1d = leaky_conv1d(inputs)
            
            # Reset for fair comparison
            utils.reset(leaky)
            
            spk_leaky, mem_leaky = leaky(inputs)

        # Calculate errors - need to handle different output formats
        # LeakyConv1d returns (seq_len, batch_size), Leaky processes timestep by timestep
        # So we need to process Leaky iteratively
        spk_leaky_seq = torch.zeros_like(spk_conv1d)
        mem_leaky_seq = torch.zeros_like(mem_conv1d)
        
        utils.reset(leaky)
        mem_state = None
        for t in range(seq_length):
            spk_t, mem_state = leaky(inputs[t], mem_state)
            spk_leaky_seq[t] = spk_t
            mem_leaky_seq[t] = mem_state

        # Calculate errors
        voltage_mse = F.mse_loss(mem_conv1d, mem_leaky_seq).item()
        voltage_mae = F.l1_loss(mem_conv1d, mem_leaky_seq).item()
        spike_accuracy = (spk_conv1d == spk_leaky_seq).float().mean().item()

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
        {"batch_size": 8, "seq_length": 100},
        {"batch_size": 32, "seq_length": 100},
        {"batch_size": 64, "seq_length": 100},
        {"batch_size": 128, "seq_length": 100},
        {"batch_size": 8, "seq_length": 500},
        {"batch_size": 32, "seq_length": 500},
        {"batch_size": 64, "seq_length": 500},
    ]

    # Parameters
    beta = 0.5

    results = []

    for config in configs:
        batch_size = config["batch_size"]
        seq_length = config["seq_length"]

        print(f"\nTesting batch_size={batch_size}, seq_length={seq_length}")

        # Create models
        leaky_conv1d = snn.LeakyConv1d(beta=beta)
        leaky = snn.Leaky(beta=beta, reset_delay=False)

        # Generate test input
        inputs = generate_test_inputs(batch_size, seq_length, "noise")

        # Warmup
        with torch.no_grad():
            utils.reset(leaky_conv1d)
            leaky_conv1d(inputs)
            
            utils.reset(leaky)
            mem_state = None
            for t in range(seq_length):
                _, mem_state = leaky(inputs[t], mem_state)

        # Benchmark LeakyConv1d model
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start_time = time.time()

        n_runs = 30
        for _ in range(n_runs):
            with torch.no_grad():
                utils.reset(leaky_conv1d)
                _, _ = leaky_conv1d(inputs)

        torch.cuda.synchronize() if torch.cuda.is_available() else None
        conv1d_time = (time.time() - start_time) / n_runs

        # Benchmark Leaky model (iterative processing)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start_time = time.time()

        for _ in range(n_runs):
            with torch.no_grad():
                utils.reset(leaky)
                mem_state = None
                for t in range(seq_length):
                    _, mem_state = leaky(inputs[t], mem_state)

        torch.cuda.synchronize() if torch.cuda.is_available() else None
        leaky_time = (time.time() - start_time) / n_runs

        speedup = leaky_time / conv1d_time

        print(f"  LeakyConv1d time: {conv1d_time*1000:.2f}ms")
        print(f"  Leaky time: {leaky_time*1000:.2f}ms")
        print(f"  Speedup: {speedup:.2f}x")

        results.append(
            {
                "batch_size": batch_size,
                "seq_length": seq_length,
                "conv1d_time": conv1d_time,
                "leaky_time": leaky_time,
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

        # Test LeakyConv1d model
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        leaky_conv1d = snn.LeakyConv1d(beta=0.5).cuda()
        inputs = generate_test_inputs(batch_size, seq_length, "noise").cuda()

        with torch.no_grad():
            utils.reset(leaky_conv1d)
            _, _ = leaky_conv1d(inputs)

        conv1d_memory = torch.cuda.max_memory_allocated() / 1024**2  # MB

        # Test Leaky model
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        leaky = snn.Leaky(beta=0.5).cuda()
        inputs = inputs.cuda()

        with torch.no_grad():
            utils.reset(leaky)
            mem_state = None
            for t in range(seq_length):
                _, mem_state = leaky(inputs[t], mem_state)

        leaky_memory = torch.cuda.max_memory_allocated() / 1024**2  # MB

        print(f"  LeakyConv1d memory: {conv1d_memory:.1f}MB")
        print(f"  Leaky memory: {leaky_memory:.1f}MB")
        print(f"  Memory ratio: {conv1d_memory/leaky_memory:.2f}x")


def visualize_comparison():
    """Create visualizations comparing the two implementations."""
    # Parameters
    beta = 0.5
    seq_length = 200

    # Create models
    leaky_conv1d = snn.LeakyConv1d(beta=beta)
    leaky = snn.Leaky(beta=beta, reset_delay=False)

    # input_types = ["step", "spiking"]
    input_types = ["spiking"]

    fig, axes = plt.subplots(2, len(input_types), figsize=(20, 8))
    fig.suptitle("LIF Neuron Comparison: LeakyConv1d vs Leaky", fontsize=14)
    for i, input_type in enumerate(input_types):
        inputs = generate_test_inputs(1, seq_length, input_type)

        with torch.no_grad():
            # LeakyConv1d processing
            utils.reset(leaky_conv1d)
            spk_conv1d, mem_conv1d = leaky_conv1d(inputs)

            # Leaky processing (timestep by timestep)
            utils.reset(leaky)
            spk_leaky_seq = torch.zeros_like(spk_conv1d)
            mem_leaky_seq = torch.zeros_like(mem_conv1d)
            
            mem_state = None
            for t in range(seq_length):
                spk_t, mem_state = leaky(inputs[t], mem_state)
                spk_leaky_seq[t] = spk_t
                mem_leaky_seq[t] = mem_state

        # Plot inputs
        t = np.arange(seq_length)
        indices = (0, i) if len(input_types) > 1 else (0,)
        axes[*indices].plot(t, inputs[:, 0].numpy(), "k-", label="Input", linewidth=2)
        axes[*indices].set_title(f"{input_type.capitalize()} Input")
        axes[*indices].set_ylabel("Current")
        axes[*indices].grid(True, alpha=0.3)
        axes[*indices].legend()

        # Plot voltages and spikes
        indices = (1, i) if len(input_types) > 1 else (1,)
        axes[*indices].plot(t, mem_conv1d[:, 0].numpy(), "b-", label="LeakyConv1d", linewidth=2)
        axes[*indices].plot(
            t,
            mem_leaky_seq[:, 0].numpy(),
            "r--",
            label="Leaky",
            linewidth=2,
            alpha=0.8,
        )

        # Add spike markers
        spike_times_conv1d = t[spk_conv1d[:, 0].numpy().astype(bool)]
        spike_times_leaky = t[spk_leaky_seq[:, 0].numpy().astype(bool)]

        if len(spike_times_conv1d) > 0:
            axes[*indices].scatter(
                spike_times_conv1d,
                [1.0] * len(spike_times_conv1d),
                color="blue",
                marker="o",
                s=50,
                label="LeakyConv1d spikes",
            )
        if len(spike_times_leaky) > 0:
            axes[*indices].scatter(
                spike_times_leaky,
                [1.0] * len(spike_times_leaky),
                color="red",
                marker="x",
                s=50,
                label="Leaky spikes",
            )

        axes[*indices].axhline(
            y=1.0,
            color="gray",
            linestyle=":",
            alpha=0.7,
            label="Threshold",
        )
        axes[*indices].set_xlabel("Time Step")
        axes[*indices].set_ylabel("Voltage")
        axes[*indices].grid(True, alpha=0.3)
        axes[*indices].legend()

    plt.tight_layout()
    plt.savefig("lif_implementations_comparison.png", dpi=150, bbox_inches="tight")
    plt.show()


def main():
    """Run all benchmarks."""
    print("=" * 60)
    print("LIF Implementations Benchmark: LeakyConv1d vs Leaky")
    print("=" * 60)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Run tests
    accuracy_test(beta=0.5, seq_length=200)
    # accuracy_test(beta=0.8, seq_length=400)  # Different parameters

    # perf_results = performance_benchmark()

    # memory_benchmark()
    visualize_comparison()

    # # Summary
    # print(f"\n=== Summary ===")
    # avg_speedup = np.mean([r["speedup"] for r in perf_results])
    # print(f"Average speedup: {avg_speedup:.2f}x")


if __name__ == "__main__":
    main()