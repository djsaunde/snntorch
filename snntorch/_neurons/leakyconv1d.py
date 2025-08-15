import torch
import torch.nn.functional as F
from .neurons import LIF


class LeakyConv1d(LIF):
    """
    An updated Leaky Integrate-and-Fire (LIF) neuron using a more efficient conv1d
    parallel processing method.

    This implementation is based on a single convolution to calculate the potential
    membrane trace, followed by an iterative voltage correction loop to apply
    spike resets.
    
    This implementation performs 10-30x better than the standard LIF implementation in
    most settings. See https://gist.github.com/djsaunde/3281daf1b527d72ad3de6c674312100b
    for benchmark results.
    """

    def __init__(
        self,
        beta,
        threshold=1.0,
        spike_grad=None,
        surrogate_disable=False,
        init_hidden=False,
        inhibition=False,
        learn_beta=False,
        learn_threshold=False,
        reset_mechanism="subtract",
        state_quant=False,
        output=False,
    ):
        super().__init__(
            beta,
            threshold,
            spike_grad,
            surrogate_disable,
            init_hidden,
            inhibition,
            learn_beta,
            learn_threshold,
            reset_mechanism,
            state_quant,
            output,
        )

        self._init_mem()

    def _init_mem(self):
        mem = torch.zeros(0)
        self.register_buffer("mem", mem, False)

    def reset_mem(self):
        self.mem = torch.zeros_like(self.mem, device=self.mem.device)
        return self.mem

    def forward(self, input_, mem=None):
        if input_.dim() < 2:
            raise ValueError(
                "LeakyConv1d expects at least 2D input (seq_len, batch_size, ...)"
            )
        
        if not mem == None:
            self.mem = mem

        if self.init_hidden and not mem == None:
            raise TypeError(
                "`mem` should not be passed as an argument while `init_hidden=True`"
            )
        
        # Store original shape for output
        original_shape = input_.shape
        seq_len = original_shape[0]
        
        # Flatten spatial dimensions if present
        if input_.dim() > 2:
            # Reshape from (seq_len, batch_size, ...) to (seq_len, batch_size *
            # spatial_dims)
            input_flat = input_.view(seq_len, -1)
        else:
            input_flat = input_
        
        device = input_.device
        beta_clamped = self.beta.clamp(0, 1)

        # Dynamically create the decay kernel: [1, β, β², ...]
        input_conv = input_flat.permute(1, 0).unsqueeze(1)
        powers = torch.arange(seq_len, device=device, dtype=torch.float32)
        decay_kernel = (beta_clamped ** powers).flip(0).unsqueeze(0).unsqueeze(0)

        # Apply causal convolution
        padding = seq_len - 1
        mem_potential = F.conv1d(
            input_conv, decay_kernel, padding=padding
        )[:, :, :seq_len]
        mem_potential = mem_potential.squeeze(1).permute(1, 0)

        # Add effect of initial membrane potential
        if mem is not None and mem.numel() > 0:
            # Flatten membrane potential if needed to match input_flat
            if mem.shape != input_flat.shape[1:]:
                mem_flat = mem.view(-1)
            else:
                mem_flat = mem.view(-1)
            
            decay_factors = (beta_clamped ** powers).view(seq_len, 1)
            mem_potential += mem_flat.view(1, -1) * decay_factors

        # Track the observable voltage trace (before resets are applied)
        voltage_trace = mem_potential.clone()
        mem = mem_potential.clone()
        spikes = torch.zeros_like(mem, device=device)

        batch_size = input_flat.shape[1]
        time_steps_vec = torch.arange(seq_len, device=device, dtype=torch.float32)

        while True:
            # Check where voltage crosses threshold (and no spike has been recorded yet)
            would_spike = (mem >= self.threshold) & (spikes < 1)
            if not torch.any(would_spike):
                break

            # Find the *first* time step of a new spike for each batch item
            # Our format is (seq_len, batch_size), so argmax over dim=0 (seq_len dimension)
            spike_times = torch.argmax(would_spike.float(), dim=0)  # Shape: (batch_size,)
            
            # Create a mask for only the new spikes found in this iteration
            # Convert (batch_size,) spike times to (seq_len, batch_size) one-hot mask
            new_spikes_mask = F.one_hot(spike_times, num_classes=seq_len).bool()  # (batch_size, seq_len)
            new_spikes_mask = new_spikes_mask.transpose(0, 1)  # (seq_len, batch_size)
            new_spikes_mask &= would_spike
            
            # Add new spikes to the output spike train
            spikes += new_spikes_mask.float()
            
            # Update voltage_trace to show the voltage at the moment of spiking
            # The voltage_trace should show what's observable before reset
            voltage_trace = mem.clone()
            
            # Calculate and subtract the reset effect for the new spikes
            # Broadcasting: (seq_len, 1) - (1, batch_size) -> (seq_len, batch_size)
            time_since_spike = time_steps_vec.view(-1, 1) - spike_times.view(1, -1)
            
            if self.reset_mechanism_val == 0:  # "subtract"
                reset_decay = torch.where(
                    time_since_spike >= 0,
                    self.threshold * (beta_clamped ** time_since_spike),
                    0.0,
                )
            elif self.reset_mechanism_val == 1:  # "zero"
                # For "zero" reset, use the membrane potential at spike time
                spike_values = mem[spike_times, torch.arange(batch_size, device=device)]  # (batch_size,)
                reset_decay = torch.where(
                    time_since_spike >= 0,
                    spike_values.view(1, -1) * (beta_clamped ** time_since_spike),
                    0.0,
                )
            else:  # "none"
                reset_decay = torch.zeros_like(time_since_spike)
            
            # Only apply the reset for batches that had a new spike
            # new_spikes_mask.any(dim=0) gives (batch_size,) mask of which batches spiked
            reset_effect = reset_decay * new_spikes_mask.any(dim=0).view(1, -1)
            
            # Subtract the reset effect from the current membrane potential
            mem -= reset_effect

        if self.state_quant:
            voltage_trace = self.state_quant(voltage_trace)

        # Reshape outputs back to original shape
        if input_.dim() > 2:
            spikes = spikes.view(original_shape)
            voltage_trace = voltage_trace.view(original_shape)

        # Store the final membrane state for persistence
        self.mem = mem.view(original_shape) if input_.dim() > 2 else mem

        if self.output:
            return spikes, voltage_trace
        elif self.init_hidden:
            return spikes
        else:
            return spikes, voltage_trace

    @classmethod
    def detach_hidden(cls):
        """Returns the hidden states, detached from the current graph.
        Intended for use in truncated backpropagation through time where
        hidden state variables are instance variables."""

        for layer in range(len(cls.instances)):
            if isinstance(cls.instances[layer], LeakyConv1d):
                cls.instances[layer].mem.detach_()

    @classmethod
    def reset_hidden(cls):
        """Used to clear hidden state variables to zero.
        Intended for use where hidden state variables are instance variables.
        Assumes hidden states have a batch dimension already."""
        for layer in range(len(cls.instances)):
            if isinstance(cls.instances[layer], LeakyConv1d):
                cls.instances[layer].mem = torch.zeros_like(
                    cls.instances[layer].mem,
                    device=cls.instances[layer].mem.device,
                )
