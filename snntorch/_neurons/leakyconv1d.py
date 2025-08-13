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
            raise ValueError("LeakyConv1d expects at least 2D input (seq_len, batch_size, ...)")
        
        if not mem == None:
            self.mem = mem

        if self.init_hidden and not mem == None:
            raise TypeError(
                "`mem` should not be passed as an argument while `init_hidden=True`"
            )
        
        # Store original shape for output
        original_shape = input_.shape
        seq_len = original_shape[0]
        batch_size = original_shape[1]
        
        # Flatten spatial dimensions if present
        if input_.dim() > 2:
            # Reshape from (seq_len, batch_size, ...) to (seq_len, batch_size * spatial_dims)
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

        mem = mem_potential.clone()
        spikes = torch.zeros_like(mem, device=device)

        while True:
            would_spike = (mem >= self.threshold) & (spikes < 1)
            if not torch.any(would_spike):
                break

            # Generate spikes using the surrogate gradient function
            new_spikes = self.fire(mem) * would_spike.float()
            spikes += new_spikes
            
            # Calculate reset effect based on the reset mechanism
            if self.reset_mechanism_val == 0:  # "subtract"
                reset_values = self.threshold
            elif self.reset_mechanism_val == 1:  # "zero"
                reset_values = mem
            else:  # "none"
                reset_values = 0

            # This creates a tensor of reset values only at spike times
            reset_pulses = new_spikes * reset_values
            
            # Reshape reset pulses for convolution: (seq, batch) -> (batch, 1, seq)
            reset_pulses_conv = reset_pulses.permute(1, 0).unsqueeze(1)
            reset_effect = F.conv1d(
                reset_pulses_conv, decay_kernel, padding=padding
            )[:, :, :seq_len]
            reset_effect = reset_effect.squeeze(1).permute(1, 0)
            
            # Subtract the reset effect from the membrane potential
            mem = mem_potential - reset_effect

        # Final spike calculation on the fully corrected membrane potential
        spikes = self.fire(mem)

        if self.state_quant:
            mem = self.state_quant(mem)

        # Reshape outputs back to original shape
        if input_.dim() > 2:
            spikes = spikes.view(original_shape)
            mem = mem.view(original_shape)

        if self.output:
            return spikes, mem
        elif self.init_hidden:
            return spikes
        else:
            return spikes, mem
