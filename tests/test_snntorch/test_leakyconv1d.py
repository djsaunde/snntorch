#!/usr/bin/env python

"""Tests for LeakyConv1d neuron."""

import pytest
import snntorch as snn
import torch
import torch._dynamo as dynamo


@pytest.fixture(scope="module")
def input_():
    return torch.Tensor([0.25, 0]).unsqueeze(-1)


@pytest.fixture(scope="module")
def input_seq():
    """Sequential input for testing conv1d functionality"""
    return torch.randn(10, 2, 3)  # (seq_len, batch_size, features)


@pytest.fixture(scope="module")
def leakyconv1d_instance():
    return snn.LeakyConv1d(beta=0.5)


@pytest.fixture(scope="module")
def leakyconv1d_instance_surrogate():
    return snn.LeakyConv1d(
        beta=0.5, surrogate_disable=True
    )


@pytest.fixture(scope="module")
def leakyconv1d_reset_zero_instance():
    return snn.LeakyConv1d(
        beta=0.5, reset_mechanism="zero"
    )


@pytest.fixture(scope="module")
def leakyconv1d_reset_none_instance():
    return snn.LeakyConv1d(
        beta=0.5, reset_mechanism="none"
    )


@pytest.fixture(scope="module")
def leakyconv1d_hidden_instance():
    return snn.LeakyConv1d(beta=0.5, init_hidden=True)


@pytest.fixture(scope="module")
def leakyconv1d_learnable_instance():
    return snn.LeakyConv1d(
        beta=0.5,
        learn_beta=True,
        learn_threshold=True,
    )


class TestLeakyConv1d:
    def test_leakyconv1d_min_2d_input_required(self, leakyconv1d_instance, input_):
        """Test that LeakyConv1d requires at least 2D input"""
        with pytest.raises(ValueError, match="LeakyConv1d expects at least 2D input"):
            leakyconv1d_instance(input_[0])  # 1D input should fail

    def test_leakyconv1d_sequence_processing(
        self, leakyconv1d_instance, input_seq
    ):
        """Test sequence processing mode"""
        spikes, mem = leakyconv1d_instance(input_seq)

        # Check output shapes
        assert spikes.shape == input_seq.shape
        assert mem.shape == input_seq.shape

        # Check that some computation occurred
        assert not torch.all(spikes == 0) or not torch.all(mem == 0)

    def test_leakyconv1d_4d_input_support(self, leakyconv1d_instance):
        """Test 4D input support (e.g., video data)"""
        # Create 4D input: (seq_len, batch, height, width)
        input_4d = torch.randn(8, 4, 16, 16)
        spikes, mem = leakyconv1d_instance(input_4d)
        
        # Output should maintain original shape
        assert spikes.shape == input_4d.shape
        assert mem.shape == input_4d.shape
        
        # Check that computation occurred
        assert not torch.all(spikes == 0) or not torch.all(mem == 0)

    def test_leakyconv1d_reset_mechanisms(
        self,
        leakyconv1d_instance,
        leakyconv1d_reset_zero_instance,
        leakyconv1d_reset_none_instance,
    ):
        """Test different reset mechanisms"""
        lif1 = leakyconv1d_instance
        lif2 = leakyconv1d_reset_zero_instance
        lif3 = leakyconv1d_reset_none_instance

        assert lif1.reset_mechanism_val == 0  # subtract
        assert lif2.reset_mechanism_val == 1  # zero
        assert lif3.reset_mechanism_val == 2  # none

        # Test reset mechanism switching
        lif1.reset_mechanism = "zero"
        lif2.reset_mechanism = "none"
        lif3.reset_mechanism = "subtract"

        assert lif1.reset_mechanism_val == 1
        assert lif2.reset_mechanism_val == 2
        assert lif3.reset_mechanism_val == 0

    def test_leakyconv1d_init_hidden(
        self, leakyconv1d_hidden_instance, input_seq
    ):
        """Test init_hidden functionality"""
        spk = leakyconv1d_hidden_instance(input_seq)

        # With init_hidden=True, only spikes are returned
        assert isinstance(spk, torch.Tensor)
        assert spk.shape == input_seq.shape

    def test_leakyconv1d_learnable_parameters(
        self, leakyconv1d_learnable_instance
    ):
        """Test learnable beta and threshold"""
        assert leakyconv1d_learnable_instance.beta.requires_grad
        assert leakyconv1d_learnable_instance.threshold.requires_grad

        # Test forward pass and gradient computation
        input_seq = torch.randn(5, 2, 3)
        spikes, mem = leakyconv1d_learnable_instance(input_seq)

        loss = spikes.sum() + mem.sum()
        loss.backward()

        assert leakyconv1d_learnable_instance.beta.grad is not None
        assert leakyconv1d_learnable_instance.threshold.grad is not None

    def test_leakyconv1d_beta_learnable(self, leakyconv1d_learnable_instance):
        """Test that learnable beta works correctly"""
        initial_beta = leakyconv1d_learnable_instance.beta.clone()

        # Run forward pass and check gradients
        input_seq = torch.randn(3, 2, 2)
        spikes, mem = leakyconv1d_learnable_instance(input_seq)
        
        loss = spikes.sum() + mem.sum()
        loss.backward()

        # Beta should have gradients
        assert leakyconv1d_learnable_instance.beta.grad is not None
        
        # Beta should be learnable
        assert leakyconv1d_learnable_instance.beta.requires_grad

    def test_leakyconv1d_max_sequence_length(self):
        """Test max_sequence_length parameter"""
        max_len = 50
        lif = snn.LeakyConv1d(beta=0.9)

        # Test with sequence longer than max_length
        long_seq = torch.randn(max_len + 10, 2, 3)
        spikes, mem = lif(long_seq)

        # Should still work but use truncated kernel
        assert spikes.shape == long_seq.shape
        assert mem.shape == long_seq.shape

    def test_leakyconv1d_input_shapes(self, leakyconv1d_instance):
        """Test different input shape formats"""
        batch_size, seq_len, features = 2, 5, 3

        # Test (seq_len, batch, features) format
        input1 = torch.randn(seq_len, batch_size, features)
        spikes1, mem1 = leakyconv1d_instance(input1)
        assert spikes1.shape == (seq_len, batch_size, features)

        # Test (seq_len, batch) format (2D)
        input2 = torch.randn(seq_len, batch_size)
        spikes2, mem2 = leakyconv1d_instance(input2)
        assert spikes2.shape == (seq_len, batch_size)

    def test_leakyconv1d_cases(self, leakyconv1d_hidden_instance, input_seq):
        """Test error cases"""
        # This should raise TypeError because init_hidden=True but mem is provided
        with pytest.raises(TypeError):
            mem_2d = torch.randn(
                input_seq.shape[1], input_seq.shape[2]
            )  # (batch, features)
            leakyconv1d_hidden_instance(input_seq, mem_2d)

    def test_leakyconv1d_reset_mem(self, leakyconv1d_instance):
        """Test membrane potential reset"""
        # Initialize with some data
        input_data = torch.randn(5, 2, 3)
        _, _ = leakyconv1d_instance(input_data)

        # Reset should zero out membrane
        mem_reset = leakyconv1d_instance.reset_mem()
        assert torch.all(mem_reset == 0)

    def test_leakyconv1d_device_compatibility(self, leakyconv1d_instance):
        """Test device compatibility"""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Move model to device
        lif_device = leakyconv1d_instance.to(device)

        # Test input on same device
        input_seq = torch.randn(5, 2, 3, device=device)
        spikes, mem = lif_device(input_seq)

        assert spikes.device.type == device.type
        assert mem.device.type == device.type

    def test_leakyconv1d_compile_fullgraph(
        self, leakyconv1d_instance_surrogate, input_seq
    ):
        """Test torch.compile compatibility"""
        explanation = dynamo.explain(leakyconv1d_instance_surrogate)(input_seq)

        # Should compile without graph breaks
        assert explanation.graph_break_count == 0
