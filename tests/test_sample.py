import torch
import pytest
from models.base_model.layers.utils.temperature import TemperatureAdjuster
from models.base_model.layers.utils.sampling import TopPSampling, TopKSampling, MixSampling


# Test Case 1: Verify TopP sampling with default parameters
def test_top_p_sampling_default():
    logits = torch.randn(1, 10)  # Random logits for vocab_size=10
    top_p_sampler = TopPSampling(top_p=0.9, temperature=1.0, dynamic_temp=False)
    sampled_token = top_p_sampler.sample(logits)

    assert sampled_token >= 0 and sampled_token < 10  # Sampled token index should be within vocab_size


# Test Case 2: Check if repetition penalty works correctly
def test_repetition_penalty():
    logits = torch.randn(1, 10)  # Random logits for vocab_size=10
    past_tokens = torch.tensor([[2, 3]])  # Sampled tokens before
    top_p_sampler = TopPSampling(top_p=0.9, temperature=1.0, repetition_penalty=2.0)

    modified_logits = top_p_sampler.apply_repetition_penalty(logits, past_tokens)

    # Check that the logits corresponding to past tokens are reduced
    assert modified_logits[0, 2] < logits[0, 2]  # Token '2' should have lower probability


# Test Case 3: Verify dynamic temperature adjustment behavior
def test_dynamic_temperature():
    logits = torch.randn(1, 10)  # Random logits for vocab_size=10
    temperature_adjuster = TemperatureAdjuster()
    top_p_sampler = TopPSampling(top_p=0.9, temperature=1.0, use_dynamic_temperature=True,
                                 temperature_adjuster=temperature_adjuster)

    initial_logits = top_p_sampler.adjust_temperature(logits, step=1, content_quality=0.8, perplexity=1.0)

    assert initial_logits.sum().item() != logits.sum().item()  # Ensure logits have changed


# Test Case 4: Test TopK Sampling with different k values
def test_top_k_sampling():
    logits = torch.randn(1, 10)  # Random logits for vocab_size=10
    top_k_sampler = TopKSampling(top_k=3, temperature=1.0, dynamic_temp=False)
    sampled_token = top_k_sampler.sample(logits)

    assert sampled_token >= 0 and sampled_token < 3  # Sampled token should be from the top-3


# Test Case 5: Verify TopK sampling when k is 0 (no truncation)
def test_top_k_sampling_zero_k():
    logits = torch.randn(1, 10)  # Random logits for vocab_size=10
    top_k_sampler = TopKSampling(top_k=0, temperature=1.0, dynamic_temp=False)
    sampled_token = top_k_sampler.sample(logits)

    assert sampled_token >= 0 and sampled_token < 10  # Sampled token should be from the entire vocab


# Test Case 6: Validate MixSampling (Top-p followed by Top-k)
def test_mix_sampling():
    logits = torch.randn(1, 10)  # Random logits for vocab_size=10
    mix_sampler = MixSampling(top_p=0.9, top_k=3, temperature=1.0, dynamic_temp=False)
    sampled_token = mix_sampler.pksample(logits)

    assert sampled_token >= 0 and sampled_token < 3  # Sampled token should be from top-p followed by top-k


# Test Case 7: Validate MixSampling when Top-k is 0 (Only Top-p applies)
def test_mix_sampling_top_k_zero():
    logits = torch.randn(1, 10)  # Random logits for vocab_size=10
    mix_sampler = MixSampling(top_p=0.9, top_k=0, temperature=1.0, dynamic_temp=False)
    sampled_token = mix_sampler.pksample(logits)

    assert sampled_token >= 0 and sampled_token < 10  # Sampled token should be from top-p selection


# Test Case 8: Test the behavior when past tokens are empty
def test_past_tokens_empty():
    logits = torch.randn(1, 10)  # Random logits for vocab_size=10
    top_p_sampler = TopPSampling(top_p=0.9, temperature=1.0, dynamic_temp=False)
    sampled_token = top_p_sampler.sample(logits, past_tokens=torch.tensor([[]]))

    assert sampled_token >= 0 and sampled_token < 10  # Should sample from the full vocab


# Test Case 9: Test behavior when logits are zero (edge case)
def test_zero_logits():
    logits = torch.zeros(1, 10)  # All logits set to 0
    top_p_sampler = TopPSampling(top_p=0.9, temperature=1.0, dynamic_temp=False)
    sampled_token = top_p_sampler.sample(logits)

    assert sampled_token >= 0 and sampled_token < 10  # Should sample from a uniform distribution


# Test Case 10: Ensure MixSampling handles all parameters correctly
def test_mix_sampling_full_flow():
    logits = torch.randn(1, 10)  # Random logits for vocab_size=10
    temperature_adjuster = TemperatureAdjuster()
    mix_sampler = MixSampling(
        top_p=0.9,
        top_k=5,
        temperature=1.0,
        dynamic_temp=True,
        temperature_adjuster=temperature_adjuster
    )
    sampled_token = mix_sampler.kpsample(logits)

    assert sampled_token >= 0 and sampled_token < 5  # Should sample from top-k and top-p mix


# Test Case 11: Test dynamic temperature adjustment for TopK sampling
def test_top_k_dynamic_temperature():
    logits = torch.randn(1, 10)  # Random logits for vocab_size=10
    temperature_adjuster = TemperatureAdjuster()
    top_k_sampler = TopKSampling(
        top_k=3,
        temperature=1.0,
        dynamic_temp=True,
        temperature_adjuster=temperature_adjuster
    )
    sampled_token = top_k_sampler.sample(logits, step=5, content_quality=0.9, perplexity=1.2)

    assert sampled_token >= 0 and sampled_token < 3  # Sampled token should be from top-3


# Test Case 12: Ensure handling of invalid parameters
def test_invalid_parameters():
    with pytest.raises(ValueError):
        top_p_sampler = TopPSampling(top_p=-0.5)  # Invalid top_p value

    with pytest.raises(ValueError):
        top_k_sampler = TopKSampling(top_k=-3)  # Invalid top_k value

    with pytest.raises(ValueError):
        mix_sampler = MixSampling(top_p=0.9, top_k=0, temperature=-1.0)  # Invalid temperature
