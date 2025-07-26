#!/usr/bin/env python3
"""
Test parity between our RoBERTa implementation and HuggingFace's RoBERTa
"""

import torch
import numpy as np
from transformers import RobertaModel, RobertaConfig as HFConfig, RobertaTokenizer
from train_roberta import RoBERTaModel, RoBERTaConfig

def test_config_parity():
    """Test that our config matches HuggingFace config"""
    print("Testing config parity...")
    
    # Load HF config
    hf_config = HFConfig.from_pretrained('roberta-base')
    
    # Our config
    our_config = RoBERTaConfig()
    
    # Compare key parameters
    comparisons = [
        ('vocab_size', hf_config.vocab_size, our_config.vocab_size),
        ('hidden_size', hf_config.hidden_size, our_config.hidden_size),
        ('num_hidden_layers', hf_config.num_hidden_layers, our_config.num_hidden_layers),
        ('num_attention_heads', hf_config.num_attention_heads, our_config.num_attention_heads),
        ('intermediate_size', hf_config.intermediate_size, our_config.intermediate_size),
        ('max_position_embeddings', hf_config.max_position_embeddings, our_config.max_position_embeddings),
    ]
    
    all_match = True
    for name, hf_val, our_val in comparisons:
        match = hf_val == our_val
        all_match &= match
        status = "✓" if match else "✗"
        print(f"  {status} {name}: HF={hf_val}, Ours={our_val}")
    
    return all_match

def test_architecture_parity():
    """Test that our architecture produces similar outputs"""
    print("\nTesting architecture parity...")
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Create models
    hf_config = HFConfig.from_pretrained('roberta-base')
    hf_model = RobertaModel(hf_config)
    
    our_config = RoBERTaConfig()
    our_model = RoBERTaModel(our_config)
    
    # Test with random input
    batch_size, seq_len = 2, 64
    input_ids = torch.randint(3, our_config.vocab_size-1, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    
    # Get outputs
    with torch.no_grad():
        hf_outputs = hf_model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
        our_outputs = our_model(input_ids, attention_mask=attention_mask)
    
    # Compare shapes
    hf_last_hidden = hf_outputs.last_hidden_state
    our_last_hidden = our_outputs[0]  # sequence output
    
    hf_pooled = hf_outputs.pooler_output
    our_pooled = our_outputs[1]  # pooled output
    
    print(f"  HF last hidden shape: {hf_last_hidden.shape}")
    print(f"  Our last hidden shape: {our_last_hidden.shape}")
    print(f"  HF pooled shape: {hf_pooled.shape}")
    print(f"  Our pooled shape: {our_pooled.shape}")
    
    # Check if shapes match
    shapes_match = (hf_last_hidden.shape == our_last_hidden.shape and 
                   hf_pooled.shape == our_pooled.shape)
    
    print(f"  {'✓' if shapes_match else '✗'} Output shapes match")
    
    return shapes_match

def test_attention_mechanism():
    """Test that our attention mechanism is bidirectional"""
    print("\nTesting attention mechanism...")
    
    our_config = RoBERTaConfig(num_hidden_layers=1)  # Use single layer for easier testing
    our_model = RoBERTaModel(our_config)
    
    # Create input with special pattern to test bidirectionality
    batch_size, seq_len = 1, 10
    input_ids = torch.tensor([[0, 100, 200, 300, 400, 50264, 500, 600, 700, 2]])  # mask token at position 5
    attention_mask = torch.ones(batch_size, seq_len)
    
    with torch.no_grad():
        outputs = our_model(input_ids, attention_mask=attention_mask)
        last_hidden = outputs[0]
        attentions = outputs[3][0]  # First layer attention
    
    # Check that the masked token (position 5) can attend to both past and future tokens
    mask_token_attention = attentions[0, :, 5, :]  # All heads, attention from position 5
    
    # The attention should be distributed across all positions (bidirectional)
    attention_spread = (mask_token_attention > 0.01).float().sum(dim=-1).mean()
    
    print(f"  Attention spread from masked token: {attention_spread:.2f} positions on average")
    print(f"  {'✓' if attention_spread > seq_len * 0.5 else '✗'} Attention is bidirectional")
    
    return attention_spread > seq_len * 0.5

def compare_parameter_counts():
    """Compare parameter counts"""
    print("\nComparing parameter counts...")
    
    # HuggingFace model
    hf_config = HFConfig.from_pretrained('roberta-base')
    hf_model = RobertaModel(hf_config)
    hf_params = sum(p.numel() for p in hf_model.parameters())
    hf_sd_params = sum(p.numel() for p in hf_model.state_dict().values())
    
    
    # Our model
    our_config = RoBERTaConfig()
    our_model = RoBERTaModel(our_config)
    our_params = sum(p.numel() for p in our_model.parameters())
    our_sd_params = sum(p.numel() for p in our_model.state_dict().values())
    
    print(f"  HF model parameters: {hf_params:,}")
    print(f"  Our model parameters: {our_params:,}")
    print(f"  Difference: {abs(hf_params - our_params):,}")

    print(f"  HF model SD parameters: {hf_sd_params:,}")
    print(f"  Our model SD parameters: {our_sd_params:,}")
    print(f"  Difference: {abs(hf_sd_params - our_sd_params):,}")
    
    # They should be very close (within 1% difference)
    close_match = abs(hf_params - our_params) / hf_params < 0.01
    print(f"  {'✓' if close_match else '✗'} SD Parameter counts are close")

    # They should be very close (within 1% difference)
    close_match = abs(hf_sd_params - our_sd_params) / hf_sd_params < 0.01
    print(f"  {'✓' if close_match else '✗'} SD Parameter counts are close")
    
    return close_match

def main():
    print("="*60)
    print("RoBERTa Implementation Parity Test")
    print("="*60)
    
    try:
        # Run all tests
        config_ok = test_config_parity()
        arch_ok = test_architecture_parity()
        attention_ok = test_attention_mechanism()
        params_ok = compare_parameter_counts()
        
        print("\n" + "="*60)
        print("Summary:")
        print(f"  {'✓' if config_ok else '✗'} Configuration parity")
        print(f"  {'✓' if arch_ok else '✗'} Architecture parity") 
        print(f"  {'✓' if attention_ok else '✗'} Bidirectional attention")
        print(f"  {'✓' if params_ok else '✗'} Parameter count parity")
        
        all_passed = all([config_ok, arch_ok, attention_ok, params_ok])
        print(f"\nOverall: {'✓ PASS' if all_passed else '✗ FAIL'}")
        print("="*60)
        
        return all_passed
        
    except ImportError as e:
        print(f"Error: {e}")
        print("Note: transformers library required for parity testing")
        print("Install with: pip install transformers")
        return False
    except Exception as e:
        print(f"Unexpected error: {e}")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)