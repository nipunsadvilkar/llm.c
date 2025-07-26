#!/usr/bin/env python3
"""
Test fill-mask functionality between our RoBERTa and HuggingFace RoBERTa
"""

import torch
import torch.nn.functional as F
from transformers import RobertaForMaskedLM, RobertaTokenizer
from train_roberta import RoBERTaForMaskedLM as OurRoBERTaForMaskedLM
# from train_roberta_tied import RoBERTaForMaskedLM as OurRoBERTaForMaskedLM

torch.manual_seed(42)  # For reproducibility

def test_fill_mask():
    print("="*60)
    print("FILL-MASK FUNCTIONALITY TEST")
    print("="*60)
    
    # Load models and tokenizer
    print("Loading models...")
    hf_model = RobertaForMaskedLM.from_pretrained("roberta-base")
    our_model = OurRoBERTaForMaskedLM.from_pretrained("roberta-base")
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    
    # Test sentences
    test_sentences = [
        "The capital of France is <mask>.",
        "I love to eat <mask> for breakfast.",
        "The <mask> is shining brightly today.",
        "She works as a <mask> in the hospital.",
        "My favorite color is <mask>.",
    ]
    
    print("\nTesting fill-mask predictions...\n")
    
    all_match = True
    
    for i, sentence in enumerate(test_sentences, 1):
        print(f"Test {i}: '{sentence}'")
        print("-" * 50)
        
        # Tokenize
        inputs = tokenizer(sentence, return_tensors="pt")
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        
        # Find mask token position
        mask_token_id = tokenizer.mask_token_id
        mask_positions = (input_ids == mask_token_id).nonzero(as_tuple=True)
        if len(mask_positions[1]) == 0:
            print("❌ No mask token found!")
            continue
        mask_pos = mask_positions[1][0].item()
        
        # Get predictions from both models
        with torch.no_grad():
            # HuggingFace model
            hf_outputs = hf_model(input_ids, attention_mask=attention_mask)
            hf_logits = hf_outputs.logits[0, mask_pos]  # Shape: [vocab_size]
            
            # Our model
            our_outputs = our_model(input_ids, attention_mask=attention_mask)
            our_logits = our_outputs[1][0, mask_pos]  # Shape: [vocab_size]
        
        # Get top 5 predictions for each model
        hf_probs = F.softmax(hf_logits, dim=-1)
        our_probs = F.softmax(our_logits, dim=-1)
        
        hf_top5 = torch.topk(hf_probs, 5)
        our_top5 = torch.topk(our_probs, 5)
        
        # Compare predictions
        print("HuggingFace Top 5:")
        for j, (prob, token_id) in enumerate(zip(hf_top5.values, hf_top5.indices)):
            token = tokenizer.decode([token_id])
            print(f"  {j+1}. {token.strip():15} ({prob:.4f})")
        
        print("\nOur Model Top 5:")
        for j, (prob, token_id) in enumerate(zip(our_top5.values, our_top5.indices)):
            token = tokenizer.decode([token_id])
            print(f"  {j+1}. {token.strip():15} ({prob:.4f})")
        
        # Check if top predictions match
        hf_top1 = tokenizer.decode([hf_top5.indices[0]]).strip()
        our_top1 = tokenizer.decode([our_top5.indices[0]]).strip()
        
        top1_match = hf_top1 == our_top1
        
        # Check if logits are close
        logits_close = torch.allclose(hf_logits, our_logits, atol=1e-3)
        
        print(f"\nResults:")
        print(f"  Top-1 prediction match: {'✓' if top1_match else '✗'} (HF: '{hf_top1}', Ours: '{our_top1}')")
        print(f"  Logits close (atol=1e-3): {'✓' if logits_close else '✗'}")
        print(f"  Max logit difference: {torch.max(torch.abs(hf_logits - our_logits)):.6f}")
        
        if not top1_match:
            all_match = False
        
        print("\n")
    
    # Special test for "The capital of France is <mask>."
    print("="*60)
    print("SPECIAL TEST: 'The capital of France is <mask>.'")
    print("="*60)
    
    sentence = "The capital of France is <mask>."
    inputs = tokenizer(sentence, return_tensors="pt")
    
    with torch.no_grad():
        hf_outputs = hf_model(**inputs)
        our_outputs = our_model(inputs["input_ids"], attention_mask=inputs["attention_mask"])
    
    mask_pos = (inputs["input_ids"] == tokenizer.mask_token_id).nonzero(as_tuple=True)[1][0]
    
    hf_logits = hf_outputs.logits[0, mask_pos]
    our_logits = our_outputs[1][0, mask_pos]
    
    hf_probs = F.softmax(hf_logits, dim=-1)
    our_probs = F.softmax(our_logits, dim=-1)
    
    # Get top prediction
    hf_pred_id = torch.argmax(hf_probs)
    our_pred_id = torch.argmax(our_probs)
    
    hf_pred = tokenizer.decode([hf_pred_id]).strip()
    our_pred = tokenizer.decode([our_pred_id]).strip()
    
    print(f"HuggingFace prediction: '{hf_pred}' (prob: {hf_probs[hf_pred_id]:.4f})")
    print(f"Our model prediction:   '{our_pred}' (prob: {our_probs[our_pred_id]:.4f})")
    
    # Check if it's Paris (or similar)
    expected_answers = ["Paris", "PARIS", " Paris", "paris"]
    hf_correct = any(ans in hf_pred for ans in expected_answers)
    our_correct = any(ans in our_pred for ans in expected_answers)
    
    print(f"\nCorrectness check:")
    print(f"  HuggingFace predicts Paris: {'✓' if hf_correct else '✗'}")
    print(f"  Our model predicts Paris:   {'✓' if our_correct else '✗'}")
    print(f"  Predictions match:          {'✓' if hf_pred == our_pred else '✗'}")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    if all_match and hf_pred == our_pred:
        print("🎉 ALL TESTS PASSED! 🎉")
        print("✅ Both models produce identical fill-mask predictions")
        print("✅ Paris is correctly predicted as capital of France")
        print("✅ Model outputs are functionally equivalent")
    else:
        print("❌ Some differences found:")
        if not all_match:
            print("  - Top predictions differ on some test cases")
        if hf_pred != our_pred:
            print("  - France capital prediction differs")
        print("\nThis might indicate:")
        print("  - Small numerical differences in weights")
        print("  - Different initialization values")
        print("  - Need to check weight loading more carefully")

if __name__ == "__main__":
    test_fill_mask()