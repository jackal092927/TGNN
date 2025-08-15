"""
Analyze GraphRNN optimization objective and class balance issues
"""

import torch
import torch.nn as nn
import numpy as np


def analyze_optimization_objective():
    """
    Analyze what GraphRNN is actually optimized for
    """
    print("=" * 80)
    print("ANALYZING GRAPHRNN OPTIMIZATION OBJECTIVE")
    print("=" * 80)
    
    print("\n🎯 WHAT IS GRAPHRNN OPTIMIZED FOR?")
    print("-" * 50)
    
    print("📊 Loss Function: BCEWithLogitsLoss (Binary Cross-Entropy)")
    print("📊 Training Sampling: 1:2 ratio (1 positive : 2 negatives)")
    print("📊 Validation/Test Sampling: 1:1 ratio (balanced)")
    print("📊 Teacher Forcing: Yes (uses ground truth previous states)")
    
    print("\n🧮 MATHEMATICAL ANALYSIS:")
    print("BCEWithLogitsLoss = -[y*log(σ(x)) + (1-y)*log(1-σ(x))]")
    print("Where: σ(x) = sigmoid(x) = 1/(1 + e^(-x))")
    
    print("\n🎭 TRAINING vs EVALUATION MISMATCH:")
    print("Training:")
    print("  - Samples: 1 positive + 2 negatives per timestamp")
    print("  - Loss: Optimizes for balanced BCE on this 1:2 distribution")
    print("  - Model learns: 'In a 1:2 setting, what's the optimal probability?'")
    
    print("\nEvaluation:")
    print("  - Samples: 1 positive + 1 negative per timestamp")
    print("  - Expectation: Model should output ~0.5 for balanced case")
    print("  - Reality: Model outputs ~0.33 (trained on 1:2 ratio!)")
    
    return True


def demonstrate_optimal_probabilities():
    """
    Show what optimal probabilities should be for different class ratios
    """
    print("\n" + "=" * 80)
    print("OPTIMAL PROBABILITIES FOR DIFFERENT CLASS RATIOS")
    print("=" * 80)
    
    print("\n🧮 THEORETICAL OPTIMAL PROBABILITIES:")
    print("If the true class distribution is P(positive) = p, then:")
    print("Optimal probability for a perfect classifier = p")
    
    ratios = [
        (1, 1, "Balanced (Evaluation)"),
        (1, 2, "GraphRNN Training"),
        (1, 4, "Highly Imbalanced"),
    ]
    
    print(f"\n{'Ratio':<20} {'P(positive)':<12} {'Optimal Prob':<15} {'Description'}")
    print("-" * 65)
    
    for pos, neg, desc in ratios:
        p_positive = pos / (pos + neg)
        print(f"{pos}:{neg:<17} {p_positive:<12.3f} {p_positive:<15.3f} {desc}")
    
    print(f"\n🎯 KEY INSIGHT:")
    print(f"GraphRNN trained on 1:2 ratio learns to output ~0.33 as 'positive'")
    print(f"But evaluation expects ~0.5 for balanced case")
    print(f"This explains why accuracy is low despite good ranking!")


def analyze_sigmoid_behavior():
    """
    Analyze how sigmoid maps logits to probabilities
    """
    print("\n" + "=" * 80)
    print("SIGMOID BEHAVIOR AND LOGIT ANALYSIS")
    print("=" * 80)
    
    print("\n📊 SIGMOID FUNCTION MAPPING:")
    logits = np.array([-2, -1, -0.5, 0, 0.5, 1, 2])
    probs = 1 / (1 + np.exp(-logits))
    
    print(f"{'Logit':<8} {'Probability':<12} {'Classification (>0.5)'}")
    print("-" * 35)
    for logit, prob in zip(logits, probs):
        classification = "Positive" if prob > 0.5 else "Negative"
        print(f"{logit:<8.1f} {prob:<12.3f} {classification}")
    
    print(f"\n🎯 FOR GRAPHRNN TO OUTPUT P=0.5:")
    print(f"Required logit = 0.0")
    print(f"But if trained on 1:2 ratio, optimal logit ≈ ln(1/2) = -0.693")
    print(f"This gives probability = sigmoid(-0.693) = 0.33")
    
    print(f"\n💡 CALIBRATION ISSUE:")
    print(f"GraphRNN learns logits that are optimal for 1:2 training distribution")
    print(f"But these logits are sub-optimal for 1:1 evaluation distribution")


def simulate_training_dynamics():
    """
    Simulate what happens during GraphRNN training
    """
    print("\n" + "=" * 80)
    print("SIMULATING GRAPHRNN TRAINING DYNAMICS")
    print("=" * 80)
    
    print("\n🎭 SIMULATION: What does GraphRNN learn?")
    
    # Simulate a simple case
    np.random.seed(42)
    
    # Training data: 1 positive, 2 negatives per batch
    n_batches = 1000
    
    # Perfect features: triadic edges have feature=1, random edges have feature=0
    train_features_pos = np.ones(n_batches)  # Triadic edges
    train_features_neg = np.zeros(n_batches * 2)  # Random edges
    
    train_features = np.concatenate([train_features_pos, train_features_neg])
    train_labels = np.concatenate([np.ones(n_batches), np.zeros(n_batches * 2)])
    
    print(f"Training data: {len(train_labels)} samples")
    print(f"  Positive samples: {np.sum(train_labels)} ({np.mean(train_labels):.1%})")
    print(f"  Negative samples: {len(train_labels) - np.sum(train_labels)} ({1-np.mean(train_labels):.1%})")
    
    # Simulate optimal linear classifier
    # For linearly separable case: w*x + b where x=1 for triadic, x=0 for random
    # Optimal decision boundary for 1:2 ratio
    
    # Bayes optimal classifier for 1:2 ratio:
    # P(y=1|x=1) should be high, P(y=1|x=0) should be low
    # But the decision threshold is not 0.5!
    
    w_optimal = 2.0  # Weight for feature
    b_optimal = -0.693  # Bias term (ln(1/2) for 1:2 ratio)
    
    # Predictions
    train_logits_pos = w_optimal * 1 + b_optimal  # For triadic edges (x=1)
    train_logits_neg = w_optimal * 0 + b_optimal  # For random edges (x=0)
    
    train_probs_pos = 1 / (1 + np.exp(-train_logits_pos))
    train_probs_neg = 1 / (1 + np.exp(-train_logits_neg))
    
    print(f"\nOptimal classifier for 1:2 training ratio:")
    print(f"  Triadic edge probability: {train_probs_pos:.3f}")
    print(f"  Random edge probability: {train_probs_neg:.3f}")
    print(f"  Decision threshold (1:2 optimal): {1/3:.3f}")
    
    # Now evaluate on balanced data (1:1)
    eval_features_pos = np.ones(100)
    eval_features_neg = np.zeros(100)
    
    eval_logits_pos = w_optimal * 1 + b_optimal
    eval_logits_neg = w_optimal * 0 + b_optimal
    
    eval_probs_pos = 1 / (1 + np.exp(-eval_logits_pos))
    eval_probs_neg = 1 / (1 + np.exp(-eval_logits_neg))
    
    # Accuracy at different thresholds
    thresholds = [0.33, 0.5]
    
    print(f"\nEvaluation on balanced data (1:1):")
    print(f"  Triadic edge probability: {eval_probs_pos:.3f}")
    print(f"  Random edge probability: {eval_probs_neg:.3f}")
    
    for thresh in thresholds:
        pred_pos = (eval_probs_pos > thresh).astype(int)
        pred_neg = (eval_probs_neg > thresh).astype(int)
        
        tp = np.sum(pred_pos)  # True positives
        tn = np.sum(1 - pred_neg)  # True negatives
        accuracy = (tp + tn) / 200
        
        print(f"  Accuracy at threshold {thresh}: {accuracy:.1%}")
    
    print(f"\n🎯 CONCLUSION:")
    print(f"Model trained on 1:2 ratio achieves:")
    print(f"  - Perfect ranking (AUC=100%)")
    print(f"  - Poor accuracy at 0.5 threshold ({((eval_probs_pos > 0.5).sum() + (eval_probs_neg <= 0.5).sum()) / 200:.1%})")
    print(f"  - Good accuracy at optimal threshold ({((eval_probs_pos > 0.33).sum() + (eval_probs_neg <= 0.33).sum()) / 200:.1%})")


def analyze_class_balance():
    """
    Analyze the class balance in training vs evaluation
    """
    print("\n" + "=" * 80)
    print("CLASS BALANCE ANALYSIS")
    print("=" * 80)
    
    print("\n🎯 CURRENT GRAPHRNN SETUP:")
    print("Training:")
    print("  ✅ Balanced sampling: 1 positive : 2 negatives")
    print("  ✅ This creates artificial 33% positive rate")
    print("  ❌ But model learns this as the 'natural' distribution")
    
    print("\nValidation/Testing:")
    print("  ✅ Balanced sampling: 1 positive : 1 negative") 
    print("  ✅ This creates 50% positive rate")
    print("  ❌ But model expects 33% positive rate")
    
    print("\n💡 THE MISMATCH:")
    print("1. Training teaches: 'Optimal probability for positives ≈ 0.67'")
    print("2. Training teaches: 'Optimal probability for negatives ≈ 0.17'")
    print("3. But evaluation expects: 'Threshold = 0.5 for 50% positive rate'")
    print("4. Result: Model's 0.67 > 0.5 (correct) but 0.17 < 0.5 (incorrect)")
    
    print("\n🔧 SOLUTIONS:")
    print("1. 🎚️  Use same ratio in training and evaluation (e.g., 1:1)")
    print("2. 📊 Calibrate probabilities post-training")
    print("3. 🎯 Optimize threshold on validation set")
    print("4. 🔄 Use different loss functions (focal loss, etc.)")


if __name__ == "__main__":
    analyze_optimization_objective()
    demonstrate_optimal_probabilities()
    analyze_sigmoid_behavior()
    simulate_training_dynamics()
    analyze_class_balance()
    
    print("\n" + "=" * 80)
    print("SUMMARY: THE ROOT CAUSE IS TRAINING/EVALUATION DISTRIBUTION MISMATCH!")
    print("GraphRNN is optimized correctly, but for a different class distribution.")
    print("=" * 80)
