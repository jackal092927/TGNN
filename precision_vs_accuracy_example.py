"""
Precision vs Accuracy: Clear Examples for Link Prediction
"""

import numpy as np
from sklearn.metrics import precision_score, accuracy_score, confusion_matrix
import pandas as pd

def precision_vs_accuracy_example():
    """Clear example showing difference between precision and accuracy"""
    
    print("=== Precision vs Accuracy in Link Prediction ===\n")
    
    # Example: Model predicting which edges will form
    print("Scenario: Predicting 10 potential edges")
    print("---------------------------------------")
    
    # Ground truth: 1 = edge will form, 0 = edge won't form
    y_true = np.array([1, 0, 1, 0, 0, 1, 0, 0, 1, 0])
    
    # Model predictions: 1 = predict edge will form, 0 = predict no edge
    y_pred = np.array([1, 0, 1, 1, 0, 1, 0, 1, 0, 0])
    
    print("Ground Truth (1=edge forms, 0=no edge):")
    print(f"  {y_true}")
    print("Model Predictions (1=predict edge, 0=predict no edge):")
    print(f"  {y_pred}")
    
    # Create detailed analysis table
    print(f"\nDetailed Analysis:")
    print("Edge | True | Pred | Result")
    print("-----|------|------|--------")
    
    tp = fp = tn = fn = 0
    
    for i, (true, pred) in enumerate(zip(y_true, y_pred)):
        if true == 1 and pred == 1:
            result = "TP (True Positive)"
            tp += 1
        elif true == 0 and pred == 1:
            result = "FP (False Positive)"
            fp += 1
        elif true == 0 and pred == 0:
            result = "TN (True Negative)"
            tn += 1
        else:  # true == 1 and pred == 0
            result = "FN (False Negative)"
            fn += 1
        
        print(f"{i+1:4d} |  {true}   |  {pred}   | {result}")
    
    print(f"\n=== Confusion Matrix ===")
    print(f"                Predicted")
    print(f"                0    1")
    print(f"Actual    0   {tn:2d}   {fp:2d}  (TN=True Neg, FP=False Pos)")
    print(f"          1   {fn:2d}   {tp:2d}  (FN=False Neg, TP=True Pos)")
    
    # Calculate metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    accuracy = (tp + tn) / (tp + fp + tn + fn)
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    print(f"\n=== Manual Calculations ===")
    print(f"True Positives (TP)  = {tp} (correctly predicted edges)")
    print(f"False Positives (FP) = {fp} (wrongly predicted edges)")
    print(f"True Negatives (TN)  = {tn} (correctly predicted no-edges)")
    print(f"False Negatives (FN) = {fn} (missed real edges)")
    
    print(f"\nPrecision = TP / (TP + FP) = {tp} / ({tp} + {fp}) = {precision:.3f}")
    print(f"Accuracy  = (TP + TN) / Total = ({tp} + {tn}) / {tp + fp + tn + fn} = {accuracy:.3f}")
    print(f"Recall    = TP / (TP + FN) = {tp} / ({tp} + {fn}) = {recall:.3f}")
    
    # Verify with sklearn
    sklearn_precision = precision_score(y_true, y_pred)
    sklearn_accuracy = accuracy_score(y_true, y_pred)
    
    print(f"\n=== Verification with sklearn ===")
    print(f"Manual Precision:  {precision:.3f}")
    print(f"Sklearn Precision: {sklearn_precision:.3f}")
    print(f"Manual Accuracy:   {accuracy:.3f}")
    print(f"Sklearn Accuracy:  {sklearn_accuracy:.3f}")
    
    return precision, accuracy

def interpretation_examples():
    """Show what precision and accuracy mean in practice"""
    
    print(f"\n=== What These Metrics Tell Us ===\n")
    
    # Scenario 1: High Precision, Lower Accuracy
    print("Scenario 1: Conservative Model (High Precision, Lower Accuracy)")
    print("--------------------------------------------------------------")
    y_true_1 = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])  # 4 real edges, 6 non-edges
    y_pred_1 = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])  # Only predicts 2 edges
    
    prec_1 = precision_score(y_true_1, y_pred_1)
    acc_1 = accuracy_score(y_true_1, y_pred_1)
    
    print(f"True:  {y_true_1}")
    print(f"Pred:  {y_pred_1}")
    print(f"Precision: {prec_1:.3f} (when model says 'edge', it's usually right)")
    print(f"Accuracy:  {acc_1:.3f} (overall correctness)")
    print("Interpretation: Model is cautious, only predicts edges when very confident")
    
    # Scenario 2: Lower Precision, Higher Accuracy  
    print(f"\nScenario 2: Aggressive Model (Lower Precision, Higher Accuracy)")
    print("--------------------------------------------------------------")
    y_true_2 = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])  # 2 real edges, 8 non-edges
    y_pred_2 = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])  # Predicts 4 edges
    
    prec_2 = precision_score(y_true_2, y_pred_2)
    acc_2 = accuracy_score(y_true_2, y_pred_2)
    
    print(f"True:  {y_true_2}")
    print(f"Pred:  {y_pred_2}")
    print(f"Precision: {prec_2:.3f} (when model says 'edge', it's often wrong)")
    print(f"Accuracy:  {acc_2:.3f} (overall correctness)")
    print("Interpretation: Model is aggressive, makes many false positive predictions")

def link_prediction_context():
    """Explain precision vs accuracy in our TGAM/TGIB context"""
    
    print(f"\n=== In Our TGAM/TGIB Context ===\n")
    
    print("What we're predicting:")
    print("• Input: Graph history + candidate edge")
    print("• Output: Probability that edge will form")
    print("• Threshold: Usually 0.5 (>0.5 = predict edge, <0.5 = predict no edge)")
    
    print(f"\nIn our experiments:")
    print("• We evaluate PAIRS: 1 positive (real edge) + 1 negative (fake edge)")
    print("• Perfect model: real edge gets high score (>0.5), fake edge gets low score (<0.5)")
    print("• This gives us precision and accuracy metrics")
    
    print(f"\nWhy precision matters for link prediction:")
    print("✅ High Precision = When we predict an edge will form, we're usually right")
    print("✅ Important for: Recommendation systems, social network analysis")
    print("✅ Avoids: False alarms, wasted resources on unlikely connections")
    
    print(f"\nWhy accuracy matters:")
    print("✅ High Accuracy = Overall correct predictions (both edges and non-edges)")
    print("✅ Important for: General model reliability")
    print("✅ Balances: Both false positives and false negatives")
    
    print(f"\nOur results interpretation:")
    results = {
        'TGAM Individual': {'acc': 0.80, 'ap': 0.925},
        'TGIB Original': {'acc': 0.775, 'ap': 0.950},
    }
    
    for model, scores in results.items():
        print(f"\n{model}:")
        print(f"  Accuracy: {scores['acc']:.3f} → Gets ~{scores['acc']*100:.0f}% of all predictions right")
        print(f"  AP Score: {scores['ap']:.3f} → Excellent at ranking real edges above fake ones")

def when_to_use_which():
    """When to focus on precision vs accuracy"""
    
    print(f"\n=== When to Focus on Which Metric ===\n")
    
    print("Focus on PRECISION when:")
    print("• False positives are costly")
    print("• You want high confidence in positive predictions") 
    print("• Example: Medical diagnosis, fraud detection, link recommendations")
    
    print(f"\nFocus on ACCURACY when:")
    print("• You care about overall performance")
    print("• False positives and false negatives are equally important")
    print("• Example: General classification tasks, balanced datasets")
    
    print(f"\nFocus on AP (Average Precision) when:")
    print("• You care about ranking quality")
    print("• You want to evaluate across different thresholds")
    print("• Example: Information retrieval, recommendation systems, link prediction")
    
    print(f"\nIn our case (link prediction):")
    print("• AP is most important (ranking quality)")
    print("• Precision is secondary (confidence in positive predictions)")
    print("• Accuracy is least important (can be skewed by class imbalance)")

if __name__ == '__main__':
    # Run all examples
    precision, accuracy = precision_vs_accuracy_example()
    interpretation_examples()
    link_prediction_context()
    when_to_use_which()
    
    print(f"\n=== Key Takeaways ===")
    print("1. Precision = 'Of my positive predictions, how many were right?'")
    print("2. Accuracy = 'Of all my predictions, how many were right?'")
    print("3. AP (Average Precision) = 'How well do I rank positives above negatives?'")
    print("4. For link prediction, AP > Precision > Accuracy in importance")
    print("5. Our models achieve high AP (~0.93-0.95), indicating excellent performance!") 