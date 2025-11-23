#!/usr/bin/env python3
"""
Analyze Android Expert misclassifications to find patterns
"""

import json
import sys
from pathlib import Path
import numpy as np

def analyze_latest_results():
    """Analyze the most recent Android Expert results"""

    results_dir = Path("results")
    android_dirs = sorted([d for d in results_dir.glob("AndroidExpert_*") if d.is_dir()])

    if not android_dirs:
        print("No Android Expert results found")
        sys.exit(1)

    latest_dir = android_dirs[-1]
    json_path = latest_dir / "android_expert_evaluation.json"

    if not json_path.exists():
        print(f"No evaluation JSON found in {latest_dir}")
        sys.exit(1)

    print(f"Analyzing: {latest_dir.name}")
    print("="*80)

    with open(json_path, 'r') as f:
        results = json.load(f)

    print(f"\nOverall Accuracy: {results['test_accuracy']*100:.2f}%\n")

    # Per-class performance
    print("Per-Class Performance:")
    print("-"*80)
    print(f"{'Class':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
    print("-"*80)

    per_class = results['per_class_metrics']
    for class_name, metrics in per_class.items():
        print(f"{class_name:<15} {metrics['precision']:<12.4f} {metrics['recall']:<12.4f} "
              f"{metrics['f1_score']:<12.4f} {metrics['support']:<10}")

    # Confusion matrix analysis
    cm = np.array(results['confusion_matrix'])
    classes = results['classes']

    print("\n\nConfusion Matrix:")
    print("-"*80)

    # Print header
    header = "Actual \\ Pred"
    print(f"{header:<15}", end="")
    for cls in classes:
        print(f"{cls:<15}", end="")
    print()
    print("-"*80)

    # Print matrix
    for i, actual_class in enumerate(classes):
        print(f"{actual_class:<15}", end="")
        for j, pred_class in enumerate(classes):
            count = cm[i][j]
            if i == j:
                # Correct predictions (diagonal) - highlight
                print(f"\033[92m{count:<15}\033[0m", end="")
            else:
                # Misclassifications
                if count > 0:
                    print(f"\033[91m{count:<15}\033[0m", end="")
                else:
                    print(f"{count:<15}", end="")
        print()

    print("\n\nMost Common Misclassifications:")
    print("-"*80)

    # Find top misclassifications
    misclassifications = []
    for i, actual in enumerate(classes):
        for j, predicted in enumerate(classes):
            if i != j and cm[i][j] > 0:
                misclassifications.append((cm[i][j], actual, predicted))

    misclassifications.sort(reverse=True)

    for count, actual, predicted in misclassifications[:5]:
        percentage = (count / cm.sum()) * 100
        print(f"  {actual} → {predicted}: {count} times ({percentage:.1f}% of test set)")

    # Insights
    print("\n\nInsights:")
    print("-"*80)

    # Find hardest class to classify (lowest recall)
    worst_recall = min(per_class.items(), key=lambda x: x[1]['recall'])
    print(f"• Hardest to identify: {worst_recall[0]} (recall: {worst_recall[1]['recall']:.2%})")

    # Find most confused pairs
    if misclassifications:
        top_confusion = misclassifications[0]
        print(f"• Most confused pair: {top_confusion[1]} ↔ {top_confusion[2]} ({top_confusion[0]} errors)")

    # Check if any class is much worse
    recalls = [m['recall'] for m in per_class.values()]
    if max(recalls) - min(recalls) > 0.15:
        print(f"• Large variance in recall ({min(recalls):.2%} to {max(recalls):.2%}) - class imbalance issue")
    else:
        print(f"• Balanced performance across classes (recall range: {min(recalls):.2%} to {max(recalls):.2%})")

    print("\n")

if __name__ == "__main__":
    analyze_latest_results()
