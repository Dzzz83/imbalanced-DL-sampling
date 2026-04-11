# compare_logs.py
import os
import re
import glob
import matplotlib.pyplot as plt
import numpy as np

def extract_best_accuracy(log_path):
    """Extract the best test accuracy (Prec@1) from a log file."""
    best_acc = None
    with open(log_path, 'r') as f:
        for line in f:
            # Look for "Best Prec@1: XX.XXX"
            match = re.search(r'Best Prec@1:\s*([\d\.]+)', line)
            if match:
                best_acc = float(match.group(1))
            # Fallback: final epoch test accuracy
            match2 = re.search(r'Testing Results: Prec@1\s+([\d\.]+)', line)
            if match2:
                best_acc = float(match2.group(1))
    return best_acc

def main():
    log_dir = '/home/phatht/phat/imbalanced-DL-sampling/results_lava_test_exp2/train'
    log_files = glob.glob(os.path.join(log_dir, '*.log'))
    if not log_files:
        print(f"No log files found in {log_dir}")
        return

    # Sort by modification time or name to have consistent order
    log_files.sort(key=os.path.getmtime)

    names = []
    accuracies = []
    for log in log_files:
        acc = extract_best_accuracy(log)
        if acc is None:
            print(f"Warning: Could not extract accuracy from {os.path.basename(log)}")
            continue
        # Use a short identifier: e.g., timestamp part or index
        # Just use index to keep names short
        names.append(f"Run {len(names)+1}")
        accuracies.append(acc)

    if not names:
        print("No valid accuracy data found.")
        return

    # Sort by accuracy descending
    sorted_indices = np.argsort(accuracies)[::-1]
    names_sorted = [names[i] for i in sorted_indices]
    acc_sorted = [accuracies[i] for i in sorted_indices]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(names_sorted, acc_sorted, color='skyblue')
    plt.ylabel('Test Accuracy (%)')
    plt.title('Comparison of Test Accuracies (Best Prec@1)')
    plt.ylim(min(acc_sorted) - 2, max(acc_sorted) + 2)
    for bar, acc in zip(bars, acc_sorted):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                 f'{acc:.2f}%', ha='center', va='bottom', fontsize=9)
    plt.tight_layout()
    output_img = 'log_comparison.png'
    plt.savefig(output_img, dpi=150)
    print(f"Bar chart saved as {output_img}")
    plt.show()

if __name__ == '__main__':
    main()