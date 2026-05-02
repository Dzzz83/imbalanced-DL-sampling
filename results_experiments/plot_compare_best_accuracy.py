import matplotlib.pyplot as plt
import os
import re
import glob

def get_metrics(file_path):
    """
    Extracts the final Train Prec@1, the best Test Prec@1, and its epoch.
    """
    best_test_prec1 = 0.0
    final_train_prec1 = 0.0
    best_epoch = 0
    current_epoch = 0
   
    if not os.path.exists(file_path):
        return None
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            # Track epoch
            epoch_match = re.search(r"Epoch[:\s]*\[(\d+)\]", line)
            if epoch_match:
                current_epoch = int(epoch_match.group(1))
            # Final Train Accuracy
            if "Training Results:" in line and "|" not in line:
                match = re.search(r"Prec@1\s+([\d\.]+)", line)
                if match:
                    final_train_prec1 = float(match.group(1))
            # Best Test Accuracy
            elif "Testing Results:" in line and "|" not in line:
                match = re.search(r"Prec@1\s+([\d\.]+)", line)
                if match:
                    test_val = float(match.group(1))
                    if test_val > best_test_prec1:
                        best_test_prec1 = test_val
                        best_epoch = current_epoch
    return {
        'name': os.path.basename(file_path),
        'train': final_train_prec1,
        'test': best_test_prec1,
        'epoch': best_epoch
    }

def clean_label(filename):
    """
    Extracts exactly the ratio number (1.0, 0.9, 0.1, etc.) from the filename.
    Now supports none, lava, AND random!
    """
    match = re.search(r'(none|lava|random)([\d\.]+)', filename.lower())
    if match:
        return match.group(2)
    return filename

def get_ratio_for_sorting(filename):
    """
    Helper function to turn the extracted text into a real decimal number for perfect sorting.
    """
    match = re.search(r'(none|lava|random)([\d\.]+)', filename.lower())
    if match:
        return float(match.group(2))
    return -1.0

def plot_multi_comparison(source_paths, output_dir):
    """
    source_paths: list of exact folders or specific file paths
    output_dir: where to save the final plot
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    all_log_files = set()
    # 1. Collect all log files ONLY from the exact paths provided (No recursive searching)
    for path in source_paths:
        if os.path.isdir(path):
            all_log_files.update(glob.glob(os.path.join(path, "*.log")))
        elif os.path.isfile(path) and path.endswith(".log"):
            all_log_files.add(path)
    all_log_files = list(all_log_files)
    if not all_log_files:
        print(f"⚠️ No log files found in the specified paths: {source_paths}")
        return
    # 2. Extract metrics for every file
    results = []
    for f_path in all_log_files:
        data = get_metrics(f_path)
        if data and (data['train'] > 0 or data['test'] > 0):
            results.append(data)
    # ---------------------------------------------------------

    # ✨ THE FIX: Sort mathematically by ratio in descending order

    # ---------------------------------------------------------
    results = sorted(results, key=lambda x: get_ratio_for_sorting(x['name']), reverse=True)
    # 3. Prepare data for plotting
    labels = [clean_label(r['name']) for r in results]
    train_scores = [r['train'] for r in results]
    test_scores = [r['test'] for r in results]
    test_epochs = [r['epoch'] for r in results]
    x = range(len(labels))
    width = 0.35  
    # Adjust figure size based on the number of methods

    plt.style.use('ggplot')
    fig_width = max(12, len(labels) * 1.5)
    fig, ax = plt.subplots(figsize=(fig_width, 7))
    bars1 = ax.bar([pos - width/2 for pos in x], train_scores, width, label='Final Train Accuracy', color='#3498db')
    bars2 = ax.bar([pos + width/2 for pos in x], test_scores, width, label='Best Test Accuracy', color='#e74c3c')

    # Add text labels
    for i, (b1, b2) in enumerate(zip(bars1, bars2)):
        ax.annotate(f'{train_scores[i]:.2f}%',
                    xy=(b1.get_x() + b1.get_width() / 2, b1.get_height()),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontweight='bold', fontsize=9)
        ax.annotate(f'{test_scores[i]:.2f}%\n(Ep: {test_epochs[i]})',
                    xy=(b2.get_x() + b2.get_width() / 2, b2.get_height()),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontweight='bold', color='#c0392b', fontsize=9)
    # Formatting
    ax.set_ylabel('Accuracy (%)', fontweight='bold')

    # ✨ Added the X-axis Title here
    ax.set_xlabel('Ratios', fontweight='bold', fontsize=12)
   
    # You might want to update this title dynamically depending on what you run!
    ax.set_title('Imbalanced cifar10 RandOversamp_noisy0.5_2 None and Random selection ratios', fontsize=16, fontweight='bold')
    ax.set_xticks(x)

    # ✨ Changed rotation to 0 since '1.0' is short and fits perfectly flat
    ax.set_xticklabels(labels, fontweight='bold', rotation=0, ha='center', fontsize=11)
    ax.set_ylim(0, max(max(train_scores), max(test_scores)) + 15)
    ax.legend(loc='lower right')
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'Imbalanced cifar10 RandOversamp_noisy0.5_2 None and Random selection ratios.png')
    plt.savefig(save_path, dpi=300)
    print(f"🎉 Successfully saved multi-comparison plot to: {save_path}")
    plt.show()

if __name__ == "__main__":
    # --- CONFIGURE YOUR PATHS HERE ---
    folders_to_compare = [
        "../experiments/cifar10_exp0.01_noisy0.5_RO_2",
        "../experiments/cifar10_exp0.01_noisy0.5_RO_2/random"
    ]
    results_output = "cifar10_exp0.01_noisy0.5_RO_2/comparisons"
    print(f"🔍 Searching exactly inside: {folders_to_compare}")
    plot_multi_comparison(folders_to_compare, results_output)