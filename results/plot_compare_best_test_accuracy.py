import matplotlib.pyplot as plt
import os
import re

def get_metrics(file_path):
    """
    Extracts the final Train Prec@1, the best Test Prec@1, AND the epoch it occurred on.
    """
    best_test_prec1 = 0.0
    final_train_prec1 = 0.0
    best_epoch = 0
    current_epoch = 0
    
    if not os.path.exists(file_path):
        print(f"❌ ERROR: Cannot find file -> {file_path}")
        return None, None, None

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            # 1. Update the current epoch tracker
            # Matches formats like "Epoch: [199]" or "Epoch [196]"
            epoch_match = re.search(r"Epoch[:\s]*\[(\d+)\]", line)
            if epoch_match:
                current_epoch = int(epoch_match.group(1))

            # 2. Get the Final Train Accuracy (Constantly overwritten until the end)
            if "Training Results:" in line and "|" not in line:
                match = re.search(r"Prec@1\s+([\d\.]+)", line)
                if match:
                    final_train_prec1 = float(match.group(1))
                    
            # 3. Look for Test Accuracy and track the Best + Epoch
            elif "Testing Results:" in line and "|" not in line:
                match = re.search(r"Prec@1\s+([\d\.]+)", line)
                if match:
                    test_val = float(match.group(1))
                    if test_val > best_test_prec1:
                        best_test_prec1 = test_val
                        best_epoch = current_epoch

    print(f"✅ Read {os.path.basename(file_path)}:")
    print(f"   -> Final Train Acc: {final_train_prec1}%")
    print(f"   -> Best Test Acc:   {best_test_prec1}% (Achieved at Epoch {best_epoch})\n")
    return final_train_prec1, best_test_prec1, best_epoch

def plot_comparison(lava_file, random_file, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Extract all data, including the best epochs
    lava_train, lava_test, lava_ep = get_metrics(lava_file)
    random_train, random_test, random_ep = get_metrics(random_file)

    if lava_test is None or random_test is None:
        print("⚠️ WARNING: One or both files were not found. Please check paths.")
        return

    # Setup Chart Data
    labels = ['Lava Selection', 'Random Selection']
    train_scores = [lava_train, random_train]
    test_scores = [lava_test, random_test]
    test_epochs = [lava_ep, random_ep] # <--- Added epoch array

    x = range(len(labels))
    width = 0.35  

    plt.style.use('ggplot')
    fig, ax = plt.subplots(figsize=(10, 6))

    bars1 = ax.bar([pos - width/2 for pos in x], train_scores, width, label='Final Train Accuracy', color='#3498db')
    bars2 = ax.bar([pos + width/2 for pos in x], test_scores, width, label='Best Test Accuracy', color='#e74c3c')

    # Add text labels for Training Bars
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  
                    textcoords="offset points",
                    ha='center', va='bottom', fontweight='bold')

    # Add text labels AND Epoch for Testing Bars
    for i, bar in enumerate(bars2):
        height = bar.get_height()
        best_ep = test_epochs[i]
        
        # Adding \n puts the epoch number on a second line above the bar
        ax.annotate(f'{height:.2f}%\n(Ep: {best_ep})', 
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  
                    textcoords="offset points",
                    ha='center', va='bottom', fontweight='bold', color='#c0392b')

    # Formatting
    ax.set_ylabel('Accuracy (%)', fontweight='bold')
    ax.set_title('Comparison between Lava Selection and Random Selection', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontweight='bold', fontsize=12)
    
    # Scale Y-axis higher so the 2-line text labels don't get cut off at the top
    max_val = max(max(train_scores), max(test_scores))
    ax.set_ylim(0, max_val + 18) 
    
    # Move legend to the lower right so it doesn't cover the high bars
    ax.legend(loc='lower right')

    plt.tight_layout()
    save_path = os.path.join(output_dir, 'comparison_lava_vs_random.png')
    plt.savefig(save_path, dpi=300)
    print(f"🎉 Successfully saved bar chart to: {save_path}")
    plt.show()

if __name__ == "__main__":
    # --- UPDATE THESE PATHS ---
    file_lava = "train/cifar10_lava_0.7_exp_0.01_ERM_200_seed.log"
    file_random = "train/cifar10_random_0.7_exp_0.01_ERM_200_seed.log"
    results_output = "plot"
    
    plot_comparison(file_lava, file_random, results_output)