import matplotlib.pyplot as plt
import os
import re

def get_final_metrics(file_path):
    """
    Extracts the strictly FINAL Train Prec@1 and FINAL Test Prec@1.
    """
    final_train_prec1 = 0.0
    final_test_prec1 = 0.0
    
    if not os.path.exists(file_path):
        print(f"❌ ERROR: Cannot find file -> {file_path}")
        return None, None

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            # 1. Continually overwrite Train Accuracy to get the final one
            if "Training Results:" in line and "|" not in line:
                match = re.search(r"Prec@1\s+([\d\.]+)", line)
                if match:
                    final_train_prec1 = float(match.group(1))
                    
            # 2. Continually overwrite Test Accuracy to get the final one
            elif "Testing Results:" in line and "|" not in line:
                match = re.search(r"Prec@1\s+([\d\.]+)", line)
                if match:
                    final_test_prec1 = float(match.group(1))

    print(f"✅ Read {os.path.basename(file_path)}:")
    print(f"   -> Final Train Acc: {final_train_prec1}%")
    print(f"   -> Final Test Acc:  {final_test_prec1}%\n")
    return final_train_prec1, final_test_prec1

def plot_final_comparison(lava_file, random_file, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Extract the final data (no epochs returned anymore)
    lava_train, lava_test = get_final_metrics(lava_file)
    random_train, random_test = get_final_metrics(random_file)

    if lava_test is None or random_test is None:
        print("⚠️ WARNING: One or both files were not found. Please check paths.")
        return

    # Setup Chart Data
    labels = ['Lava Selection', 'Random Selection']
    train_scores = [lava_train, random_train]
    test_scores = [lava_test, random_test]

    x = range(len(labels))
    width = 0.35  

    plt.style.use('ggplot')
    fig, ax = plt.subplots(figsize=(10, 6))

    bars1 = ax.bar([pos - width/2 for pos in x], train_scores, width, label='Final Train Accuracy', color='#3498db')
    bars2 = ax.bar([pos + width/2 for pos in x], test_scores, width, label='Final Test Accuracy', color='#e74c3c')

    # Add text labels for Training Bars
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  
                    textcoords="offset points",
                    ha='center', va='bottom', fontweight='bold')

    # Add text labels for Testing Bars (Cleaned up, no epochs)
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}%', 
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  
                    textcoords="offset points",
                    ha='center', va='bottom', fontweight='bold', color='#c0392b')

    # Formatting
    ax.set_ylabel('Accuracy (%)', fontweight='bold')
    ax.set_title('Comparison between Lava Selection and Random Selection', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontweight='bold', fontsize=12)
    
    # Scale Y-axis slightly higher so labels don't get cut off
    max_val = max(max(train_scores), max(test_scores))
    ax.set_ylim(0, max_val + 10) 
    
    ax.legend(loc='lower right')

    plt.tight_layout()
    save_path = os.path.join(output_dir, 'comparison_lava_vs_random_final.png')
    plt.savefig(save_path, dpi=300)
    print(f"🎉 Successfully saved bar chart to: {save_path}")
    plt.show()

if __name__ == "__main__":
    # --- UPDATE THESE PATHS ---
    file_lava = "train/cifar10_lava_0.7_exp_0.01_ERM_200_seed.log"
    file_random = "train/cifar10_random_0.7_exp_0.01_ERM_200_seed.log"
    results_output = "plot"
    
    plot_final_comparison(file_lava, file_random, results_output)