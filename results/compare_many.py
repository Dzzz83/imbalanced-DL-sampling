import matplotlib.pyplot as plt
import os
import re
import glob

def get_final_metrics(file_path):
    """
    Extracts the strictly FINAL Train Prec@1 and FINAL Test Prec@1.
    """
    final_train_prec1 = 0.0
    final_test_prec1 = 0.0
    
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

    return final_train_prec1, final_test_prec1

def plot_multi_method_comparison(log_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 1. Find all log files in the folder
    log_files = glob.glob(os.path.join(log_dir, "*.log"))
    
    if not log_files:
        print(f"⚠️ WARNING: No .log files found in '{log_dir}'.")
        return

    labels = []
    train_scores = []
    test_scores = []

    # 2. Extract data from every file found
    print("📊 Scanning files and extracting data...")
    for file_path in sorted(log_files): # Sorting keeps the chart order alphabetical
        file_name = os.path.basename(file_path)
        
        train_acc, test_acc = get_final_metrics(file_path)
        
        if train_acc > 0 or test_acc > 0:
            # Clean up the label name (remove .log and maybe some common redundant text if you want)
            clean_label = file_name.replace('.log', '').replace('cifar10_', '')
            
            labels.append(clean_label)
            train_scores.append(train_acc)
            test_scores.append(test_acc)
            print(f"✅ {clean_label} -> Train: {train_acc}%, Test: {test_acc}%")

    if not labels:
        print("⚠️ No valid data found in the logs.")
        return

    # 3. Setup Chart Data
    x = range(len(labels))
    width = 0.35  

    plt.style.use('ggplot')
    # Dynamically scale the width of the chart based on how many methods you are comparing
    fig_width = max(10, len(labels) * 2) 
    fig, ax = plt.subplots(figsize=(fig_width, 6))

    bars1 = ax.bar([pos - width/2 for pos in x], train_scores, width, label='Final Train Accuracy', color='#3498db')
    bars2 = ax.bar([pos + width/2 for pos in x], test_scores, width, label='Final Test Accuracy', color='#e74c3c')

    # 4. Add text labels for Training Bars
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  
                    textcoords="offset points",
                    ha='center', va='bottom', fontweight='bold', fontsize=9)

    # 5. Add text labels for Testing Bars
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}%', 
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  
                    textcoords="offset points",
                    ha='center', va='bottom', fontweight='bold', fontsize=9, color='#c0392b')

    # 6. Formatting
    ax.set_ylabel('Accuracy (%)', fontweight='bold')
    ax.set_title('Comparison of Multiple Selection Methods', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    
    # Rotate the x-axis labels if there are many methods, so they don't overlap
    ax.set_xticklabels(labels, fontweight='bold', fontsize=10, rotation=25, ha='right')
    
    max_val = max(max(train_scores), max(test_scores))
    ax.set_ylim(0, max_val + 15) 
    
    # Put legend outside or at the lower right to avoid hiding bars
    ax.legend(loc='lower right')

    plt.tight_layout()
    save_path = os.path.join(output_dir, 'multi_method_comparison.png')
    plt.savefig(save_path, dpi=300)
    print(f"\n🎉 Successfully saved bar chart to: {save_path}")
    plt.show()

if __name__ == "__main__":
    # --- UPDATE THESE PATHS ---
    # This should be the folder containing all your Random, Lava, DRW, etc. logs
    source_folder = "train" 
    results_output = "plot"
    
    plot_multi_method_comparison(source_folder, results_output)