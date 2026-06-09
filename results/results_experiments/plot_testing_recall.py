import matplotlib.pyplot as plt
import numpy as np
import os
import re

def get_best_recall_metrics(file_path):
    """
    Finds the epoch with the highest Prec@1 test score, 
    then flexibly extracts the Testing Class Recall for that specific epoch.
    """
    best_test_prec1 = 0.0
    best_epoch = -1
    current_epoch = -1
    
    class_recalls_by_epoch = {}
    
    if not os.path.exists(file_path):
        print(f"❌ ERROR: Cannot find file -> {file_path}")
        return None

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            lower_line = line.lower()
            
            # 1. Catch the epoch number
            epoch_match = re.search(r"epoch.*?(\d+)", lower_line)
            if epoch_match:
                current_epoch = int(epoch_match.group(1))

            # 2. Catch Best Test Accuracy (Ignores "Train" lines)
            if "prec@1" in lower_line and "train" not in lower_line:
                match = re.search(r"prec@1\s+([\d\.]+)", lower_line)
                if match:
                    test_val = float(match.group(1))
                    if test_val > best_test_prec1:
                        best_test_prec1 = test_val
                        best_epoch = current_epoch
                        
            # 3. Catch Testing Class Recall Array
            if "recall" in lower_line and "train" not in lower_line and "[" in line:
                matches = re.findall(r"\[(.*?)\]", line)
                if matches:
                    # Look at the LAST set of brackets on the line to avoid [183]
                    recall_string = matches[-1] 
                    raw_nums = re.findall(r"[\d\.]+", recall_string)
                    if len(raw_nums) >= 10:
                        recalls = [float(x) for x in raw_nums[:10]]
                        class_recalls_by_epoch[current_epoch] = recalls

    # Retrieve the best array for the best epoch
    if best_epoch != -1 and best_epoch in class_recalls_by_epoch:
        best_recalls = class_recalls_by_epoch[best_epoch]
    else:
        print(f"⚠️ Warning: Found Best Epoch {best_epoch}, but couldn't find its recall array in: {file_path}")
        best_recalls = [0.0] * 10  # Fallback

    return {
        'name': os.path.basename(file_path),
        'best_test_acc': best_test_prec1,
        'epoch': best_epoch,
        'recalls': best_recalls
    }

def plot_two_files_recall(file1_path, file2_path, label1, label2, output_dir, plot_title):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Extract data from both files
    print("📊 Extracting data...")
    data1 = get_best_recall_metrics(file1_path)
    data2 = get_best_recall_metrics(file2_path)

    if not data1 or not data2:
        print("⚠️ Aborting plot. Please check the file paths above.")
        return

    print(f"✅ {label1} Best Epoch: {data1['epoch']} (Acc: {data1['best_test_acc']}%)")
    print(f"✅ {label2} Best Epoch: {data2['epoch']} (Acc: {data2['best_test_acc']}%)")

    # Setup Chart Data
    classes = [f"Class {i}" for i in range(10)]
    y = np.arange(len(classes))  # Changed to Y-axis
    bar_height = 0.35            # Used for height of horizontal bars

    plt.style.use('ggplot')
    
    # Changed figure size to be taller than wide for horizontal bars
    fig, ax = plt.subplots(figsize=(10, 8)) 

    # Plot bars horizontally using barh
    bars1 = ax.barh(y - bar_height/2, data1['recalls'], bar_height, label=f'{label1} (Ep: {data1["epoch"]})', color='#3498db')
    bars2 = ax.barh(y + bar_height/2, data2['recalls'], bar_height, label=f'{label2} (Ep: {data2["epoch"]})', color='#e74c3c')

    # Add numbers next to the bars
    for bars in [bars1, bars2]:
        for bar in bars:
            width_val = bar.get_width()
            ax.annotate(f'{width_val:.3f}',
                        xy=(width_val, bar.get_y() + bar.get_height() / 2),
                        xytext=(5, 0),  # 5 points to the right
                        textcoords="offset points",
                        ha='left', va='center', fontsize=9, fontweight='bold')

    # Formatting (Axes swapped)
    ax.set_xlabel('Testing Recall Score', fontweight='bold', fontsize=12)
    ax.set_ylabel('Classes', fontweight='bold', fontsize=12)
    ax.set_title(plot_title, fontsize=16, fontweight='bold')
    
    ax.set_yticks(y)
    ax.set_yticklabels(classes, fontweight='bold', fontsize=11)
    
    # Invert the Y-axis so Class 0 is at the top
    ax.invert_yaxis()
    
    # Set X limit higher so annotations fit inside the graph
    ax.set_xlim(0, 1.15)
    
    ax.legend(loc='lower right')  # Moved legend to lower right to avoid blocking bars

    plt.tight_layout()
    
    # Dynamically name the saved file based on the plot title
    save_path = os.path.join(output_dir, f'{plot_title}.png')
    plt.savefig(save_path, dpi=300)
    print(f"\n🎉 Successfully saved horizontal recall comparison plot to: {save_path}")
    plt.show()

if __name__ == "__main__":
    # --- 1. UPDATE THESE EXACT FILE PATHS ---
    file_1 = "../experiments/cifar10_noisy_exp0.01_MaMix_drw/cifar10_noisy_none1.0_mamix_drw_exp0.01.log"
    file_2 = "../experiments/cifar10_noisy_exp0.01_MaMix_drw/lava/cifar10_noisy_lava0.3_mamix_drw_exp0.01_seed42_20260505_031900.log"
    
    # --- 2. GIVE THEM CUSTOM CLEAN NAMES FOR THE LEGEND ---
    name_1 = "None Selection"
    name_2 = "Lava Selection"
    
    # --- 3. SET THE EXACT TITLE & FILENAME HERE ---
    title_text = "Cifar10 exp 0.01 Mamix_drw noisy None and Lava selection 0.3"
    
    # --- 4. WHERE TO SAVE ---
    results_output = "cifar10_noisy_exp0.01_MaMix_drw/comparisons" 
    
    plot_two_files_recall(file_1, file_2, name_1, name_2, results_output, title_text)