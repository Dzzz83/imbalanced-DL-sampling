import matplotlib.pyplot as plt
import numpy as np
import os
import re

def get_best_metrics(file_path):
    """
    Finds the epoch with the highest Prec@1 test score, 
    then checks if 'Testing Group Acc' exists. If it does, it extracts it.
    Otherwise, it falls back to extracting 'Testing Class Recall'.
    """
    best_test_prec1 = 0.0
    best_epoch = -1
    current_epoch = -1
    
    class_recalls_by_epoch = {}
    group_accs_by_epoch = {}
    
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

            # 2. Catch Best Test Accuracy to lock in the best epoch
            if "testing results:" in lower_line and "|" not in lower_line:
                match = re.search(r"prec@1\s+([\d\.]+)", lower_line)
                if match:
                    test_val = float(match.group(1))
                    if test_val > best_test_prec1:
                        best_test_prec1 = test_val
                        best_epoch = current_epoch
                        
            # 3. Catch Testing Group Acc Array (High Priority)
            if "testing group acc" in lower_line and "[" in line:
                matches = re.findall(r"\[(.*?)\]", line)
                if matches:
                    acc_string = matches[-1]
                    raw_nums = re.findall(r"[\d\.]+", acc_string)
                    if raw_nums:
                        group_accs = [float(x) for x in raw_nums]
                        group_accs_by_epoch[current_epoch] = group_accs

            # 4. Catch Testing Class Recall Array (Fallback)
            if "testing class recall" in lower_line and "[" in line:
                matches = re.findall(r"\[(.*?)\]", line)
                if matches:
                    recall_string = matches[-1] 
                    raw_nums = re.findall(r"[\d\.]+", recall_string)
                    if len(raw_nums) >= 10:
                        recalls = [float(x) for x in raw_nums[:10]]
                        class_recalls_by_epoch[current_epoch] = recalls

    # Determine which metric to return (Prioritize Group Accuracy)
    if best_epoch != -1 and best_epoch in group_accs_by_epoch:
        best_values = group_accs_by_epoch[best_epoch]
        metric_type = "group_acc"
    elif best_epoch != -1 and best_epoch in class_recalls_by_epoch:
        best_values = class_recalls_by_epoch[best_epoch]
        metric_type = "recall"
    else:
        print(f"⚠️ Warning: Found Best Epoch {best_epoch}, but couldn't find any metrics in: {file_path}")
        best_values = [0.0] * 10  # Fallback
        metric_type = "recall"

    return {
        'name': os.path.basename(file_path),
        'best_test_acc': best_test_prec1,
        'epoch': best_epoch,
        'values': best_values,
        'metric_type': metric_type
    }

def plot_two_files_comparison(file1_path, file2_path, label1, label2, output_dir, plot_title):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Extract data from both files
    print("📊 Extracting data...")
    data1 = get_best_metrics(file1_path)
    data2 = get_best_metrics(file2_path)

    if not data1 or not data2:
        print("⚠️ Aborting plot. Please check the file paths above.")
        return

    print(f"✅ {label1} Best Epoch: {data1['epoch']} (Acc: {data1['best_test_acc']}%)")
    print(f"✅ {label2} Best Epoch: {data2['epoch']} (Acc: {data2['best_test_acc']}%)")

    plt.style.use('ggplot')

    # =========================================================================
    # ✨ VERTICAL PLOT LOGIC (For Testing Group Acc)
    # =========================================================================
    if data1['metric_type'] == "group_acc":
        print("🎯 'Testing Group Acc' found! Plotting VERTICAL custom groups.")
        num_items = len(data1['values'])
        
        if num_items == 3:
            labels = ["Majority Group", "Middle Group", "Minority Group"]
        else:
            labels = [f"Group {i}" for i in range(num_items)]
            
        x = np.arange(len(labels))  
        bar_width = 0.35            

        fig, ax = plt.subplots(figsize=(8, 6)) 

        # Plot bars vertically
        bars1 = ax.bar(x - bar_width/2, data1['values'], bar_width, label=f'{label1} (Ep: {data1["epoch"]})', color='#3498db')
        bars2 = ax.bar(x + bar_width/2, data2['values'], bar_width, label=f'{label2} (Ep: {data2["epoch"]})', color='#e74c3c')

        # Add numbers on top of the bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height_val = bar.get_height()
                ax.annotate(f'{height_val:.3f}',
                            xy=(bar.get_x() + bar.get_width() / 2, height_val),
                            xytext=(0, 5),  # 5 points above
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=9, fontweight='bold')

        # Formatting Vertical Chart
        ax.set_ylabel('Testing Group Accuracy', fontweight='bold', fontsize=12)
        ax.set_xlabel('Groups', fontweight='bold', fontsize=12)
        ax.set_title(plot_title, fontsize=16, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontweight='bold', fontsize=11)
        
        # Set Y limit higher so annotations fit inside the graph
        max_val = max(max(data1['values']), max(data2['values']))
        ax.set_ylim(0, max_val * 1.2 + 0.05)
        ax.legend(loc='upper right') 

    # =========================================================================
    # ⏪ HORIZONTAL PLOT LOGIC (Fallback for Class Recall)
    # =========================================================================
    else:
        print("📊 'Testing Class Recall' found. Plotting HORIZONTALLY.")
        num_items = len(data1['values'])
        labels = [f"Class {i}" for i in range(num_items)]
        
        y = np.arange(len(labels))  
        bar_height = 0.35            

        fig, ax = plt.subplots(figsize=(10, 8)) 

        # Plot bars horizontally
        bars1 = ax.barh(y - bar_height/2, data1['values'], bar_height, label=f'{label1} (Ep: {data1["epoch"]})', color='#3498db')
        bars2 = ax.barh(y + bar_height/2, data2['values'], bar_height, label=f'{label2} (Ep: {data2["epoch"]})', color='#e74c3c')

        # Add numbers next to the bars
        for bars in [bars1, bars2]:
            for bar in bars:
                width_val = bar.get_width()
                ax.annotate(f'{width_val:.3f}',
                            xy=(width_val, bar.get_y() + bar.get_height() / 2),
                            xytext=(5, 0),  
                            textcoords="offset points",
                            ha='left', va='center', fontsize=9, fontweight='bold')

        # Formatting Horizontal Chart
        ax.set_xlabel('Testing Recall Score', fontweight='bold', fontsize=12)
        ax.set_ylabel('Classes', fontweight='bold', fontsize=12)
        ax.set_title(plot_title, fontsize=16, fontweight='bold')
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontweight='bold', fontsize=11)
        ax.invert_yaxis() # Item 0 at top
        ax.set_xlim(0, max(max(data1['values']), max(data2['values'])) * 1.2 + 0.1)
        ax.legend(loc='lower right') 

    # =========================================================================
    
    plt.tight_layout()
    
    # Save the file
    save_path = os.path.join(output_dir, f'{plot_title}.png')
    plt.savefig(save_path, dpi=300)
    print(f"\n🎉 Successfully saved comparison plot to: {save_path}")
    plt.show()

if __name__ == "__main__":
    # --- 1. UPDATE THESE EXACT FILE PATHS ---
    file_1 = "./seed9999/cifar100_drw/baselines/cifar100_none1.0_drw_exp0.01_seed9999_20260626_023559.log"
    file_2 = "./seed9999/cifar100_drw/cifar100_sava1.0_sava_reweight_drw_exp0.01_seed9999_20260626_025550.log"
    
    # --- 2. GIVE THEM CUSTOM CLEAN NAMES FOR THE LEGEND ---
    name_1 = "Erm"
    name_2 = "Sava_reweight"
    
    # --- 3. SET THE EXACT TITLE & FILENAME HERE ---
    title_text = "Cifar100 Drw_seed9999: Erm and Sava_reweight"
    
    # --- 4. WHERE TO SAVE ---
    results_output = "./comparisons/seed9999" 
    
    plot_two_files_comparison(file_1, file_2, name_1, name_2, results_output, title_text)