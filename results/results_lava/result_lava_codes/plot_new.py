import matplotlib.pyplot as plt
import os
import re

def get_experts_metrics(file_path):
    """
    Extracts the final best model metrics from the Experts log file.
    Example line to parse: 
    => Best Model: Avg Acc = 38.73% | CE Head = 36.20% | LA Head = 39.13% | BS Head = 39.20%
    """
    metrics = {
        'CE Head': 0.0,
        'LA Head': 0.0,
        'BS Head': 0.0,
        'Avg': 0.0
    }
    
    if not os.path.exists(file_path):
        print(f"❌ ERROR: Cannot find file -> {file_path}")
        return None
        
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if "=> Best Model:" in line:
                # Use regex to dynamically pull out the numbers next to each metric
                ce_match = re.search(r"CE Head =\s*([\d\.]+)%", line)
                la_match = re.search(r"LA Head =\s*([\d\.]+)%", line)
                bs_match = re.search(r"BS Head =\s*([\d\.]+)%", line)
                avg_match = re.search(r"Avg Acc =\s*([\d\.]+)%", line)
                
                if ce_match: metrics['CE Head'] = float(ce_match.group(1))
                if la_match: metrics['LA Head'] = float(la_match.group(1))
                if bs_match: metrics['BS Head'] = float(bs_match.group(1))
                if avg_match: metrics['Avg'] = float(avg_match.group(1))
                
    return metrics

def plot_expert_heads(file_path, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    print(f"📊 Extracting data from: {os.path.basename(file_path)}...")
    metrics = get_experts_metrics(file_path)
    
    if not metrics or sum(metrics.values()) == 0:
        print("⚠️ Aborting plot. Could not find '=> Best Model:' line in the file.")
        return

    # Set up the labels in the exact order requested
    labels = ['CE Head', 'LA Head', 'BS Head', 'Avg']
    scores = [metrics[label] for label in labels]
    
    plt.style.use('ggplot')
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # ✨ Define four distinct colors: Red, Blue, Green, Purple
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6']
    
    # Plot the bars
    bars = ax.bar(labels, scores, color=colors, width=0.5)

    # Add the percentage values on top of each bar
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.annotate(f'{height:.2f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 5),  # 5 points above the bar
                    textcoords="offset points",
                    ha='center', va='bottom', fontweight='bold', fontsize=11, color='#2c3e50')

    # Formatting the Chart
    ax.set_ylabel('Accuracy (%)', fontweight='bold', fontsize=12)
    ax.set_title('Best Model Accuracies by Head', fontsize=16, fontweight='bold')
    
    ax.set_xticklabels(labels, fontweight='bold', fontsize=12)
    
    # Set the Y-limit slightly higher than the max score so the text isn't cut off
    ax.set_ylim(0, max(scores) + 10)
    
    plt.tight_layout()
    
    # Save the plot
    save_path = os.path.join(output_dir, 'Experts_Heads_Comparison.png')
    plt.savefig(save_path, dpi=300)
    print(f"🎉 Successfully saved plot to: {save_path}")
    plt.show()

if __name__ == "__main__":
    # --- 1. UPDATE THIS PATH TO YOUR EXACT LOG FILE ---
    target_log_file = "cifar100_none1.0_experts_exp0.01_seed42_20260702_032216.log"
    
    # --- 2. WHERE TO SAVE THE IMAGE ---
    results_output = "." 
    
    plot_expert_heads(target_log_file, results_output)