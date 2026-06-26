import matplotlib.pyplot as plt
import os
import re
import glob

def get_metrics(file_path):
    """
    Extracts ONLY the best Test Prec@1 and its epoch.
    """
    best_test_prec1 = 0.0
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
               
            # Best Test Accuracy Only
            if "Testing Results:" in line and "|" not in line:
                match = re.search(r"Prec@1\s+([\d\.]+)", line)
                if match:
                    test_val = float(match.group(1))
                    if test_val > best_test_prec1:
                        best_test_prec1 = test_val
                        best_epoch = current_epoch
                   
    return {
        'path': file_path,
        'name': os.path.basename(file_path),
        'test': best_test_prec1,
        'epoch': best_epoch
    }

def get_method_info(filename):
    """
    Parses the filename to return the exact label and forces the sorting order.
    Order: erm (1) -> classbalanced (2) -> sava_reweight (3)
    """
    name_lower = filename.lower()
   
    # We must check 'classbalanced' first, because 'classbalanced_erm' contains BOTH words.
    if 'classbalanced' in name_lower:
        return 'classbalanced', 2
    elif 'sava_reweight' in name_lower:
        return 'sava_reweight', 3
    else:
        # If it doesn't have classbalanced or sava_reweight, it's the base 'erm' file.
        return 'erm', 1

def plot_multi_comparison(source_paths, output_dir):
    """
    source_paths: list of exact folders or specific file paths
    output_dir: where to save the final plot
    """

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    all_log_files = set()

    # 1. Collect all log files
    for path in source_paths:
        if os.path.isdir(path):
            all_log_files.update(glob.glob(os.path.join(path, "*.log")))
        elif os.path.isfile(path) and path.endswith(".log"):
            all_log_files.add(path)         

    all_log_files = list(all_log_files)

    if not all_log_files:
        print(f"⚠️ No log files found in the specified paths: {source_paths}")
        return     

    # 2. Extract metrics
    results = []
    for f_path in all_log_files:
        data = get_metrics(f_path)
        if data and data['test'] > 0:
            results.append(data)

    if not results:
        print("❌ ERROR: No valid testing results found to plot.")
        return

    # Sort by our custom method index (1, 2, 3) to guarantee Left-to-Right order
    results = sorted(results, key=lambda x: get_method_info(x['name'])[1])  

    # 3. Prepare data for plotting
    labels = [get_method_info(r['name'])[0] for r in results]
    test_scores = [r['test'] for r in results]
    test_epochs = [r['epoch'] for r in results] 
    x = range(len(labels))
    width = 0.4  
    plt.style.use('ggplot')
    fig_width = max(10, len(labels) * 2)
    fig, ax = plt.subplots(figsize=(fig_width, 7))

    # ✨ Define a list of distinct colors (Blue, Green, Red, Purple, Orange)
    palette = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6', '#f39c12']
    # Match the colors to the number of bars we have
    bar_colors = [palette[i % len(palette)] for i in range(len(test_scores))]

    # ✨ Pass the list of colors to the bar chart
    bars = ax.bar(x, test_scores, width, color=bar_colors)

    # Add text labels on top of the bars
    for i, b in enumerate(bars):
        ax.annotate(f'{test_scores[i]:.2f}%\n(Ep: {test_epochs[i]})',
                    xy=(b.get_x() + b.get_width() / 2, b.get_height()),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom',
                    fontweight='bold', color='#2c3e50', fontsize=10) # Used neutral dark text for readability
                
    # Formatting
    ax.set_ylabel('Accuracy (%)', fontweight='bold')
    ax.set_xlabel('Methods', fontweight='bold', fontsize=12)
    ax.set_title('Comparison of Cifar 10 Drw_seed9999', fontsize=16, fontweight='bold')  
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontweight='bold', rotation=0, ha='center', fontsize=12)
    ax.set_ylim(0, max(test_scores) + 15)  # Add padding to top of chart   
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'Comparison of Cifar 10 Drw_seed999.png')
    plt.savefig(save_path, dpi=300)
    print(f"🎉 Successfully saved testing multi-comparison plot to: {save_path}")
    plt.show()

if __name__ == "__main__":
    # --- CONFIGURE YOUR PATHS HERE ---
    folders_to_compare = [
        "./seed9999/cifar10_drw",
        "./seed9999/cifar10_drw/baselines",
    ]
    results_output = "./comparisons/seed9999"   
    print(f"🔍 Searching exactly inside: {folders_to_compare}")
    plot_multi_comparison(folders_to_compare, results_output)