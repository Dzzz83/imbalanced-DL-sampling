import matplotlib.pyplot as plt
import os
import re
import glob

def parse_log_file(file_path):
    """
    Extracts both Training and Testing summary lines from a single log file.
    """
    data = {'train_prec1': [], 'train_loss': [], 'test_prec1': [], 'test_loss': []}
    
    # 1. Loudly announce if the file is missing!
    if not os.path.exists(file_path):
        print(f"❌ ERROR: Cannot find file -> {file_path}")
        return data

    # 2. Safely read the single log file and separate Train vs Test
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if "Training Results:" in line:
                p1_match = re.search(r"Prec@1\s+([\d\.]+)", line)
                loss_match = re.search(r"Loss\s+([\d\.]+)", line)
                if p1_match and loss_match:
                    data['train_prec1'].append(float(p1_match.group(1)))
                    data['train_loss'].append(float(loss_match.group(1)))
                    
            elif "Testing Results:" in line:
                p1_match = re.search(r"Prec@1\s+([\d\.]+)", line)
                loss_match = re.search(r"Loss\s+([\d\.]+)", line)
                if p1_match and loss_match:
                    data['test_prec1'].append(float(p1_match.group(1)))
                    data['test_loss'].append(float(loss_match.group(1)))
                    
    print(f"✅ SUCCESS: Read {len(data['train_loss'])} Train & {len(data['test_loss'])} Test entries from {os.path.basename(file_path)}")
    return data

def plot_and_save(log_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 3. Find all .log files in your target folder
    log_files = glob.glob(os.path.join(log_dir, "*.log"))
    
    if not log_files:
        print(f"⚠️ WARNING: No .log files found in '{log_dir}'. Plotting cancelled.")
        return

    # Loop through each log file and generate a separate plot
    for file_path in log_files:
        file_name = os.path.basename(file_path)
        data = parse_log_file(file_path)

        # Stop if the file didn't have usable data
        if not data['train_loss'] and not data['test_loss']:
            print(f"⚠️ WARNING: {file_name} was empty. Skipping.")
            continue

        # Create independent X-axes for Train and Test
        train_epochs = range(1, len(data['train_loss']) + 1)
        test_epochs = range(1, len(data['test_loss']) + 1)

        plt.style.use('ggplot') 
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Plot 1: Loss
        if data['train_loss']:
            ax1.plot(train_epochs, data['train_loss'], label='Train Loss', color="#2424c8", linewidth=2)
        if data['test_loss']:
            ax1.plot(test_epochs, data['test_loss'], label='Test Loss', color='#e74c3c', linestyle='--', linewidth=2)
        ax1.set_title('Training vs Testing Loss', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss Value')
        ax1.legend()

        # Plot 2: Accuracy (Prec@1)
        if data['train_prec1']:
            ax2.plot(train_epochs, data['train_prec1'], label='Train Prec@1', color="#15AB54", linewidth=2)
        if data['test_prec1']:
            ax2.plot(test_epochs, data['test_prec1'], label='Test Prec@1', color='#f39c12', linestyle='--', linewidth=2)
        ax2.set_title('Training vs Testing Accuracy (Top-1)', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()

        # 4. Dynamically set title and filename based on the log file name!
        base_name = file_name.replace(".log", "")
        plt.suptitle(base_name, fontsize=16)
        plt.tight_layout()
        
        save_path = os.path.join(output_dir, f'{base_name}.png')
        plt.savefig(save_path, dpi=300)
        print(f"🎉 Successfully saved plot to: {save_path}")
        
        # Close the plot to save memory before moving to the next file
        plt.close()

if __name__ == "__main__":
    source_logs = "train/cifar100_random0.7_mixup_drw_exp0.01_seed42" 
    results_output = "plot"
    
    print(f"Looking for logs in: {os.path.abspath(source_logs)}")
    plot_and_save(source_logs, results_output)