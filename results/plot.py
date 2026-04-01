import matplotlib.pyplot as plt
import os
import re

def parse_final_results(file_path):
    """
    Extracts ONLY the final summary lines for each epoch.
    """
    data = {'prec1': [], 'loss': []}
    
    # 1. Loudly announce if the file is missing!
    if not os.path.exists(file_path):
        print(f"❌ ERROR: Cannot find file -> {file_path}")
        return data

    # 2. Safely read the file
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if "Results:" in line:
                p1_match = re.search(r"Prec@1\s+([\d\.]+)", line)
                loss_match = re.search(r"Loss\s+([\d\.]+)", line)
                if p1_match and loss_match:
                    data['prec1'].append(float(p1_match.group(1)))
                    data['loss'].append(float(loss_match.group(1)))
                    
    print(f"✅ SUCCESS: Read {len(data['loss'])} entries from {os.path.basename(file_path)}")
    return data

def plot_and_save(log_dir, output_dir):
    train_path = os.path.join(log_dir, 'log_train.csv')
    test_path = os.path.join(log_dir, 'log_test.csv')
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    train_data = parse_final_results(train_path)
    test_data = parse_final_results(test_path)

    # 3. Stop before making a blank graph
    if not train_data['loss'] and not test_data['loss']:
        print("⚠️ WARNING: Both files were empty or missing. Plotting cancelled.")
        return

    # 4. Create independent X-axes for Train and Test
    train_epochs = range(1, len(train_data['loss']) + 1)
    test_epochs = range(1, len(test_data['loss']) + 1)

    plt.style.use('ggplot') 
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: Loss
    if train_data['loss']:
        ax1.plot(train_epochs, train_data['loss'], label='Train Loss', color="#2424c8", linewidth=2)
    if test_data['loss']:
        ax1.plot(test_epochs, test_data['loss'], label='Test Loss', color='#e74c3c', linestyle='--', linewidth=2)
    ax1.set_title('Training vs Testing Loss', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss Value')
    ax1.legend()

    # Plot 2: Accuracy (Prec@1)
    if train_data['prec1']:
        ax2.plot(train_epochs, train_data['prec1'], label='Train Prec@1', color="#15AB54", linewidth=2)
    if test_data['prec1']:
        ax2.plot(test_epochs, test_data['prec1'], label='Test Prec@1', color='#f39c12', linestyle='--', linewidth=2)
    ax2.set_title('Training vs Testing Accuracy (Top-1)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()

    plt.suptitle('CIFAR-10_exp_0.01_DeepSMOTE', fontsize=16)
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, 'CIFAR-10_exp_0.01_DeepSMOTE_200_None.png')
    plt.savefig(save_path, dpi=300)
    print(f"🎉 Successfully saved plot to: {save_path}")
    plt.show()

if __name__ == "__main__":
    # CHECK THIS PATH CAREFULLY!
    source_logs = "../example/checkpoint_cifar10_deepsmote/cifar10_exp_0.01_DeepSMOTE_200_None"
    results_output = "results_plot"
    
    print(f"Looking for logs in: {os.path.abspath(source_logs)}")
    plot_and_save(source_logs, results_output)