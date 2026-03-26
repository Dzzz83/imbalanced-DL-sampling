import numpy as np

def random_selection(train_dataset, keep_ratio = 0.7):
    # get the number of total samples
    total_sample_num = len(train_dataset)
    # get the number of samples after selection
    num_to_keep = int(total_sample_num * keep_ratio)
    # generate indices
    indices = np.arange(total_sample_num)
    # shuffle indices
    np.random.shuffle(indices)
    # take the top 70%
    selected_indices = indices[:num_to_keep].tolist()
    
    print(f"---Random Selection Completed---")
    print(f"Selected {len(selected_indices)} / {total_sample_num}. Ratio: {keep_ratio}")

    return selected_indices



