import os
import numpy as np
import collections
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class CustomImageDataset(Dataset):
    def __init__(self, X, Y, transform=None):
        self.X = X
        self.Y = Y
        self.transform = transform
        self.targets = Y
        self.cls_num_list = np.bincount(Y, minlength=len(np.unique(Y))).tolist()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        _input = self.X[idx]
        target = self.Y[idx]
        if self.transform:
            _input = self.transform(_input)
        return _input, target
    
    def get_cls_num_list(self):
        return self._cls_num_list
    
    def targets(self):
        return self.Y

def get_balanced_deep_smote(dataset, batch_size, imb_type, imb_factor, num_workers=0):
    deepsmote_folder = 'deepsmote_models'
    train_data = 'train_data'
    train_label = 'train_label'

    path_prefix = './' + deepsmote_folder + '/' + dataset + '/' + dataset + '_' + imb_type + '_' + 'R' + str(int(1/imb_factor)) + '_'
    dtrnimg = path_prefix + train_data + '.txt'
    dtrnlab = path_prefix + train_label + '.txt'

    if not os.path.exists(dtrnimg):
        print(f"CRITICAL ERROR: Data file not found at {os.path.abspath(dtrnimg)}")
        print("Check if 'deepsmote_models' is in the same folder as main.py")
        return None
    
    if dataset == 'cifar10':
        # dtrnimg = '../' + deepsmote_folder + '/' + dataset + '/' + dataset + '_' + imb_type + '_' + 'R' + str(int(1/imb_factor)) + '_' + train_data + '.txt'
        # dtrnlab = '../' + deepsmote_folder + '/' + dataset + '/' + dataset + '_' + imb_type + '_' + 'R' + str(int(1/imb_factor)) + '_' + train_label + '.txt'
        val_transform = transforms.Compose(
        [transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    elif dataset == 'cifar100':
        # dtrnimg = '../' + deepsmote_folder + '/' + dataset + '/' + dataset + '_' + imb_type + '_' + 'R' + str(int(1/imb_factor)) + '_' + train_data + '.txt'
        # dtrnlab = '../' + deepsmote_folder + '/' + dataset + '/' + dataset + '_' + imb_type + '_' + 'R' + str(int(1/imb_factor)) + '_' + train_label + '.txt'
        val_transform = transforms.Compose(
        [transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])
    elif dataset == 'cinic10':
        # dtrnimg = '../' + deepsmote_folder + '/' + dataset + '/' + dataset + '_' + imb_type + '_' + 'R' + str(int(1/imb_factor)) + '_' + train_data + '.txt'
        # dtrnlab = '../' + deepsmote_folder + '/' + dataset + '/' + dataset + '_' + imb_type + '_' + 'R' + str(int(1/imb_factor)) + '_' + train_label + '.txt'
        val_transform = transforms.Compose(
        [transforms.ToTensor(),
            transforms.Normalize((.47889522, 0.47227842, 0.43047404), (0.24205776, 0.23828046, 0.25874835))])
    elif dataset == 'svhn10':
        # dtrnimg = '../' + deepsmote_folder + '/' + dataset + '/' + dataset + '_' + imb_type + '_' + 'R' + str(int(1/imb_factor)) + '_' + train_data + '.txt'
        # dtrnlab = '../' + deepsmote_folder + '/' + dataset + '/' + dataset + '_' + imb_type + '_' + 'R' + str(int(1/imb_factor)) + '_' + train_label + '.txt'
        val_transform = transforms.Compose(
        [transforms.ToTensor(),
            transforms.Normalize((.5, .5, .5), (.5, .5, .5))])
    elif dataset == 'tiny200':
        # dtrnimg = '../' + deepsmote_folder + '/' + dataset + '/' + dataset + '_' + imb_type + '_' + 'R' + str(int(1/imb_factor)) + '_' + train_data + '.txt'
        # dtrnlab = '../' + deepsmote_folder + '/' + dataset + '/' + dataset + '_' + imb_type + '_' + 'R' + str(int(1/imb_factor)) + '_' + train_label + '.txt'
        val_transform = transforms.Compose(
        [transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    print(dtrnimg)
    print(dtrnlab)
    dec_x = np.loadtxt(dtrnimg) 
    dec_y = np.loadtxt(dtrnlab)

    print('train imgs before reshape ',dec_x.shape) #(44993, 3072) 45500, 3072)
    print('train labels ',dec_y.shape) #(44993,) (45500,)

    print(collections.Counter(dec_y))
    dec_y = torch.tensor(dec_y,dtype=torch.long)

    if dataset == 'mnist':
        dec_x = dec_x.reshape(dec_x.shape[0],1,28,28) #(50000, 32, 32, 3)
        print('train imgs after reshape ',dec_x.shape)
    else:
        dec_x = dec_x.reshape(dec_x.shape[0],3,32,32) #(50000, 32, 32, 3)
        print('train imgs after reshape ',dec_x.shape) 

    dec_x = np.transpose(dec_x, axes= (0, 2, 3, 1)) # 0, 32, 32, 3
    print('train imgs after reshape ',dec_x.shape) 

    dec_x *=255.
    dec_x = np.clip(dec_x, 0, 255)
    dec_x = np.asarray(dec_x, dtype=np.uint8)

    balance_dataset = CustomImageDataset(dec_x, dec_y, val_transform)
    train_smote_loader = torch.utils.data.DataLoader(balance_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

    return train_smote_loader

def load_and_cap_deepsmote(dataset, imb_type, imb_factor, class_caps=None):
    """
    Load raw DeepSMOTE data and apply class capping.
    Returns (X_capped, Y_capped) as numpy arrays.
    """
    deepsmote_folder = 'deepsmote_models'
    path_prefix = f'./{deepsmote_folder}/{dataset}/{dataset}_{imb_type}_R{int(1/imb_factor)}_'
    data_file = path_prefix + "train_data.txt"
    label_file = path_prefix + "train_label.txt"

    if not os.path.exists(data_file):
        raise FileNotFoundError(f"DeepSMOTE data not found at {os.path.abspath(data_file)}")

    X = np.loadtxt(data_file)   # (N, 3072)
    Y = np.loadtxt(label_file)  # (N,)

    # Reshape to (N, 3, 32, 32)
    X = X.reshape(-1, 3, 32, 32)
    # Convert to HWC for PIL compatibility (uint8)
    X = np.transpose(X, (0, 2, 3, 1))
    X = np.clip(X * 255, 0, 255).astype(np.uint8)

    num_classes = len(np.unique(Y))
    # Determine per-class caps
    if class_caps is None:
        # NEW DEFAULT: all classes capped to 2000
        caps = [5000] * num_classes
    elif isinstance(class_caps, list):
        caps = class_caps
    elif isinstance(class_caps, dict):
        caps = []
        for c in range(num_classes):
            caps.append(class_caps.get(c, len(np.where(Y == c)[0])))
    else:
        raise TypeError("class_caps must be None, list, or dict")

    # Subsample each class
    keep = []
    for c in range(num_classes):
        idx = np.where(Y == c)[0]
        cap = caps[c]
        if len(idx) > cap:
            idx = np.random.choice(idx, cap, replace=False)
        keep.extend(idx)

    keep = np.array(keep)
    X = X[keep]
    Y = Y[keep].astype(int)
    print(f"Capped dataset size: {len(X)} samples (all classes capped to {caps[0] if caps else '?'})")
    return X, Y

def load_deepsmote_dataset(dataset, imb_type, imb_factor, transform=None, class_caps=None):
    """Load pre‑generated DeepSMOTE balanced dataset as a CustomImageDataset."""
    deepsmote_folder = 'deepsmote_models'
    path_prefix = f'./{deepsmote_folder}/{dataset}/{dataset}_{imb_type}_R{int(1/imb_factor)}_'
    data_file = path_prefix + "train_data.txt"
    label_file = path_prefix + "train_label.txt"

    if not os.path.exists(data_file):
        raise FileNotFoundError(f"DeepSMOTE data not found at {os.path.abspath(data_file)}")

    X = np.loadtxt(data_file)   # (N, 3072) for CIFAR
    Y = np.loadtxt(label_file)  # (N,)

    # Reshape and convert to HWC uint8
    X = X.reshape(-1, 3, 32, 32)               # (N, 3, 32, 32)
    X = np.transpose(X, (0, 2, 3, 1))          # (N, 32, 32, 3) for PIL
    X = np.clip(X * 255, 0, 255).astype(np.uint8)

    num_classes = len(np.unique(Y))
    if class_caps is None:
        # num_per_class = [5000] + [4000] * (num_classes - 1) # [5000, 4000, ...., 4000]
        num_per_class = [5000] * num_classes
    elif isinstance(class_caps, list):
        num_per_class = class_caps  
    elif isinstance(class_caps, dict):
        num_per_class = []
        for i in range(num_classes):
            if i in class_caps:
                num_per_class.append(class_caps[i])
            else:
                num_per_class.append(len(np.where(Y == i)[0]))
    else:
        raise TypeError("class_caps must be None, list, or dict")
    
    keep_indices = []
    for c in range(num_classes):
        # find all positions where the label equals c
        class_idx = np.where(Y == c)[0] # Y is array of labels
        cap = num_per_class[c]
        # if there are more samples in that class, cap it down
        if len(class_idx) > cap:
            # choose randomly without replacement
            selected = np.random.choice(class_idx, cap, replace=False)
        else:
            selected = class_idx
        keep_indices.extend(selected)

    keep_indices = np.array(keep_indices)
    X = X[keep_indices]
    Y = Y[keep_indices].astype(int)

    print(f"Dataset size after capping: {len(keep_indices)} samples")  

    # Default transform if none provided
    if transform is None:
        if dataset == 'cifar10':
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
        elif dataset == 'cifar100':
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
            ])
        else:
            raise NotImplementedError(f"Default transform for {dataset} not defined.")

    return CustomImageDataset(X, Y.astype(int), transform)