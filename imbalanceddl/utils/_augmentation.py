import torchvision.transforms as transforms

# CIFAR‑10 statistics
MEAN_CIFAR10 = (0.4914, 0.4822, 0.4465)
STD_CIFAR10  = (0.2023, 0.1994, 0.2010)

# CIFAR‑100 statistics
MEAN_CIFAR100 = (0.5071, 0.4867, 0.4408)
STD_CIFAR100  = (0.2675, 0.2565, 0.2761)

def get_weak_augmentation(dataset='cifar10'):
    """
    Standard augmentation: random horizontal flip + random crop.
    Args:
        dataset: 'cifar10' or 'cifar100'
    """
    if dataset == 'cifar10':
        mean, std = MEAN_CIFAR10, STD_CIFAR10
    elif dataset == 'cifar100':
        mean, std = MEAN_CIFAR100, STD_CIFAR100
    else:
        raise NotImplementedError(f"Dataset {dataset} not supported for weak augmentation")

    train_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    val_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    return train_transforms, val_transforms

def get_trivial_augmentation(dataset='cifar10'):
    """
    TrivialAugmentWide.
    Args:
        dataset: 'cifar10' or 'cifar100'
    """
    if dataset == 'cifar10':
        mean, std = MEAN_CIFAR10, STD_CIFAR10
    elif dataset == 'cifar100':
        mean, std = MEAN_CIFAR100, STD_CIFAR100
    else:
        raise NotImplementedError(f"Dataset {dataset} not supported for trivial augmentation")

    train_transforms = transforms.Compose([
        transforms.TrivialAugmentWide(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    val_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    return train_transforms, val_transforms