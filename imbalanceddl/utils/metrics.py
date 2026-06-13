import numpy as np
import torch
from torch.utils.data import Subset
from imbalanceddl.utils.debug_logger import get_debug_logger   # ADDED

def accuracy(output, target, topk=(1, )):
    """
    Function to compute topk accuracy for evaluation
    Common Usage is to use top 1 and top 5
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def shot_acc(args,
             preds,
             labels,
             train_data,
             many_shot_thr=100,
             low_shot_thr=20,
             acc_per_cls=False):
    """
    Function to compute many shot, median_shot, and low shot accuracy
    Typically used when the class number is huge, ex. CIFAR-100
    """
    debug = getattr(args, 'debug', False)
    logger = get_debug_logger(debug=debug)   # ADDED

    # ---------- BEGIN FIX: handle Subset objects ----------
    if isinstance(train_data, Subset):
        # Get original dataset and indices
        orig_dataset = train_data.dataset
        indices = train_data.indices
        # Extract targets from original dataset
        if hasattr(orig_dataset, 'targets'):
            all_targets = np.array(orig_dataset.targets)
        elif hasattr(orig_dataset, 'Y'):
            all_targets = np.array(orig_dataset.Y)
        else:
            # Fallback: iterate (slow but safe)
            all_targets = np.array([orig_dataset[i][1] for i in range(len(orig_dataset))])
        training_labels = all_targets[indices]
        if debug:
            logger.debug(f"train_data is Subset, len(training_labels)={len(training_labels)}")
    else:
        # Original behaviour for non-Subset datasets
        if isinstance(train_data, np.ndarray):
            training_labels = np.array(train_data).astype(int)
        else:
            if args.dataset == 'svhn':
                training_labels = np.array(train_data.labels).astype(int)
            else:
                training_labels = np.array(train_data.targets).astype(int)
        if debug:
            logger.debug(f"train_data is not Subset, type={type(train_data)}")
    # ---------- END FIX ----------

    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()
    elif isinstance(preds, np.ndarray):
        pass
    else:
        raise TypeError('Type ({}) of preds not supported'.format(type(preds)))

    # Compute per‑class statistics
    unique_labels = np.unique(labels)
    train_class_count = []
    test_class_count = []
    class_correct = []

    for l in unique_labels:
        train_class_count.append(len(training_labels[training_labels == l]))
        test_class_count.append(len(labels[labels == l]))
        class_correct.append((preds[labels == l] == labels[labels == l]).sum())

    if debug:
        # Show first 10 classes to avoid flooding
        logger.debug(f"unique_labels (first 10): {unique_labels[:10]}")
        logger.debug(f"train_class_count (first 10): {train_class_count[:10]}")
        logger.debug(f"test_class_count (first 10): {test_class_count[:10]}")
        logger.debug(f"per-class accuracy (first 10): "
                     f"{[c/tc if tc>0 else 0.0 for c,tc in zip(class_correct[:10], test_class_count[:10])]}")
        logger.debug(f"many_shot_thr={many_shot_thr}, low_shot_thr={low_shot_thr}")

    many_shot = []
    median_shot = []
    low_shot = []
    many_indices = []
    median_indices = []
    low_indices = []

    for i, cnt in enumerate(train_class_count):
        acc = class_correct[i] / test_class_count[i] if test_class_count[i] > 0 else 0.0
        if cnt > many_shot_thr:
            many_shot.append(acc)
            many_indices.append(i)
        elif cnt < low_shot_thr:
            low_shot.append(acc)
            low_indices.append(i)
        else:
            median_shot.append(acc)
            median_indices.append(i)

    if debug:
        logger.debug(f"many_shot classes (first 10 indices): {many_indices[:10]} count={len(many_indices)}")
        logger.debug(f"median_shot classes (first 10 indices): {median_indices[:10]} count={len(median_indices)}")
        logger.debug(f"low_shot classes (first 10 indices): {low_indices[:10]} count={len(low_indices)}")
        logger.debug(f"many_shot acc (first 10): {many_shot[:10]}")
        logger.debug(f"median_shot acc (first 10): {median_shot[:10]}")
        logger.debug(f"low_shot acc (first 10): {low_shot[:10]}")

    if len(many_shot) == 0:
        many_shot.append(0)
    if len(median_shot) == 0:
        median_shot.append(0)
    if len(low_shot) == 0:
        low_shot.append(0)

    many_acc = np.mean(many_shot)
    median_acc = np.mean(median_shot)
    low_acc = np.mean(low_shot)

    if debug:
        logger.debug(f"many_acc={many_acc:.4f}, median_acc={median_acc:.4f}, low_acc={low_acc:.4f}")

    if acc_per_cls:
        class_accs = [c / cnt for c, cnt in zip(class_correct, test_class_count)]
        return many_acc, median_acc, low_acc, class_accs
    else:
        return many_acc, median_acc, low_acc
