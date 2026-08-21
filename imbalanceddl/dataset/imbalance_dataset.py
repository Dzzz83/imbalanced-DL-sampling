import torch
import numpy as np
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from imbalanceddl.dataset import IMBALANCECIFAR10
from imbalanceddl.dataset import IMBALANCECIFAR100
from imbalanceddl.utils import get_weak_augmentation, get_trivial_augmentation

class ImbalancedDataset:
    def __init__(self, cfg, dataset_name, augmentation='weak'):
        self.cfg = cfg
        self.dataset_name = dataset_name
        self.imb_type = cfg.imb_type
        self.imb_factor = cfg.imb_factor
        self.augmentation = augmentation
        self.data_transform = self._get_data_transform()

    def _get_base_dataset_name(self):
        """Return base dataset name for augmentation (cifar10 or cifar100)."""
        if 'cifar10' in self.dataset_name:
            return 'cifar10'
        elif 'cifar100' in self.dataset_name:
            return 'cifar100'
        else:
            raise NotImplementedError(f"Dataset {self.dataset_name} not supported.")

    def _get_data_transform(self):
        """
        Return data transform by dataset name
        """
        data_transform = dict()

        if self.dataset_name in ['cifar10', 'cifar100']:
            print("=> Get {} data transform".format(self.dataset_name))
            base_dataset = self._get_base_dataset_name()
            if self.augmentation == 'weak':
                print(f"Applying Weak Augmentation to the {self.dataset_name}")
                train_transform, val_transform = get_weak_augmentation(base_dataset)
            elif self.augmentation == 'trivial':
                print(f"Applying Trivial Augmentation to the {self.dataset_name}")
                train_transform, val_transform = get_trivial_augmentation(base_dataset)
            elif self.augmentation == 'none':
                print(f"Not applying augmentation to the {self.dataset_name}")
                if self.dataset_name == 'cifar10':
                    mean = (0.4914, 0.4822, 0.4465)
                    std = (0.2023, 0.1994, 0.2010)
                elif self.dataset_name == 'cifar100':
                    mean = (0.5071, 0.4867, 0.4408)
                    std = (0.2675, 0.2565, 0.2761)
                else:
                    mean = (0.5, 0.5, 0.5)
                    std = (0.5, 0.5, 0.5)
                train_transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std)
                ])
                val_transform = train_transform
            else:
                raise NotImplementedError(f"The augmentation '{self.augmentation}' is not implemented")
            data_transform['train'] = train_transform
            data_transform['val'] = val_transform
        else:
            raise NotImplementedError(f"Dataset {self.dataset_name} not supported.")

        return data_transform

    @property
    def train_val_sets(self):
        if self.dataset_name == 'cifar10':
            train_dataset, val_dataset = self._cifar10()
        elif self.dataset_name == 'cifar100':
            train_dataset, val_dataset = self._cifar100()
        else:
            raise NotImplementedError

        return train_dataset, val_dataset

    def _cifar10(self):
        print("=> Preparing IMBALANCECIFAR10 {} | {} !".format(
            self.imb_type, self.imb_factor))
        train_dataset = IMBALANCECIFAR10(
            root='./data',
            imb_type=self.imb_type,
            imb_factor=self.imb_factor,
            rand_number=self.cfg.rand_number,
            train=True,
            download=True,
            transform=self.data_transform['train'])

        self.cfg.cls_num_list = train_dataset.get_cls_num_list()
        val_dataset = datasets.CIFAR10(root='./data',
                                       train=False,
                                       download=True,
                                       transform=self.data_transform['val'])

        return train_dataset, val_dataset

    def _cifar100(self):
        print("=> Preparing IMBALANCECIFAR100 {} | {} !".format(
            self.imb_type, self.imb_factor))
        train_dataset = IMBALANCECIFAR100(
            root='./data',
            imb_type=self.imb_type,
            imb_factor=self.imb_factor,
            rand_number=self.cfg.rand_number,
            train=True,
            download=True,
            transform=self.data_transform['train'])

        self.cfg.cls_num_list = train_dataset.get_cls_num_list()
        val_dataset = datasets.CIFAR100(root='./data',
                                        train=False,
                                        download=True,
                                        transform=self.data_transform['val'])

        return train_dataset, val_dataset