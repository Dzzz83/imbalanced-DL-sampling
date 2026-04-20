import torch
import numpy as np
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from imbalanceddl.dataset import IMBALANCECIFAR10
from imbalanceddl.dataset import IMBALANCECIFAR100
from imbalanceddl.dataset import IMBALANCECINIC10
from imbalanceddl.dataset import IMBALANCETINY
from imbalanceddl.dataset import IMBALANCESVHN
from imbalanceddl.dataset import IMBALANCECIFAR10_NOISY
from imbalanceddl.utils import get_weak_augmentation, get_trivial_augmentation


class ImbalancedDataset:
    def __init__(self, cfg, dataset_name, augmentation='weak'):
        self.cfg = cfg
        self.dataset_name = dataset_name
        self.imb_type = cfg.imb_type
        self.imb_factor = cfg.imb_factor
        self.augmentation = augmentation
        self.data_transform = self._get_data_transform()

    def _get_data_transform(self):
        """
        Return data transform by dataset name

        """
        data_transform = dict()

        if self.dataset_name in ['cifar10', 'cifar100']:
            print("=> Get {} data transform".format(self.dataset_name))
            if self.augmentation == 'weak':
                print(f"Applying Weak Augmentation to the {self.dataset_name}")
                train_transform, val_transform = get_weak_augmentation()
            elif self.augmentation == 'trivial':
                print(f"Applying Trivial Augmentation to the {self.dataset_name}")
                train_transform, val_transform = get_trivial_augmentation()
            elif self.augmentation == 'none':
                print(f"Not applying augmentation to the {self.dataset_name}")
                if self.dataset_name == 'cifar10':
                    mean = (0.4914, 0.4822, 0.4465)
                    std = (0.2023, 0.1994, 0.2010)
                elif self.dataset_name == 'cifar100':
                    mean = (0.5071, 0.4867, 0.4408)
                    std = (0.2675, 0.2565, 0.2761)
                else:
                    # fallback or raise error
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
            # data_transform['train'] = transforms.Compose([
            #     transforms.RandomCrop(32, padding=4),
            #     transforms.RandomHorizontalFlip(),
            #     transforms.ToTensor(),
            #     transforms.Normalize((0.4914, 0.4822, 0.4465),
            #                          (0.2023, 0.1994, 0.2010)),
            # ])
            # data_transform['val'] = transform_val = transforms.Compose([
            #     transforms.ToTensor(),
            #     transforms.Normalize((0.4914, 0.4822, 0.4465),
            #                          (0.2023, 0.1994, 0.2010)),
            # ])
        elif self.dataset_name == "cinic10":
            print("=> Get {} data transform".format(self.dataset_name))
            data_transform['train'] = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.47889522, 0.47227842, 0.43047404),
                                     (0.24205776, 0.23828046, 0.25874835)),
            ])
            data_transform['val'] = transform_val = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.47889522, 0.47227842, 0.43047404),
                                     (0.24205776, 0.23828046, 0.25874835)),
            ])
        elif self.dataset_name == "tiny200":
            print("=> Get {} data transform".format(self.dataset_name))
            data_transform['train'] = transforms.Compose([
                transforms.RandomCrop(64, padding=8),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225)),
            ])
            data_transform['val'] = transform_val = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225)),
            ])
        elif self.dataset_name == "svhn10":
            print("=> Get {} data transform".format(self.dataset_name))
            data_transform['train'] = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((.5, .5, .5), (.5, .5, .5)),
            ])
            data_transform['val'] = transform_val = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((.5, .5, .5), (.5, .5, .5)),
            ])
        else:
            raise NotImplementedError

        return data_transform

    @property
    def train_val_sets(self):
        if self.dataset_name == 'cifar10':
            return self._cifar10()
        elif self.dataset_name == 'cifar100':
            return self._cifar100()
        elif self.dataset_name == 'cifar10_noisy':
            return self._cifar10_noisy()
        elif self.dataset_name == 'cinic10':
            return self._cinic10()
        elif self.dataset_name == 'tiny200':
            return self._tiny200()
        elif self.dataset_name == 'svhn10':
            return self._svhn10()
        else:
            raise NotImplementedError

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

    def _cifar10_noisy(self):
        print("=> Preparing IMBALANCECIFAR10_NOISY {} | {} with noise_ratio={}".format(
            self.imb_type, self.imb_factor, getattr(self.cfg, 'noise_ratio', 0.25)))
        train_dataset = IMBALANCECIFAR10_NOISY(
            root='./data',
            imb_type=self.imb_type,
            imb_factor=self.imb_factor,
            rand_number=self.cfg.rand_number,
            train=True,
            download=True,
            transform=self.data_transform['train'],
            noise_ratio=getattr(self.cfg, 'noise_ratio', 0.25),
            num_classes=self.cfg.num_classes,
            seed=self.cfg.rand_number
        )
        self.cfg.cls_num_list = train_dataset.get_cls_num_list()
        val_dataset = datasets.CIFAR10(root='./data', train=False, download=True,
                                    transform=self.data_transform['val'])
        return train_dataset, val_dataset 

    def _cinic10(self):
        print("=> Preparing IMBALANCECINIC100 {} | {} !".format(
            self.imb_type, self.imb_factor))
        # Change to your path into configuration file
        # cinic_root = "/tmp2/wccheng/cinic/"
        cinic_root = self.cfg.cinic_root
        train_dataset = IMBALANCECINIC10(
            cinic_root + "train",
            imb_type=self.imb_type,
            imb_factor=self.imb_factor,
            rand_number=self.cfg.rand_number,
            transform=self.data_transform['train'])

        self.cfg.cls_num_list = train_dataset.get_cls_num_list()
        val_dataset = datasets.ImageFolder(
            cinic_root + "valid", transform=self.data_transform['val'])

        return train_dataset, val_dataset

    def _tiny200(self):
        print("=> Preparing IMBALANCETINY {} | {} !".format(
            self.imb_type, self.imb_factor))
        # Change to your path into configuration file
        # tiny_root = "/tmp2/wccheng/tiny/tiny-imagenet-200/"
        tiny_root = self.cfg.tiny_root
        train_dataset = IMBALANCETINY(tiny_root + "train",
                                      imb_type=self.imb_type,
                                      imb_factor=self.imb_factor,
                                      rand_number=self.cfg.rand_number,
                                      transform=self.data_transform['train'])

        self.cfg.cls_num_list = train_dataset.get_cls_num_list()
        val_dataset = datasets.ImageFolder(
            tiny_root + "val/images", transform=self.data_transform['val'])

        return train_dataset, val_dataset

    def _svhn10(self):
        print("=> Preparing IMBALANCESVHN {} | {} !".format(
            self.imb_type, self.imb_factor))
        train_dataset = IMBALANCESVHN(root='./data',
                                      imb_type=self.imb_type,
                                      imb_factor=self.imb_factor,
                                      rand_number=self.cfg.rand_number,
                                      split='train',
                                      download=True,
                                      transform=self.data_transform['train'])

        self.cfg.cls_num_list = train_dataset.get_cls_num_list()
        val_dataset = datasets.SVHN(root='./data',
                                    split='test',
                                    download=True,
                                    transform=self.data_transform['val'])

        return train_dataset, val_dataset

