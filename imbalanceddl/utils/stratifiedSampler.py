from torch.utils.data import Sampler
import torch
from typing import Iterator, Sequence, Optional
from collections import Counter

from typing import Iterator, Iterable, Optional, Sequence, List, TypeVar, Generic, Sized, Union
import numpy as np
class StratifiedSampler(Sampler[int]):
    def __init__(self, 
                 labels: Sequence[int], 
                 num_samples: Optional[int] = None, 
                 generator=None) -> None:
        """
        Args:
            labels (Sequence[int]): The class labels for each data point.
            num_samples (Optional[int]): Total number of samples to generate. Defaults to the size of labels.
            generator (torch.Generator, optional): Random number generator.
        """
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.num_samples = num_samples or len(self.labels)
        self.generator = generator

        # Calculate stratify values and proportions
        label_counts = Counter(self.labels.tolist())
        total_count = sum(label_counts.values())

        self.stratify_values = torch.tensor(list(label_counts.keys()), dtype=torch.long)
        self.stratify_proportions = torch.tensor([count / total_count for count in label_counts.values()], dtype=torch.float)

        # Calculate weights for stratified sampling
        self.weights = torch.zeros_like(self.labels, dtype=torch.float)
        total_samples = self.num_samples

        for label, proportion in zip(self.stratify_values, self.stratify_proportions):
            label_mask = self.labels == label
            label_count = label_mask.sum().item()
            stratified_count = int(total_samples * proportion)

            if stratified_count > 0 and label_count > 0:
                self.weights[label_mask] = stratified_count / label_count

    def __iter__(self) -> Iterator[int]:
        # Normalize weights and sample indices
        normalized_weights = self.weights / self.weights.sum()
        sampled_indices = torch.multinomial(normalized_weights, self.num_samples, replacement=True, generator=self.generator)
        return iter(sampled_indices.tolist())

    def __len__(self) -> int:
        return self.num_samples
# class StratifiedSampler(Sampler[int]):
#     r"""Samples elements randomly. If without replacement, then sample from a shuffled dataset.
#     If with replacement, then user can specify :attr:`num_samples` to draw.

#     Args:
#         data_source (Dataset): dataset to sample from
#         replacement (bool): samples are drawn on-demand with replacement if ``True``, default=``False``
#         num_samples (int): number of samples to draw, default=`len(dataset)`.
#         generator (Generator): Generator used in sampling.
#     """
#     data_source: Sized
#     replacement: bool

#     def __init__(self, data_source: Sized, replacement: bool = False,
#                  num_samples: Optional[int] = None, num_samples_per_class: Optional[int] = None,
#                  batch_size: Optional[int] = None, alpha: Optional[int] = None, generator=None) -> None:
#         self.data_source = data_source
#         self.replacement = replacement
#         self._num_samples = num_samples
#         self.num_samples_per_class = num_samples_per_class
#         self.batch_size = batch_size
#         self.alpha = alpha
#         self.generator = generator
#         print(self.batch_size)

#         self.number_samples_mini_batch = []
#         number_samples_mini_batch_uniform = []
#         number_samples_mini_batch_imbalance = []
#         alpha_balance = []

#         if len(self.num_samples_per_class) == 10:
#             for i in range(len(self.num_samples_per_class)):
#                 x = self.batch_size/len(self.num_samples_per_class)
#                 number_samples_mini_batch_uniform.append(x)
#                 y = (self.batch_size/self._num_samples)*self.num_samples_per_class[i]
#                 number_samples_mini_batch_imbalance.append(y)
            
#             # Alpha_balance = stratified(alpha * uniform + (1-alpha) * imbalanced)
#             alpha_balance = self.alpha * np.asarray(number_samples_mini_batch_uniform) + (1 - self.alpha) * np.asarray(number_samples_mini_batch_imbalance)
#             alpha_balance = np.round(alpha_balance).astype(int)

#             for i in range(len(alpha_balance)-1, 0, -1):
#                 x = alpha_balance[i]
#                 self.number_samples_mini_batch.append(x)
            
#             self.number_samples_mini_batch.append(self.batch_size - sum(self.number_samples_mini_batch))
#             self.number_samples_mini_batch.reverse()
#             # self.number_samples_mini_batch = [32, 22, 16, 12, 10, 8, 7, 7, 7, 7]
#         else:
#             num_group = 5 #fine tunning depends on each Dataset
#             self.num_class_per_group = len(self.num_samples_per_class) // num_group
#             for i in range(num_group):
#                 total_num_group = sum(self.num_samples_per_class[i*self.num_class_per_group:(i+1)*self.num_class_per_group])
#                 num_each_group = []
#                 for j in range(self.num_class_per_group-1):
#                     # x = round(self.num_samples_per_class[i*self.num_class_per_group + j] * batch_size / total_num_group)
#                     x = round(self.batch_size / self.num_class_per_group)
#                     num_each_group.append(x)
#                 num_each_group.append(batch_size - sum(num_each_group))
#                 self.number_samples_mini_batch+=num_each_group
#                 # self.number_samples_mini_batch.append(num_each_group)

#         if not isinstance(self.replacement, bool):
#             raise TypeError("replacement should be a boolean value, but got "
#                             "replacement={}".format(self.replacement))

#         if not isinstance(self.num_samples, int) or self.num_samples <= 0:
#             raise ValueError("num_samples should be a positive integer "
#                              "value, but got num_samples={}".format(self.num_samples))

#     @property
#     def num_samples(self) -> int:
#         # dataset size might change at runtime
#         if self._num_samples is None:
#             return len(self.data_source)
#         return self._num_samples

#     def __iter__(self) -> Iterator[int]:
#         n = len(self.data_source)
#         if self.generator is None:
#             seed = int(torch.empty((), dtype=torch.int64).random_().item())
#             generator = torch.Generator()
#             generator.manual_seed(seed)
#         else:
#             generator = self.generator
        
#         if len(self.num_samples_per_class) == 10:
#             for _ in range(self._num_samples // self.batch_size):
#                 total_samples = 0
#                 for i in range(len(self.num_samples_per_class)):
#                     yield from torch.randint(low=total_samples, high=self.num_samples_per_class[i]+total_samples, size=(self.number_samples_mini_batch[i],), dtype=torch.int64).tolist()
#                     total_samples +=self.num_samples_per_class[i]
#         else:
#             # for _ in range((self._num_samples // self.batch_size)/self.num_class_per_group): # For CIFAR100
#             for _ in range(18):
#                 total_samples = 0
#                 for i in range(len(self.num_samples_per_class)):
#                     yield from torch.randint(low=total_samples, high=self.num_samples_per_class[i]+total_samples, size=(self.number_samples_mini_batch[i],), dtype=torch.int64).tolist()
#                     total_samples +=self.num_samples_per_class[i]

#     def __len__(self) -> int:
#         return self.num_samples
# from torch.utils.data import Sampler
# import torch
# from typing import Iterator, Sequence, Optional
# import numpy as np
# from collections import Counter

# class StratifiedSampler(Sampler[int]):
#     def __init__(self, 
#                  labels: Sequence[int], 
#                  num_samples: Optional[int] = None, 
#                  batch_size: Optional[int] = None, 
#                  alpha: Optional[float] = None,
#                  generator=None) -> None:
#         """
#         Args:
#             labels (Sequence[int]): The class labels for each data point.
#             num_samples (Optional[int]): Total number of samples to generate. Defaults to the size of labels.
#             batch_size (Optional[int]): Batch size for sampling.
#             alpha (Optional[float]): Proportion between uniform and imbalanced sampling (0 to 1).
#             generator (torch.Generator, optional): Random number generator.
#         """
#         self.labels = torch.tensor(labels, dtype=torch.long)
#         self.num_samples = num_samples or len(self.labels)
#         self.batch_size = batch_size or len(self.labels)
#         self.alpha = alpha or 0.5  # Default to 50% blending
#         self.generator = generator or torch.Generator().manual_seed(42)

#         # Validate inputs
#         assert len(self.labels) > 0, "Labels cannot be empty."
#         assert self.num_samples <= len(self.labels), "num_samples exceeds the number of available samples."
#         assert self.batch_size >= 1, "Batch size must be >= 1."

#         # Calculate class counts and proportions
#         label_counts = Counter(self.labels.tolist())
#         self.classes = list(label_counts.keys())
#         self.num_samples_per_class = [label_counts[cls] for cls in self.classes]
        
#         self.num_classes = len(self.classes)

#         # Precompute number of samples per batch for each class
#         uniform_per_class = [self.batch_size / self.num_classes] * self.num_classes
#         imbalance_per_class = [(self.batch_size / self.num_samples) * count for count in self.num_samples_per_class]

#         # Blend uniform and imbalanced distributions based on alpha
#         self.samples_per_batch = np.round(
#             self.alpha * np.asarray(uniform_per_class) + (1 - self.alpha) * np.asarray(imbalance_per_class)
#         ).astype(int)

#         # Adjust to ensure the total matches the batch size
#         self.samples_per_batch[-1] += self.batch_size - sum(self.samples_per_batch)

#         # Ensure no class is allocated zero samples
#         self.samples_per_batch = np.maximum(self.samples_per_batch, 1)

#     def __iter__(self) -> Iterator[int]:
#         indices = torch.arange(len(self.labels))

#         for _ in range(self.num_samples // self.batch_size):
#             batch_indices = []
#             for cls, num_samples in zip(self.classes, self.samples_per_batch):
#                 cls_indices = indices[self.labels == cls]
#                 if len(cls_indices) > 0:
#                     sampled_count = min(num_samples, len(cls_indices))
#                     batch_indices.extend(
#                         cls_indices[torch.randint(0, len(cls_indices), (sampled_count,), generator=self.generator)].tolist()
#                     )
#                 else:
#                     print(f"Warning: No samples found for class {cls}.")
#             yield from batch_indices

#     def __len__(self) -> int:
#         return self.num_samples
# class StratifiedSampler(Sampler[int]):
#     r"""Samples elements randomly. If without replacement, then sample from a shuffled dataset.
#     If with replacement, then user can specify :attr:`num_samples` to draw.

#     Args:
#         data_source (Dataset): dataset to sample from
#         replacement (bool): samples are drawn on-demand with replacement if ``True``, default=``False``
#         num_samples (int): number of samples to draw, default=`len(dataset)`.
#         generator (Generator): Generator used in sampling.
#     """
#     data_source: Sized
#     replacement: bool

#     def __init__(self, data_source: Sized, replacement: bool = False,
#                  num_samples: Optional[int] = None, num_samples_per_class: Optional[int] = None,
#                  batch_size: Optional[int] = None, alpha: Optional[int] = None, generator=None) -> None:
#         self.data_source = data_source
#         self.replacement = replacement
#         self._num_samples = num_samples
#         self.num_samples_per_class = num_samples_per_class
#         self.batch_size = batch_size
#         self.alpha = alpha
#         self.generator = generator
#         print(self.batch_size)

#         self.number_samples_mini_batch = []
#         number_samples_mini_batch_uniform = []
#         number_samples_mini_batch_imbalance = []
#         alpha_balance = []

#         if len(self.num_samples_per_class) == 10:
#             for i in range(len(self.num_samples_per_class)):
#                 x = self.batch_size/len(self.num_samples_per_class)
#                 number_samples_mini_batch_uniform.append(x)
#                 y = (self.batch_size/self._num_samples)*self.num_samples_per_class[i]
#                 number_samples_mini_batch_imbalance.append(y)
            
#             # Alpha_balance = stratified(alpha * uniform + (1-alpha) * imbalanced)
#             alpha_balance = self.alpha * np.asarray(number_samples_mini_batch_uniform) + (1 - self.alpha) * np.asarray(number_samples_mini_batch_imbalance)
#             alpha_balance = np.round(alpha_balance).astype(int)

#             for i in range(len(alpha_balance)-1, 0, -1):
#                 x = alpha_balance[i]
#                 self.number_samples_mini_batch.append(x)
            
#             self.number_samples_mini_batch.append(self.batch_size - sum(self.number_samples_mini_batch))
#             self.number_samples_mini_batch.reverse()
#             # self.number_samples_mini_batch = [32, 22, 16, 12, 10, 8, 7, 7, 7, 7]
#         else:
#             num_group = 5 #fine tunning depends on each Dataset
#             self.num_class_per_group = len(self.num_samples_per_class) // num_group
#             for i in range(num_group):
#                 total_num_group = sum(self.num_samples_per_class[i*self.num_class_per_group:(i+1)*self.num_class_per_group])
#                 num_each_group = []
#                 for j in range(self.num_class_per_group-1):
#                     # x = round(self.num_samples_per_class[i*self.num_class_per_group + j] * batch_size / total_num_group)
#                     x = round(self.batch_size / self.num_class_per_group)
#                     num_each_group.append(x)
#                 num_each_group.append(batch_size - sum(num_each_group))
#                 self.number_samples_mini_batch+=num_each_group
#                 # self.number_samples_mini_batch.append(num_each_group)

#         if not isinstance(self.replacement, bool):
#             raise TypeError("replacement should be a boolean value, but got "
#                             "replacement={}".format(self.replacement))

#         if not isinstance(self.num_samples, int) or self.num_samples <= 0:
#             raise ValueError("num_samples should be a positive integer "
#                              "value, but got num_samples={}".format(self.num_samples))
#     @property
#     def num_samples(self) -> int:
#         # dataset size might change at runtime
#         if self._num_samples is None:
#             return len(self.data_source)
#         return self._num_samples

#     # def __iter__(self) -> Iterator[int]:
#     #     n = len(self.data_source)
#     #     if self.generator is None:
#     #         seed = int(torch.empty((), dtype=torch.int64).random_().item())
#     #         generator = torch.Generator()
#     #         generator.manual_seed(seed)
#     #     else:
#     #         generator = self.generator
        
#     #     if len(self.num_samples_per_class) == 10:
#     #         for _ in range(self._num_samples // self.batch_size):
#     #             total_samples = 0
#     #             for i in range(len(self.num_samples_per_class)):
#     #                 yield from torch.randint(low=total_samples, high=self.num_samples_per_class[i]+total_samples, size=(self.number_samples_mini_batch[i],), dtype=torch.int64).tolist()
#     #                 total_samples +=self.num_samples_per_class[i]
#     #     else:
#     #         # for _ in range((self._num_samples // self.batch_size)/self.num_class_per_group): # For CIFAR100
#     #         for _ in range(18):
#     #             total_samples = 0
#     #             for i in range(len(self.num_samples_per_class)):
#     #                 yield from torch.randint(low=total_samples, high=self.num_samples_per_class[i]+total_samples, size=(self.number_samples_mini_batch[i],), dtype=torch.int64).tolist()
#     #                 total_samples +=self.num_samples_per_class[i]
#     def __iter__(self) -> Iterator[int]:
#         total_samples = 0
#         for _ in range(self.num_samples // self.batch_size):
#             for i in range(len(self.num_samples_per_class)):
#                 if self.num_samples_per_class[i] > 0:
#                     upper_bound = min(self.num_samples_per_class[i] + total_samples, len(self.data_source))
#                     yield from torch.randint(low=total_samples, high=upper_bound, size=(self.number_samples_mini_batch[i],), dtype=torch.int64)
#                 total_samples += self.num_samples_per_class[i]


#     def __len__(self) -> int:
#         return self.num_samples