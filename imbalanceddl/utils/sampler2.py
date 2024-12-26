import torch
import numpy as np
from torch.utils.data import Sampler
from typing import Iterator, Iterable, Optional, Sequence, List, TypeVar, Generic, Sized, Union

class BalancedSampler(Sampler[int]):
    r"""Samples elements from ``[0,..,len(weights)-1]`` with given probabilities (weights),
    adjusted to account for the provided algorithm for class-based sampling.

    Args:
        weights (sequence): a sequence of weights, not necessarily summing up to one
        num_samples_per_class (sequence): list of number of samples per class
        num_classes (int): total number of classes
        M (int): number of classes per subgroup
        batch_size (int): number of samples to draw in a batch
        replacement (bool): if ``True``, samples are drawn with replacement.
            If not, they are drawn without replacement.
        generator (Generator): Generator used in sampling.

    Example:
        >>> num_samples_per_class = [500, 497, 100]
        >>> weights = [0.0002, 0.0002, 0.02]
        >>> sampler = WeightedRandomSampler(weights, num_samples_per_class, 3, 2, 40, replacement=True)
        >>> list(sampler)
        [0, 1, 2, ...]
    """

    def __init__(
        self,
        weights: Sequence[float],
        num_samples_per_class: Sequence[int],
        num_classes: int,
        M: int,
        batch_size: int,
        replacement: bool = True,
        generator=None,
    ) -> None:
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError(
                f"batch_size should be a positive integer value, but got batch_size={batch_size}"
            )
        if not isinstance(replacement, bool):
            raise ValueError(
                f"replacement should be a boolean value, but got replacement={replacement}"
            )
        if len(num_samples_per_class) != num_classes:
            raise ValueError(
                "Length of num_samples_per_class should match num_classes"
            )
        self.weights = torch.tensor(weights, dtype=torch.float32)
        self.num_samples_per_class = num_samples_per_class
        self.num_classes = num_classes
        self.M = M
        self.batch_size = batch_size
        self.replacement = replacement
        self.generator = generator
    def __iter__(self) -> Iterator[int]:
        
        for _ in range(sum(self.num_samples_per_class) // self.batch_size):
                # Step 1: Randomly select M classes
            selected_classes = np.zeros(shape=(self.num_classes))
            choice = np.random.choice(
                np.arange(0, self.num_classes), size=self.M, replace=False)
            selected_classes[choice] = 1  # Mark selected classes

            # Step 2: Create mask based on selected classes
            mask = []
            # for i in range(self.num_classes):
            #     mask += [selected_classes[i]] * self.num_samples_per_class[i]
            mask = np.repeat(selected_classes, self.num_samples_per_class)
            mask = torch.tensor(mask, dtype=torch.float32)
                

        
            # Step 3: Adjust weights using mask
            new_weights = mask * self.weights
            
        # Step 4: Sample based on adjusted weights
            rand_tensor = torch.multinomial(new_weights, self.batch_size, self.replacement, generator=self.generator)
            yield from iter(rand_tensor.tolist())

    def __len__(self) -> int:
        return self.batch_size

