"""
Cache key generation for SAVA (no LAVA dependencies).
"""
class SavaCacheKey:
    """
    Generates a unique cache key for SAVA scores based on dataset configuration.
    """
    def __init__(self, config, is_deepsmote=False, is_noisy=False, is_oversampled=False,
                 is_noise_first=False, is_selection_first=False):
        """
        Args:
            config: The configuration object (e.g., from get_args()).
            is_deepsmote: Whether the dataset is DeepSMOTE‑balanced.
            is_noisy: Whether the dataset has label noise.
            is_oversampled: Whether random oversampling was applied.
            is_noise_first: Whether noise was added before oversampling.
            is_selection_first: Whether selection is applied before other ops.
        """
        self.config = config
        self.is_deepsmote = is_deepsmote
        self.is_noisy = is_noisy
        self.is_oversampled = is_oversampled
        self.is_noise_first = is_noise_first
        self.is_selection_first = is_selection_first

    def generate(self) -> str:
        """
        Returns a unique string key for caching SAVA scores.
        Example: 'cifar10_exp_0.01_deepsmote_0_sava'
        """
        parts = [self.config.dataset, self.config.imb_type, str(self.config.imb_factor)]

        if self.is_noisy and hasattr(self.config, 'noise_ratio') and self.config.noise_ratio > 0:
            parts.append(f"noise{self.config.noise_ratio}")

        if self.is_noise_first:
            parts.append("noise_first")

        if self.is_selection_first:
            parts.append("selection_first")

        if self.is_deepsmote:
            parts.append("deepsmote")

        if self.is_oversampled:
            parts.append("oversampled")

        # Add random seed and a SAVA suffix to avoid mixing with LAVA caches
        parts.append(str(self.config.rand_number))
        parts.append("sava")

        return "_".join(parts)