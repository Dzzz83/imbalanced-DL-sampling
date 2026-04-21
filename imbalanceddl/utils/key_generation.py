class LavaCacheKey:
    """
    Generates a unique cache key for LAVA scores based on dataset configuration.
    """
    def __init__(self, config, is_deepsmote=False, is_noisy=False):
        """
        Args:
            config: The configuration object (e.g., from get_args()).
            is_deepsmote: Whether the dataset is DeepSMOTE‑balanced (synthetic).
            is_noisy: Whether the dataset has label noise.
        """
        self.config = config
        self.is_deepsmote = is_deepsmote
        self.is_noisy = is_noisy

    def generate(self) -> str:
        """
        Returns a unique string key for caching LAVA scores.
        Example: 'cifar10_exp_0.01_deepsmote_0' or 'cifar10_exp_1.0_noise0.25_0'
        """
        parts = [self.config.dataset, self.config.imb_type, str(self.config.imb_factor)]

        if self.is_noisy and hasattr(self.config, 'noise_ratio') and self.config.noise_ratio > 0:
            parts.append(f"noise{self.config.noise_ratio}")

        if self.is_deepsmote:
            parts.append("deepsmote")

        parts.append(str(self.config.rand_number))

        return "_".join(parts)