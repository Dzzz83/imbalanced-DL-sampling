import numpy as np
import torch
class BaseDataset:
    """Base Dataset (Mixin)

    Base dataset for creating imbalanced dataset
    """
    def get_img_num_per_cls(self, cls_num, imb_type, imb_factor):
        # cifar10, cifar100, svhn
        if hasattr(self, "data"):
            img_max = len(self.data) / cls_num
        # cinic10, tiny-imagenet
        elif hasattr(self, "samples"):
            img_max = len(self.samples) / cls_num
        else:
            raise AttributeError("[Warning] Check your data or customize !")
        img_num_per_cls = []
        if imb_type == 'exp':
            for cls_idx in range(cls_num):
                num = img_max * (imb_factor**(cls_idx / (cls_num - 1.0)))
                img_num_per_cls.append(int(num))
        elif imb_type == 'step':
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max))
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max * imb_factor))
        else:
            img_num_per_cls.extend([int(img_max)] * cls_num)
        return img_num_per_cls

    def gen_imbalanced_data(self, img_num_per_cls):
        new_data = []
        new_targets = []
        targets_np = np.array(self.targets, dtype=np.int64)
        classes = np.unique(targets_np)
        # np.random.shuffle(classes)
        self.num_per_cls_dict = dict()
        for the_class, the_img_num in zip(classes, img_num_per_cls):
            self.num_per_cls_dict[the_class] = the_img_num
            idx = np.where(targets_np == the_class)[0]
            np.random.shuffle(idx)
            selec_idx = idx[:the_img_num]
            new_data.append(self.data[selec_idx, ...])
            new_targets.extend([
                the_class,
            ] * the_img_num)
        new_data = np.vstack(new_data)
        assert new_data.shape[0] == len(new_targets)
        self.data = new_data
        self.targets = new_targets

    def get_cls_num_list(self):
        cls_num_list = []
        for i in range(self.cls_num):
            cls_num_list.append(self.num_per_cls_dict[i])
        return cls_num_list
    
    def get_weights(self) -> np.ndarray:
        """
        Compute weights for each class inversely proportional to the number of samples,
        with safeguards to avoid zero or extreme values.

        Returns
        -------
        np.ndarray:
            An array where each element is the weight of a class.
        """
        # Step 1: Get the total number of samples
        total_samples = sum(self.num_per_cls_dict.values())

        # Step 2: Compute weight for each class
        epsilon = 1e-8  # Small value to avoid division by zero
        class_weights = np.array([
            total_samples / (len(self.num_per_cls_dict) * max(count, epsilon))
            for count in self.num_per_cls_dict.values()
        ], dtype=np.float32)

        # Step 3: Clip weights to a minimum threshold to avoid zero
        min_weight = 1e-6  # Replace with your desired minimum weight
        class_weights = np.clip(class_weights, min_weight, None)

        return class_weights

    def get_sample_weights(self):
        """
        Generate weights for each sample based on the class it belongs to,
        with safeguards to avoid zero or extreme values.

        Returns
        -------
        list of float:
            A list where each element is the weight of a sample.
        """
        # Step 1: Get class weights
        class_weights = self.get_weights()

        # Step 2: Get the class of each sample
        targets_np = np.array(self.targets, dtype=np.int64)

        # Step 3: Assign weights for each sample based on its class
        sample_weights = [class_weights[class_idx] for class_idx in targets_np]

        return sample_weights

    
    def get_class_idxs(self):
            """
            Generate class indices (class_idxs) for the dataset.

            Returns
            -------
            class_idxs : list of lists
                Each sublist contains the indices of samples for a particular class.
            """
            targets_np = np.array(self.targets, dtype=np.int64)
            classes = np.unique(targets_np)
            class_idxs = [np.where(targets_np == the_class)[0].tolist() for the_class in classes]
            return class_idxs
    def get_class_idxs2(self):
        """
        Generate class indices (class_idxs) for the dataset.

        Returns
        -------
        class_idxs : list of lists
            Each sublist contains the indices of samples for a particular class.
        """
        # Ensure self.targets is a 1D list or array of integers
        if not isinstance(self.targets, (list, np.ndarray)):
            raise ValueError("Expected self.targets to be a list or numpy array.")

        # Convert self.targets to a numpy array
        targets_np = np.array(self.targets, dtype=np.int64)

        # Validate that targets_np contains valid integer class labels
        if not np.issubdtype(targets_np.dtype, np.integer):
            raise ValueError("Targets must contain integer class labels.")

        # Find unique classes and generate indices for each class
        classes = np.unique(targets_np)
        class_idxs = [np.where(targets_np == the_class)[0].tolist() for the_class in classes]

        # Validate that class_idxs is a list of lists and contains indices
        if not all(isinstance(idxs, list) for idxs in class_idxs):
            raise ValueError("Generated class_idxs is not a list of lists.")
        if not all(all(isinstance(idx, int) for idx in idxs) for idxs in class_idxs):
            raise ValueError("Class indices must be integers.")

        return class_idxs
