"""
Dataset generation for quadratic equations.
"""
import torch
from torch.utils.data import Dataset
import numpy as np
from sklearn.preprocessing import StandardScaler


class QuadraticDataset(Dataset):
    """
    Generates synthetic quadratic equations (ax^2 + bx + c = 0)
    and solves for roots (including complex numbers).
    """

    def __init__(self, samples=10000, range_val=10, seed=None):
        if seed is not None:
            np.random.seed(seed)
        
        self.samples = samples
        # Generate coefficients
        # We ensure 'a' is not 0 to maintain quadratic nature
        self.a = np.random.uniform(-range_val, range_val, samples)
        self.a[np.abs(self.a) < 0.1] = 1.0  # Fix near-zero a
        self.b = np.random.uniform(-range_val, range_val, samples)
        self.c = np.random.uniform(-range_val, range_val, samples)

        self.inputs = np.stack([self.a, self.b, self.c], axis=1)
        self.targets = self._solve_quadratic(self.a, self.b, self.c)

        # Preprocessing: Standard Scaling for faster convergence
        self.scaler_in = StandardScaler()
        self.inputs_scaled = self.scaler_in.fit_transform(self.inputs)

        # Convert to PyTorch tensors
        self.X = torch.FloatTensor(self.inputs_scaled)
        self.y = torch.FloatTensor(self.targets)

    def _solve_quadratic(self, a, b, c):
        """
        Analytic solution for ground truth.
        Returns [Root1_Real, Root1_Imag, Root2_Real, Root2_Imag]
        """
        delta = b ** 2 - 4 * a * c

        # Prepare arrays for solutions
        r1_real = np.zeros_like(delta)
        r1_imag = np.zeros_like(delta)
        r2_real = np.zeros_like(delta)
        r2_imag = np.zeros_like(delta)

        # Case 1: Real Roots (delta >= 0)
        mask_real = delta >= 0
        if np.any(mask_real):
            sqrt_delta = np.sqrt(delta[mask_real])
            r1_real[mask_real] = (-b[mask_real] + sqrt_delta) / (2 * a[mask_real])
            r2_real[mask_real] = (-b[mask_real] - sqrt_delta) / (2 * a[mask_real])

        # Case 2: Complex Roots (delta < 0)
        mask_imag = delta < 0
        if np.any(mask_imag):
            sqrt_delta = np.sqrt(-delta[mask_imag])
            # Real part is -b / 2a
            real_part = -b[mask_imag] / (2 * a[mask_imag])
            # Imaginary part is sqrt(-delta) / 2a
            imag_part = sqrt_delta / (2 * a[mask_imag])

            r1_real[mask_imag] = real_part
            r2_real[mask_imag] = real_part
            r1_imag[mask_imag] = imag_part
            r2_imag[mask_imag] = -imag_part

        return np.stack([r1_real, r1_imag, r2_real, r2_imag], axis=1)

    def add_samples(self, a, b, c):
        """Add new samples to the dataset."""
        a = np.array(a) if not isinstance(a, np.ndarray) else a
        b = np.array(b) if not isinstance(b, np.ndarray) else b
        c = np.array(c) if not isinstance(c, np.ndarray) else c
        
        # Ensure a is not zero
        a = np.where(np.abs(a) < 0.1, 1.0, a)
        
        new_inputs = np.stack([a, b, c], axis=1)
        new_targets = self._solve_quadratic(a, b, c)
        
        # Combine with existing data
        self.inputs = np.vstack([self.inputs, new_inputs])
        self.targets = np.vstack([self.targets, new_targets])
        
        # Re-fit scaler and transform
        self.inputs_scaled = self.scaler_in.fit_transform(self.inputs)
        self.X = torch.FloatTensor(self.inputs_scaled)
        self.y = torch.FloatTensor(self.targets)
        self.samples = len(self.inputs)

    @classmethod
    def from_equation(cls, a_center, b_center, c_center, samples=10000, variation=0.5, seed=None):
        """
        Generate dataset centered around a specific equation with variation.
        
        Args:
            a_center: Center value for coefficient a
            b_center: Center value for coefficient b
            c_center: Center value for coefficient c
            samples: Number of samples to generate
            variation: Relative variation (0.5 = ±50% variation)
            seed: Random seed for reproducibility
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Ensure a_center is not zero
        if abs(a_center) < 0.1:
            a_center = 1.0 if a_center >= 0 else -1.0
        
        # Generate coefficients around the center with variation
        a_range = abs(a_center) * variation
        b_range = abs(b_center) * variation if b_center != 0 else abs(a_center) * variation
        c_range = abs(c_center) * variation if c_center != 0 else abs(a_center) * variation
        
        # Ensure minimum range
        min_range = 0.1
        a_range = max(a_range, min_range)
        b_range = max(b_range, min_range)
        c_range = max(c_range, min_range)
        
        a = np.random.normal(a_center, a_range, samples)
        b = np.random.normal(b_center, b_range, samples)
        c = np.random.normal(c_center, c_range, samples)
        
        # Ensure a is not zero
        a = np.where(np.abs(a) < 0.1, np.sign(a) * 0.1, a)
        
        # Create dataset
        dataset = cls.__new__(cls)
        dataset.samples = samples
        dataset.a = a
        dataset.b = b
        dataset.c = c
        
        dataset.inputs = np.stack([dataset.a, dataset.b, dataset.c], axis=1)
        dataset.targets = dataset._solve_quadratic(dataset.a, dataset.b, dataset.c)
        
        # Preprocessing: Standard Scaling for faster convergence
        dataset.scaler_in = StandardScaler()
        dataset.inputs_scaled = dataset.scaler_in.fit_transform(dataset.inputs)
        
        # Convert to PyTorch tensors
        dataset.X = torch.FloatTensor(dataset.inputs_scaled)
        dataset.y = torch.FloatTensor(dataset.targets)
        
        return dataset

    def __len__(self):
        return self.samples

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

