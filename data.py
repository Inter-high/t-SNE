"""
This script provides a utility function to retrieve a random sample from
the MNIST dataset. The function fetches the dataset, shuffles it, and
returns the specified number of samples with normalized features and
integer labels.

Author: yumemonzo@gmail.com
Date: 2024-12-19
"""

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.utils import shuffle
from typing import Tuple


def get_mnist_sample(sample_size: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fetch a sample of the MNIST dataset with the specified sample size.

    Parameters:
        sample_size (int): The number of samples to retrieve from the dataset.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing the sampled features (x) and labels (y).
            - x: A NumPy array of shape (sample_size, 784), normalized to the range [0, 1].
            - y: A NumPy array of shape (sample_size,) containing integer labels for the samples.
    """
    data = fetch_openml("mnist_784", version=1)

    x, y = data.data, data.target
    x, y = shuffle(x, y, random_state=42)

    y = y.astype(int)

    x = x.to_numpy() if hasattr(x, "to_numpy") else np.array(x)

    x = x[:sample_size]
    y = y[:sample_size]

    x = x / 255.0

    return x, y


def get_cifar10_sample(sample_size: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fetch a sample of the CIFAR-10 dataset with the specified sample size using fetch_openml.

    Parameters:
        sample_size (int): The number of samples to retrieve from the dataset.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing the sampled features (x) and labels (y).
            - x: A NumPy array of shape (sample_size, 3072), normalized to the range [0, 1].
            - y: A NumPy array of shape (sample_size,) containing integer labels for the samples.
    """
    data = fetch_openml("CIFAR_10", version=1, as_frame=False)

    x, y = data.data, data.target
    x, y = shuffle(x, y, random_state=42)

    y = y.astype(int)  # CIFAR-10의 레이블은 정수형으로 변환

    x = x[:sample_size]
    y = y[:sample_size]

    x = x / 255.0  # 픽셀 값을 [0, 1]로 정규화

    return x, y
