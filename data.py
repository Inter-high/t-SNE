import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.utils import shuffle


def get_mnist_sample(sample_size):
    data = fetch_openml('mnist_784', version=1)

    x, y = data.data, data.target
    x, y = shuffle(x, y, random_state=42)
    y = y.astype(int)

    x = x.to_numpy() if hasattr(x, "to_numpy") else np.array(x)

    x = x[:sample_size]
    y = y[:sample_size]

    x = x / 255.0

    return x, y
