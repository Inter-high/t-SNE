"""
This script demonstrates the application of t-SNE on a sample of the MNIST dataset.
It fetches a random sample, applies t-SNE dimensionality reduction, and prepares
the transformed data for visualization.

The script allows configuration of key t-SNE hyperparameters such as perplexity,
learning rate, maximum iterations, and early exaggeration through command-line
arguments.

Author: yumemonzo@gmail.com
Date: 2024-12-19
"""

import argparse
from data import get_mnist_sample
from tsne import TSNE


def get_args() -> argparse.Namespace:
    """
    Parse and retrieve command-line arguments.

    Returns:
        argparse.Namespace: The parsed arguments with the following attributes:
            - sample_size (int): Number of MNIST samples to retrieve (default: 1000).
            - perplexity (int): Perplexity parameter for t-SNE (default: 30).
            - learning_rate (float): Learning rate for t-SNE optimization (default: 200.0).
            - max_iter (int): Maximum number of t-SNE iterations (default: 2000).
            - early_exaggeration (int): Early exaggeration factor for t-SNE (default: 4).
    """
    parser = argparse.ArgumentParser(description="t-SNE Training for MNIST Dataset")
    parser.add_argument(
        "--sample_size",
        type=int,
        help="Number of MNIST samples to retrieve",
        default=1000,
    )
    parser.add_argument(
        "--perplexity", type=int, help="Perplexity parameter for t-SNE", default=30
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        help="Learning rate for t-SNE optimization",
        default=200.0,
    )
    parser.add_argument(
        "--max_iter", type=int, help="Maximum number of t-SNE iterations", default=2000
    )
    parser.add_argument(
        "--early_exaggeration",
        type=int,
        help="Early exaggeration factor for t-SNE",
        default=4,
    )

    args, _ = parser.parse_known_args()
    return args


def main() -> None:
    """
    Main function to perform t-SNE training on the MNIST dataset.

    Steps:
        1. Parse command-line arguments.
        2. Fetch a random sample of MNIST data.
        3. Initialize and train the t-SNE model with specified hyperparameters.
    """
    args = get_args()

    x, y = get_mnist_sample(args.sample_size)
    t_sne = TSNE()

    _ = t_sne.train(
        x, args.perplexity, args.learning_rate, args.max_iter, args.early_exaggeration
    )


if __name__ == "__main__":
    main()
