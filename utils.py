"""
This script provides utility functions for t-SNE dimensionality reduction tasks.
It includes functionalities for logging, folder creation with timestamps, and
visualizing both KL divergence and t-SNE transformed data.

The utilities are designed to assist in organizing experiments, saving results,
and generating visual insights from t-SNE processes.

Author: yumemonzo@gmail.com
Date: 2024-12-19
"""

import logging
import numpy as np
import matplotlib.pyplot as plt


def get_logger() -> logging.Logger:
    """
    Creates and returns a logger object for logging messages to both console and file.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger("tsne_train_logger")
    logger.setLevel(logging.DEBUG)

    return logger


def plot_kl_divergence(kl_divergences: list, file_path: str) -> None:
    """
    Plots KL divergence values over iterations and saves the plot as an image.

    Parameters:
        kl_divergences (list): List of KL divergence values over iterations.
        file_path (str): Path to save the plot image.

    Returns:
        None
    """
    plt.figure(figsize=(8, 6))
    plt.plot(kl_divergences)
    plt.xlabel("Iteration")
    plt.ylabel("KL Divergence")
    plt.title("KL Divergence over Iterations")
    plt.grid()
    plt.savefig(file_path)
    plt.close()


def plot_tsne(y: np.ndarray, labels: np.ndarray, file_path: str) -> None:
    """
    Plots the t-SNE transformed data points with labels and saves the plot as an image.

    Parameters:
        y (np.ndarray): 2D array of t-SNE transformed data points (shape: [n_samples, 2]).
        labels (np.ndarray): Array of labels corresponding to the data points.
        file_path (str): Path to save the plot image.

    Returns:
        None
    """
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(y[:, 0], y[:, 1], c=labels, cmap="tab10", s=10, alpha=0.7)
    plt.colorbar(scatter, label="Label")
    plt.title("t-SNE Result", fontsize=14)
    plt.axis("off")
    plt.savefig(file_path, dpi=300)
    plt.close()
