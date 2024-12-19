"""
This script provides utility functions for t-SNE dimensionality reduction tasks.
It includes functionalities for logging, folder creation with timestamps, and
visualizing both KL divergence and t-SNE transformed data.

The utilities are designed to assist in organizing experiments, saving results,
and generating visual insights from t-SNE processes.

Author: yumemonzo@gmail.com
Date: 2024-12-19
"""

import os
import logging
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime


def get_logger(file_path: str) -> logging.Logger:
    """
    Creates and returns a logger object for logging messages to both console and file.

    Parameters:
        file_path (str): Path to the directory where the log file will be saved.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger("tsne_train_logger")
    logger.setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler()
    file_handler = logging.FileHandler(os.path.join(file_path, "train.log"))
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return logger


def create_folder_with_timestamp(base_path: str) -> str:
    """
    Creates a timestamped folder inside a base directory.

    Parameters:
        base_path (str): The base directory where the result folder will be created.

    Returns:
        str: Path to the created timestamped folder.
    """
    result_folder = os.path.join(base_path, "result")
    os.makedirs(result_folder, exist_ok=True)
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    folder_name = os.path.join(result_folder, timestamp)
    os.makedirs(folder_name, exist_ok=True)
    print(f"Folder created: {folder_name}")
    return folder_name


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
