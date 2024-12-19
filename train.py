"""
This script demonstrates the application of t-SNE on a sample of the MNIST dataset.
It fetches a random sample, applies t-SNE dimensionality reduction, and prepares
the transformed data for visualization.

The script allows configuration of key t-SNE hyperparameters such as perplexity,
learning rate, maximum iterations, and early exaggeration through Hydra configuration.

Author: yumemonzo@gmail.com
Date: 2024-12-19
"""

import os
from data import get_mnist_sample
from tsne import TSNE
from utils import (
    get_logger,
    plot_tsne,
    plot_kl_divergence,
)
import hydra
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="configs", config_name="train")
def main(cfg: DictConfig) -> None:
    """
    Main function to perform t-SNE dimensionality reduction on the MNIST dataset.

    Workflow:
        1. Retrieve the output directory generated by Hydra for saving results.
        2. Set up a logger to log the training process.
        3. Fetch a random sample of the MNIST dataset based on the specified sample size.
        4. Initialize and train the t-SNE model using the provided hyperparameters.
        5. Save and visualize the KL divergence and the 2D t-SNE embedding.

    Args:
        cfg (DictConfig): Configuration object containing t-SNE hyperparameters and other settings.
    """
    file_path = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    logger = get_logger()

    x, y = get_mnist_sample(cfg.sample_size)
    tsne = TSNE(logger)

    tsne_y, kl_divergence = tsne.train(
        x, cfg.perplexity, cfg.learning_rate, cfg.max_iter, cfg.early_exaggeration
    )

    plot_kl_divergence(kl_divergence, os.path.join(file_path, "kl_divergence.jpg"))
    plot_tsne(tsne_y, y, os.path.join(file_path, "result.jpg"))


if __name__ == "__main__":
    main()
