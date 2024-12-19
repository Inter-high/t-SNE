"""
This class provides an implementation of t-SNE (t-Distributed Stochastic Neighbor Embedding),
a dimensionality reduction algorithm primarily used for visualizing high-dimensional data.

It includes methods to calculate distance matrices, determine optimal sigmas based on
perplexity, compute joint probability distributions (P and Q matrices), and perform
gradient descent to optimize the embeddings.

Author: yumemonzo@gmail.com
Date: 2024-12-19
"""

import numpy as np
from scipy.spatial.distance import pdist, squareform


class TSNE:
    """
    Implementation of t-SNE (t-Distributed Stochastic Neighbor Embedding).

    This class provides the functionality to perform dimensionality reduction
    using t-SNE. The algorithm reduces high-dimensional data to a low-dimensional
    space (typically 2D or 3D) for visualization while preserving the local
    structure of the data.

    Methods:
        - calc_distance_matrix: Calculate pairwise squared Euclidean distances.
        - find_sigma: Find optimal sigma values for each data point.
        - calc_p_matrix: Construct the P matrix (high-dimensional joint probabilities).
        - calc_q_matrix: Construct the Q matrix (low-dimensional joint probabilities).
        - calc_kl_divergence: Compute the KL divergence between P and Q matrices.
        - calc_gradient: Compute gradients for t-SNE optimization.
        - train: Perform t-SNE training to produce low-dimensional embeddings.

    Usage:
        - Instantiate the class: `t_sne = TSNE()`.
        - Call the `train` method with high-dimensional data and hyperparameters:
          ```
          y_tsne = t_sne.train(x, target_perplexity=30, learning_rate=200.0,
                               max_iter=2000, early_exaggeration=4)
          ```
          where `x` is a NumPy array of shape (n_samples, n_features).
    """

    def __init__(self) -> None:
        pass

    def calc_distance_matrix(self, arr: np.ndarray) -> np.ndarray:
        """
        Compute the squared Euclidean distance matrix for the input array.

        Parameters:
            arr (np.ndarray): Input array of shape (n_samples, n_features).

        Returns:
            np.ndarray: Pairwise squared Euclidean distance matrix of shape (n_samples, n_samples).
        """
        return squareform(pdist(arr, metric="sqeuclidean"))

    def find_sigma(
        self, distance_matrix: np.ndarray, target_perplexity: int
    ) -> np.ndarray:
        """
        Find the optimal sigma values for each data point to achieve the target perplexity.

        Parameters:
            distance_matrix (np.ndarray): Pairwise squared Euclidean distance matrix.
            target_perplexity (int): Desired perplexity for the probability distribution.

        Returns:
            np.ndarray: Array of sigma values for each data point.
        """
        num_points = distance_matrix.shape[0]
        sigmas = np.zeros(num_points)

        for i in range(num_points):
            norm = distance_matrix[i]
            std_norm = np.std(norm)

            best_sigma = None
            best_diff = np.inf

            for sample_sigma in np.linspace(0.01 * std_norm, 5 * std_norm, 200):
                p = np.exp(-norm / (2 * sample_sigma**2))
                p[i] = 0

                epsilon = np.nextafter(0, 1)  # Closest positive float to zero.
                p_sum = np.sum(p)
                if p_sum == 0.0:
                    p_sum = epsilon

                p_new = np.maximum(p / p_sum, epsilon)

                H_pi = -np.sum(p_new * np.log2(p_new))
                diff = np.abs(
                    np.log(target_perplexity) - (H_pi * np.log(2))
                )  # log(target_perplexity) - log(perplexity)

                if diff < best_diff:
                    best_diff = diff
                    best_sigma = sample_sigma

            sigmas[i] = best_sigma

        return sigmas

    def calc_p_matrix(
        self, distance_matrix: np.ndarray, sigmas: np.ndarray
    ) -> np.ndarray:
        """
        Calculate the P matrix (joint probability matrix) based on the distance matrix and sigmas.

        Parameters:
            distance_matrix (np.ndarray): Pairwise squared Euclidean distance matrix.
            sigmas (np.ndarray): Array of sigma values for each data point.

        Returns:
            np.ndarray: Symmetric joint probability matrix (P matrix).
        """
        p_matrix = np.exp(-distance_matrix / (2 * sigmas[:, np.newaxis] ** 2))
        np.fill_diagonal(p_matrix, 0)  # Set diagonal elements (self-distances) to zero.

        p_matrix /= (
            np.sum(p_matrix, axis=1, keepdims=True) + 1e-10
        )  # Normalize rows to sum to 1.
        p_matrix = (p_matrix + p_matrix.T) / (2 * len(distance_matrix))  # Symmetrize.

        return p_matrix

    def calc_q_matrix(self, y: np.ndarray) -> np.ndarray:
        """
        Calculate the Q matrix (low-dimensional joint probability matrix).

        Parameters:
            y (np.ndarray): Low-dimensional data points of shape (n_samples, 2).

        Returns:
            np.ndarray: Q matrix representing pairwise similarities in the low-dimensional space.
        """
        distances = 1 + self.calc_distance_matrix(y)
        q_matrix = 1 / distances  # Inverse squared distances.
        np.fill_diagonal(q_matrix, 0)

        q_matrix /= np.sum(q_matrix)

        return q_matrix

    def calc_kl_divergence(self, p_matrix: np.ndarray, q_matrix: np.ndarray) -> float:
        """
        Calculate the KL divergence between the P and Q matrices.

        Parameters:
            p_matrix (np.ndarray): P matrix (high-dimensional joint probability matrix).
            q_matrix (np.ndarray): Q matrix (low-dimensional joint probability matrix).

        Returns:
            float: The KL divergence value.
        """
        return np.sum(p_matrix * np.log((p_matrix + 1e-10) / (q_matrix + 1e-10)))

    def calc_gradient(
        self, p_matrix: np.ndarray, q_matrix: np.ndarray, y: np.ndarray
    ) -> np.ndarray:
        """
        Compute the gradient for t-SNE optimization.

        Parameters:
            p_matrix (np.ndarray): P matrix (high-dimensional joint probability matrix).
            q_matrix (np.ndarray): Q matrix (low-dimensional joint probability matrix).
            y (np.ndarray): Low-dimensional data points of shape (n_samples, 2).

        Returns:
            np.ndarray: Gradient array of shape (n_samples, 2).
        """
        pq_diff = (p_matrix - q_matrix)[:, :, np.newaxis]
        distances = 1 + squareform(pdist(y, metric="sqeuclidean"))[:, :, np.newaxis]
        grad = np.sum(
            pq_diff * (y[:, np.newaxis, :] - y[np.newaxis, :, :]) / distances, axis=1
        )

        return 4 * grad

    def train(
        self,
        x: np.ndarray,
        target_perplexity: int,
        learning_rate: float,
        max_iter: int,
        early_exaggeration: int,
    ) -> np.ndarray:
        """
        Train the t-SNE model and return the low-dimensional embeddings.

        Parameters:
            x (np.ndarray): High-dimensional input data of shape (n_samples, n_features).
            target_perplexity (int): Desired perplexity for the probability distribution.
            learning_rate (float): Learning rate for gradient descent.
            max_iter (int): Maximum number of iterations for optimization.
            early_exaggeration (int): Factor to exaggerate the P matrix early in training.

        Returns:
            np.ndarray: Low-dimensional embeddings of shape (n_samples, 2).
        """
        distance_matrix = self.calc_distance_matrix(x)
        sigmas = self.find_sigma(distance_matrix, target_perplexity)

        p_matrix = self.calc_p_matrix(distance_matrix, sigmas)
        p_matrix *= (
            early_exaggeration  # Exaggerate P and Q differences early in training.
        )

        y = (
            np.random.randn(x.shape[0], 2) * 1e-4
        )  # Initialize with small random values.
        momentum = np.zeros_like(y)  # Initialize momentum.
        alpha = 0.5  # Initial momentum value.

        for iteration in range(max_iter):
            q_matrix = self.calc_q_matrix(y)
            gradient = self.calc_gradient(p_matrix, q_matrix, y)

            # Update using momentum
            momentum = alpha * momentum - learning_rate * gradient
            y += momentum

            if iteration == 250:
                p_matrix /= early_exaggeration  # Stop early exaggeration
                print("Early exaggeration ended.")

            # Increase momentum dynamically
            if iteration >= 250:
                alpha = 0.8

            if iteration % 100 == 0 or iteration == max_iter - 1:
                kl_divergence = self.calc_kl_divergence(p_matrix, q_matrix)
                print(f"Iteration {iteration}: KL Divergence = {kl_divergence:.4f}")

        return y
