import numpy as np
from scipy.spatial.distance import pdist, squareform


class TSNE:
    def __init__(self):
        pass

    def calc_distance_matrix(self, arr):
        return squareform(pdist(arr, metric="sqeuclidean"))

    def find_sigma(self, distance_matrix, target_perplexity):
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

                epsilon = np.nextafter(0, 1)  # 0에서 1 방향으로 가장 가까운 값
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

    def calc_p_matrix(self, distance_matrix, sigmas):
        p_matrix = np.exp(-distance_matrix / (2 * sigmas[:, np.newaxis] ** 2))
        np.fill_diagonal(
            p_matrix, 0
        )  # 대각선 요소를 0으로 채움(=자기 자신과의 거리를 0으로 만듬)

        p_matrix /= (
            np.sum(p_matrix, axis=1, keepdims=True) + 1e-10
        )  # 안전성을 위한 추가
        p_matrix = (p_matrix + p_matrix.T) / (2 * len(distance_matrix))  # 대칭화

        return p_matrix

    def calc_q_matrix(self, y):
        distances = 1 + self.calc_distance_matrix(y)
        q_matrix = 1 / distances  # -1 제곱근
        np.fill_diagonal(q_matrix, 0)

        q_matrix /= np.sum(q_matrix)

        return q_matrix

    def calc_kl_divergence(self, p_matrix, q_matrix):
        return np.sum(p_matrix * np.log((p_matrix + 1e-10) / (q_matrix + 1e-10)))

    def calc_gradient(self, p_matrix, q_matrix, y):
        pq_diff = (p_matrix - q_matrix)[:, :, np.newaxis]
        distances = 1 + squareform(pdist(y, metric="sqeuclidean"))[:, :, np.newaxis]
        grad = np.sum(
            pq_diff * (y[:, np.newaxis, :] - y[np.newaxis, :, :]) / distances, axis=1
        )

        return 4 * grad

    def train(self, x, target_perplexity, learning_rate, max_iter, early_exaggeration):
        distance_matrix = self.calc_distance_matrix(x)
        sigmas = self.find_sigma(distance_matrix, target_perplexity)

        p_matrix = self.calc_p_matrix(distance_matrix, sigmas)
        p_matrix *= early_exaggeration  # 학습 초기 P와 Q의 차이를 명확하게 구별

        y = np.random.randn(x.shape[0], 2) * 1e-4  # 작은 값으로 초기화
        momentum = np.zeros_like(y)  # 모멘텀 초기화
        alpha = 0.5  # 초기 모멘텀 값

        for iteration in range(max_iter):
            q_matrix = self.calc_q_matrix(y)
            gradient = self.calc_gradient(p_matrix, q_matrix, y)

            # 모멘텀을 사용해 업데이트
            momentum = alpha * momentum - learning_rate * gradient
            y += momentum

            if iteration == 250:
                p_matrix /= early_exaggeration  # early exaggeration 해제
                print("Early exaggeration ended.")

            # 동적으로 모멘텀 증가
            if iteration >= 250:
                alpha = 0.8

            if iteration % 100 == 0 or iteration == max_iter - 1:
                kl_divergence = np.sum(
                    p_matrix * np.log((p_matrix + 1e-10) / (q_matrix + 1e-10))
                )
                print(f"Iteration {iteration}: KL Divergence = {kl_divergence:.4f}")

        return y
