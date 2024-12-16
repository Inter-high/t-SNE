"""
TODO: epoch 증가 시, cost 값 증가하는 버그 디버깅
"""

import itertools
import numpy as np

x = np.array([[1, 1, 0, 0],
              [0, 1, 1, 0],
              [1, 0, 0, 1]])
y = np.array([[0.1, 0.2],
              [0.4, 0.3],
              [0.9, 0.7]])
sigma = 1


class TSNE:
    def __init__(self, y):
        self.updated_y = y

    def calc_distance(self, i, j):
        distance = np.sum((i -j) ** 2)

        return distance

    def calc_sigma(self, x, i_idx, j_idx, target_perplexity=2, sigma_min=1e-5, sigma_max=50.0):
        while True:
            sigma = (sigma_min + sigma_max) / 2
            numerator = np.exp(-1 * ((self.calc_distance(x[i_idx], x[j_idx])) / (2 * sigma ** 2)))
            denominator = 0.0
            for i, x_k in enumerate(x):
                if i != i_idx:
                    denominator += np.exp(-1 * ((self.calc_distance(x[i_idx], x_k)) / (2 * sigma ** 2)))
                
            p_ji = numerator / denominator
            h_pi = p_ji * np.log2(p_ji)
            perplexity = 2 ** h_pi
            
            if perplexity > target_perplexity:
                sigma_max = sigma
            elif perplexity < target_perplexity:
                sigma_min = sigma

            if perplexity - target_perplexity < 1e-4:
                break

        return sigma

    def calc_high_dim_sim(self, x, x_i, x_j, sigma):
        numerator = np.exp(-1 * (self.calc_distance(x_i, x_j)) / (2 * sigma ** 2))
        denominator = 0.0
        for x_k in x:
            denominator += np.exp(-1 * (self.calc_distance(x_i, x_k) / (2 * sigma ** 2)))
        denominator -= np.exp(-1 * (self.calc_distance(x_i, x_i) / (2 * sigma ** 2)))

        return numerator / denominator

    def calc_low_dim_sim(self, y, y_i, y_j):
        numerator = (1 + self.calc_distance(y_i, y_j)) ** -1
        denominator = 0.0
        indices = list(itertools.combinations(range(y.shape[0]), 2))
        for i, j in indices:
            denominator += (1 + self.calc_distance(y[i], y[j])) ** -1

        return numerator / denominator

    def calc_cost(self, p, q):
        indices = [(0, 1), (0, 2), (1, 2)]
        cost = 0.0
        for i, j in indices:
            cost += p[i, j] * round(np.log(p[i, j] / q[i, j]), 3)
        
        return round(cost, 3)

    def calc_gradient(self, p, q, y):
        """
        TODO: 함수 정리 필요
        """
        indices = [(0, 1), (0, 2), (1, 2)]
        gradient_list = []
        minus_list = []
        weight_list = []
        vector_diff_list = []
        for i, j in indices:
            minus = round(p[i, j] - q[i, j], 3)
            weight = round((1 + self.calc_distance(y[i], y[j])) ** -1, 3)
            vector_diff = y[i] - y[j]
            minus_list.append(minus)
            weight_list.append(weight)
            vector_diff_list.append(vector_diff)

        for i, j in indices:
            gradient_list.append(4 * (
                (minus_list[i] * np.dot(vector_diff_list[i], weight_list[i])) +
                (minus_list[j] * np.dot(vector_diff_list[j], weight_list[j]))
            ))
            # print(f"4 * {minus_list[i]} * {vector_diff_list[i]} * {weight_list[i]} + {minus_list[j]} * {vector_diff_list[j]} * {weight_list[j]}")

        return [np.round(arr, 3) for arr in gradient_list]

    def data_step(self, y, gradient, learning_rate=0.1):
        updated_y = []
        for y_value, gradient_value in zip(y, gradient):
            update_value = y_value - np.dot(learning_rate, gradient_value)
            updated_y.append(update_value)

        return updated_y

    def train(self, x, epoch=100):
        for num in range(epoch):
            indices = [(0, 1), (0, 2), (1, 2)]
            sigma_list = []
            for i, j in indices:
                sigma = self.calc_sigma(x, i, j)
                sigma_list.append(sigma)

            P = np.zeros((3, 3))
            indices = [(0, 1), (0, 2), (1, 2)]
            for idx, (i, j) in enumerate(indices):
                P[i, j] = round(self.calc_high_dim_sim(x, x[i], x[j], sigma_list[idx]), 3)
                P[j, i] = P[i, j]  # 대칭 행렬로 가정
            # print(P)

            Q = np.zeros((3, 3))
            indices = [(0, 1), (0, 2), (1, 2)]
            for i, j in indices:
                Q[i, j] = round(self.calc_low_dim_sim(self.updated_y, self.updated_y[i], self.updated_y[j]), 3)
                Q[j, i] = Q[i, j]  # 대칭 행렬로 가정
            # print(Q)

            cost = self.calc_cost(P, Q)

            gradient = self.calc_gradient(P, Q, self.updated_y)
            # print(gradient)

            updated_y = self.data_step(self.updated_y, gradient)
            # print(updated_y)
            print(f'epoch: {num+1}/{epoch} cost::{cost} updated_y:: {updated_y}')
            self.updated_y = np.array(updated_y)


tsne = TSNE(y)
tsne.train(x)
