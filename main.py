import itertools
import numpy as np

x = np.array([[1, 1, 0, 0],
              [0, 1, 1, 0],
              [1, 0, 0, 1]])
y = np.array([[0.1, 0.2],
              [0.4, 0.3],
              [0.9, 0.7]])
sigma = 1


def calc_distance(i, j):
    distance = np.sum((i -j) ** 2)

    return distance


def calc_high_dim_sim(x, x_i, x_j, sigma):
    numerator = np.exp(-1 * (calc_distance(x_i, x_j)) / (2 * sigma ** 2))
    denominator = 0.0
    for x_k in x:
        denominator += np.exp(-1 * (calc_distance(x_i, x_k) / (2 * sigma ** 2)))
    denominator -= np.exp(-1 * (calc_distance(x_i, x_i) / (2 * sigma ** 2)))

    return numerator / denominator

P = np.zeros((3, 3))
indices = [(0, 1), (0, 2), (1, 2)]
for i, j in indices:
    P[i, j] = calc_high_dim_sim(x, x[i], x[j], sigma)
    P[j, i] = P[i, j]  # 대칭 행렬로 가정
print(P)


def calc_low_dim_sim(y, y_i, y_j):
    numerator = (1 + calc_distance(y_i, y_j)) ** -1
    denominator = 0.0
    indices = list(itertools.combinations(range(y.shape[0]), 2))
    for i, j in indices:
        denominator += (1 + calc_distance(y[i], y[j])) ** -1

    return numerator / denominator

Q = np.zeros((3, 3))
indices = [(0, 1), (0, 2), (1, 2)]
for i, j in indices:
    Q[i, j] = calc_low_dim_sim(y, y[i], y[j])
    Q[j, i] = Q[i, j]  # 대칭 행렬로 가정
print(Q)