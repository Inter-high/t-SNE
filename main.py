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
    P[i, j] = round(calc_high_dim_sim(x, x[i], x[j], sigma), 3)
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
    Q[i, j] = round(calc_low_dim_sim(y, y[i], y[j]), 3)
    Q[j, i] = Q[i, j]  # 대칭 행렬로 가정
print(Q)


def calc_cost(p, q):
    indices = [(0, 1), (0, 2), (1, 2)]
    cost = 0.0
    for i, j in indices:
        cost += p[i, j] * round(np.log(p[i, j] / q[i, j]), 3)
    
    return cost

cost = calc_cost(P, Q)
print(f'{cost}\n')


def calc_gradient(p, q, y):
    """
    TODO: gradient_list 할당 수식 부분 디버깅
    """
    indices = [(0, 1), (0, 2), (1, 2)]
    gradient_list = []
    minus_list = []
    weight_list = []
    vector_diff_list = []
    for i, j in indices:
        minus = round(p[i, j] - q[i, j], 3)
        weight = round((1 + calc_distance(y[i], y[j])) ** -1, 3)
        vector_diff = y[i] - y[j]
        minus_list.append(minus)
        weight_list.append(weight)
        vector_diff_list.append(vector_diff)

    for i, j in indices:
        gradient_list.append(4 * (
            (minus_list[i] * np.dot(vector_diff_list[i], weight_list[i])) +
            (minus_list[j] * np.dot(vector_diff_list[j], weight_list[j]))
        ))

    return [np.round(arr, 3) for arr in gradient_list]

gradient = calc_gradient(P, Q, y)
print(gradient)