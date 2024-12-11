import numpy as np


def calc_similarity(x, i_idx, sigma):
    distances = np.linalg.norm(x - x[i_idx], axis=1)

    similarities = np.exp(-distances / (2 * sigma ** 2))
    similarities[i_idx] = 0

    return similarities / (np.sum(similarities) + 1e-10)


def calc_entropy(sim_ji):
    return -np.sum(sim_ji * np.log2(sim_ji + 1e-10))


def calc_perplexity(entropy):
    return 2 ** entropy


def calc_sigma(x, i_idx, perplexity_target, epsilon=1e-5):
    sigma_min, sigma_max = 1e-5, 50
    perplexity = np.inf

    while abs(perplexity - perplexity_target) > epsilon:
        sigma = (sigma_min + sigma_max) / 2

        similarities = calc_similarity(x, i_idx, sigma)
        entropy = calc_entropy(similarities)
        perplexity = calc_perplexity(entropy)

        if perplexity > perplexity_target:
            sigma_max = sigma
        else:
            sigma_min = sigma
    
    return sigma


x = np.array([[1, 1, 0, 0],
              [0, 1, 1, 0],
              [1, 0, 0, 1]])

sigma = calc_sigma(x, 0, perplexity_target=2)
print(f"sigma: {sigma}")