import numpy as np


def define_multinomial_probs(values, dif_prob=2):
    interval_size = values[-1] - values[0] + 1

    general_prob = 1.0 / float(interval_size)
    max_prob = general_prob * dif_prob  # for values

    probs = np.full(interval_size, (1.0 - max_prob * len(values)) / float(interval_size - len(values)))
    for i in range(len(values)):
        probs[values[i] - values[0]] = max_prob

    return probs
