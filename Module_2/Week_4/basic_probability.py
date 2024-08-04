import numpy as np # type: ignore

def compute_mean(sample):
    return np.mean(sample)

sample = [2, 0, 2, 2, 7, 4, -2, 5, -1, -1]
print(compute_mean(sample))

def compute_median(sample):
    size = len(sample)
    sample = np.sort(sample)
    if(size % 2 == 0):
        return (sample[size // 2 - 1] + sample[size // 2]) / 2
    else:
        return sample[size // 2]

sample = [1, 5, 4, 4, 9, 13]
print(compute_median(sample))

def compute_std(sample):
    mean = compute_mean(sample)
    variance = 0
    for i in range(len(sample)):
        variance += np.power(sample[i] - mean, 2)
    variance /= len(sample)

    return np.sqrt(variance)

sample = [171, 176, 155, 167, 169, 182]
print(compute_std(sample))

def compute_correlation_cofficient(sample1, sample2):
    N = len(sample1)

    sum_x = np.sum(sample1)
    sum_y = np.sum(sample2)
    sum_xy = sample1 @ sample2
    square_x = np.sum(sample1 ** 2)
    square_y = np.sum(sample2 ** 2)

    numerator = N * sum_xy - sum_x * sum_y
    denominator = np.sqrt(N * square_x - sum_x ** 2) * np.sqrt(N * square_y - sum_y ** 2)

    return np.round(numerator / denominator, 2)

sample1 = np.asarray ([-2, -5, -11, 6, 4, 15, 9])
sample2 = np.asarray ([4, 25, 121, 36, 16, 225, 81])
print(compute_correlation_cofficient(sample1, sample2))