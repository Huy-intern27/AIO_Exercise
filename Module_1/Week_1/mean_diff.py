def mean_difference(y_true, y_pred, n, p):
    return ((y_true ** (1 / n)) - (y_pred ** (1 / n))) ** p

if __name__ == "__main__":
    print(mean_difference(100, 99.5, 2, 1))