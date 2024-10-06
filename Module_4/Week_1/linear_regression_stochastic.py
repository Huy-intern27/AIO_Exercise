import numpy as np # type: ignore
import random
import matplotlib.pyplot as plt # type: ignore

def get_column(data, index):
    result = [row[index] for row in data]
    return result

def prepare_data(file_name_dataset):
    data = np.genfromtxt(file_name_dataset, delimiter=',', skip_header=1).tolist()

    tv_data = get_column(data, 0)
    radio_data = get_column(data, 1)
    newspaper_data = get_column(data, 2)
    sales_data = get_column(data, 3)

    X = [[1, x1, x2, x3] for x1, x2, x3 in zip(tv_data, radio_data, newspaper_data)]
    y = sales_data

    return X, y

def initialize_params():
    bias = 0
    w1 = random.gauss(mu=0.0, sigma=0.01)
    w2 = random.gauss(mu=0.0, sigma=0.01)
    w3 = random.gauss(mu=0.0, sigma=0.01)

    return [bias, w1, w2, w3]

def predict(X_features, weights):
    result = sum([(i * j) for i, j in zip(X_features, weights)])

    return result

def compute_loss(y_hat, y):
    loss = (y_hat - y) ** 2
    return loss

def compute_gradient_w(X_feature, y, y_hat):
    dl_db = 2 * (y_hat - y)
    dl_dw = [2 * x * (y_hat - y) for x in X_feature[1:]]
    dl_dweights = [dl_db, *(x for x in dl_dw)]

    return dl_dweights

def update_weight(weights, dl_dweights, lr):
    for i in range(len(weights)):
        weights[i] -= lr * dl_dweights[i]

    return weights

def implement_linear_regression(X_features, y_output, epoch_max=50, lr=1e-5):
    weights = initialize_params()
    N = len(y_output)
    losses = []
    for _ in range(epoch_max):
        for i in range(N):
            feature_i = X_features[i]
            y = y_output[i]
            y_hat = predict(feature_i, weights)

            loss = compute_loss(y_hat, y)
            dl_dweights = compute_gradient_w(feature_i, y, y_hat)
            weights = update_weight(weights, dl_dweights, lr)

            losses.append(loss)

    return weights, losses

X, y = prepare_data('data/advertising.csv')
W, L = implement_linear_regression(X, y)
print(L[9999])
plt.plot(L[0:100])
plt.xlabel('iteration')
plt.ylabel('Loss')
plt.show()