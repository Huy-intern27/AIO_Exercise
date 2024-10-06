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

    X = [tv_data, radio_data, newspaper_data]
    y = sales_data

    return X, y

def initialize_params():
    w1, w2, w3, b = (0.016992259082509283, 0.0070783670518262355, -0.002307860847821344, 0)

    return w1, w2, w3, b

def predict(x1, x2, x3, w1, w2, w3, b):
    result = x1 * w1 + x2 * w2 + x3 * w3 + b

    return result

def compute_loss(y, y_hat):
    loss = (y_hat - y) ** 2

    return loss

def compute_gradient_wi(xi, y, y_hat):
    dl_dwi = 2 * xi *(y_hat - y)

    return dl_dwi

def compute_gradient_b(y, y_hat):
    dl_db = 2 * (y_hat - y)

    return dl_db

def update_weight_wi(wi, dl_dwi, lr):
    wi = wi - lr * dl_dwi

    return wi

def update_weight_b(b, dl_db, lr):
    b = b - lr * dl_db

    return b

def implement_linear_regression(X_data, y_data, epoch_max = 50, lr = 1e-5):
    losses = []

    w1, w2, w3, b = initialize_params()
    N = len(y_data)

    for _ in range(epoch_max):
        for i in range(N):
            x1 = X_data[0][i]
            x2 = X_data[1][i]
            x3 = X_data[2][i]
            y = y_data[i]

            y_hat = predict(x1, x2, x3, w1, w2, w3, b)
            loss = compute_loss(y_hat, y)

            dl_dw1 = compute_gradient_wi(x1, y, y_hat)
            dl_dw2 = compute_gradient_wi(x2, y, y_hat)
            dl_dw3 = compute_gradient_wi(x3, y, y_hat)
            dl_db = compute_gradient_b(y, y_hat)

            w1 = update_weight_wi(w1, dl_dw1, lr)
            w2 = update_weight_wi(w2, dl_dw2, lr)
            w3 = update_weight_wi(w3, dl_dw3, lr)
            b = update_weight_b(b, dl_db, lr)

            losses.append(loss)
    return (w1, w2, w3, b, losses)

X, y = prepare_data('data/advertising.csv')
(w1, w2, w3, b, losses) = implement_linear_regression(X, y)
plt.plot(losses[:100])
plt.xlabel('#iteration')
plt.ylabel('Loss')
plt.show()

sales = predict(19.2, 35.9, 51.3, w1, w2, w3, b)
print(sales)

after_wi = update_weight_wi(wi=1.0, dl_dwi=-0.5, lr=1e-5)
after_b = update_weight_b(b=0.5, dl_db=-1.0, lr=1e-5)
print(after_b)