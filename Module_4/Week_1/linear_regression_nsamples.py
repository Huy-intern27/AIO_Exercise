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

def implement_linear_regression_nsamples(X_data, y_data, epoch_max=50, lr=1e-5):
    losses = []

    w1, w2, w3, b = initialize_params()
    N = len(y_data)

    for _ in range(epoch_max):
        losses_total = 0.0
        dw1_total = 0.0
        dw2_total = 0.0
        dw3_total = 0.0
        db_total = 0.0
        for i in range(N):
            x1 = X_data[0][i]
            x2 = X_data[1][i]
            x3 = X_data[2][i]

            y = y_data[i]
            y_hat = predict(x1, x2, x3, w1, w2, w3, b)

            loss = compute_loss(y, y_hat)
            losses_total += loss

            dl_dw1 = compute_gradient_wi(x1, y, y_hat)
            dl_dw2 = compute_gradient_wi(x2, y, y_hat)
            dl_dw3 = compute_gradient_wi(x3, y, y_hat)
            dl_db = compute_gradient_b(y, y_hat)

            dw1_total += dl_dw1
            dw2_total += dl_dw2
            dw3_total += dl_dw3
            db_total += dl_db

        w1 = update_weight_wi(w1, dw1_total / N, lr)
        w2 = update_weight_wi(w2, dw2_total / N, lr)
        w3 = update_weight_wi(w3, dw3_total / N, lr)
        b = update_weight_b(b, db_total / N, lr)

        losses.append(losses_total / N)

    return (w1, w2, w3, b, losses)

X, y = prepare_data('data/advertising.csv')
(w1, w2, w3, b, losses) = implement_linear_regression_nsamples(X, y, 1000, 1e-5)
print(w1, w2, w3)
plt.plot(losses)
plt.xlabel('#epoch')
plt.ylabel('MSE Loss')
plt.show()