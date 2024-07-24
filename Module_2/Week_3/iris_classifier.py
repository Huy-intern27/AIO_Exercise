import numpy as np  # type: ignore
import pandas as pd  # type: ignore


def create_train_data():
    data = pd.read_csv('data/iris.data.txt', header=None)

    return np.array(data)


def compute_prior_probability(train_data):
    y_unique = np.unique(train_data[:, -1])
    prior_probability = np.zeros(len(y_unique))
    for i in range(len(y_unique)):
        prior_probability[i] = np.sum(
            train_data[:, -1] == y_unique[i]) / len(train_data)

    return prior_probability


def compute_conditional_probability(train_data):
    conditional_probability = []
    list_x_name = []
    for i in range(0, train_data.shape[1] - 1):
        x_unique = np.unique(train_data[:, i])
        list_x_name.append(x_unique)

    sample_spaces_setosa = np.sum(train_data[:, -1] == 'Iris-setosa')
    sample_spaces_versicolor = np.sum(train_data[:, -1] == 'Iris-versicolor')
    sample_spaces_virginica = np.sum(train_data[:, -1] == 'Iris-virginica')
    for i in range(len(list_x_name)):
        x_conditional_probability = np.zeros((3, len(list_x_name[i])))
        for j in range(len(list_x_name[i])):
            x_conditional_probability[0, j] = np.sum((train_data[:, i] == list_x_name[i][j]) & (
                train_data[:, -1] == 'Iris-setosa')) / sample_spaces_setosa
            x_conditional_probability[1, j] = np.sum((train_data[:, i] == list_x_name[i][j]) & (
                train_data[:, -1] == 'Iris-versicolor')) / sample_spaces_versicolor
            x_conditional_probability[2, j] = np.sum((train_data[:, i] == list_x_name[i][j]) & (
                train_data[:, -1] == 'Iris-virginica')) / sample_spaces_virginica
        conditional_probability.append(x_conditional_probability)

    return conditional_probability, list_x_name


def get_index_from_value(feature_name, list_features):
    return np.nonzero(list_features == feature_name)[0][0]


def train_naive_bayes(train_data):
    prior_probability = compute_prior_probability(train_data)

    conditional_probability, list_x_name = compute_conditional_probability(
        train_data)

    return prior_probability, conditional_probability, list_x_name


def prediction_play_tennis(sample, list_x_name, prior_probability, conditional_probability):
    p_setosa = 1
    p_versicolor = 1
    p_virginica = 1

    for i in range(len(sample)):
        sample_i = get_index_from_value(sample[i], list_x_name[i])
        p_setosa *= conditional_probability[i][0, sample_i]
        p_versicolor *= conditional_probability[i][1, sample_i]
        p_virginica *= conditional_probability[i][2, sample_i]

    p_setosa *= prior_probability[0]
    p_versicolor *= prior_probability[1]
    p_virginica *= prior_probability[2]

    p_array = np.array([p_setosa, p_versicolor, p_virginica])
    pred_index = np.argmax(p_array)

    return pred_index


sample = [5.0, 2.0, 3.5, 1.0]
data = create_train_data()
y_unique = np.unique(data[:, -1])
prior_probability, conditional_probability, list_x_name = train_naive_bayes(
    data)
pred_index = prediction_play_tennis(
    sample, list_x_name, prior_probability, conditional_probability)
print(y_unique[pred_index])
