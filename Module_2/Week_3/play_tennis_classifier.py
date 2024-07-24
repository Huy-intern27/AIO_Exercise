import numpy as np  # type: ignore
import pandas as pd  # type: ignore


def create_train_data():
    data = pd.read_csv('data/play_tennis.csv')
    data = data.iloc[:10, 1:]

    return np.array(data)


def compute_prior_probability(train_data):
    y_unique = ['No', 'Yes']
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

    sample_spaces_yes = np.sum(train_data[:, -1] == 'Yes')
    sample_spaces_no = np.sum(train_data[:, -1] == 'No')
    for i in range(len(list_x_name)):
        x_conditional_probability = np.zeros((2, len(list_x_name[i])))
        for j in range(len(list_x_name[i])):
            x_conditional_probability[0, j] = np.sum((train_data[:, i] == list_x_name[i][j]) & (
                train_data[:, -1] == 'No')) / sample_spaces_no
            x_conditional_probability[1, j] = np.sum((train_data[:, i] == list_x_name[i][j]) & (
                train_data[:, -1] == 'Yes')) / sample_spaces_yes
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
    p0 = 1
    p1 = 1

    for i in range(len(sample)):
        sample_i = get_index_from_value(sample[i], list_x_name[i])
        p0 *= conditional_probability[i][0, sample_i]
        p1 *= conditional_probability[i][1, sample_i]

    p0 *= prior_probability[0]
    p1 *= prior_probability[1]
    if p0 > p1:
        y_pred = 0
    else:
        y_pred = 1

    return y_pred


sample = ['Sunny', 'Cool', 'High', 'Strong']
data = create_train_data()
prior_probability, conditional_probability, list_x_name = train_naive_bayes(
    data)
pred = prediction_play_tennis(
    sample, list_x_name, prior_probability, conditional_probability)

if (pred):
    print('Ad should go!')
else:
    print('Ad should not go!')
