import numpy as np  # type: ignore
import pandas as pd  # type: ignore

def gaussian_function(mean, variance, variable):
    return (1 / np.sqrt(2 * np.pi * variance)) * np.exp(-(((variable - mean) ** 2) / (2 * variance)))

#read data
def create_train_data():
    data = pd.read_csv('data/iris.data.txt', header=None)
    data = data.to_numpy()
    return data

#compute label_probability
def compute_prior_probability(train_data):
    y_unique = np.unique(train_data[:, -1])
    prior_probability = np.zeros(len(y_unique))
    for i in range(len(y_unique)):
        prior_probability[i] = np.sum(
            train_data[:, -1] == y_unique[i]) / len(train_data)

    return prior_probability


def compute_conditional_probability(train_data, sample):
    #compute mean, variance on each label on dataset
    y_unique = np.unique(train_data[:, -1])
    conditional_probability = []
    list_mean_variance = []
    for i in range(len(y_unique)):
        conditional_sample_spaces = train_data[train_data[:, -1] == y_unique[i]]
        element = []
        for j in range(train_data.shape[1] - 1):
            mean = np.mean(conditional_sample_spaces[:, j])
            variance = np.var(conditional_sample_spaces[:, j])
            element.append(np.array([mean, variance]))
        list_mean_variance.append(element)

    #compute contional_probability on each feature on each label
    for i in range(len(list_mean_variance)):
        probability = np.zeros(len(list_mean_variance[i]))
        for j in range(len(list_mean_variance[i])):
            probability[j] = gaussian_function(mean=list_mean_variance[i][j][0], variance=list_mean_variance[i][j][1], variable=sample[j])
        conditional_probability.append(probability)

    return conditional_probability

#general function
def train_gaussian_naive_bayes(train_data, sample):
    prior_probability = compute_prior_probability(train_data)
    conditional_probability = compute_conditional_probability(train_data, sample)

    return prior_probability, conditional_probability

#prediction sample
def prediction_iris(sample, prior_probability, conditional_probability):
    p_setosa = 1
    p_versicolor = 1
    p_virginica = 1
    sum_probability = 0

    for i in range(len(sample)):
        p_setosa *= conditional_probability[0][i]
        p_versicolor *= conditional_probability[1][i]
        p_virginica *= conditional_probability[2][i]

    p_setosa *= prior_probability[0]
    p_versicolor *= prior_probability[1]
    p_virginica *= prior_probability[2]

    sum_probability += p_setosa
    sum_probability += p_versicolor
    sum_probability += p_virginica
    p_setosa /= sum_probability
    p_versicolor /= sum_probability
    p_virginica /= sum_probability

    p_array = np.array([p_setosa, p_versicolor, p_virginica])
    pred_index = np.argmax(p_array)

    return pred_index

sample = [6.3, 3.3, 6.0, 2.5]
data = create_train_data()
y_unique = np.unique(data[:, -1])
prior_probability, conditional_probability = train_gaussian_naive_bayes(data, sample)
pred_index = prediction_iris(sample, prior_probability, conditional_probability)
print(y_unique[pred_index])