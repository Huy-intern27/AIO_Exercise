import pandas as pd # type: ignore
import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore
import seaborn as sns # type: ignore

data = pd.read_csv('data/advertising.csv')

def correlation(feature_1, feature_2):
    feature_1 = feature_1.to_numpy()
    feature_2 = feature_2.to_numpy()
    size = len(feature_1)

    sum_xy = feature_1 @ feature_2
    sum_x = np.sum(feature_1)
    sum_y = np.sum(feature_2)
    square_x = np.sum(feature_1 ** 2)
    square_y = np.sum(feature_2 ** 2)

    numerator = size * sum_xy - sum_x * sum_y
    denominator = np.sqrt(size * square_x - np.power(sum_x, 2)) * np.sqrt(size * square_y - np.power(sum_y, 2))
    return numerator / denominator

feature_1 = data['TV']
feature_2 = data['Radio']
corr_feature_1_2 = correlation(feature_1, feature_2)
print(round(corr_feature_1_2, 2))

features = ['TV', 'Radio', 'Newspaper']
for feature_1 in features:
    for feature_2 in features:
        correlation_value = correlation(data[feature_1], data[feature_2])
        print(f'{feature_1} and {feature_2}: {correlation_value}')

feature_1 = data['Radio']
feature_2 = data['Newspaper']

result = np.corrcoef(feature_1, feature_2)
print(result)

print(data.corr())

plt.figure(figsize=(10,8))
data_corr = data.corr()
sns.heatmap(data_corr, annot=True, fmt='.2f', linewidths=.5)
plt.show()