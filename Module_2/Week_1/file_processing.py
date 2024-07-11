import pandas as pd  # type: ignore
import numpy as np  # type: ignore

df = pd.read_csv('Module_2/Week_1/data/advertising.csv')
data = df.to_numpy()
max_sale = np.max(data[:, 3])
index = np.nonzero(data == max_sale)[0][0]
print(max_sale, index)

mean_tv = np.mean(data[:, 0])
print(mean_tv)

count_sale = np.sum(data[:, 3] >= 20)
print(count_sale)

radio_condition = data[data[:, 3] >= 15]
mean_radio = np.mean(radio_condition[:, 1])
print(mean_radio)

news_condition = data[data[:, 2] > np.mean(data[:, 2])]
sum_sale = np.sum(news_condition[:, 3])
print(sum_sale)

mean_sale = np.mean(data[:, 3])
boxes = data[7:10, 3]
scores = np.where(boxes > mean_sale, 'Good', np.where(
    boxes == mean_sale, 'Average', 'Bad'))
print(scores)

sales = data[:, 3]
closest_mean_sale = sales[np.abs(sales - mean_sale).argmin()]
boxes2 = data[7:10, 3]
scores2 = np.where(boxes2 > closest_mean_sale, 'Good', np.where(
    boxes2 == closest_mean_sale, 'Average', 'Bad'))
print(scores2)
