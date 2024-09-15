import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import matplotlib.pyplot as plt  # type: ignore

from sklearn.model_selection import train_test_split  # type: ignore
from sklearn.metrics import mean_absolute_error, mean_squared_error  # type: ignore
from sklearn.preprocessing import StandardScaler, OrdinalEncoder  # type: ignore
from sklearn.tree import DecisionTreeRegressor  # type: ignore
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor  # type: ignore

dataset_path = '/content/Housing.csv'
df = pd.read_csv(dataset_path)
print(df)

catergorical_cols = df.select_dtypes(include='object').columns.to_list()
print(catergorical_cols)

ordinal_encoder = OrdinalEncoder()
encoded_catergorical_cols = ordinal_encoder.fit_transform(
    df[catergorical_cols]
)
encoded_catergorical_df = pd.DataFrame(
    encoded_catergorical_cols,
    columns=catergorical_cols
)
numerical_df = df.drop(columns=catergorical_cols, axis=1)
encoded_df = pd.concat(
    [numerical_df, encoded_catergorical_df],
    axis=1
)
print(encoded_df)

nomalizer = StandardScaler()
dataset_arr = nomalizer.fit_transform(encoded_df)
print(dataset_arr)

X, y = dataset_arr[:, 1:], dataset_arr[:, 0]
random_state = 1
is_shuffle = True
test_size = 0.3
X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=test_size,
    random_state=random_state,
    shuffle=is_shuffle
)

regressor = GradientBoostingRegressor(random_state=random_state)
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_val)
mae = mean_absolute_error(y_val, y_pred)
mse = mean_squared_error(y_val, y_pred)
print(y_pred)
print(mae, mse)
