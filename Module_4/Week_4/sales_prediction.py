import pandas as pd # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.preprocessing import StandardScaler  # type: ignore
from sklearn.preprocessing import PolynomialFeatures # type: ignore
from sklearn.linear_model import LinearRegression # type: ignore
from sklearn.metrics import r2_score # type: ignore

df = pd.read_csv('./data/SalesPrediction.csv')
df = pd.get_dummies(df)
df = df.fillna(df.mean())

X = df[['TV', 'Radio', 'Social Media', 'Influencer_Macro', 'Influencer_Mega', 
        'Influencer_Micro', 'Influencer_Nano']]
y = df[['Sales']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)

scaler = StandardScaler()
X_train_processed = scaler.fit_transform(X_train)
X_test_processed = scaler.transform(X_test)

poly_feature = PolynomialFeatures(degree=2)
X_train_poly = poly_feature.fit_transform(X_train_processed)
X_test_poly = poly_feature.transform(X_test_processed)

model = LinearRegression()
model.fit(X_train_poly, y_train)
y_pred = model.predict(X_test_poly)
r2_score(y_test, y_pred)

model2 = LinearRegression()
model2.fit(X_train_processed, y_train)
y_pred2 = model2.predict(X_test_processed)
r2_score(y_test, y_pred2)