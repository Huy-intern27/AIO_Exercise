from sklearn.tree import DecisionTreeRegressor  # type: ignore
from sklearn.model_selection import train_test_split  # type: ignore
from sklearn.metrics import mean_squared_error  # type: ignore
from sklearn.datasets import fetch_openml  # type: ignore
import matplotlib.pyplot as plt  # type: ignore

machine_cpu = fetch_openml(name='machine_cpu')
machine_data = machine_cpu.data
machine_labels = machine_cpu.target
X_train, X_test, y_train, y_test = train_test_split(machine_data,
                                                    machine_labels,
                                                    test_size=0.2,
                                                    random_state=42)

tree_rg = DecisionTreeRegressor()
tree_rg.fit(X_train, y_train)

y_pred = tree_rg.predict(X_test)
print(mean_squared_error(y_pred, y_test))

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue', label='Predicted vs Actual')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)],
         color='red', lw=2, label='Perfect Prediction Line')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Decision Tree Regressor: Predicted vs Actual')
plt.legend()
plt.show()
