import matplotlib.pyplot as plt  # type: ignore
import seaborn as sns  # type: ignore
from sklearn import datasets  # type: ignore
from sklearn.tree import DecisionTreeClassifier  # type: ignore
from sklearn.model_selection import train_test_split  # type: ignore
from sklearn.metrics import accuracy_score, confusion_matrix  # type: ignore

iris_X, iris_y = datasets.load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(iris_X,
                                                    iris_y,
                                                    test_size=0.2,
                                                    random_state=42)

dt_classifier = DecisionTreeClassifier()
dt_classifier.fit(X_train, y_train)

y_pred = dt_classifier.predict(X_test)
accuracy = accuracy_score(y_pred, y_test)
print("Accuracy:", accuracy)

# Compute confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Visualize the confusion matrix using a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=datasets.load_iris(
).target_names, yticklabels=datasets.load_iris().target_names)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
