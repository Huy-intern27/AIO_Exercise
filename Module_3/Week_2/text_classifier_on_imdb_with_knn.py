import datasets  # type: ignore
import numpy as np  # type: ignore
from datasets import load_dataset  # type: ignore
from sklearn.neighbors import KNeighborsClassifier  # type: ignore
from sklearn.preprocessing import StandardScaler  # type: ignore
from sklearn.feature_extraction.text import CountVectorizer  # type: ignore
from sklearn.metrics import accuracy_score  # type: ignore

imdb = load_dataset('imdb')
imdb_train, imdb_test = imdb['train'], imdb['test']

vectorizer = CountVectorizer(max_features=1000)
X_train = vectorizer.fit_transform(imdb_train['text']).toarray()
X_test = vectorizer.transform(imdb_test['text']).toarray()
y_train = np.array(imdb_train['label'])
y_test = np.array(imdb_test['label'])

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

knn_classifier = KNeighborsClassifier()
knn_classifier.fit(X_train, y_train)

y_pred = knn_classifier.predict(X_test)
print(accuracy_score(y_test, y_pred))
