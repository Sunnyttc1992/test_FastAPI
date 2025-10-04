from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle


X,y = fetch_openml('mnist_784', version=1, return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

print(f"Model accuracy: {clf.score(X_test, y_test)}")

with open('mnist_rf_model.pkl', 'wb') as f:
    pickle.dump(clf, f)
print("Model trained and saved as mnist_rf_model.pkl")


