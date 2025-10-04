from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / 'mnist_rf_model.pkl'


def main() -> None:
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
    y = y.astype(np.int8)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)

    accuracy = clf.score(X_test, y_test)
    print(f'Model accuracy: {accuracy:.4f}')

    with MODEL_PATH.open('wb') as f:
        pickle.dump(clf, f)
    print(f'Model trained and saved as {MODEL_PATH.name}')


if __name__ == '__main__':
    main()
