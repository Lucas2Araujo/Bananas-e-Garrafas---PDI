import pickle
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict

DATASET_DIR = Path(__file__).resolve().parents[2] / "dataset"


def treinar(npz_path: Path = DATASET_DIR / "features_xy.npz", saida_modelo: Path = DATASET_DIR / "modelo_rf.pkl"):
    dados = np.load(npz_path)
    X, y = dados["X"], dados["y"]

    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_pred = cross_val_predict(clf, X, y, cv=cv)

    clf.fit(X, y)
    with open(saida_modelo, "wb") as f:
        pickle.dump(clf, f)

    print(f"K-Fold (5 splits) concluído em {len(y)} amostras. Modelo salvo em {saida_modelo}.")
    return clf, y, y_pred


if __name__ == "__main__":
    treinar()
