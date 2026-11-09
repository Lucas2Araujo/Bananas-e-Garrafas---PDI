import csv
import cv2
import joblib
import numpy as np
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix

import config
from pipeline import HybridPDIPipeline

FEATURE_NAMES = [
    "hu_1", "hu_2", "hu_3", "hu_4", "hu_5", "hu_6", "hu_7",
    "circularity", "aspect_ratio", "solidity",
    "glcm_contrast", "glcm_dissimilarity", "glcm_homogeneity", "glcm_energy",
    "hsv_mean_h", "hsv_mean_s", "hsv_mean_v",
    "hsv_std_h", "hsv_std_s", "hsv_std_v",
    "label"
]

def load_dataset_features(pipeline: HybridPDIPipeline):
    """Varre as pastas de dataset, extrai ROIs e features de cada imagem."""
    classes = [
        (config.DATASET_ORGANICO, 0),
        (config.DATASET_RECICLAVEL, 1)
    ]
    
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    x_data, y_data = [], []

    for folder, label in classes:
        image_paths = [p for p in folder.iterdir() if p.suffix.lower() in valid_exts]
        print(f"[*] Processando {len(image_paths)} imagens da classe {config.CLASS_NAMES[label]}...")
        
        for p in image_paths:
            img = cv2.imread(str(p))
            if img is None:
                continue
            try:
                roi, _, _ = pipeline.extract_roi(img, force_infer=True)
                if roi is None or roi.size == 0:
                    continue
                
                # Extração direta das 20 features (invariantes a rotação por formulação analítica)
                features, _ = pipeline.segment_and_extract(roi)
                if features is not None:
                    x_data.append(features)
                    y_data.append(label)

            except Exception as e:
                print(f"[!] Erro ao processar {p.name}: {e}")

    return np.array(x_data, dtype=np.float32), np.array(y_data, dtype=np.int32)

def save_features_csv(x_data, y_data, output_path: Path):
    """Exporta matriz de características e rótulos para auditoria acadêmica em CSV."""
    with open(output_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(FEATURE_NAMES)
        for feats, lbl in zip(x_data, y_data):
            writer.writerow(list(feats) + [int(lbl)])
    print(f"[+] Features salvas com sucesso em: {output_path}")

def build_classifier():
    """Constrói o Pipeline scikit-learn com normalização StandardScaler e SVM RBF."""
    base_svc = SVC(kernel="rbf", C=1.0, random_state=42)
    clf = CalibratedClassifierCV(estimator=base_svc, ensemble=False)
    
    return Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", clf)
    ])

def train(log_fn=print):
    """Executa o pipeline completo de treinamento, avaliação e exportação do modelo."""
    log_fn("=" * 60)
    log_fn("INICIANDO TREINAMENTO DO MODELO DE CLASSIFICAÇÃO PDI (SVM)")
    log_fn("=" * 60)

    pdi_pipe = HybridPDIPipeline(use_yolo=True)
    X, y = load_dataset_features(pdi_pipe)

    if len(X) == 0:
        log_fn("[!] Nenhuma amostra válida encontrada no dataset. Adicione imagens antes de treinar.")
        return None

    # Salva features para auditoria acadêmica
    save_features_csv(X, y, config.FEATURES_CSV)

    unique, counts = np.unique(y, return_counts=True)
    class_counts = dict(zip(unique, counts))
    log_fn(f"[*] Distribuição de amostras: {class_counts}")

    if len(unique) < 2:
        log_fn("[!] Necessário ao menos uma amostra de cada classe (Organico e Reciclavel) para treinar.")
        return None

    min_class_samples = min(counts)
    clf_pipeline = build_classifier()

    # Validação Cruzada se houver dados suficientes
    if min_class_samples >= 2:
        n_splits = min(5, min_class_samples)
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

        # 1. Comparativo rápido de baselines (justificativa experimental)
        knn_pipe = Pipeline([("scaler", StandardScaler()), ("clf", KNeighborsClassifier(n_neighbors=min(5, min_class_samples)))])
        rf_clf = RandomForestClassifier(n_estimators=50, random_state=42)

        svm_scores = cross_val_score(clf_pipeline, X, y, cv=cv, scoring="accuracy")
        knn_scores = cross_val_score(knn_pipe, X, y, cv=cv, scoring="accuracy")
        rf_scores = cross_val_score(rf_clf, X, y, cv=cv, scoring="accuracy")

        log_fn(f"\n[+] Comparativo de Baselines (StratifiedKFold - {n_splits} folds):")
        log_fn(f"    • SVM (RBF + Scaler) : {svm_scores.mean() * 100:.2f}% (± {svm_scores.std() * 100:.2f}%)")
        log_fn(f"    • KNN (k=5 + Scaler) : {knn_scores.mean() * 100:.2f}% (± {knn_scores.std() * 100:.2f}%)")
        log_fn(f"    • Random Forest      : {rf_scores.mean() * 100:.2f}% (± {rf_scores.std() * 100:.2f}%)")

        # 2. Avaliação Out-of-Fold honesta para o modelo principal (SVM)
        oof_preds = cross_val_predict(clf_pipeline, X, y, cv=cv)
        log_fn("\n--- Relatório de Classificação Out-of-Fold (Validação Cruzada) ---")
        log_fn(classification_report(y, oof_preds, target_names=[config.CLASS_NAMES[0], config.CLASS_NAMES[1]], zero_division=0))
        log_fn("--- Matriz de Confusão Out-of-Fold ---")
        log_fn(str(confusion_matrix(y, oof_preds)))

    # Ajusta o modelo final com todos os dados
    clf_pipeline.fit(X, y)

    # Persiste o modelo
    joblib.dump(clf_pipeline, config.MODEL_PATH)
    log_fn(f"\n[✔] Modelo (SVM RBF) salvo em: {config.MODEL_PATH}")
    return clf_pipeline

if __name__ == "__main__":
    train()
