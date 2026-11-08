import csv
import cv2
import joblib
import numpy as np
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix

import config
from pipeline import HybridPDIPipeline

FEATURE_NAMES = [
    "hu_1", "hu_2", "hu_3", "hu_4", "hu_5", "hu_6", "hu_7",
    "circularity", "aspect_ratio",
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
                
                # 1. Amostra original
                features, _ = pipeline.segment_and_extract(roi)
                if features is not None:
                    x_data.append(features)
                    y_data.append(label)

                # 2. Data Augmentation: Rotação de 90 graus (invariância vertical/horizontal)
                roi_rot = cv2.rotate(roi, cv2.ROTATE_90_CLOCKWISE)
                features_rot, _ = pipeline.segment_and_extract(roi_rot)
                if features_rot is not None:
                    x_data.append(features_rot)
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

def build_classifier(model_type: str = "SVM"):
    """Constrói o Pipeline scikit-learn com normalização StandardScaler."""
    if model_type.upper() == "KNN":
        clf = KNeighborsClassifier(n_neighbors=5)
    else:
        base_svc = SVC(kernel="rbf", C=1.0, random_state=42)
        clf = CalibratedClassifierCV(estimator=base_svc, ensemble=False)
    
    return Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", clf)
    ])

def train(model_type: str = None, log_fn=print):
    """Executa o pipeline completo de treinamento, avaliação e exportação do modelo."""
    m_type = (model_type or config.MODEL_TYPE).upper()
    log_fn("=" * 60)
    log_fn(f"INICIANDO TREINAMENTO DO MODELO DE CLASSIFICAÇÃO PDI ({m_type})")
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
    clf_pipeline = build_classifier(m_type)

    # Validação Cruzada se houver dados suficientes
    if min_class_samples >= 2:
        n_splits = min(5, min_class_samples)
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        scores = cross_val_score(clf_pipeline, X, y, cv=cv, scoring="accuracy")
        log_fn(f"\n[+] Validação Cruzada StratifiedKFold ({n_splits} folds):")
        log_fn(f"    Acurácia Média: {scores.mean() * 100:.2f}% (± {scores.std() * 100:.2f}%)")

    # Ajusta o modelo final com todos os dados
    clf_pipeline.fit(X, y)
    preds = clf_pipeline.predict(X)

    log_fn("\n--- Relatório de Classificação (Treino Completo) ---")
    log_fn(classification_report(y, preds, target_names=[config.CLASS_NAMES[0], config.CLASS_NAMES[1]], zero_division=0))
    log_fn("--- Matriz de Confusão ---")
    log_fn(str(confusion_matrix(y, preds)))

    # Persiste o modelo
    joblib.dump(clf_pipeline, config.MODEL_PATH)
    log_fn(f"\n[✔] Modelo ({m_type}) salvo em: {config.MODEL_PATH}")
    return clf_pipeline

if __name__ == "__main__":
    train()
