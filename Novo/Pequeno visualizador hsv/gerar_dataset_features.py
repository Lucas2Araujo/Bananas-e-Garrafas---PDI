from pathlib import Path
import cv2
import numpy as np

from caracteristicas import extrair_vetor_caracteristicas

DATASET_DIR = Path(__file__).resolve().parents[2] / "dataset"
CLASSES = [("reciclavel", 0), ("organico", 1)]  # 0 = garrafa, 1 = banana
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def gerar_xy(saida_npz: Path = DATASET_DIR / "features_xy.npz") -> tuple[np.ndarray, np.ndarray]:
    """Extrai x = [v_chain_code, v_LBP] (29 dims) e y para todas as imagens rotuladas."""
    X_list, y_list = [], []

    for sub, label in CLASSES:
        img_dir = DATASET_DIR / sub
        mask_dir = DATASET_DIR / "masks" / sub
        for img_path in sorted(img_dir.iterdir()):
            if img_path.suffix.lower() not in IMG_EXTS:
                continue
            gt_path = mask_dir / f"{img_path.stem}.png"
            if not gt_path.exists():
                continue

            img = cv2.imread(str(img_path))
            mask = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
            if img is None or mask is None:
                continue

            X_list.append(extrair_vetor_caracteristicas(img, mask))
            y_list.append(label)

    X = np.vstack(X_list).astype(np.float32) if X_list else np.empty((0, 29), dtype=np.float32)
    y = np.array(y_list, dtype=np.int64)
    np.savez(saida_npz, X=X, y=y)
    print(f"Salvo {saida_npz}: X={X.shape}, y={y.shape} (garrafa=0: {(y == 0).sum()}, banana=1: {(y == 1).sum()})")
    return X, y


if __name__ == "__main__":
    gerar_xy()
