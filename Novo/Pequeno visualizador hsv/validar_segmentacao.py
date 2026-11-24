from pathlib import Path
import cv2
import numpy as np

from segmentacao import aplicar_segmentacao

DATASET_DIR = Path(__file__).resolve().parents[2] / "dataset"
SUBDIRS = ("organico", "reciclavel")
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def calcular_iou_dice(pred: np.ndarray, gt: np.ndarray) -> tuple[float, float]:
    """Calcula IoU e Dice entre duas máscaras binárias (0/255)."""
    p = pred > 0
    g = gt > 0
    intersecao = np.logical_and(p, g).sum()
    uniao = np.logical_or(p, g).sum()
    soma = p.sum() + g.sum()

    iou = float(intersecao / uniao) if uniao > 0 else 1.0
    dice = float(2 * intersecao / soma) if soma > 0 else 1.0
    return iou, dice


def validar(canal: str = "GRAY", inverter_limiar: bool = True) -> None:
    resultados = {"mask_otsu": [], "mask_adaptativa": [], "mask_watershed": []}

    for sub in SUBDIRS:
        img_dir = DATASET_DIR / sub
        mask_dir = DATASET_DIR / "masks" / sub
        for img_path in sorted(img_dir.iterdir()):
            if img_path.suffix.lower() not in IMG_EXTS:
                continue
            gt_path = mask_dir / f"{img_path.stem}.png"
            if not gt_path.exists():
                continue

            img = cv2.imread(str(img_path))
            gt = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
            seg = aplicar_segmentacao(img, canal=canal, inverter_limiar=inverter_limiar)

            for metodo in resultados:
                pred = seg.get(metodo)
                if pred is not None:
                    resultados[metodo].append(calcular_iou_dice(pred, gt))

    print(f"Validação ({len(resultados['mask_otsu'])} imagens | canal={canal}):")
    for metodo, pares in resultados.items():
        ious = [p[0] for p in pares]
        dices = [p[1] for p in pares]
        print(f"  {metodo:10s} -> IoU médio: {np.mean(ious):.4f} | Dice médio: {np.mean(dices):.4f}")


if __name__ == "__main__":
    validar()
