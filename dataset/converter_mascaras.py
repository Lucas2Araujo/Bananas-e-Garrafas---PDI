import json
from pathlib import Path

import cv2
import numpy as np

DATASET_DIR = Path(__file__).resolve().parent
SUBDIRS = ("organico", "reciclavel")
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CLASS_MAP = {
    "Banana": 1,
    "Garrafa": 2,
}


def limpar_jsons_orfaos(dataset_dir: Path = DATASET_DIR) -> int:
    """Remove arquivos .json que não possuem imagem correspondente."""
    removidos = 0
    for sub in SUBDIRS:
        pasta = dataset_dir / sub
        if not pasta.exists():
            continue
        imgs = {p.stem for p in pasta.iterdir() if p.suffix.lower() in IMG_EXTS}
        for json_path in pasta.glob("*.json"):
            if json_path.stem not in imgs:
                json_path.unlink()
                removidos += 1
                print(f"Removido órfão: {sub}/{json_path.name}")
    return removidos


def labelme_para_mascara(json_path: Path, binario: bool = True) -> np.ndarray:
    """Converte um arquivo .json do Labelme em máscara (0/255 se binario=True, ou IDs de classe)."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    h, w = data["imageHeight"], data["imageWidth"]
    mascara = np.zeros((h, w), dtype=np.uint8)

    for shape in data.get("shapes", []):
        label = shape.get("label")
        if label not in CLASS_MAP:
            continue
        pts = np.array(shape["points"], dtype=np.int32)
        valor = 255 if binario else CLASS_MAP[label]
        cv2.fillPoly(mascara, [pts], color=valor)

    return mascara


def converter_todas_mascaras(dataset_dir: Path = DATASET_DIR, binario: bool = True) -> int:
    """Gera máscaras PNG em dataset/masks/<subpasta>/<nome>.png a partir dos .json."""
    total = 0
    for sub in SUBDIRS:
        pasta = dataset_dir / sub
        if not pasta.exists():
            continue
        saida_dir = dataset_dir / "masks" / sub
        saida_dir.mkdir(parents=True, exist_ok=True)

        for json_path in sorted(pasta.glob("*.json")):
            mascara = labelme_para_mascara(json_path, binario=binario)
            out_path = saida_dir / f"{json_path.stem}.png"
            cv2.imwrite(str(out_path), mascara)
            total += 1
    return total


if __name__ == "__main__":
    removidos = limpar_jsons_orfaos()
    convertidas = converter_todas_mascaras(binario=True)
    print(f"Limpeza concluída: {removidos} JSON(s) órfão(s) removido(s).")
    print(f"Conversão concluída: {convertidas} máscara(s) salva(s) em {DATASET_DIR / 'masks'}.")
