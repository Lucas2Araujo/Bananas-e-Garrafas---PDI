import csv
import cv2
import numpy as np
from pathlib import Path
from icrawler.builtin import BingImageCrawler

try:
    from rembg import remove as rembg_remove
    from PIL import Image
    import io
    HAS_REMBG = True
except ImportError:
    HAS_REMBG = False


VALID_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def get_dark_border_ratio(img: np.ndarray, threshold: int = 55, border_px: int = 6) -> float:
    """Retorna a proporção de pixels escuros nas bordas da imagem."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    bp = min(border_px, h // 4, w // 4)
    if bp < 1:
        return 0.0

    top = gray[:bp, :].ravel()
    bottom = gray[-bp:, :].ravel()
    left = gray[bp:-bp, :bp].ravel()
    right = gray[bp:-bp, -bp:].ravel()
    borders = np.concatenate([top, bottom, left, right])
    return float(np.mean(borders <= threshold))


def enforce_pure_black_background(img_path: Path) -> bool:
    """Remove o fundo usando rembg (se disponível) e compõe sobre preto absoluto (#000000)."""
    if not HAS_REMBG:
        return False
    try:
        with open(img_path, "rb") as f:
            input_bytes = f.read()
        output_bytes = rembg_remove(input_bytes)
        rgba = Image.open(io.BytesIO(output_bytes)).convert("RGBA")
        black_bg = Image.new("RGBA", rgba.size, (0, 0, 0, 255))
        composited = Image.alpha_composite(black_bg, rgba).convert("RGB")
        composited.save(img_path, quality=95)
        return True
    except Exception:
        return False


def clean_and_standardize_folder(
    folder: Path,
    min_border_ratio: float = 0.80,
    enforce_black_bg: bool = True,
) -> tuple[int, int]:
    """Remove imagens corrompidas, pequenas ou sem fundo escuro nas bordas e padroniza o fundo."""
    if not folder.exists():
        return 0, 0

    removed = 0
    kept = 0
    for p in sorted(folder.iterdir()):
        if not p.is_file() or p.suffix.lower() not in VALID_EXTS:
            continue
        try:
            img = cv2.imread(str(p))
            if img is None or img.shape[0] < 100 or img.shape[1] < 100:
                p.unlink()
                removed += 1
                continue

            ratio = get_dark_border_ratio(img)
            if ratio < min_border_ratio:
                p.unlink()
                removed += 1
                continue

            if enforce_black_bg:
                enforce_pure_black_background(p)
            kept += 1
        except Exception:
            p.unlink(missing_ok=True)
            removed += 1

    print(f"[*] {folder.name}: {removed} removidas, {kept} mantidas.")
    return kept, removed


def generate_catalog(base_dir: Path) -> Path:
    """Gera dataset/catalog.csv com metadados das imagens válidas para ML."""
    catalog_path = base_dir / "catalog.csv"
    rows = []
    for cls_name in ("organico", "reciclavel"):
        cls_dir = base_dir / cls_name
        if not cls_dir.exists():
            continue
        for p in sorted(cls_dir.iterdir()):
            if not p.is_file() or p.suffix.lower() not in VALID_EXTS:
                continue
            img = cv2.imread(str(p))
            if img is None:
                continue
            h, w = img.shape[:2]
            ratio = get_dark_border_ratio(img, threshold=15)
            rows.append([str(p.relative_to(base_dir)), cls_name, w, h, f"{ratio:.3f}"])

    with open(catalog_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["filepath", "class", "width", "height", "pure_black_border_ratio"])
        writer.writerows(rows)

    print(f"[*] Catálogo salvo em {catalog_path} ({len(rows)} registros).")
    return catalog_path


def populate_initial_dataset(samples_per_query: int = 30, download: bool = True):
    base_dir = Path("dataset")
    org_dir = base_dir / "organico"
    rec_dir = base_dir / "reciclavel"

    org_dir.mkdir(parents=True, exist_ok=True)
    rec_dir.mkdir(parents=True, exist_ok=True)

    wm_filter = "-shutterstock -istock -alamy -depositphotos -dreamstime"

    queries_organico = [
        f"single yellow banana on black background {wm_filter}",
        f"single green banana on black background {wm_filter}",
        f"single ripe banana black background studio {wm_filter}",
        f"single banana peel on black background {wm_filter}",
        f"brown banana peel on black background {wm_filter}",
        f"single banana isolated dark background {wm_filter}",
    ]

    queries_reciclavel = [
        f"single clear plastic water bottle on black background {wm_filter}",
        f"single empty pet bottle black background {wm_filter}",
        f"single crushed plastic bottle isolated black background {wm_filter}",
        f"single twisted plastic bottle black background {wm_filter}",
        f"single green plastic bottle on black background {wm_filter}",
        f"single blue plastic bottle dark background {wm_filter}",
    ]

    if download:
        print("=" * 65)
        print("DOWNLOAD E HIGIENIZAÇÃO - FUNDO PRETO ABSOLUTO (#000000)")
        print("=" * 65)

        print("\n[1/2] Baixando amostras de Bananas (objeto único, fundo escuro)...")
        for idx, q in enumerate(queries_organico):
            print(f"  -> [{idx+1}/{len(queries_organico)}] Busca: '{q}'...")
            crawler = BingImageCrawler(storage={"root_dir": str(org_dir)}, log_level="ERROR")
            crawler.crawl(keyword=q, max_num=samples_per_query, file_idx_offset=(idx + 1) * 300)

        print("\n[2/2] Baixando amostras de Garrafas (objeto único, fundo escuro)...")
        for idx, q in enumerate(queries_reciclavel):
            print(f"  -> [{idx+1}/{len(queries_reciclavel)}] Busca: '{q}'...")
            crawler = BingImageCrawler(storage={"root_dir": str(rec_dir)}, log_level="ERROR")
            crawler.crawl(keyword=q, max_num=samples_per_query, file_idx_offset=(idx + 1) * 300)

    print("\n[*] Higienizando e padronizando fundo preto absoluto...")
    if not HAS_REMBG:
        print("[!] Aviso: 'rembg' não instalado. Aplicando apenas filtro de bordas escuras.")
    org_kept, _ = clean_and_standardize_folder(org_dir)
    rec_kept, _ = clean_and_standardize_folder(rec_dir)
    generate_catalog(base_dir)

    print("\n" + "=" * 65)
    print(f"Total em Organico (Banana): {org_kept} imagens")
    print(f"Total em Reciclavel (Garrafa): {rec_kept} imagens")
    print("=" * 65)


if __name__ == "__main__":
    populate_initial_dataset(samples_per_query=30, download=True)
