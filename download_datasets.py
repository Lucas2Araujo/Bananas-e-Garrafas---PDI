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


def compute_dhash(img: np.ndarray, hash_size: int = 8) -> int:
    """Calcula o dHash (difference hash) de uma imagem para detectar duplicatas visuais."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
    diff = resized[:, 1:] > resized[:, :-1]
    return sum(1 << i for i, v in enumerate(diff.flatten()) if v)


def hamming_distance(h1: int, h2: int) -> int:
    return (h1 ^ h2).bit_count()


def load_existing_hashes(base_dir: Path, staging_dir: Path) -> list[int]:
    """Carrega hashes das imagens já existentes no dataset e das rejeitadas anteriormente."""
    hashes: list[int] = []
    for cls_name in ("organico", "reciclavel"):
        cls_dir = base_dir / cls_name
        if cls_dir.exists():
            for p in cls_dir.iterdir():
                if p.is_file() and p.suffix.lower() in VALID_EXTS:
                    img = cv2.imread(str(p))
                    if img is not None:
                        hashes.append(compute_dhash(img))

    rejected_file = staging_dir / "rejected_hashes.txt"
    if rejected_file.exists():
        for line in rejected_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                try:
                    hashes.append(int(line))
                except ValueError:
                    pass
    return hashes


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


def clean_and_deduplicate_folder(
    folder: Path,
    existing_hashes: list[int],
    min_border_ratio: float = 0.75,
    enforce_black_bg: bool = True,
    max_hamming: int = 5,
) -> tuple[int, int, int]:
    """Remove imagens corrompidas, pequenas, sem fundo escuro ou duplicadas."""
    if not folder.exists():
        return 0, 0, 0

    removed_quality = 0
    removed_dup = 0
    kept = 0

    for p in sorted(folder.iterdir()):
        if not p.is_file() or p.suffix.lower() not in VALID_EXTS:
            continue
        try:
            img = cv2.imread(str(p))
            if img is None or img.shape[0] < 100 or img.shape[1] < 100:
                p.unlink()
                removed_quality += 1
                continue

            ratio = get_dark_border_ratio(img)
            if ratio < min_border_ratio:
                p.unlink()
                removed_quality += 1
                continue

            h = compute_dhash(img)
            if any(hamming_distance(h, eh) <= max_hamming for eh in existing_hashes):
                p.unlink()
                removed_dup += 1
                continue

            existing_hashes.append(h)
            if enforce_black_bg:
                enforce_pure_black_background(p)
            kept += 1
        except Exception:
            p.unlink(missing_ok=True)
            removed_quality += 1

    print(
        f"[*] {folder.name}: {kept} novas prontas para revisão | "
        f"{removed_dup} duplicadas removidas | {removed_quality} fora do padrão."
    )
    return kept, removed_dup, removed_quality


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


def download_batch(
    target: str = "ambas",
    custom_query: str = "",
    samples_per_query: int = 15,
    progress_cb=None,
) -> tuple[int, int]:
    """Baixa um lote de imagens para dataset_staging/ com deduplicação e filtro de fundo escuro."""
    base_dir = Path("dataset")
    staging_dir = Path("dataset_staging")
    org_staging = staging_dir / "organico"
    rec_staging = staging_dir / "reciclavel"

    org_staging.mkdir(parents=True, exist_ok=True)
    rec_staging.mkdir(parents=True, exist_ok=True)

    if progress_cb:
        progress_cb("Indexando hashes existentes...")
    existing_hashes = load_existing_hashes(base_dir, staging_dir)

    wm_filter = "-shutterstock -istock -alamy -depositphotos -dreamstime -gettyimages -pngtree -freepik"

    queries_organico = (
        [f"{custom_query.strip()} {wm_filter}"]
        if custom_query.strip()
        else [
            f"single yellow banana on black background {wm_filter}",
            f"single green banana on black background {wm_filter}",
            f"single ripe spotted banana black background {wm_filter}",
            f"single peeled banana on black background {wm_filter}",
            f"single banana peel on black background {wm_filter}",
            f"banana tilted angled black background studio {wm_filter}",
            f"banana close up cropped black background {wm_filter}",
            f"single overripe brown banana black background {wm_filter}",
        ]
    )

    queries_reciclavel = (
        [f"{custom_query.strip()} {wm_filter}"]
        if custom_query.strip()
        else [
            f"single clear plastic water bottle on black background {wm_filter}",
            f"single green plastic bottle on black background {wm_filter}",
            f"single blue plastic bottle dark background {wm_filter}",
            f"single crushed plastic bottle isolated black background {wm_filter}",
            f"plastic bottle lying down black background {wm_filter}",
            f"empty pet bottle angled tilted black background {wm_filter}",
            f"plastic bottle with label on black background {wm_filter}",
            f"plastic bottle without cap black background {wm_filter}",
        ]
    )

    if target in ("ambas", "organico"):
        for idx, q in enumerate(queries_organico):
            if progress_cb:
                progress_cb(f"Baixando Bananas ({idx+1}/{len(queries_organico)})...")
            crawler = BingImageCrawler(storage={"root_dir": str(org_staging)}, log_level="ERROR")
            crawler.crawl(keyword=q, max_num=samples_per_query, file_idx_offset=(idx + 1) * 300)

    if target in ("ambas", "reciclavel"):
        for idx, q in enumerate(queries_reciclavel):
            if progress_cb:
                progress_cb(f"Baixando Garrafas ({idx+1}/{len(queries_reciclavel)})...")
            crawler = BingImageCrawler(storage={"root_dir": str(rec_staging)}, log_level="ERROR")
            crawler.crawl(keyword=q, max_num=samples_per_query, file_idx_offset=(idx + 1) * 300)

    if progress_cb:
        progress_cb("Filtrando fundo escuro e removendo duplicatas...")

    org_kept = 0
    rec_kept = 0
    if target in ("ambas", "organico"):
        org_kept, _, _ = clean_and_deduplicate_folder(org_staging, existing_hashes)
    if target in ("ambas", "reciclavel"):
        rec_kept, _, _ = clean_and_deduplicate_folder(rec_staging, existing_hashes)

    return org_kept, rec_kept


def populate_staging_dataset(samples_per_query: int = 25, download: bool = True):
    if not download:
        return
    print("=" * 65)
    print("DOWNLOAD PARA REVISÃO (STAGING) - FUNDO PRETO & DEDUPLICAÇÃO")
    print("=" * 65)
    org_kept, rec_kept = download_batch(
        target="ambas",
        samples_per_query=samples_per_query,
        progress_cb=lambda msg: print(f"[*] {msg}"),
    )
    print("\n" + "=" * 65)
    print(f"Prontas para revisão em dataset_staging/organico:   {org_kept} imagens")
    print(f"Prontas para revisão em dataset_staging/reciclavel: {rec_kept} imagens")
    print("Execute '.venv/bin/python tinder_rotulador.py' para aprovar e rotular!")
    print("=" * 65)


if __name__ == "__main__":
    populate_staging_dataset(samples_per_query=25, download=True)
