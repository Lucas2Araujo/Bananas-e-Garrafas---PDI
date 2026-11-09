import cv2
from pathlib import Path
from icrawler.builtin import BingImageCrawler

def clean_corrupted_images(folder: Path):
    """Remove imagens corrompidas, miniaturas ou menores que 100x100 pixels."""
    valid_exts = {".jpg", ".jpeg", ".png", ".webp"}
    removed = 0
    for p in folder.iterdir():
        if p.suffix.lower() in valid_exts:
            try:
                img = cv2.imread(str(p))
                if img is None or img.shape[0] < 100 or img.shape[1] < 100:
                    p.unlink()
                    removed += 1
            except Exception:
                p.unlink()
                removed += 1
    if removed > 0:
        print(f"[*] Limpeza: {removed} arquivos inválidos removidos de {folder.name}.")

def populate_initial_dataset(samples_per_query=30):
    base_dir = Path("dataset")
    org_dir = base_dir / "organico"
    rec_dir = base_dir / "reciclavel"
    
    org_dir.mkdir(parents=True, exist_ok=True)
    rec_dir.mkdir(parents=True, exist_ok=True)

    # Buscas em estúdio / fundo escuro isolado com múltiplas orientações, cores e deformações
    queries_organico = [
        "single yellow banana on black background",
        "vertical banana fruit black background",
        "hanging banana isolated black background",
        "ripe banana horizontal black background",
        "banana peel on black background",
        "single banana plain dark background studio",
        "brown banana peel on black background",
        "old oxidized banana peel dark background"
    ]
    
    queries_reciclavel = [
        "clear plastic water bottle on black background",
        "horizontal plastic water bottle black background",
        "empty transparent pet bottle black background",
        "crushed plastic bottle isolated black background",
        "vertical water bottle black background studio",
        "plastic water bottle lying flat dark background",
        "green plastic bottle on black background",
        "crushed plastic bottle with label black background"
    ]

    print("=" * 65)
    print("DOWNLOAD DE DATASET EXPANDIDO - ESTÚDIO / FUNDO PRETO ISOLADO")
    print("=" * 65)

    print("\n[1/2] Baixando amostras de Bananas (fundo preto, verticais e horizontais)...")
    for idx, q in enumerate(queries_organico):
        print(f"  -> [{idx+1}/{len(queries_organico)}] Busca: '{q}'...")
        crawler = BingImageCrawler(
            storage={"root_dir": str(org_dir)},
            log_level="ERROR"
        )
        crawler.crawl(keyword=q, max_num=samples_per_query, file_idx_offset=(idx + 1) * 300)

    clean_corrupted_images(org_dir)

    print("\n[2/2] Baixando amostras de Garrafas (fundo preto, verticais e horizontais)...")
    for idx, q in enumerate(queries_reciclavel):
        print(f"  -> [{idx+1}/{len(queries_reciclavel)}] Busca: '{q}'...")
        crawler = BingImageCrawler(
            storage={"root_dir": str(rec_dir)},
            log_level="ERROR"
        )
        crawler.crawl(keyword=q, max_num=samples_per_query, file_idx_offset=(idx + 1) * 300)

    clean_corrupted_images(rec_dir)

    org_count = len(list(org_dir.glob("*.jpg")) + list(org_dir.glob("*.png")))
    rec_count = len(list(rec_dir.glob("*.jpg")) + list(rec_dir.glob("*.png")))

    print("\n" + "=" * 65)
    print("Download e higienização concluídos com sucesso!")
    print(f"Total em Organico (Banana): {org_count} imagens")
    print(f"Total em Reciclavel (Garrafa): {rec_count} imagens")
    print("Pronto para rodar 'python train_model.py' ou apertar 'T' no painel!")
    print("=" * 65)

if __name__ == "__main__":
    populate_initial_dataset(samples_per_query=30)
