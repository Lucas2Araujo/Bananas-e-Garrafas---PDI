from pathlib import Path
from icrawler.builtin import BingImageCrawler

def populate_initial_dataset(samples_per_query=35):
    base_dir = Path("dataset")
    org_dir = base_dir / "organico"
    rec_dir = base_dir / "reciclavel"
    
    org_dir.mkdir(parents=True, exist_ok=True)
    rec_dir.mkdir(parents=True, exist_ok=True)

    # Buscas focadas em fundo escuro/isolado e variações reais para ajudar o Otsu inicial
    queries_organico = [
        "single yellow banana on black background",
        "ripe banana fruit isolated",
        "banana peel organic waste"
    ]
    
    queries_reciclavel = [
        "plastic water bottle on dark background",
        "clear pet water bottle isolated",
        "crushed plastic water bottle"
    ]

    print("Baixando amostras de Bananas (Orgânico)...")
    for idx, q in enumerate(queries_organico):
        crawler = BingImageCrawler(storage={"root_dir": str(org_dir)})
        crawler.crawl(keyword=q, max_num=samples_per_query, file_idx_offset=idx * 100)

    print("Baixando amostras de Garrafas D'água (Reciclável)...")
    for idx, q in enumerate(queries_reciclavel):
        crawler = BingImageCrawler(storage={"root_dir": str(rec_dir)})
        crawler.crawl(keyword=q, max_num=samples_per_query, file_idx_offset=idx * 100)

    print("\nConcluído! Agora basta rodar 'python train_model.py' ou apertar 't' no painel!")

if __name__ == "__main__":
    populate_initial_dataset(samples_per_query=30)