from pathlib import Path
import cv2
import numpy as np

_YOLO_MODEL = None


def _garantir_impar_minimo(valor, minimo=1):
    """Garante que o valor seja inteiro, ímpar e maior ou igual ao mínimo."""
    v = max(minimo, int(valor))
    if v % 2 == 0:
        v += 1
    return v


def _extrair_canal(img_bgr, canal):
    """Extrai o canal desejado ('GRAY', 'R', 'G', 'B', 'H', 'S' ou 'V') como matriz 2D."""
    if canal in ("R", "G", "B"):
        idx = {"B": 0, "G": 1, "R": 2}[canal]
        return img_bgr[:, :, idx]
    if canal in ("H", "S", "V"):
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        idx = {"H": 0, "S": 1, "V": 2}[canal]
        return hsv[:, :, idx]
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)


def aplicar_segmentacao(
    img_objeto_bgr,
    img_fundo_bgr=None,
    canal="GRAY",
    usar_blur=True,
    blur_ksize=5,
    limiar_absdiff=30,
    bloco_adaptativo=21,
    constante_c=5,
    inverter_limiar=True,
    incluir_grabcut=False,
):
    """
    Aplica três métodos de segmentação na imagem do objeto (e opcionalmente fundo):
      1. Subtração de Fundo (absdiff) - requer img_fundo_bgr
      2. Limiarização de Otsu
      3. Limiarização Adaptativa (Gaussiana)

    Retorna um dicionário com as imagens em escala de cinza (ou None para absdiff se não houver fundo).
    """
    if img_objeto_bgr is None:
        return None

    k = _garantir_impar_minimo(blur_ksize, minimo=1)
    bloco = _garantir_impar_minimo(bloco_adaptativo, minimo=3)
    tipo_thresh = cv2.THRESH_BINARY_INV if inverter_limiar else cv2.THRESH_BINARY

    objeto_gray = _extrair_canal(img_objeto_bgr, canal)
    objeto_blur = cv2.GaussianBlur(objeto_gray, (k, k), 0) if (usar_blur and k > 1) else objeto_gray

    # Método 1: Subtração de Fundo (absdiff)
    mask_subtracao = None
    if img_fundo_bgr is not None:
        fundo_gray = _extrair_canal(img_fundo_bgr, canal)
        if fundo_gray.shape != objeto_gray.shape:
            fundo_gray = cv2.resize(fundo_gray, (objeto_gray.shape[1], objeto_gray.shape[0]))
        fundo_blur = cv2.GaussianBlur(fundo_gray, (k, k), 0) if (usar_blur and k > 1) else fundo_gray
        diferenca = cv2.absdiff(fundo_blur, objeto_blur)
        _, mask_subtracao = cv2.threshold(diferenca, int(limiar_absdiff), 255, cv2.THRESH_BINARY)

    # Método 2: Limiarização de Otsu
    _, mask_otsu = cv2.threshold(objeto_blur, 0, 255, tipo_thresh + cv2.THRESH_OTSU)

    # Método 3: Limiarização Adaptativa
    mask_adaptativa = cv2.adaptiveThreshold(
        objeto_blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        tipo_thresh,
        bloco,
        int(constante_c),
    )

    mask_grabcut = None
    if incluir_grabcut:
        h_img, w_img = img_objeto_bgr.shape[:2]
        gc_res = aplicar_grabcut(img_objeto_bgr, (1, 1, w_img - 2, h_img - 2), iteracoes=3, label="Sem ROI")
        if gc_res is not None:
            mask_grabcut = gc_res["mask_grabcut"]

    return {
        "objeto_gray": objeto_blur,
        "mask_subtracao": mask_subtracao,
        "mask_otsu": mask_otsu,
        "mask_adaptativa": mask_adaptativa,
        "mask_grabcut": mask_grabcut,
    }


def _obter_yolo():
    """Carrega preguiçosamente o modelo YOLO da raiz do repositório."""
    global _YOLO_MODEL
    if _YOLO_MODEL is not None:
        return _YOLO_MODEL

    from ultralytics import YOLO

    raiz = Path(__file__).resolve().parents[2]
    for nome in ("yolo26n.pt", "yolo11n.pt"):
        caminho = raiz / nome
        if caminho.exists():
            _YOLO_MODEL = YOLO(str(caminho))
            return _YOLO_MODEL
    _YOLO_MODEL = YOLO("yolo26n.pt")
    return _YOLO_MODEL


def detectar_roi_yolo(img_bgr, padding_pct=10, espaco_yolo="BGR"):
    """
    Detecta a ROI principal via YOLO (prioriza garrafa=39 e banana=46; fallback para qualquer classe ou centro 70%).
    espaco_yolo: 'BGR' (padrão), 'HSV' ou 'GRAY'.
    Retorna ((x, y, w, h), texto_deteccao).
    """
    h_img, w_img = img_bgr.shape[:2]
    if espaco_yolo == "HSV":
        img_yolo = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    elif espaco_yolo == "GRAY":
        img_yolo = cv2.cvtColor(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    else:
        img_yolo = img_bgr

    modelo = _obter_yolo()
    results = modelo(img_yolo, conf=0.20, verbose=False)
    boxes = results[0].boxes

    melhor = None
    origem = "YOLO"
    if boxes is not None and len(boxes) > 0:
        alvo = [b for b in boxes if int(b.cls[0]) in (39, 46)]
        melhor = max(alvo or list(boxes), key=lambda b: float(b.conf[0]))

    if melhor is not None:
        x1, y1, x2, y2 = map(int, melhor.xyxy[0])
        cls_id = int(melhor.cls[0])
        nome_cls = results[0].names.get(cls_id, str(cls_id))
        conf = float(melhor.conf[0])
        texto = f"{origem}: {nome_cls} ({conf:.0%})"
    else:
        mx, my = int(w_img * 0.15), int(h_img * 0.15)
        x1, y1, x2, y2 = mx, my, w_img - mx, h_img - my
        texto = "FALLBACK_CENTER (70%)"

    pad_frac = max(0.0, float(padding_pct)) / 100.0
    pad_x = int((x2 - x1) * pad_frac)
    pad_y = int((y2 - y1) * pad_frac)
    x1 = max(1, x1 - pad_x)
    y1 = max(1, y1 - pad_y)
    x2 = min(w_img - 1, x2 + pad_x)
    y2 = min(h_img - 1, y2 + pad_y)

    w = max(2, x2 - x1)
    h = max(2, y2 - y1)
    return (x1, y1, w, h), texto


def aplicar_grabcut(img_bgr, rect_xywh, iteracoes=5, label="ROI"):
    """Executa GrabCut a partir de rect_xywh=(x, y, w, h) e retorna visualizações RGB/máscara."""
    if img_bgr is None:
        return None

    h_img, w_img = img_bgr.shape[:2]
    x, y, w, h = map(int, rect_xywh)
    # Garante margem mínima de 1px para que haja amostras de fundo fora do retângulo
    x = max(1, min(x, w_img - 3))
    y = max(1, min(y, h_img - 3))
    w = max(2, min(w, w_img - x - 1))
    h = max(2, min(h, h_img - y - 1))

    # ponytail: reduz escala > 512px para interatividade em tempo real no Tkinter
    escala = min(1.0, 512.0 / max(h_img, w_img))
    if escala < 1.0:
        pw, ph = max(8, int(w_img * escala)), max(8, int(h_img * escala))
        img_proc = cv2.resize(img_bgr, (pw, ph), interpolation=cv2.INTER_AREA)
        rx = max(1, min(int(x * escala), pw - 3))
        ry = max(1, min(int(y * escala), ph - 3))
        rw = max(2, min(int(w * escala), pw - rx - 1))
        rh = max(2, min(int(h * escala), ph - ry - 1))
        rect_proc = (rx, ry, rw, rh)
    else:
        img_proc = img_bgr
        rect_proc = (x, y, w, h)

    mask_proc = np.zeros(img_proc.shape[:2], np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    cv2.grabCut(img_proc, mask_proc, rect_proc, bgd_model, fgd_model, max(1, int(iteracoes)), cv2.GC_INIT_WITH_RECT)

    mask_bin_proc = np.where((mask_proc == cv2.GC_FGD) | (mask_proc == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    if escala < 1.0:
        mask_bin = cv2.resize(mask_bin_proc, (w_img, h_img), interpolation=cv2.INTER_NEAREST)
    else:
        mask_bin = mask_bin_proc

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    bbox_vis = img_rgb.copy()
    cv2.rectangle(bbox_vis, (x, y), (x + w, y + h), (40, 220, 80), 2)
    cv2.putText(bbox_vis, label, (x, max(20, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (40, 220, 80), 2, cv2.LINE_AA)

    roi_crop = img_rgb[y : y + h, x : x + w].copy()
    objeto_isolado = cv2.bitwise_and(img_rgb, img_rgb, mask=mask_bin)

    return {
        "bbox_vis": bbox_vis,
        "roi_crop": roi_crop,
        "mask_grabcut": mask_bin,
        "objeto_isolado": objeto_isolado,
    }


def _aplicar_prefiltro(img, tipo="Gaussiano", ksize=5):
    """Aplica pré-filtro ('Nenhum', 'Gaussiano', 'Mediana' ou 'Bilateral') na imagem."""
    k = _garantir_impar_minimo(ksize, minimo=1)
    if tipo == "Nenhum" or k <= 1:
        return img
    if tipo == "Mediana":
        return cv2.medianBlur(img, k)
    if tipo == "Bilateral":
        return cv2.bilateralFilter(img, k, 75, 75)
    return cv2.GaussianBlur(img, (k, k), 0)


def segmentar_roi(
    img_bgr,
    rect_xywh,
    metodo="GrabCut",
    iteracoes=5,
    usar_gauss=False,
    gauss_ksize=5,
    usar_mediana=False,
    usar_bilateral=False,
    label="ROI",
):
    """
    Aplica GrabCut, Otsu ou Limiarização Adaptativa apenas dentro da ROI definida por rect_xywh.
    Permite combinar múltiplos pré-filtros (Gaussiano, Mediana, Bilateral) somente na ROI.
    """
    if img_bgr is None:
        return None

    h_img, w_img = img_bgr.shape[:2]
    x, y, w, h = map(int, rect_xywh)
    x = max(1, min(x, w_img - 3))
    y = max(1, min(y, h_img - 3))
    w = max(2, min(w, w_img - x - 1))
    h = max(2, min(h, h_img - y - 1))

    img_proc = img_bgr.copy()
    roi_sub = img_proc[y : y + h, x : x + w]
    if usar_gauss:
        roi_sub = _aplicar_prefiltro(roi_sub, tipo="Gaussiano", ksize=gauss_ksize)
    if usar_mediana:
        roi_sub = _aplicar_prefiltro(roi_sub, tipo="Mediana", ksize=5)
    if usar_bilateral:
        roi_sub = _aplicar_prefiltro(roi_sub, tipo="Bilateral", ksize=9)
    img_proc[y : y + h, x : x + w] = roi_sub

    if metodo == "GrabCut":
        res = aplicar_grabcut(img_proc, (x, y, w, h), iteracoes=iteracoes, label=label)
        return {
            "bbox_vis": res["bbox_vis"],
            "roi_crop": res["roi_crop"],
            "mask_resultado": res["mask_grabcut"],
            "objeto_isolado": res["objeto_isolado"],
        }

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    bbox_vis = img_rgb.copy()
    cv2.rectangle(bbox_vis, (x, y), (x + w, y + h), (40, 220, 80), 2)
    cv2.putText(bbox_vis, label, (x, max(20, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (40, 220, 80), 2, cv2.LINE_AA)

    roi_bgr = img_proc[y : y + h, x : x + w]
    roi_rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
    roi_gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)

    if metodo == "Otsu":
        _, mask_roi = cv2.threshold(roi_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    else:
        mask_roi = cv2.adaptiveThreshold(
            roi_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 5
        )

    mask_full = np.zeros((h_img, w_img), dtype=np.uint8)
    mask_full[y : y + h, x : x + w] = mask_roi
    objeto_isolado = cv2.bitwise_and(img_rgb, img_rgb, mask=mask_full)

    return {
        "bbox_vis": bbox_vis,
        "roi_crop": roi_rgb,
        "mask_resultado": mask_full,
        "objeto_isolado": objeto_isolado,
    }


if __name__ == "__main__":
    from processamento import gerar_histograma_canal, processar_imagem

    img_teste = Path(__file__).resolve().parents[2] / "testes" / "banana2.png"
    res = processar_imagem(str(img_teste))
    assert res is not None and "bgr" in res
    assert gerar_histograma_canal(res["h"], "Canal H", (200, 20, 200), 180).shape == (480, 800, 3)

    img_bgr = res["bgr"]
    for canal in ("GRAY", "R", "G", "B", "H", "S", "V"):
        seg = aplicar_segmentacao(img_bgr, canal=canal)
        assert seg["mask_otsu"].shape == img_bgr.shape[:2]
        assert seg["mask_adaptativa"].shape == img_bgr.shape[:2]

    rect, label = detectar_roi_yolo(img_bgr, padding_pct=10)
    gc = aplicar_grabcut(img_bgr, rect, iteracoes=3, label=label)
    assert gc["mask_grabcut"].shape == img_bgr.shape[:2] and np.any(gc["mask_grabcut"] > 0)
    assert gc["roi_crop"].size > 0
    print(f"Self-check OK | {label} | ROI={rect}")
