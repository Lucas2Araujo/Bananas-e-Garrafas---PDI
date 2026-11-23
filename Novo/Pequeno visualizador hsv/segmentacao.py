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


def aplicar_watershed(img_bgr, mask_otsu):
    """Aplica Watershed usando Distance Transform sobre a máscara de Otsu."""
    if img_bgr is None or mask_otsu is None:
        return None
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(mask_otsu, cv2.MORPH_OPEN, kernel, iterations=2)
    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    max_dist = float(dist_transform.max())
    if max_dist <= 0:
        return np.zeros_like(mask_otsu)

    _, sure_fg = cv2.threshold(dist_transform, 0.4 * max_dist, 255, 0)
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    markers = cv2.watershed(img_bgr.copy(), markers)
    return np.where(markers > 1, 255, 0).astype(np.uint8)


def aplicar_segmentacao(
    img_objeto_bgr,
    canal="GRAY",
    usar_blur=True,
    blur_ksize=5,
    bloco_adaptativo=21,
    constante_c=5,
    inverter_limiar=True,
):
    """
    Aplica três métodos de segmentação na imagem do objeto:
      1. Limiarização de Otsu
      2. Limiarização Adaptativa (Gaussiana)
      3. Watershed (baseado em Distance Transform sobre Otsu)

    Retorna um dicionário com as máscaras binárias e o limiar de Otsu calculado.
    """
    if img_objeto_bgr is None:
        return None

    k = _garantir_impar_minimo(blur_ksize, minimo=1)
    bloco = _garantir_impar_minimo(bloco_adaptativo, minimo=3)
    tipo_thresh = cv2.THRESH_BINARY_INV if inverter_limiar else cv2.THRESH_BINARY

    objeto_gray = _extrair_canal(img_objeto_bgr, canal)
    objeto_blur = cv2.GaussianBlur(objeto_gray, (k, k), 0) if (usar_blur and k > 1) else objeto_gray

    # Método 1: Limiarização de Otsu
    limiar_otsu, mask_otsu = cv2.threshold(objeto_blur, 0, 255, tipo_thresh + cv2.THRESH_OTSU)

    # Método 2: Limiarização Adaptativa
    mask_adaptativa = cv2.adaptiveThreshold(
        objeto_blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        tipo_thresh,
        bloco,
        int(constante_c),
    )

    # Método 3: Watershed
    mask_watershed = aplicar_watershed(img_objeto_bgr, mask_otsu)

    return {
        "objeto_gray": objeto_blur,
        "limiar_otsu": float(limiar_otsu),
        "mask_otsu": mask_otsu,
        "mask_adaptativa": mask_adaptativa,
        "mask_watershed": mask_watershed,
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
    metodo="Otsu",
    canal="GRAY",
    usar_gauss=False,
    gauss_ksize=5,
    usar_mediana=False,
    usar_bilateral=False,
    inverter_otsu=False,
    label="ROI",
):
    """
    Aplica Otsu, Limiarização Adaptativa ou Watershed apenas dentro da ROI definida por rect_xywh.
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

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    bbox_vis = img_rgb.copy()
    cv2.rectangle(bbox_vis, (x, y), (x + w, y + h), (40, 220, 80), 2)

    roi_bgr = img_proc[y : y + h, x : x + w]
    roi_rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
    roi_gray = _extrair_canal(roi_bgr, canal)

    tipo_otsu = cv2.THRESH_BINARY_INV if inverter_otsu else cv2.THRESH_BINARY
    if metodo == "Otsu":
        _, mask_roi = cv2.threshold(roi_gray, 0, 255, tipo_otsu + cv2.THRESH_OTSU)
    elif metodo == "Watershed":
        _, mask_otsu_roi = cv2.threshold(roi_gray, 0, 255, tipo_otsu + cv2.THRESH_OTSU)
        mask_roi = aplicar_watershed(roi_bgr, mask_otsu_roi)
    else:
        mask_roi = cv2.adaptiveThreshold(
            roi_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, tipo_otsu, 21, 5
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
        assert seg["mask_watershed"].shape == img_bgr.shape[:2]

    rect, label = detectar_roi_yolo(img_bgr, padding_pct=10)
    roi_res = segmentar_roi(img_bgr, rect, metodo="Watershed", label=label)
    assert roi_res["mask_resultado"].shape == img_bgr.shape[:2] and np.any(roi_res["mask_resultado"] > 0)
    assert roi_res["roi_crop"].size > 0
    print(f"Self-check OK | {label} | ROI={rect}")
