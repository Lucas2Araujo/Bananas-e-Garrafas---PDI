import cv2
import numpy as np


_FREEMAN_DIRS = {
    (1, 0): 0,
    (1, -1): 1,
    (0, -1): 2,
    (-1, -1): 3,
    (-1, 0): 4,
    (-1, 1): 5,
    (0, 1): 6,
    (1, 1): 7,
}


def aplicar_bifurcacao(img_bgr, mask_binaria):
    """
    Bifurcação crucial do pipeline:
      - Forma: usa diretamente a máscara binária.
      - Textura: aplica bitwise_and sobre a imagem em escala de cinza.
    """
    if img_bgr is None or mask_binaria is None:
        return None

    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    mask_u8 = np.where(mask_binaria > 0, 255, 0).astype(np.uint8)
    textura_gray = cv2.bitwise_and(img_gray, img_gray, mask=mask_u8)

    return {
        "img_gray": img_gray,
        "entrada_forma": mask_u8,
        "entrada_textura": textura_gray,
    }


def extrair_momentos_hu(mask_binaria):
    """Calcula os 7 Momentos invariantes de Hu (brutos e em escala logarítmica)."""
    mask_u8 = np.where(mask_binaria > 0, 255, 0).astype(np.uint8)
    momentos = cv2.moments(mask_u8)
    hu = cv2.HuMoments(momentos).flatten()
    hu_log = [float(-np.sign(h) * np.log10(abs(h))) if h != 0 else 0.0 for h in hu]
    return {
        "brutos": [float(h) for h in hu],
        "log": hu_log,
    }


def _desenhar_histograma_barras(valores, labels, titulo, cor=(50, 140, 220), largura=640, altura=320):
    """Desenha um gráfico de barras simples em RGB."""
    canvas = np.full((altura, largura, 3), 250, dtype=np.uint8)
    m_esq, m_dir, m_sup, m_inf = 50, 20, 45, 40
    pw = largura - m_esq - m_dir
    ph = altura - m_sup - m_inf

    cv2.rectangle(canvas, (m_esq, m_sup), (m_esq + pw, m_sup + ph), (240, 243, 247), -1)
    cv2.putText(canvas, titulo, (m_esq, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30, 30, 30), 2, cv2.LINE_AA)

    n = len(valores)
    if n == 0:
        return canvas

    max_v = float(max(valores)) if max(valores) > 0 else 1.0
    step = pw / n
    bar_w = max(2, int(step * 0.7))

    for i, val in enumerate(valores):
        x1 = int(m_esq + i * step + (step - bar_w) / 2)
        x2 = x1 + bar_w
        h_bar = int((val / max_v) * (ph - 10))
        y1 = m_sup + ph - h_bar
        y2 = m_sup + ph
        cv2.rectangle(canvas, (x1, y1), (x2, y2), cor, -1)
        if labels and i < len(labels):
            cv2.putText(canvas, str(labels[i]), (x1, y2 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (60, 60, 60), 1, cv2.LINE_AA)

    return canvas


def extrair_codigo_cadeia(mask_binaria):
    """Extrai o Código de Cadeia de Freeman (8 direções) do maior contorno externo."""
    mask_u8 = np.where(mask_binaria > 0, 255, 0).astype(np.uint8)
    contornos, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    h, w = mask_u8.shape
    img_contorno = np.zeros((h, w, 3), dtype=np.uint8)

    if not contornos:
        img_hist = _desenhar_histograma_barras([0] * 8, list(range(8)), "Codigo de Cadeia (8 direcoes)")
        return {
            "sequencia": [],
            "histograma": [0] * 8,
            "img_contorno": img_contorno,
            "img_histograma": img_hist,
        }

    maior = max(contornos, key=cv2.contourArea)
    pts = maior.reshape(-1, 2)
    cv2.drawContours(img_contorno, [maior], -1, (40, 220, 120), 2)
    if len(pts) > 0:
        cv2.circle(img_contorno, tuple(pts[0]), 5, (255, 80, 80), -1)

    sequencia = []
    hist = [0] * 8
    for i in range(len(pts)):
        p1 = pts[i]
        p2 = pts[(i + 1) % len(pts)]
        dx = int(np.clip(p2[0] - p1[0], -1, 1))
        dy = int(np.clip(p2[1] - p1[1], -1, 1))
        codigo = _FREEMAN_DIRS.get((dx, dy))
        if codigo is not None:
            sequencia.append(codigo)
            hist[codigo] += 1

    img_hist = _desenhar_histograma_barras(hist, [f"D{i}" for i in range(8)], "Histograma - Codigo de Cadeia (Freeman 8-dir)")
    return {
        "sequencia": sequencia,
        "histograma": hist,
        "img_contorno": img_contorno,
        "img_histograma": img_hist,
    }


def extrair_lbp(img_gray_mascarada, mask_binaria):
    """Calcula Local Binary Patterns (P=8, R=1) vetorizado sobre os pixels da máscara."""
    img = img_gray_mascarada.astype(np.int16)
    padded = np.pad(img, 1, mode="edge")
    centro = padded[1:-1, 1:-1]

    vizinhos = [
        padded[0:-2, 0:-2],  # NW
        padded[0:-2, 1:-1],  # N
        padded[0:-2, 2:],    # NE
        padded[1:-1, 2:],    # E
        padded[2:, 2:],      # SE
        padded[2:, 1:-1],    # S
        padded[2:, 0:-2],    # SW
        padded[1:-1, 0:-2],  # W
    ]

    lbp = np.zeros_like(centro, dtype=np.uint8)
    for bit, viz in enumerate(vizinhos):
        lbp |= ((viz >= centro).astype(np.uint8) << bit)

    mask_bool = mask_binaria > 0
    lbp_mascarado = np.where(mask_bool, lbp, 0).astype(np.uint8)

    if np.any(mask_bool):
        hist, _ = np.histogram(lbp_mascarado[mask_bool], bins=256, range=(0, 256))
        hist_norm = hist.astype(np.float64) / float(hist.sum())
    else:
        hist_norm = np.zeros(256, dtype=np.float64)

    # Agrupa em 16 bins para visualização limpa no gráfico de barras
    hist_16 = hist_norm.reshape(16, 16).sum(axis=1).tolist()
    img_hist = _desenhar_histograma_barras(
        hist_16,
        [f"{i*16}" for i in range(16)],
        "Histograma LBP (16 faixas de 0-255)",
        cor=(180, 90, 220),
    )

    mapa_rgb = cv2.cvtColor(cv2.applyColorMap(lbp_mascarado, cv2.COLORMAP_INFERNO), cv2.COLOR_BGR2RGB)
    mapa_rgb[~mask_bool] = 0

    return {
        "mapa_lbp": mapa_rgb,
        "histograma": hist_norm.tolist(),
        "img_histograma": img_hist,
    }


def extrair_haralick_glcm(img_gray_mascarada, mask_binaria, niveis=16):
    """Calcula a matriz GLCM (horizontal dx=1, simétrica) e descritores de Haralick."""
    mask_bool = mask_binaria > 0
    q = (img_gray_mascarada.astype(np.int32) * niveis) // 256
    q = np.clip(q, 0, niveis - 1)

    glcm = np.zeros((niveis, niveis), dtype=np.float64)
    validos = mask_bool[:, :-1] & mask_bool[:, 1:]
    if np.any(validos):
        p1 = q[:, :-1][validos]
        p2 = q[:, 1:][validos]
        np.add.at(glcm, (p1, p2), 1.0)
        np.add.at(glcm, (p2, p1), 1.0)
        total = glcm.sum()
        if total > 0:
            glcm /= total

    i_idx, j_idx = np.indices((niveis, niveis))
    diff = i_idx - j_idx
    contraste = float(np.sum(glcm * (diff ** 2)))
    dissimilaridade = float(np.sum(glcm * np.abs(diff)))
    homogeneidade = float(np.sum(glcm / (1.0 + diff ** 2)))
    asm = float(np.sum(glcm ** 2))
    energia = float(np.sqrt(asm))

    mu_i = float(np.sum(i_idx * glcm))
    mu_j = float(np.sum(j_idx * glcm))
    sigma_i = float(np.sqrt(np.sum(glcm * ((i_idx - mu_i) ** 2))))
    sigma_j = float(np.sqrt(np.sum(glcm * ((j_idx - mu_j) ** 2))))
    if sigma_i > 1e-12 and sigma_j > 1e-12:
        correlacao = float(np.sum(glcm * (i_idx - mu_i) * (j_idx - mu_j)) / (sigma_i * sigma_j))
    else:
        correlacao = 0.0

    # Renderiza heatmap da matriz GLCM
    glcm_max = float(glcm.max())
    if glcm_max > 0:
        norm_img = np.uint8((glcm / glcm_max) * 255)
    else:
        norm_img = np.zeros((niveis, niveis), dtype=np.uint8)
    heatmap = cv2.resize(norm_img, (320, 320), interpolation=cv2.INTER_NEAREST)
    heatmap_rgb = cv2.cvtColor(cv2.applyColorMap(heatmap, cv2.COLORMAP_VIRIDIS), cv2.COLOR_BGR2RGB)

    return {
        "glcm": glcm,
        "metricas": {
            "Contraste": contraste,
            "Dissimilaridade": dissimilaridade,
            "Homogeneidade": homogeneidade,
            "Energia": energia,
            "ASM": asm,
            "Correlacao": correlacao,
        },
        "img_heatmap": heatmap_rgb,
    }


def extrair_todas_caracteristicas(img_bgr, mask_binaria):
    """Executa o pipeline completo de bifurcação e extração de forma e textura."""
    bif = aplicar_bifurcacao(img_bgr, mask_binaria)
    if bif is None:
        return None

    hu = extrair_momentos_hu(bif["entrada_forma"])
    cadeia = extrair_codigo_cadeia(bif["entrada_forma"])
    lbp = extrair_lbp(bif["entrada_textura"], bif["entrada_forma"])
    haralick = extrair_haralick_glcm(bif["entrada_textura"], bif["entrada_forma"])

    return {
        "bifurcacao": bif,
        "hu": hu,
        "cadeia": cadeia,
        "lbp": lbp,
        "haralick": haralick,
    }
