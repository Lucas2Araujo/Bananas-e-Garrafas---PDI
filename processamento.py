import cv2
import numpy as np


def processar_imagem(caminho):
    """Lê a imagem e gera todas as variações RGB, HSV e histogramas detalhados."""
    img_bgr = cv2.imread(caminho)
    if img_bgr is None:
        return None

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    r_gray, g_gray, b_gray = cv2.split(img_rgb)
    h, s, v = cv2.split(img_hsv)

    zeros = np.zeros_like(r_gray)
    r_color = cv2.merge([r_gray, zeros, zeros])
    g_color = cv2.merge([zeros, g_gray, zeros])
    b_color = cv2.merge([zeros, zeros, b_gray])

    return {
        "rgb": img_rgb,
        "r_color": r_color,
        "g_color": g_color,
        "b_color": b_color,
        "r_gray": r_gray,
        "g_gray": g_gray,
        "b_gray": b_gray,
        "hist_rgb": gerar_histograma_rgb(img_rgb),
        "hsv": img_hsv,
        "h": h,
        "s": s,
        "v": v,
        "hist_hsv": gerar_histograma_hsv(img_hsv),
    }


def gerar_histograma_rgb(img_rgb):
    """Gera histograma RGB detalhado (800x480) com área preenchida, médias e estatísticas."""
    largura, altura = 800, 480
    margem_esq, margem_dir = 65, 25
    margem_sup, margem_inf = 85, 45

    plot_w = largura - margem_esq - margem_dir
    plot_h = altura - margem_sup - margem_inf

    canvas = np.full((altura, largura, 3), 252, dtype=np.uint8)
    overlay = canvas.copy()

    # Fundo e grade da área do gráfico
    cv2.rectangle(canvas, (margem_esq, margem_sup), (margem_esq + plot_w, margem_sup + plot_h), (245, 247, 250), -1)
    for i in range(5):
        y = margem_sup + int(i * plot_h / 4)
        cv2.line(canvas, (margem_esq, y), (margem_esq + plot_w, y), (220, 225, 230), 1)
        pct = f"{100 - i * 25}%"
        cv2.putText(canvas, pct, (12, y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (100, 100, 100), 1, cv2.LINE_AA)

    for val in [0, 32, 64, 96, 128, 160, 192, 224, 255]:
        x = margem_esq + int((val / 255.0) * plot_w)
        cv2.line(canvas, (x, margem_sup), (x, margem_sup + plot_h), (220, 225, 230), 1)
        cv2.putText(canvas, str(val), (x - 10, margem_sup + plot_h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (80, 80, 80), 1, cv2.LINE_AA)

    cv2.putText(canvas, "Intensidade do Pixel (0 - 255)", (margem_esq + plot_w // 2 - 95, altura - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (60, 60, 60), 1, cv2.LINE_AA)

    cores = [(220, 40, 40), (30, 175, 50), (35, 90, 230)]
    nomes = ["R", "G", "B"]

    for idx, (cor, nome) in enumerate(zip(cores, nomes)):
        canal = img_rgb[:, :, idx]
        media = float(np.mean(canal))
        desvio = float(np.std(canal))
        mediana = float(np.median(canal))

        hist = cv2.calcHist([img_rgb], [idx], None, [256], [0, 256])
        cv2.normalize(hist, hist, 0, plot_h - 10, cv2.NORM_MINMAX)
        hist_1d = hist.flatten()

        pontos = [(margem_esq, margem_sup + plot_h)]
        for x in range(256):
            px = margem_esq + int((x / 255.0) * plot_w)
            py = margem_sup + plot_h - int(hist_1d[x])
            pontos.append((px, py))
        pontos.append((margem_esq + plot_w, margem_sup + plot_h))

        pts_array = np.array(pontos, dtype=np.int32)
        cv2.fillPoly(overlay, [pts_array], cor)
        cv2.polylines(canvas, [pts_array[1:-1]], False, cor, 2, cv2.LINE_AA)

        # Linha vertical tracejada indicando a média do canal
        x_media = margem_esq + int((media / 255.0) * plot_w)
        for y_dash in range(margem_sup, margem_sup + plot_h, 8):
            cv2.line(canvas, (x_media, y_dash), (x_media, min(y_dash + 4, margem_sup + plot_h)), cor, 1)

        # Cabeçalho estatístico do canal
        y_txt = 22 + idx * 20
        info = f"{nome}: Media={media:5.1f} | Desvio={desvio:5.1f} | Mediana={mediana:5.1f}"
        cv2.putText(canvas, info, (margem_esq, y_txt), cv2.FONT_HERSHEY_SIMPLEX, 0.46, cor, 1, cv2.LINE_AA)

    # Mistura transparência do preenchimento sob as curvas
    cv2.addWeighted(overlay, 0.16, canvas, 0.84, 0, canvas)
    cv2.rectangle(canvas, (margem_esq, margem_sup), (margem_esq + plot_w, margem_sup + plot_h), (140, 140, 140), 1)
    return canvas


def gerar_histograma_hsv(img_hsv):
    """Gera histograma HSV detalhado (800x480) respeitando escala 0-179 para H e 0-255 para S/V."""
    largura, altura = 800, 480
    margem_esq, margem_dir = 65, 25
    margem_sup, margem_inf = 85, 45

    plot_w = largura - margem_esq - margem_dir
    plot_h = altura - margem_sup - margem_inf

    canvas = np.full((altura, largura, 3), 252, dtype=np.uint8)
    overlay = canvas.copy()

    cv2.rectangle(canvas, (margem_esq, margem_sup), (margem_esq + plot_w, margem_sup + plot_h), (245, 247, 250), -1)
    for i in range(5):
        y = margem_sup + int(i * plot_h / 4)
        cv2.line(canvas, (margem_esq, y), (margem_esq + plot_w, y), (220, 225, 230), 1)
        pct = f"{100 - i * 25}%"
        cv2.putText(canvas, pct, (12, y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (100, 100, 100), 1, cv2.LINE_AA)

    for frac in np.linspace(0, 1, 9):
        x = margem_esq + int(frac * plot_w)
        cv2.line(canvas, (x, margem_sup), (x, margem_sup + plot_h), (220, 225, 230), 1)
        label_x = f"{int(frac * 179)}/{int(frac * 255)}"
        cv2.putText(canvas, label_x, (x - 22, margem_sup + plot_h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (80, 80, 80), 1, cv2.LINE_AA)

    cv2.putText(canvas, "Escala Normalizada: Matiz H (0-179) / Saturacao S e Valor V (0-255)",
                (margem_esq + 115, altura - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (60, 60, 60), 1, cv2.LINE_AA)

    configs = [
        (0, 180, (200, 20, 200), "H (Matiz)"),
        (1, 256, (0, 160, 185), "S (Saturacao)"),
        (2, 256, (55, 55, 55), "V (Brilho)"),
    ]

    for idx, bins, cor, legenda in configs:
        canal = img_hsv[:, :, idx]
        media = float(np.mean(canal))
        desvio = float(np.std(canal))
        mediana = float(np.median(canal))

        hist = cv2.calcHist([img_hsv], [idx], None, [bins], [0, bins])
        cv2.normalize(hist, hist, 0, plot_h - 10, cv2.NORM_MINMAX)
        hist_1d = hist.flatten()

        pontos = [(margem_esq, margem_sup + plot_h)]
        for x in range(bins):
            px = margem_esq + int((x / (bins - 1)) * plot_w)
            py = margem_sup + plot_h - int(hist_1d[x])
            pontos.append((px, py))
        pontos.append((margem_esq + plot_w, margem_sup + plot_h))

        pts_array = np.array(pontos, dtype=np.int32)
        cv2.fillPoly(overlay, [pts_array], cor)
        cv2.polylines(canvas, [pts_array[1:-1]], False, cor, 2, cv2.LINE_AA)

        x_media = margem_esq + int((media / (bins - 1)) * plot_w)
        for y_dash in range(margem_sup, margem_sup + plot_h, 8):
            cv2.line(canvas, (x_media, y_dash), (x_media, min(y_dash + 4, margem_sup + plot_h)), cor, 1)

        y_txt = 22 + idx * 20
        info = f"{legenda:<13}: Media={media:5.1f} | Desvio={desvio:5.1f} | Mediana={mediana:5.1f}"
        cv2.putText(canvas, info, (margem_esq, y_txt), cv2.FONT_HERSHEY_SIMPLEX, 0.45, cor, 1, cv2.LINE_AA)

    cv2.addWeighted(overlay, 0.16, canvas, 0.84, 0, canvas)
    cv2.rectangle(canvas, (margem_esq, margem_sup), (margem_esq + plot_w, margem_sup + plot_h), (140, 140, 140), 1)
    return canvas