import cv2
import numpy as np
from pathlib import Path
from skimage.feature import graycomatrix, graycoprops
from ultralytics import YOLO

import config

class HybridPDIPipeline:
    def __init__(self, yolo_weights=None, use_yolo=True, skip_frames=3, imgsz=320, device="intel:gpu", min_contour_area=1500):
        self.use_yolo = use_yolo
        self.skip_frames = max(1, skip_frames)
        self.imgsz = imgsz
        self.device = device
        self.min_contour_area = min_contour_area
        self.frame_count = 0
        self.last_bbox = None
        self.last_source = None
        self.last_contour_area = 0.0
        self.last_is_valid = False
        
        self.target_classes = [39, 46]

        if self.use_yolo:
            weights_to_load = yolo_weights or getattr(config, "YOLO_MODEL_PATH", "yolo26n.pt")
            self.detector = YOLO(str(weights_to_load))
            self.is_openvino = "openvino" in str(weights_to_load).lower()
        else:
            self.detector = None
            self.is_openvino = False

    def extract_roi(self, frame, force_infer=False, allow_fallback=True):
        """
        1. LOCALIZAÇÃO: Tenta isolar o objeto via YOLO com suporte a Frame Skipping.
        Se force_infer=False, infere a cada self.skip_frames e reutiliza a última caixa nos frames intermediários.
        Se falhar e allow_fallback=True, aplica Fallback de Center Crop (70% central da imagem).
        """
        h_img, w_img = frame.shape[:2]
        
        if self.use_yolo and self.detector is not None:
            should_infer = force_infer or (self.frame_count % self.skip_frames == 0) or (self.last_bbox is None)
            self.frame_count += 1

            if should_infer:
                predict_kwargs = {
                    "classes": self.target_classes,
                    "conf": 0.20,
                    "imgsz": self.imgsz,
                    "verbose": False
                }
                if self.is_openvino and self.device:
                    predict_kwargs["device"] = self.device

                try:
                    results = self.detector(frame, **predict_kwargs)
                except Exception:
                    # Fallback suave caso o driver da GPU falhe momentaneamente
                    predict_kwargs.pop("device", None)
                    results = self.detector(frame, **predict_kwargs)

                boxes = results[0].boxes
                
                if len(boxes) > 0:
                    best_box = max(boxes, key=lambda b: float(b.conf[0]))
                    x1, y1, x2, y2 = map(int, best_box.xyxy[0])
                    
                    # Adiciona 10% de margem (padding) para não cortar bordas na morfologia
                    pad_x = int((x2 - x1) * 0.10)
                    pad_y = int((y2 - y1) * 0.10)
                    x1 = max(0, x1 - pad_x)
                    y1 = max(0, y1 - pad_y)
                    x2 = min(w_img, x2 + pad_x)
                    y2 = min(h_img, y2 + pad_y)
                    
                    self.last_bbox = (x1, y1, x2, y2)
                    self.last_source = "YOLO"
                    return frame[y1:y2, x1:x2], self.last_bbox, "YOLO"
                else:
                    self.last_bbox = None
            
            elif self.last_bbox is not None and self.last_source == "YOLO":
                # Frame Skipping: Reutiliza a última detecção válida
                x1, y1, x2, y2 = self.last_bbox
                return frame[y1:y2, x1:x2], self.last_bbox, "YOLO"

        # Se não detectou e fallback não for permitido, não força processamento
        if not allow_fallback:
            self.last_bbox = None
            self.last_source = None
            return None, (0, 0, 0, 0), "NONE"

        # FALLBACK: Recorte central (70% da imagem) caso o YOLO falhe ou esteja desligado
        self.last_bbox = None
        margin_x = int(w_img * 0.15)
        margin_y = int(h_img * 0.15)
        x1, y1, x2, y2 = margin_x, margin_y, w_img - margin_x, h_img - margin_y
        return frame[y1:y2, x1:x2], (x1, y1, x2, y2), "FALLBACK_CENTER"

    def resize_keep_aspect(self, roi, max_dim=256):
        """Redimensiona mantendo a proporção original para não distorcer os Momentos de Hu."""
        h, w = roi.shape[:2]
        scale = max_dim / float(max(h, w))
        new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
        return cv2.resize(roi, (new_w, new_h), interpolation=cv2.INTER_AREA)

    def segment_and_extract(self, roi):
        """
        2. PDI CLÁSSICO: Pré-processamento, Otsu Duplo (V + S), Morfologia Matemática e Features.
        Retorna o vetor 1D de características (20 features) e as imagens intermediárias para debug.
        """
        roi_norm = self.resize_keep_aspect(roi, max_dim=256)
        
        # Filtro Gaussiano 5x5 para suavizar ruído do sensor da câmera
        blurred = cv2.GaussianBlur(roi_norm, (5, 5), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        _, s_chan, v_chan = cv2.split(hsv)

        # Limiarização de Otsu combinada (Canal V para brilho/reflexo + Canal S para saturação/rótulo)
        _, thresh_v = cv2.threshold(v_chan, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        _, thresh_s = cv2.threshold(s_chan, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Inversão automática se a borda externa for clara (fundo iluminado)
        border_mask = np.zeros_like(thresh_v, dtype=bool)
        border_mask[:2, :] = border_mask[-2:, :] = border_mask[:, :2] = border_mask[:, -2:] = True
        if np.mean(thresh_v[border_mask] == 255) > 0.50:
            thresh_v = cv2.bitwise_not(thresh_v)

        otsu_combined = cv2.bitwise_or(thresh_v, thresh_s)

        # --- MORFOLOGIA MATEMÁTICA ---
        # 1. Abertura: remove ruídos brancos isolados
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        opened = cv2.morphologyEx(otsu_combined, cv2.MORPH_OPEN, kernel_open)

        # 2. Fechamento: une paredes translúcidas e reflexos
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19))
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel_close)

        # 3. Preenchimento de buracos internos via maior contorno
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        solid_mask = np.zeros_like(closed)
        
        if contours:
            largest_cnt = max(contours, key=cv2.contourArea)
            cv2.drawContours(solid_mask, [largest_cnt], -1, 255, thickness=cv2.FILLED)
            area = cv2.contourArea(largest_cnt)
            perimeter = cv2.arcLength(largest_cnt, True)
            circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0.0
            
            # Razão de aspecto invariante a rotação via minAreaRect (independe de estar vertical ou horizontal)
            rect = cv2.minAreaRect(largest_cnt)
            dim1, dim2 = rect[1]
            minor_axis = min(dim1, dim2)
            major_axis = max(dim1, dim2)
            aspect_ratio = minor_axis / max(major_axis, 1e-5)

            # Solidez via Fecho Convexo (Convex Hull): diferencia corpos compactos/garrafas de cascas abertas/fatias
            hull = cv2.convexHull(largest_cnt)
            hull_area = cv2.contourArea(hull)
            solidity = float(area) / max(hull_area, 1e-5)
        else:
            solid_mask = np.zeros_like(closed)
            area = 0.0
            circularity = 0.0
            aspect_ratio = 1.0
            solidity = 0.0

        self.last_contour_area = area
        self.last_is_valid = bool(area >= self.min_contour_area)

        # --- EXTRAÇÃO DE ATRIBUTOS (20 FEATURES) ---
        # A) FORMA (10 features): 7 Momentos de Hu + Circularidade + Razão de Aspecto Invariante + Solidez
        moments = cv2.moments(solid_mask)
        hu_moments = cv2.HuMoments(moments).flatten()
        hu_log = [-np.sign(m) * np.log10(abs(m)) if m != 0 else 0.0 for m in hu_moments]
        shape_features = hu_log + [circularity, aspect_ratio, solidity]

        # B) TEXTURA - GLCM (4 features em 4 ângulos: 0°, 45°, 90°, 135° restrito ao objeto)
        gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
        if np.any(solid_mask):
            gray = cv2.bitwise_and(gray, gray, mask=solid_mask)
        gray_quant = (gray // 8).astype(np.uint8)
        glcm = graycomatrix(gray_quant, distances=[1, 3], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], 
                            levels=32, symmetric=True, normed=True)
        texture_features = [
            float(graycoprops(glcm, 'contrast').mean()),
            float(graycoprops(glcm, 'dissimilarity').mean()),
            float(graycoprops(glcm, 'homogeneity').mean()),
            float(graycoprops(glcm, 'energy').mean())
        ]

        # C) COR NO ESPAÇO HSV (6 features)
        mean_hsv, std_hsv = cv2.meanStdDev(hsv, mask=solid_mask if np.any(solid_mask) else None)
        color_features = mean_hsv.flatten().tolist() + std_hsv.flatten().tolist()

        h_mean, s_mean, v_mean = mean_hsv.flatten()[:3]
        # Validação de assinatura HSV de banana ou garrafa PET no fallback
        is_banana = (12.0 <= h_mean <= 50.0 and s_mean >= 40.0 and v_mean >= 35.0) or (s_mean >= 30.0 and 20.0 <= v_mean <= 95.0 and 10.0 <= h_mean <= 45.0)
        is_bottle = (s_mean <= 85.0 and v_mean >= 40.0) or (35.0 <= h_mean <= 110.0 and v_mean >= 35.0)
        self.last_color_valid = bool((is_banana or is_bottle) and v_mean >= 25.0)

        feature_vector = np.array(shape_features + texture_features + color_features, dtype=np.float32)

        # HUD didático na visualização da máscara sólida
        solid_mask_vis = cv2.cvtColor(solid_mask, cv2.COLOR_GRAY2BGR)
        if contours:
            cv2.drawContours(solid_mask_vis, [hull], -1, (0, 255, 0), 2)
            box_pts = cv2.boxPoints(rect).astype(np.int32)
            cv2.drawContours(solid_mask_vis, [box_pts], -1, (255, 200, 0), 2)
            hud1 = f"Area: {int(area)} | Solidez: {solidity:.2f}"
            hud2 = f"Circ: {circularity:.2f} | Aspecto: {aspect_ratio:.2f}"
            cv2.putText(solid_mask_vis, hud1, (6, solid_mask_vis.shape[0] - 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(solid_mask_vis, hud2, (6, solid_mask_vis.shape[0] - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 255), 1, cv2.LINE_AA)

        debug_views = {
            "roi_norm": roi_norm,
            "otsu_raw": otsu_combined,
            "morph_closed": closed,
            "solid_mask": solid_mask_vis
        }

        # Validação de objeto real: se a área for inferior ao mínimo, descarta vetor para evitar falsos positivos
        if not self.last_is_valid:
            return None, debug_views

        return feature_vector, debug_views
