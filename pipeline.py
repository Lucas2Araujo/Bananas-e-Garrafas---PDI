import cv2
import numpy as np
from pathlib import Path
from skimage.feature import graycomatrix, graycoprops
from ultralytics import YOLO

class HybridPDIPipeline:
    def __init__(self, yolo_weights="yolo11n.pt", use_yolo=True, skip_frames=3, imgsz=320, device="intel:gpu"):
        self.use_yolo = use_yolo
        self.skip_frames = max(1, skip_frames)
        self.imgsz = imgsz
        self.device = device
        self.frame_count = 0
        self.last_bbox = None
        self.last_source = None
        
        # Classes COCO: 39 = bottle (garrafa), 46 = banana
        self.target_classes = [39, 46]

        if self.use_yolo:
            # Prioriza modelo exportado OpenVINO se existir na pasta do projeto
            ov_dir = Path("yolo11n_openvino_model")
            weights_to_load = str(ov_dir) if ov_dir.is_dir() else yolo_weights
            self.detector = YOLO(weights_to_load)
            self.is_openvino = ov_dir.is_dir() or "openvino" in str(weights_to_load).lower()
        else:
            self.detector = None
            self.is_openvino = False

    def extract_roi(self, frame, force_infer=False):
        """
        1. LOCALIZAÇÃO: Tenta isolar o objeto via YOLO com suporte a Frame Skipping.
        Se force_infer=False, infere a cada self.skip_frames e reutiliza a última caixa nos frames intermediários.
        Se falhar (Falso Negativo), aplica Fallback de Center Crop (70% central da caixa).
        """
        h_img, w_img = frame.shape[:2]
        
        if self.use_yolo and self.detector is not None:
            should_infer = force_infer or (self.frame_count % self.skip_frames == 0) or (self.last_bbox is None)
            self.frame_count += 1

            if should_infer:
                # Dispara inferência rápida (OpenVINO GPU se disponível, imgsz=320)
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
        Retorna o vetor 1D de características (19 features) e as imagens intermediárias para debug.
        """
        roi_norm = self.resize_keep_aspect(roi, max_dim=256)
        
        # Filtro Gaussiano 5x5 para suavizar ruído do sensor da câmera
        blurred = cv2.GaussianBlur(roi_norm, (5, 5), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        _, s_chan, v_chan = cv2.split(hsv)

        # Limiarização de Otsu combinada (Canal V para brilho/reflexo + Canal S para saturação/rótulo)
        _, thresh_v = cv2.threshold(v_chan, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        _, thresh_s = cv2.threshold(s_chan, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
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
        else:
            solid_mask = closed
            circularity = 0.0

        # --- EXTRAÇÃO DE ATRIBUTOS (19 FEATURES) ---
        # A) FORMA (9 features): 7 Momentos de Hu + Circularidade + Razão de Aspecto
        moments = cv2.moments(solid_mask)
        hu_moments = cv2.HuMoments(moments).flatten()
        hu_log = [-np.sign(m) * np.log10(abs(m)) if m != 0 else 0.0 for m in hu_moments]
        
        h_roi, w_roi = roi_norm.shape[:2]
        aspect_ratio = float(w_roi) / float(h_roi)
        shape_features = hu_log + [circularity, aspect_ratio]

        # B) TEXTURA - GLCM (4 features)
        gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
        gray_quant = (gray // 8).astype(np.uint8)
        glcm = graycomatrix(gray_quant, distances=[1, 3], angles=[0, np.pi/4, np.pi/2], 
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

        feature_vector = np.array(shape_features + texture_features + color_features, dtype=np.float32)

        debug_views = {
            "roi_norm": roi_norm,
            "otsu_raw": otsu_combined,
            "morph_closed": closed,
            "solid_mask": solid_mask
        }
        return feature_vector, debug_views
