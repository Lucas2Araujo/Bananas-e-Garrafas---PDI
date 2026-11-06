import os
# Silencia logs ruidosos do backend C++ do OpenCV durante o scan de câmeras
os.environ.setdefault("OPENCV_LOG_LEVEL", "OFF")
# Garante compatibilidade do OpenCV GUI no Linux Wayland (evita erro de plugin Qt wayland)
os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

import cv2
from pathlib import Path

def scan_available_cameras(max_index=5):
    """
    Testa portas de vídeo locais de 0 até max_index.
    Verifica abertura e leitura de frame real para ignorar loopbacks virtuais (como OBS em /dev/video0).
    """
    available = []
    backend = cv2.CAP_V4L2 if hasattr(cv2, "CAP_V4L2") and os.name != "nt" else cv2.CAP_ANY
    for idx in range(max_index + 1):
        cap = cv2.VideoCapture(idx, backend)
        if not cap.isOpened() and backend != cv2.CAP_ANY:
            cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                available.append(idx)
            cap.release()
    return available

# --- FONTE DE VÍDEO ---
_cams = scan_available_cameras(max_index=5)
CAMERA_SOURCE = _cams[0] if _cams else 0

# --- ACELERAÇÃO YOLO / OPENVINO / FRAME SKIPPING ---
# intel:gpu usa a Intel Iris Xe Graphics integrada; fallback automático para CPU
YOLO_DEVICE = "intel:gpu"
YOLO_IMGSZ = 320         # 320px quadruplica o FPS sem perder bananas e garrafas
YOLO_SKIP_FRAMES = 3     # Executa o YOLO a cada 3 frames e reutiliza a bbox nos demais

# --- DIRETÓRIOS E ARQUIVOS DO PROJETO ---
BASE_DIR = Path(__file__).resolve().parent

DATASET_ORGANICO = BASE_DIR / "dataset" / "organico"
DATASET_RECICLAVEL = BASE_DIR / "dataset" / "reciclavel"
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "classifier_pipeline.pkl"
FEATURES_CSV = BASE_DIR / "dataset" / "features.csv"

# Garante a existência dos diretórios
DATASET_ORGANICO.mkdir(parents=True, exist_ok=True)
DATASET_RECICLAVEL.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# --- COMUNICAÇÃO SERIAL (ESP32) ---
ENABLE_SERIAL = False
SERIAL_PORT = "/dev/ttyUSB0"  # No Windows altere para "COM3", "COM4", etc.
BAUD_RATE = 115200

# --- CLASSIFICADOR CLÁSSICO ---
MODEL_TYPE = "SVM"  # Opções: "SVM" ou "KNN"
CLASS_NAMES = {0: "Organico (Banana)", 1: "Reciclavel (Garrafa)"}
SERIAL_CODES = {0: b"O", 1: b"R"}
