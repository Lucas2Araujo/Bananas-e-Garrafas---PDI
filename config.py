import os
# Silencia logs ruidosos do backend C++ do OpenCV durante o scan de câmeras
os.environ.setdefault("OPENCV_LOG_LEVEL", "OFF")
# Garante compatibilidade do OpenCV GUI no Linux Wayland (evita erro de plugin Qt wayland)
os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

import cv2
from pathlib import Path

def scan_available_cameras(max_index=5):
    """
    Retorna lista de tuplas (indice, nome_amigavel).
    No Linux, usa /sys/class/video4linux para detecção instantânea sem travar em buffers de loopback.
    """
    cameras = []
    if os.name != "nt" and os.path.isdir("/sys/class/video4linux"):
        import glob
        for dev_path in sorted(glob.glob("/sys/class/video4linux/video*")):
            try:
                base = os.path.basename(dev_path)
                idx = int(base.replace("video", ""))
                name_file = os.path.join(dev_path, "name")
                name = open(name_file).read().strip() if os.path.exists(name_file) else f"Camera {idx}"
                cameras.append((idx, f"video{idx}: {name}"))
            except Exception:
                pass
    if not cameras:
        backend = cv2.CAP_V4L2 if hasattr(cv2, "CAP_V4L2") and os.name != "nt" else cv2.CAP_ANY
        for idx in range(max_index + 1):
            cap = cv2.VideoCapture(idx, backend)
            if cap.isOpened():
                cameras.append((idx, f"Camera {idx}"))
                cap.release()
    return cameras

def get_default_camera_index():
    cams = scan_available_cameras()
    if not cams:
        return 0
    # Prioriza câmeras reais (evita Virtual Camera e Dummy se houver webcam física)
    for idx, name in cams:
        name_lower = name.lower()
        if "virtual" not in name_lower and "dummy" not in name_lower and "obs" not in name_lower:
            return idx
    return cams[0][0]

# --- FONTE DE VÍDEO ---
CAMERA_SOURCE = get_default_camera_index()

# --- MODELO YOLO (YOLO 26n) ---
YOLO_MODEL_PATH = "yolo26n.pt"

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
MODEL_TYPE = "SVM"  # Classificador Supervisionado SVM (RBF + StandardScaler)
CLASS_NAMES = {0: "Organico (Banana)", 1: "Reciclavel (Garrafa)", 2: "Indefinido"}
SERIAL_CODES = {0: b"O", 1: b"R", 2: b"I"}
