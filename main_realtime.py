import os
# Silencia logs ruidosos do backend C++ do OpenCV
os.environ.setdefault("OPENCV_LOG_LEVEL", "OFF")
# Garante compatibilidade do OpenCV GUI no Linux Wayland (evita erro de plugin Qt wayland)
os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

import cv2
import time
import joblib
import numpy as np
from datetime import datetime
from pathlib import Path

import config
from pipeline import HybridPDIPipeline
import train_model

try:
    import serial
except ImportError:
    serial = None

def make_tile(img, title, size):
    """Converte imagem para BGR 3-canais, redimensiona e adiciona título legível."""
    if len(img.shape) == 2:
        img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        img_bgr = img.copy()
    
    resized = cv2.resize(img_bgr, size, interpolation=cv2.INTER_AREA)
    cv2.rectangle(resized, (0, 0), (size[0], 28), (30, 30, 30), -1)
    cv2.putText(resized, title, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return resized

def init_camera(source):
    """Inicializa captura com V4L2 no Linux se for índice numérico e define buffer unitário."""
    backend = cv2.CAP_V4L2 if isinstance(source, int) and hasattr(cv2, "CAP_V4L2") and os.name != "nt" else cv2.CAP_ANY
    cap = cv2.VideoCapture(source, backend)
    if not cap.isOpened() and backend != cv2.CAP_ANY:
        cap = cv2.VideoCapture(source)
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap

def main():
    print("=" * 60)
    print("INICIANDO SISTEMA PDI - BANANAS & GARRAFAS EM TEMPO REAL")
    print("=" * 60)

    # 1. Pipeline de PDI com Aceleração OpenVINO e Frame Skipping
    pdi_pipeline = HybridPDIPipeline(
        use_yolo=True,
        skip_frames=config.YOLO_SKIP_FRAMES,
        imgsz=config.YOLO_IMGSZ,
        device=config.YOLO_DEVICE
    )

    # 2. Carrega Classificador se disponível
    clf_model = None
    if config.MODEL_PATH.exists():
        try:
            clf_model = joblib.load(config.MODEL_PATH)
            print(f"[✔] Classificador ({config.MODEL_TYPE}) carregado com sucesso.")
        except Exception as e:
            print(f"[!] Erro ao carregar modelo: {e}")
    else:
        print("[!] Modelo ainda não treinado. Operando em modo de Coleta/Debug.")

    # 3. Comunicação Serial com ESP32
    ser = None
    if config.ENABLE_SERIAL:
        if serial is None:
            print("[!] pyserial não está instalado. Desabilitando serial.")
        else:
            try:
                ser = serial.Serial(config.SERIAL_PORT, config.BAUD_RATE, timeout=0.01)
                print(f"[✔] Porta Serial aberta em {config.SERIAL_PORT} @ {config.BAUD_RATE} bps.")
            except Exception as e:
                print(f"[!] Falha ao conectar na porta serial {config.SERIAL_PORT}: {e}")

    # 4. Inicializa Câmera
    # Prioriza câmeras reais funcionais
    available_cams = config.scan_available_cameras(max_index=5)
    camera_sources = list(available_cams)
    if config.CAMERA_SOURCE not in camera_sources:
        camera_sources.insert(0, config.CAMERA_SOURCE)
    if not camera_sources:
        camera_sources = [0]

    current_cam_idx = 0
    cap = init_camera(camera_sources[current_cam_idx])

    # Se a primeira falhar ao abrir, tenta a próxima disponível
    while not cap.isOpened() and current_cam_idx + 1 < len(camera_sources):
        current_cam_idx += 1
        print(f"[*] Tentando câmera alternativa: {camera_sources[current_cam_idx]}")
        cap = init_camera(camera_sources[current_cam_idx])

    if cap.isOpened():
        print(f"[✔] Câmera conectada com sucesso na fonte: {camera_sources[current_cam_idx]}")
    else:
        print(f"[!] Erro ao conectar em qualquer fonte de vídeo: {camera_sources}")

    cv2.namedWindow("Painel Unificado PDI", cv2.WINDOW_NORMAL)

    # Variáveis de Controle e Feedback
    prev_time = time.time()
    fps = 0.0
    status_msg = "Pronto"
    status_expiry = 0.0

    def trigger_decision(pred_class, pred_prob):
        """Dispara a decisão final para o ESP32 e atualiza status."""
        nonlocal status_msg, status_expiry
        class_name = config.CLASS_NAMES.get(pred_class, "Desconhecido")
        msg = f"GATILHO: {class_name} ({pred_prob * 100:.1f}%)"
        if ser and ser.is_open:
            code = config.SERIAL_CODES.get(pred_class)
            if code:
                try:
                    ser.write(code)
                    msg += f" -> Serial '{code.decode()}' enviada!"
                except Exception as ex:
                    msg += f" -> Erro serial: {ex}"
        print(f"[+] {msg}")
        status_msg = msg
        status_expiry = time.time() + 3.0

    while True:
        curr_time = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / max(curr_time - prev_time, 1e-6))
        prev_time = curr_time

        # Checa Gatilho Serial do ESP32 ('G')
        sim_trigger = False
        if ser and ser.is_open:
            try:
                if ser.in_waiting > 0:
                    incoming = ser.read(ser.in_waiting).decode(errors="ignore")
                    if "G" in incoming:
                        print("[ESP32] Gatilho 'G' recebido!")
                        sim_trigger = True
            except Exception:
                sim_trigger = False

        ret, frame = cap.read()
        if not ret or frame is None:
            # Em caso de falha de leitura, desenha tela de espera
            blank = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(blank, f"Aguardando camera: {camera_sources[current_cam_idx]} (pressione 'c' p/ alternar)", 
                        (20, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
            cv2.imshow("Painel Unificado PDI", blank)
            key = cv2.waitKey(30) & 0xFF
            if key in (27, ord('q')):
                break
            elif key == ord('c'):
                current_cam_idx = (current_cam_idx + 1) % len(camera_sources)
                cap.release()
                cap = init_camera(camera_sources[current_cam_idx])
            continue

        raw_frame = frame.copy()

        # Extração de ROI e Features
        roi, bbox, source_str = pdi_pipeline.extract_roi(frame)
        x1, y1, x2, y2 = bbox

        features, debug_views = None, None
        pred_label, pred_prob = None, None

        if roi is not None and roi.size > 0:
            features, debug_views = pdi_pipeline.segment_and_extract(roi)
            if clf_model is not None:
                try:
                    probs = clf_model.predict_proba([features])[0]
                    pred_label = int(np.argmax(probs))
                    pred_prob = float(probs[pred_label])
                except Exception:
                    pred_label = int(clf_model.predict([features])[0])
                    pred_prob = 1.0

        # Se houve gatilho do ESP32 durante este ciclo
        if sim_trigger and pred_label is not None:
            trigger_decision(pred_label, pred_prob)

        # Desenho no Frame Principal
        main_view = frame.copy()
        box_color = (0, 255, 0) if source_str == "YOLO" else (0, 140, 255)
        cv2.rectangle(main_view, (x1, y1), (x2, y2), box_color, 2)
        cv2.putText(main_view, f"ROI: {source_str}", (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, box_color, 2, cv2.LINE_AA)

        # Painel Superior com Status
        cv2.rectangle(main_view, (0, 0), (main_view.shape[1], 40), (20, 20, 20), -1)
        accel_info = "OpenVINO GPU" if getattr(pdi_pipeline, "is_openvino", False) else "PyTorch"
        mode_text = f"YOLO: ON [{accel_info}, Skip {pdi_pipeline.skip_frames}x]" if pdi_pipeline.use_yolo else "YOLO: OFF (Center Crop)"
        cv2.putText(main_view, f"FPS: {fps:.1f} | {mode_text} | Cam: {camera_sources[current_cam_idx]}",
                    (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)

        # Predição no rodapé do frame principal
        if pred_label is not None:
            pred_text = f"{config.CLASS_NAMES[pred_label]} ({pred_prob * 100:.1f}%)"
            text_color = (0, 255, 0) if pred_label == 0 else (255, 200, 0)
        elif clf_model is None:
            pred_text = "Modo Coleta/Debug (Pressione 't' para treinar)"
            text_color = (0, 215, 255)
        else:
            pred_text = "Objeto nao identificado"
            text_color = (0, 0, 255)

        cv2.rectangle(main_view, (0, main_view.shape[0] - 40), (main_view.shape[1], main_view.shape[0]), (20, 20, 20), -1)
        cv2.putText(main_view, pred_text, (10, main_view.shape[0] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2, cv2.LINE_AA)

        # Mensagem de Notificação Temporária (Toast)
        if time.time() < status_expiry:
            cv2.rectangle(main_view, (0, 45), (main_view.shape[1], 80), (0, 100, 0), -1)
            cv2.putText(main_view, status_msg, (15, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

        # Padroniza dimensões para o grid unificado
        target_h, target_w = 480, 640
        main_resized = cv2.resize(main_view, (target_w, target_h), interpolation=cv2.INTER_AREA)

        # Monta Grid 2x2 dos 4 Debug Views
        tile_w, tile_h = target_w // 2, target_h // 2
        dummy_tile = np.zeros((tile_h, tile_w, 3), dtype=np.uint8)

        if debug_views:
            t1 = make_tile(debug_views["roi_norm"], "1. ROI Norm (256px)", (tile_w, tile_h))
            t2 = make_tile(debug_views["otsu_raw"], "2. Otsu Duplo (V+S)", (tile_w, tile_h))
            t3 = make_tile(debug_views["morph_closed"], "3. Morfologia Fechamento", (tile_w, tile_h))
            t4 = make_tile(debug_views["solid_mask"], "4. Mascara Solida", (tile_w, tile_h))
        else:
            t1, t2, t3, t4 = dummy_tile, dummy_tile, dummy_tile, dummy_tile

        debug_grid = np.vstack([
            np.hstack([t1, t2]),
            np.hstack([t3, t4])
        ])

        # Painel Final: Frame Principal + Grid de Debug lado a lado
        dashboard = np.hstack([main_resized, debug_grid])
        cv2.imshow("Painel Unificado PDI", dashboard)

        # Captura de Teclado
        key = cv2.waitKey(1) & 0xFF

        if key in (27, ord('q')):
            print("[*] Encerrando aplicacao...")
            break

        elif key == ord('o'):
            # Salva frame completo em dataset/organico
            filename = config.DATASET_ORGANICO / f"organico_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg"
            cv2.imwrite(str(filename), raw_frame)
            status_msg = f"Salvo em Organico: {filename.name}"
            status_expiry = time.time() + 1.5
            print(f"[+] {status_msg}")

        elif key == ord('r'):
            # Salva frame completo em dataset/reciclavel
            filename = config.DATASET_RECICLAVEL / f"reciclavel_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg"
            cv2.imwrite(str(filename), raw_frame)
            status_msg = f"Salvo em Reciclavel: {filename.name}"
            status_expiry = time.time() + 1.5
            print(f"[+] {status_msg}")

        elif key == ord('t'):
            # Treina modelo e recarrega em memória dinamicamente
            status_msg = "Treinando modelo... Aguarde no terminal."
            status_expiry = time.time() + 2.0
            print("\n[*] Disparando treinamento do classificador...")
            trained_model = train_model.train()
            if trained_model is not None:
                clf_model = trained_model
                status_msg = f"Modelo {config.MODEL_TYPE} treinado e carregado com sucesso!"
                status_expiry = time.time() + 3.0
            else:
                status_msg = "Falha no treino: verifique amostras no dataset."
                status_expiry = time.time() + 3.0

        elif key == ord(' '):
            # Simula gatilho ultrassônico
            if pred_label is not None:
                trigger_decision(pred_label, pred_prob)
            else:
                status_msg = "Gatilho acionado, mas nenhuma predicao disponivel."
                status_expiry = time.time() + 2.0
                print(f"[!] {status_msg}")

        elif key == ord('y'):
            # Alterna YOLO / Fallback
            pdi_pipeline.use_yolo = not pdi_pipeline.use_yolo
            mode = "YOLO Ativado" if pdi_pipeline.use_yolo else "YOLO Desativado (Center Fallback)"
            status_msg = f"Modo alterado: {mode}"
            status_expiry = time.time() + 1.5
            print(f"[*] {mode}")

        elif key == ord('c'):
            # Alterna índice de câmera
            current_cam_idx = (current_cam_idx + 1) % len(camera_sources)
            new_source = camera_sources[current_cam_idx]
            print(f"[*] Alternando para camera: {new_source}")
            cap.release()
            cap = init_camera(new_source)
            status_msg = f"Camera alterada para: {new_source}"
            status_expiry = time.time() + 1.5

    # Limpeza de Recursos
    cap.release()
    cv2.destroyAllWindows()
    if ser and ser.is_open:
        ser.close()
        print("[*] Conexao serial encerrada.")
    print("[✔] Sistema finalizado.")

if __name__ == "__main__":
    main()
