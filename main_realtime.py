import os
# Silencia logs ruidosos do backend C++ do OpenCV
os.environ.setdefault("OPENCV_LOG_LEVEL", "OFF")
# Garante compatibilidade do OpenCV GUI no Linux Wayland/X11 (evita erro de plugin Qt wayland)
os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

import cv2
import time
import joblib
import threading
import numpy as np
from collections import deque
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog

import config
from pipeline import HybridPDIPipeline
import train_model

try:
    import serial
except ImportError:
    serial = None

class CameraThread(threading.Thread):
    """
    Thread dedicada para captura de frames em background.
    Elimina travamentos da interface ao conectar no DroidCam (IP/WiFi) ou V4L2.
    Garante latência zero lendo continuamente o buffer mais recente.
    """
    def __init__(self, source, on_frame_cb, on_error_cb):
        super().__init__(daemon=True)
        self.source = source
        self.on_frame_cb = on_frame_cb
        self.on_error_cb = on_error_cb
        self.running = True
        self.cap = None

    def run(self):
        backend = cv2.CAP_ANY
        # Se for índice numérico e estiver no Linux, usa V4L2
        if isinstance(self.source, int) and hasattr(cv2, "CAP_V4L2") and os.name != "nt":
            backend = cv2.CAP_V4L2

        try:
            self.cap = cv2.VideoCapture(self.source, backend)
            if not self.cap.isOpened() and backend != cv2.CAP_ANY:
                self.cap = cv2.VideoCapture(self.source)
            if self.cap.isOpened():
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception as e:
            self.on_error_cb(f"Exceção ao abrir fonte: {e}")
            return

        if not self.cap.isOpened():
            self.on_error_cb(f"Não foi possível abrir a fonte: {self.source}")
            return

        while self.running:
            try:
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    self.on_frame_cb(frame)
                else:
                    time.sleep(0.01)
            except Exception:
                break

        if self.cap is not None:
            self.cap.release()

    def stop(self):
        self.running = False


class PDIApplication:
    def __init__(self, root):
        self.root = root
        self.root.title("Lixeira Inteligente PDI - Controle (YOLO26n + DroidCam)")
        self.root.geometry("640x760")
        self.root.minsize(580, 680)

        # Pipeline e Modelos
        self.pdi_pipeline = HybridPDIPipeline(
            yolo_weights=config.YOLO_MODEL_PATH,
            use_yolo=True,
            skip_frames=config.YOLO_SKIP_FRAMES,
            imgsz=config.YOLO_IMGSZ,
            device=config.YOLO_DEVICE,
            min_contour_area=1500
        )

        self.clf_model = None
        self.load_classifier()

        # Comunicação Serial
        self.ser = None
        if config.ENABLE_SERIAL and serial:
            try:
                self.ser = serial.Serial(config.SERIAL_PORT, config.BAUD_RATE, timeout=0.01)
                self.log(f"[✔] Serial conectada: {config.SERIAL_PORT}")
            except Exception as e:
                self.log(f"[!] Erro ao abrir serial: {e}")

        # Gerenciamento de Câmera (Inicia DESCONECTADA)
        self.cam_thread = None
        self.is_camera_connected = False
        self.is_static_mode = False
        self.camera_sources = []
        self.latest_frame = None
        self.frame_lock = threading.Lock()
        
        # Métricas de execução
        self.prev_time = time.time()
        self.fps = 0.0
        self.last_pred_label = None
        self.last_pred_prob = 0.0
        self.prob_history = deque(maxlen=4)
        self.status_msg = "Aguardando conexão da câmera"
        self.status_expiry = 0.0

        # Janelas controláveis
        self.window_defs = {
            "main": {"title": "Visao Principal (Lixeira Inteligente)", "default": True},
            "roi": {"title": "1. ROI Normalizada (256px)", "default": False},
            "otsu": {"title": "2. Otsu Duplo (V+S)", "default": False},
            "morph": {"title": "3. Morfologia Fechamento", "default": False},
            "mask": {"title": "4. Mascara Solida", "default": False},
        }
        self.win_vars = {}
        self.win_created = set()

        # Resoluções configuráveis para a janela de vídeo
        self.WINDOW_SIZES = {
            "800x450 (Recomendado Notebook)": (800, 450),
            "640x360 (Compacto)": (640, 360),
            "960x540 (Médio)": (960, 540),
            "1280x720 (HD)": (1280, 720),
            "Nativo (Resolução da Câmera)": None
        }
        self.window_size_var = tk.StringVar(value="800x450 (Recomendado Notebook)")

        # Monta Interface
        self.setup_ui()

        # Popula lista de câmeras locais
        self.reload_cameras()

        # Inicia loop de renderização (acoplado ao Tkinter, ~60 FPS)
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.after(15, self.render_loop)

    def load_classifier(self):
        if config.MODEL_PATH.exists():
            try:
                self.clf_model = joblib.load(config.MODEL_PATH)
                return True
            except Exception:
                self.clf_model = None
        return False

    def setup_ui(self):
        style = ttk.Style()
        style.theme_use("clam")

        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # --- SEÇÃO 1: FONTE DE VÍDEO (DROIDCAM / LOCAL) ---
        cam_frame = ttk.LabelFrame(main_frame, text=" 📹 Conexão da Câmera (Celular / DroidCam / USB) ", padding=8)
        cam_frame.pack(fill=tk.X, pady=(0, 8))

        # Seletor de Tipo de Conexão
        self.source_mode = tk.StringVar(value="local")
        mode_row = ttk.Frame(cam_frame)
        mode_row.pack(fill=tk.X, pady=(0, 4))
        
        rb_local = ttk.Radiobutton(mode_row, text="Dispositivo Local / V4L2 (/dev/video*)",
                                   variable=self.source_mode, value="local", command=self.on_mode_change)
        rb_local.pack(side=tk.LEFT, padx=(0, 16))
        
        rb_droid = ttk.Radiobutton(mode_row, text="DroidCam WiFi / IP (URL de Rede)",
                                   variable=self.source_mode, value="ip", command=self.on_mode_change)
        rb_droid.pack(side=tk.LEFT)

        # Container para Câmera Local
        self.frame_local_cfg = ttk.Frame(cam_frame)
        self.frame_local_cfg.pack(fill=tk.X, pady=2)
        ttk.Label(self.frame_local_cfg, text="Dispositivo:").pack(side=tk.LEFT, padx=(0, 6))
        self.cam_combo = ttk.Combobox(self.frame_local_cfg, state="readonly", width=34)
        self.cam_combo.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 6))
        ttk.Button(self.frame_local_cfg, text="🔄 Atualizar", command=self.reload_cameras, width=10).pack(side=tk.LEFT)

        # Container para DroidCam IP
        self.frame_ip_cfg = ttk.Frame(cam_frame)
        ttk.Label(self.frame_ip_cfg, text="URL DroidCam:").pack(side=tk.LEFT, padx=(0, 6))
        self.entry_ip = ttk.Entry(self.frame_ip_cfg, width=34)
        self.entry_ip.insert(0, "http://192.168.1.100:4747/video")
        self.entry_ip.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 6))
        ttk.Label(self.frame_ip_cfg, text="(Ex: porta 4747)", font=("Sans", 8, "italic")).pack(side=tk.LEFT)

        # Linha de Ação de Conexão
        act_row = ttk.Frame(cam_frame)
        act_row.pack(fill=tk.X, pady=(6, 0))

        self.btn_connect = tk.Button(act_row, text="▶ Conectar Câmera", bg="#28a745", fg="white",
                                     font=("Sans", 9, "bold"), relief="raised", padx=12, pady=4,
                                     command=self.toggle_camera_connection)
        self.btn_connect.pack(side=tk.LEFT, padx=(0, 6))

        self.btn_static = ttk.Button(act_row, text="📂 Testar Imagem", command=self.load_static_image)
        self.btn_static.pack(side=tk.LEFT, padx=(0, 10))

        self.lbl_cam_status = ttk.Label(act_row, text="Câmera: DESCONECTADA (clique em Conectar)",
                                        font=("Sans", 9, "bold"), foreground="#d9534f")
        self.lbl_cam_status.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # --- SEÇÃO 2: VISUALIZAÇÃO E JANELAS PDI ---
        win_frame = ttk.LabelFrame(main_frame, text=" 🪟 Visualização e Filtros PDI (Janelas Separadas) ", padding=8)
        win_frame.pack(fill=tk.X, pady=(0, 8))

        self.win_vars["main"] = tk.BooleanVar(value=True)
        cb_main = ttk.Checkbutton(win_frame, text="🖥  Janela Principal (Câmera + YOLO26n + Diagnóstico)",
                                  variable=self.win_vars["main"], command=lambda: self.on_toggle_window("main"))
        cb_main.pack(anchor="w", pady=(0, 2))

        # Seletor de Resolução / Tamanho da Janela de Vídeo
        res_row = ttk.Frame(win_frame)
        res_row.pack(fill=tk.X, pady=(2, 6))
        ttk.Label(res_row, text="📐 Tamanho da Janela de Vídeo:").pack(side=tk.LEFT, padx=(0, 6))
        combo_res = ttk.Combobox(res_row, textvariable=self.window_size_var,
                                 values=list(self.WINDOW_SIZES.keys()), state="readonly", width=28)
        combo_res.pack(side=tk.LEFT)
        combo_res.bind("<<ComboboxSelected>>", self.on_window_size_changed)

        sep = ttk.Separator(win_frame, orient="horizontal")
        sep.pack(fill=tk.X, pady=4)

        ttk.Label(win_frame, text="Filtros Intermediários (ativáveis sob demanda):",
                  font=("Sans", 8, "italic")).pack(anchor="w", pady=(0, 4))

        filters_grid = ttk.Frame(win_frame)
        filters_grid.pack(fill=tk.X)

        filter_keys = [
            ("roi", "1. ROI Normalizada (256px)", 0, 0),
            ("otsu", "2. Otsu Duplo (V+S)", 0, 1),
            ("morph", "3. Morfologia Fechamento", 1, 0),
            ("mask", "4. Máscara Sólida", 1, 1),
        ]
        for key, text, r, c in filter_keys:
            self.win_vars[key] = tk.BooleanVar(value=False)
            cb = ttk.Checkbutton(filters_grid, text=text, variable=self.win_vars[key],
                                 command=lambda k=key: self.on_toggle_window(k))
            cb.grid(row=r, column=c, sticky="w", padx=8, pady=2)

        # Configuração de Fallback / YOLO
        ctrl_grid = ttk.Frame(win_frame)
        ctrl_grid.pack(fill=tk.X, pady=(6, 0))

        self.var_use_yolo = tk.BooleanVar(value=True)
        cb_yolo = ttk.Checkbutton(ctrl_grid, text="⚡ Detector YOLO26n Ativo",
                                  variable=self.var_use_yolo, command=self.on_toggle_yolo)
        cb_yolo.pack(side=tk.LEFT, padx=(0, 14))

        self.var_allow_fallback = tk.BooleanVar(value=True)
        cb_fb = ttk.Checkbutton(ctrl_grid, text="🛡 Fallback Central Inteligente (com validação de área)",
                                variable=self.var_allow_fallback)
        cb_fb.pack(side=tk.LEFT)

        # --- SEÇÃO 3: COLETA DE DADOS REALISTAS ---
        collect_frame = ttk.LabelFrame(main_frame, text=" 📸 Coleta Direta na Lixeira (Imagens Reais do Celular) ", padding=8)
        collect_frame.pack(fill=tk.X, pady=(0, 8))

        col_row = ttk.Frame(collect_frame)
        col_row.pack(fill=tk.X)

        self.btn_save_org = tk.Button(col_row, text="🍌 Salvar Orgânico (Banana) [Tecla O]", bg="#d4edda", fg="#155724",
                                      font=("Sans", 9, "bold"), relief="raised", padx=8, pady=4,
                                      command=lambda: self.capture_sample(0))
        self.btn_save_org.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))

        self.btn_save_rec = tk.Button(col_row, text="🍾 Salvar Reciclável (Garrafa) [Tecla R]", bg="#cce5ff", fg="#004085",
                                      font=("Sans", 9, "bold"), relief="raised", padx=8, pady=4,
                                      command=lambda: self.capture_sample(1))
        self.btn_save_rec.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(4, 0))

        self.lbl_dataset_count = ttk.Label(collect_frame, text="Contagem: Carregando...", font=("Sans", 8))
        self.lbl_dataset_count.pack(anchor="w", pady=(6, 0))
        self.update_dataset_counts()

        # --- SEÇÃO 4: TREINAMENTO & GATILHO ---
        train_frame = ttk.LabelFrame(main_frame, text=" 🧠 Classificador & Teste de Gatilho ", padding=8)
        train_frame.pack(fill=tk.X, pady=(0, 8))

        tr_row = ttk.Frame(train_frame)
        tr_row.pack(fill=tk.X)

        ttk.Label(tr_row, text="Modelo: SVM RBF (20 Features PDI)", font=("Sans", 9, "bold")).pack(side=tk.LEFT, padx=(0, 10))

        self.btn_train = tk.Button(tr_row, text="🚀 Treinar Modelo Agora [Tecla T]", bg="#fff3cd", fg="#856404",
                                   font=("Sans", 9, "bold"), relief="raised", padx=10, pady=4,
                                   command=self.start_training_thread)
        self.btn_train.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.btn_trigger = ttk.Button(train_frame, text="🎯 Simular Gatilho Ultrassônico / ESP32 [Barra de Espaço]",
                                      command=self.manual_trigger)
        self.btn_trigger.pack(fill=tk.X, pady=(6, 0))

        # --- SEÇÃO 5: CONSOLE DE LOGS ---
        log_frame = ttk.LabelFrame(main_frame, text=" 📋 Console de Logs & Status ", padding=6)
        log_frame.pack(fill=tk.BOTH, expand=True)

        self.txt_log = scrolledtext.ScrolledText(log_frame, wrap=tk.WORD, height=7, font=("Monospace", 8))
        self.txt_log.pack(fill=tk.BOTH, expand=True)

        # Atalhos globais
        self.root.bind("<Key-o>", lambda e: self.capture_sample(0))
        self.root.bind("<Key-r>", lambda e: self.capture_sample(1))
        self.root.bind("<Key-t>", lambda e: self.start_training_thread())
        self.root.bind("<space>", lambda e: self.manual_trigger())

        self.log("[✔] Painel de Controle carregado com sucesso.")
        self.log("[*] Dica: Escolha a câmera (USB ou DroidCam WiFi) e clique em 'Conectar Câmera'.")

    def log(self, text):
        def _append():
            self.txt_log.insert(tk.END, text + "\n")
            self.txt_log.see(tk.END)
        self.root.after(0, _append)

    def on_mode_change(self):
        if self.source_mode.get() == "local":
            self.frame_ip_cfg.pack_forget()
            self.frame_local_cfg.pack(fill=tk.X, pady=2)
        else:
            self.frame_local_cfg.pack_forget()
            self.frame_ip_cfg.pack(fill=tk.X, pady=2)

    def reload_cameras(self):
        self.camera_sources = config.scan_available_cameras()
        if not self.camera_sources:
            self.camera_sources = [(0, "video0: Padrão")]
        display_values = [name for _, name in self.camera_sources]
        self.cam_combo["values"] = display_values
        self.cam_combo.current(0)
        self.log(f"[*] Câmeras locais detectadas: {len(self.camera_sources)}")

    def toggle_camera_connection(self):
        if self.is_camera_connected:
            self.disconnect_camera()
        else:
            self.connect_camera()

    def connect_camera(self):
        self.is_static_mode = False
        if self.source_mode.get() == "local":
            sel_idx = self.cam_combo.current()
            if sel_idx < 0 or sel_idx >= len(self.camera_sources):
                messagebox.showwarning("Aviso", "Selecione uma câmera local válida.")
                return
            source = self.camera_sources[sel_idx][0]
            display_name = self.camera_sources[sel_idx][1]
        else:
            source = self.entry_ip.get().strip()
            if not source:
                messagebox.showwarning("Aviso", "Digite a URL do DroidCam (ex: http://192.168.1.100:4747/video).")
                return
            display_name = f"DroidCam ({source})"

        self.btn_connect.config(state="disabled", text="⏳ Conectando...")
        self.lbl_cam_status.config(text=f"Conectando a {display_name}...", foreground="#ec971f")
        self.log(f"[*] Tentando conexão assíncrona com: {display_name}...")

        def _on_frame(frame):
            with self.frame_lock:
                self.latest_frame = frame

        def _on_error(msg):
            def _ui():
                self.log(f"[!] {msg}")
                self.disconnect_camera()
                messagebox.showerror("Erro de Conexão", f"{msg}\n\nVerifique se o DroidCam está ativo no celular ou se o cabo USB está conectado.")
            self.root.after(0, _ui)

        self.cam_thread = CameraThread(source, _on_frame, _on_error)
        self.cam_thread.start()

        # Checa em 1 segundo se começou a receber frames
        def _verify():
            if self.cam_thread and self.cam_thread.running:
                self.is_camera_connected = True
                self.btn_connect.config(state="normal", text="⏹ Desconectar Câmera", bg="#dc3545")
                self.lbl_cam_status.config(text=f"Câmera CONECTADA: {display_name}", foreground="#28a745")
                self.log(f"[✔] Câmera conectada e transmitindo em tempo real.")
            else:
                self.btn_connect.config(state="normal", text="▶ Conectar Câmera", bg="#28a745")
        self.root.after(1200, _verify)

    def load_static_image(self):
        filepath = filedialog.askopenfilename(
            initialdir=str(config.BASE_DIR),
            title="Selecione uma imagem para teste PDI",
            filetypes=[("Imagens", "*.jpg *.jpeg *.png *.bmp *.webp")]
        )
        if not filepath:
            return
        img = cv2.imread(filepath)
        if img is None:
            messagebox.showerror("Erro", "Não foi possível abrir a imagem selecionada.")
            return
        if self.is_camera_connected:
            self.disconnect_camera()
        with self.frame_lock:
            self.latest_frame = img
        self.is_static_mode = True
        self.lbl_cam_status.config(text=f"Modo Estático: {Path(filepath).name}", foreground="#0275d8")
        self.log(f"[*] Imagem estática carregada: {Path(filepath).name}")

    def disconnect_camera(self):
        self.is_camera_connected = False
        self.is_static_mode = False
        if self.cam_thread is not None:
            self.cam_thread.stop()
            self.cam_thread = None
        with self.frame_lock:
            self.latest_frame = None

        self.btn_connect.config(state="normal", text="▶ Conectar Câmera", bg="#28a745")
        self.lbl_cam_status.config(text="Câmera: DESCONECTADA", foreground="#d9534f")
        self.log("[*] Câmera desconectada.")
        self.close_all_opencv_windows()

    def close_all_opencv_windows(self):
        for k in list(self.win_created):
            title = self.window_defs[k]["title"]
            try:
                cv2.destroyWindow(title)
            except Exception:
                pass
        self.win_created.clear()

    def on_toggle_window(self, win_key):
        is_active = self.win_vars[win_key].get()
        title = self.window_defs[win_key]["title"]
        if not is_active:
            try:
                cv2.destroyWindow(title)
            except Exception:
                pass
            if win_key in self.win_created:
                self.win_created.remove(win_key)

    def on_window_size_changed(self, event=None):
        target = self.WINDOW_SIZES.get(self.window_size_var.get())
        main_title = self.window_defs["main"]["title"]
        if "main" in self.win_created and target is not None:
            try:
                cv2.resizeWindow(main_title, target[0], target[1])
            except Exception:
                pass
        self.log(f"[*] Resolução da janela ajustada para: {self.window_size_var.get()}")

    def on_toggle_yolo(self):
        self.pdi_pipeline.use_yolo = self.var_use_yolo.get()
        state = "ATIVADO" if self.pdi_pipeline.use_yolo else "DESATIVADO"
        self.log(f"[*] Detector YOLO26n {state}.")

    def update_dataset_counts(self):
        org = len(list(config.DATASET_ORGANICO.glob("*.jpg")) + list(config.DATASET_ORGANICO.glob("*.png")))
        rec = len(list(config.DATASET_RECICLAVEL.glob("*.jpg")) + list(config.DATASET_RECICLAVEL.glob("*.png")))
        self.lbl_dataset_count.config(
            text=f"Amostras salvas: 🍌 Orgânico = {org}  |  🍾 Reciclável = {rec}  |  Total = {org + rec}"
        )

    def capture_sample(self, label_idx):
        with self.frame_lock:
            if self.latest_frame is None:
                self.log("[!] Nenhum frame disponível. Conecte a câmera primeiro.")
                return
            frame_to_save = self.latest_frame.copy()

        folder = config.DATASET_ORGANICO if label_idx == 0 else config.DATASET_RECICLAVEL
        cls_name = "organico" if label_idx == 0 else "reciclavel"
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = folder / f"{cls_name}_{ts}.jpg"
        cv2.imwrite(str(filename), frame_to_save)
        self.log(f"[+] Amostra salva ({cls_name.upper()}): {filename.name}")
        self.update_dataset_counts()

    def start_training_thread(self):
        self.btn_train.config(state="disabled", text="⏳ Treinando...")
        self.log("\n[*] Iniciando processo de treinamento (SVM RBF)...")

        def _worker():
            try:
                trained = train_model.train(log_fn=self.log)
                if trained is not None:
                    self.clf_model = trained
                    self.log("[✔] Novo classificador recarregado dinamicamente na memória!")
                else:
                    self.log("[!] Falha no treinamento: dados insuficientes.")
            except Exception as e:
                self.log(f"[!] Erro no treinamento: {e}")
            finally:
                def _done():
                    self.btn_train.config(state="normal", text="🚀 Treinar Modelo Agora [Tecla T]")
                    self.update_dataset_counts()
                self.root.after(0, _done)

        t = threading.Thread(target=_worker, daemon=True)
        t.start()

    def manual_trigger(self):
        if self.last_pred_label is not None:
            self.send_serial_decision(self.last_pred_label, self.last_pred_prob)
        else:
            self.log("[!] Gatilho acionado: Lixeira vazia ou predição incerta no momento.")

    def send_serial_decision(self, pred_class, pred_prob):
        class_name = config.CLASS_NAMES.get(pred_class, "Desconhecido")
        msg = f"DECISÃO LIXEIRA: {class_name} ({pred_prob * 100:.1f}%)"
        if self.ser and self.ser.is_open:
            code = config.SERIAL_CODES.get(pred_class)
            if code:
                try:
                    self.ser.write(code)
                    msg += f" -> Serial '{code.decode()}' enviada!"
                except Exception as ex:
                    msg += f" -> Erro serial: {ex}"
        self.log(f"[+] {msg}")
        self.status_msg = msg
        self.status_expiry = time.time() + 3.0

    def check_x11_window_closed(self, win_key, title):
        if win_key in self.win_created:
            try:
                prop = cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE)
                if prop < 1:
                    cv2.destroyWindow(title)
                    self.win_created.remove(win_key)
                    self.win_vars[win_key].set(False)
                    self.log(f"[*] Janela '{title}' fechada pelo usuário.")
            except Exception:
                pass

    def render_loop(self):
        """Loop contínuo de renderização acoplado ao Tkinter."""
        frame = None
        with self.frame_lock:
            if self.latest_frame is not None:
                frame = self.latest_frame.copy()

        is_active = self.is_camera_connected or self.is_static_mode
        if frame is not None and is_active:
            curr_time = time.time()
            self.fps = 0.9 * self.fps + 0.1 * (1.0 / max(curr_time - self.prev_time, 1e-6))
            self.prev_time = curr_time

            self.process_and_render_frame(frame)
        else:
            # Se não houver fonte ativa, garante janelas fechadas
            if not is_active and self.win_created:
                self.close_all_opencv_windows()

        self.root.after(15, self.render_loop)

    def process_and_render_frame(self, frame):
        # 1. Localização via YOLO ou Fallback Central (respeita checkbox)
        roi, bbox, source_str = self.pdi_pipeline.extract_roi(
            frame, allow_fallback=self.var_allow_fallback.get()
        )
        x1, y1, x2, y2 = bbox

        features, debug_views = None, None
        pred_label, pred_prob = None, 0.0
        status_category = "Lixeira Vazia / Sem Objeto"

        if roi is not None and roi.size > 0:
            features, debug_views = self.pdi_pipeline.segment_and_extract(roi)
            
            # Validação anti-falso positivo: se a máscara for insignificante, considera vazio
            if features is None:
                self.prob_history.clear()
                status_category = "Lixeira Vazia (Área insuficiente)"
            else:
                if self.clf_model is not None:
                    # Se for Fallback, verifica se a cor bate com o escopo de banana/garrafa
                    if source_str == "FALLBACK_CENTER" and not getattr(self.pdi_pipeline, "last_color_valid", True):
                        self.prob_history.clear()
                        pred_label = 2  # Indefinido
                        pred_prob = 0.0
                        status_category = "Indefinido (Cor fora do escopo)"
                    else:
                        try:
                            raw_probs = self.clf_model.predict_proba([features])[0]
                            self.prob_history.append(raw_probs)
                            probs = np.mean(self.prob_history, axis=0)
                            best_idx = int(np.argmax(probs))
                            confidence = float(probs[best_idx])
                            # Limiar de certeza estrito (70%) para validar classe alvo
                            if confidence >= 0.70:
                                pred_label = best_idx
                                pred_prob = confidence
                                status_category = f"{config.CLASS_NAMES[pred_label]} ({pred_prob * 100:.1f}%)"
                            else:
                                pred_label = 2  # Indefinido
                                pred_prob = confidence
                                status_category = f"Indefinido (Confiança baixa: {confidence * 100:.1f}%)"
                        except Exception:
                            pred_label = int(self.clf_model.predict([features])[0])
                            pred_prob = 1.0
                            status_category = f"{config.CLASS_NAMES[pred_label]}"
                else:
                    status_category = "Objeto Detectado (Aguardando Treino)"
        else:
            self.prob_history.clear()

        self.last_pred_label = pred_label
        self.last_pred_prob = pred_prob

        # 2. Janela Principal
        if self.win_vars["main"].get():
            main_title = self.window_defs["main"]["title"]
            main_view = frame.copy()

            # Cor da Bounding Box: verde se banana, azul se garrafa, cinza/laranja se indefinido
            if pred_label == 0:
                box_color = (0, 255, 0)
            elif pred_label == 1:
                box_color = (255, 200, 0)
            elif pred_label == 2:
                box_color = (0, 140, 255)
            else:
                box_color = (120, 120, 120)

            # Desenha ROI
            cv2.rectangle(main_view, (x1, y1), (x2, y2), box_color, 2)
            cv2.putText(main_view, f"ROI: {source_str}", (x1, max(20, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, box_color, 2, cv2.LINE_AA)

            # Barra superior informativa
            cv2.rectangle(main_view, (0, 0), (main_view.shape[1], 38), (20, 20, 20), -1)
            cv2.putText(main_view, f"FPS: {self.fps:.1f} | {status_category}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.60, (230, 230, 230), 2, cv2.LINE_AA)

            # Toast temporário de gatilho
            if time.time() < self.status_expiry:
                cv2.rectangle(main_view, (0, 42), (main_view.shape[1], 78), (0, 100, 0), -1)
                cv2.putText(main_view, self.status_msg, (15, 68),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

            # Redimensionamento inteligente conforme tamanho configurado
            target_res = self.WINDOW_SIZES.get(self.window_size_var.get())
            if target_res is not None:
                display_view = cv2.resize(main_view, target_res, interpolation=cv2.INTER_AREA)
            else:
                display_view = main_view

            if "main" not in self.win_created:
                cv2.namedWindow(main_title, cv2.WINDOW_NORMAL)
                if target_res is not None:
                    cv2.resizeWindow(main_title, target_res[0], target_res[1])
                self.win_created.add("main")

            cv2.imshow(main_title, display_view)
            self.check_x11_window_closed("main", main_title)

        # 3. Janelas Individuais dos Filtros
        if debug_views:
            views_map = {
                "roi": debug_views.get("roi_norm"),
                "otsu": debug_views.get("otsu_raw"),
                "morph": debug_views.get("morph_closed"),
                "mask": debug_views.get("solid_mask"),
            }
            for k, img in views_map.items():
                title = self.window_defs[k]["title"]
                if self.win_vars[k].get() and img is not None:
                    cv2.imshow(title, img)
                    self.win_created.add(k)
                    self.check_x11_window_closed(k, title)
                elif not self.win_vars[k].get() and k in self.win_created:
                    try:
                        cv2.destroyWindow(title)
                        self.win_created.remove(k)
                    except Exception:
                        pass

        # Captura de Teclado no OpenCV
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q')):
            self.win_vars["main"].set(False)
            self.on_toggle_window("main")
        elif key == ord('o'):
            self.capture_sample(0)
        elif key == ord('r'):
            self.capture_sample(1)
        elif key == ord('t'):
            self.start_training_thread()
        elif key == ord(' '):
            self.manual_trigger()

    def on_closing(self):
        self.disconnect_camera()
        self.close_all_opencv_windows()
        if self.ser and self.ser.is_open:
            try:
                self.ser.close()
            except Exception:
                pass
        self.root.destroy()


def main():
    root = tk.Tk()
    app = PDIApplication(root)
    root.mainloop()

if __name__ == "__main__":
    main()
