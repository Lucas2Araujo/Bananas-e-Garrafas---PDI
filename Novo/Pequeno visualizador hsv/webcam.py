import glob
import os
import threading
import time
import tkinter as tk
from tkinter import messagebox, ttk

import cv2
from PIL import Image, ImageTk


def escanear_cameras_locais(max_index=5):
    """Retorna lista de tuplas (indice, nome_amigavel) para webcams locais."""
    cameras = []
    if os.name != "nt" and os.path.isdir("/sys/class/video4linux"):
        for dev_path in sorted(glob.glob("/sys/class/video4linux/video*")):
            try:
                base = os.path.basename(dev_path)
                idx = int(base.replace("video", ""))
                name_file = os.path.join(dev_path, "name")
                with open(name_file, encoding="utf-8", errors="ignore") as f:
                    name = f.read().strip() if os.path.exists(name_file) else f"Camera {idx}"
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


def escolher_indice_padrao(cameras):
    """Seleciona o índice na lista priorizando câmeras físicas."""
    if not cameras:
        return 0
    for i, (_idx, nome) in enumerate(cameras):
        nome_lower = nome.lower()
        if "virtual" not in nome_lower and "dummy" not in nome_lower and "obs" not in nome_lower:
            return i
    return 0


class CameraThread(threading.Thread):
    """Thread dedicada para captura contínua de frames locais sem travar a UI."""

    def __init__(self, source_idx, on_frame_cb, on_error_cb):
        super().__init__(daemon=True)
        self.source_idx = source_idx
        self.on_frame_cb = on_frame_cb
        self.on_error_cb = on_error_cb
        self.running = True
        self.cap = None

    def run(self):
        backend = cv2.CAP_V4L2 if hasattr(cv2, "CAP_V4L2") and os.name != "nt" else cv2.CAP_ANY
        try:
            self.cap = cv2.VideoCapture(self.source_idx, backend)
            if not self.cap.isOpened() and backend != cv2.CAP_ANY:
                self.cap = cv2.VideoCapture(self.source_idx)
            if self.cap.isOpened():
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception as exc:
            if self.running:
                self.on_error_cb(f"Erro ao abrir câmera {self.source_idx}: {exc}")
            return

        if not self.cap.isOpened():
            if self.running:
                self.on_error_cb(f"Não foi possível abrir a câmera {self.source_idx}.")
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


class JanelaWebcam(tk.Toplevel):
    """Janela dedicada para seleção de webcam local, preview ao vivo e captura de frames."""

    def __init__(self, parent, destino_label, on_capturar_cb, on_fechar_cb=None):
        super().__init__(parent)
        self.title(f"Webcam ao Vivo — {destino_label}")
        self.geometry("720x600")
        self.minsize(560, 480)
        self.transient(parent)

        self.on_capturar_cb = on_capturar_cb
        self.on_fechar_cb = on_fechar_cb

        self.camera_sources = []
        self.cam_thread = None
        self.latest_frame = None
        self.frame_lock = threading.Lock()
        self._photo_ref = None
        self._after_id = None
        self._fechada = False
        self._nome_cam_atual = ""

        self._setup_ui()
        self.protocol("WM_DELETE_WINDOW", self.fechar)

        self.atualizar_cameras(conectar=True)
        self._atualizar_preview()

    def _setup_ui(self):
        topo = tk.Frame(self, padx=12, pady=10, bg="#f8fafc", bd=1, relief=tk.GROOVE)
        topo.pack(fill=tk.X)

        tk.Label(topo, text="Dispositivo:", font=("Arial", 10, "bold"), bg="#f8fafc").pack(side=tk.LEFT, padx=(0, 6))

        self.cam_combo = ttk.Combobox(topo, state="readonly", width=36)
        self.cam_combo.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 8))
        self.cam_combo.bind("<<ComboboxSelected>>", self._ao_trocar_camera)

        tk.Button(
            topo, text="🔄 Atualizar", command=lambda: self.atualizar_cameras(conectar=True),
            font=("Arial", 9, "bold"), bg="#e2e8f0", padx=10, pady=4, cursor="hand2",
        ).pack(side=tk.LEFT)

        self.lbl_status = tk.Label(
            self, text="Procurando webcams locais...", font=("Consolas", 9),
            bg="#f1f5f9", fg="#334155", anchor=tk.W, padx=12, pady=4,
        )
        self.lbl_status.pack(fill=tk.X)

        self.frame_preview = tk.Frame(self, bg="#111827")
        self.frame_preview.pack(expand=True, fill=tk.BOTH, padx=12, pady=10)

        self.lbl_preview = tk.Label(
            self.frame_preview, text="Aguardando vídeo da webcam...",
            fg="#9ca3af", bg="#111827", font=("Arial", 11),
        )
        self.lbl_preview.pack(expand=True, fill=tk.BOTH)

        rodape = tk.Frame(self, padx=12, pady=10)
        rodape.pack(fill=tk.X)

        self.btn_capturar = tk.Button(
            rodape, text="📸 Capturar Frame", command=self.capturar_frame,
            font=("Arial", 11, "bold"), bg="#4CAF50", fg="white",
            padx=18, pady=7, cursor="hand2",
        )
        self.btn_capturar.pack(side=tk.LEFT)

        tk.Button(
            rodape, text="Fechar", command=self.fechar,
            font=("Arial", 10), bg="#64748b", fg="white",
            padx=14, pady=7, cursor="hand2",
        ).pack(side=tk.RIGHT)

    def atualizar_cameras(self, conectar=True):
        self._parar_thread()
        self.camera_sources = escanear_cameras_locais()

        if not self.camera_sources:
            self.cam_combo["values"] = ["Nenhuma webcam detectada"]
            self.cam_combo.current(0)
            self.btn_capturar.config(state=tk.DISABLED)
            self.lbl_status.config(text="Nenhuma webcam local encontrada. Conecte uma câmera e clique em Atualizar.", fg="#b91c1c")
            return

        self.btn_capturar.config(state=tk.NORMAL)
        self.cam_combo["values"] = [nome for _idx, nome in self.camera_sources]
        self.cam_combo.current(escolher_indice_padrao(self.camera_sources))

        if conectar:
            self._conectar_selecionada()

    def _ao_trocar_camera(self, _event=None):
        if self.camera_sources:
            self._conectar_selecionada()

    def _conectar_selecionada(self):
        sel_idx = self.cam_combo.current()
        if sel_idx < 0 or sel_idx >= len(self.camera_sources):
            return

        self._parar_thread()
        source_idx, display_name = self.camera_sources[sel_idx]
        self._nome_cam_atual = display_name
        self.lbl_status.config(text=f"Conectando a {display_name}...", fg="#b45309")

        def _on_frame(frame):
            with self.frame_lock:
                self.latest_frame = frame

        def _on_error(msg):
            def _ui():
                if self._fechada:
                    return
                self.lbl_status.config(text=f"Erro: {msg}", fg="#b91c1c")
                self.lbl_preview.config(image="", text=msg)
                self._photo_ref = None
            self.after(0, _ui)

        self.cam_thread = CameraThread(source_idx, _on_frame, _on_error)
        self.cam_thread.start()

    def _atualizar_preview(self):
        if self._fechada:
            return

        frame = None
        with self.frame_lock:
            if self.latest_frame is not None:
                frame = self.latest_frame.copy()

        if frame is not None:
            w_box = max(320, self.frame_preview.winfo_width() - 8)
            h_box = max(240, self.frame_preview.winfo_height() - 8)
            h_img, w_img = frame.shape[:2]
            escala = min(w_box / w_img, h_box / h_img)
            novo_w = max(1, int(w_img * escala))
            novo_h = max(1, int(h_img * escala))

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(rgb).resize((novo_w, novo_h), Image.Resampling.LANCZOS)
            self._photo_ref = ImageTk.PhotoImage(img_pil)
            self.lbl_preview.config(image=self._photo_ref, text="")
            self.lbl_status.config(text=f"Transmitindo ao vivo: {self._nome_cam_atual} ({w_img}x{h_img})", fg="#15803d")

        self._after_id = self.after(30, self._atualizar_preview)

    def capturar_frame(self):
        frame = None
        with self.frame_lock:
            if self.latest_frame is not None:
                frame = self.latest_frame.copy()

        if frame is None:
            messagebox.showwarning("Aviso", "Nenhum frame disponível ainda. Aguarde a câmera iniciar.", parent=self)
            return

        self.on_capturar_cb(frame, self._nome_cam_atual)
        self.lbl_status.config(text=f"✔ Frame capturado de {self._nome_cam_atual} e enviado ao workspace!", fg="#1d4ed8")

    def _parar_thread(self):
        if self.cam_thread is not None:
            self.cam_thread.stop()
            self.cam_thread = None
        with self.frame_lock:
            self.latest_frame = None

    def fechar(self):
        if self._fechada:
            return
        self._fechada = True
        if self._after_id is not None:
            try:
                self.after_cancel(self._after_id)
            except Exception:
                pass
            self._after_id = None
        self._parar_thread()
        if self.on_fechar_cb is not None:
            self.on_fechar_cb()
        self.destroy()
