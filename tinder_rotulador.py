import json
import logging
import shutil
import threading
import tkinter as tk
from tkinter import ttk, messagebox
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageTk

from download_datasets import VALID_EXTS, compute_dhash, download_batch, generate_catalog


STAGING_DIR = Path("dataset_staging")
DATASET_DIR = Path("dataset")
MASKS_DIR = DATASET_DIR / "masks"
REJECTED_FILE = STAGING_DIR / "rejected_hashes.txt"

CLASS_TO_LABEL = {
    "organico": "Banana",
    "reciclavel": "Garrafa",
}

TARGET_MAP = {
    "Ambas": "ambas",
    "Só Bananas": "organico",
    "Só Garrafas": "reciclavel",
}


def get_next_index(cls_dir: Path) -> int:
    """Retorna o próximo índice numérico disponível na pasta da classe."""
    max_idx = 2000
    if cls_dir.exists():
        for p in cls_dir.iterdir():
            if p.is_file() and p.stem.isdigit():
                max_idx = max(max_idx, int(p.stem))
    return max_idx + 1


def extract_auto_polygon(img_bgr: np.ndarray, v_thresh: int = 25) -> tuple[list[list[float]], np.ndarray]:
    """Extrai automaticamente o polígono do objeto principal em fundo escuro."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    v_channel = hsv[:, :, 2]

    _, mask = cv2.threshold(v_channel, v_thresh, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return [], mask

    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < 200:
        return [], mask

    epsilon = 0.003 * cv2.arcLength(largest, True)
    approx = cv2.approxPolyDP(largest, epsilon, True)
    points = [[float(pt[0][0]), float(pt[0][1])] for pt in approx]

    clean_mask = np.zeros_like(mask)
    cv2.drawContours(clean_mask, [largest], -1, 255, thickness=cv2.FILLED)
    return points, clean_mask


def save_labelme_json(
    json_path: Path,
    image_filename: str,
    label: str,
    points: list[list[float]],
    height: int,
    width: int,
) -> None:
    """Salva anotação no formato LabelMe compatível com o dataset existente."""
    data = {
        "version": "7.11.0",
        "flags": {},
        "shapes": [
            {
                "label": label,
                "points": points,
                "group_id": None,
                "description": "",
                "shape_type": "polygon",
                "flags": {},
                "mask": None,
            }
        ],
        "imagePath": image_filename,
        "imageData": None,
        "imageHeight": height,
        "imageWidth": width,
    }
    json_path.write_text(json.dumps(data, indent=2), encoding="utf-8")


class TinderRotuladorApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Dataset Studio: Download + Tinder + Auto-Rotulação")
        self.root.geometry("960x780")
        self.root.configure(bg="#1e1e1e")

        self.queue: list[tuple[Path, str]] = []
        self.history: list[dict] = []
        self.current_img_bgr: np.ndarray | None = None
        self.current_points: list[list[float]] = []
        self.current_mask: np.ndarray | None = None
        self.photo_ref = None
        self.is_downloading = False

        self._cleanup_undo_files()
        self._load_queue()
        self._build_ui()
        self._bind_keys()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        if self.queue:
            self._show_current()
        else:
            self._show_empty()

    def _cleanup_undo_files(self):
        """Remove arquivos temporários .undo_* deixados em dataset_staging."""
        if STAGING_DIR.exists():
            for p in STAGING_DIR.glob(".undo_*"):
                if p.is_file():
                    p.unlink(missing_ok=True)

    def _on_close(self):
        self._cleanup_undo_files()
        self.root.destroy()

    def _load_queue(self):
        existing_paths = {p for p, _ in self.queue}
        for cls_name in ("organico", "reciclavel"):
            cls_staging = STAGING_DIR / cls_name
            if cls_staging.exists():
                for p in sorted(cls_staging.iterdir()):
                    if p.is_file() and p.suffix.lower() in VALID_EXTS and p not in existing_paths:
                        self.queue.append((p, cls_name))

    def _build_ui(self):
        # Barra de Download Integrada
        dl_frame = tk.Frame(self.root, bg="#252526", bd=1, relief=tk.SOLID)
        dl_frame.pack(fill=tk.X, padx=16, pady=(12, 6))

        tk.Label(dl_frame, text="Baixar:", fg="#cccccc", bg="#252526", font=("Segoe UI", 10, "bold")).pack(
            side=tk.LEFT, padx=(10, 4), pady=8
        )

        self.target_var = tk.StringVar(value="Ambas")
        self.combo_target = ttk.Combobox(
            dl_frame,
            textvariable=self.target_var,
            values=list(TARGET_MAP.keys()),
            state="readonly",
            width=13,
        )
        self.combo_target.pack(side=tk.LEFT, padx=4, pady=8)

        tk.Label(dl_frame, text="Busca (opcional):", fg="#cccccc", bg="#252526", font=("Segoe UI", 10)).pack(
            side=tk.LEFT, padx=(10, 4), pady=8
        )

        self.entry_query = tk.Entry(dl_frame, bg="#3c3c3c", fg="#ffffff", insertbackground="#ffffff", width=28)
        self.entry_query.pack(side=tk.LEFT, padx=4, pady=8, fill=tk.X, expand=True)

        self.btn_download = tk.Button(
            dl_frame,
            text="⬇ Baixar mais",
            command=self.start_download,
            bg="#1565c0",
            fg="white",
            font=("Segoe UI", 10, "bold"),
            padx=12,
            pady=4,
            relief=tk.FLAT,
        )
        self.btn_download.pack(side=tk.LEFT, padx=(6, 6), pady=8)

        tk.Label(dl_frame, text="Qtd/busca:", fg="#cccccc", bg="#252526", font=("Segoe UI", 9)).pack(
            side=tk.LEFT, padx=(4, 2), pady=8
        )
        self.qty_var = tk.IntVar(value=10)
        self.spin_qty = tk.Spinbox(
            dl_frame,
            from_=1,
            to=50,
            textvariable=self.qty_var,
            width=4,
            bg="#3c3c3c",
            fg="#ffffff",
            buttonbackground="#555555",
        )
        self.spin_qty.pack(side=tk.LEFT, padx=(0, 8), pady=8)

        self.progress = ttk.Progressbar(dl_frame, mode="indeterminate", length=110)
        self.progress.pack(side=tk.RIGHT, padx=(4, 10), pady=8)

        self.lbl_dl_status = tk.Label(dl_frame, text="", fg="#64b5f6", bg="#252526", font=("Segoe UI", 9))
        self.lbl_dl_status.pack(side=tk.RIGHT, padx=6)

        # Painel compacto de log de download
        self.log_text = tk.Text(
            self.root,
            height=4,
            bg="#141414",
            fg="#90caf9",
            font=("Consolas", 9),
            bd=0,
            padx=8,
            pady=4,
            state=tk.DISABLED,
        )
        self.log_text.pack(fill=tk.X, padx=16, pady=(0, 4))

        # Barra de Status da Imagem Atual
        top_frame = tk.Frame(self.root, bg="#1e1e1e")
        top_frame.pack(fill=tk.X, padx=16, pady=(4, 4))

        self.lbl_status = tk.Label(
            top_frame, text="", fg="#ffffff", bg="#1e1e1e", font=("Segoe UI", 11, "bold")
        )
        self.lbl_status.pack(side=tk.LEFT)

        self.lbl_class = tk.Label(
            top_frame, text="", fg="#ffd54f", bg="#1e1e1e", font=("Segoe UI", 12, "bold")
        )
        self.lbl_class.pack(side=tk.RIGHT)

        # Área da Imagem
        self.canvas = tk.Label(self.root, bg="#111111")
        self.canvas.pack(expand=True, fill=tk.BOTH, padx=16, pady=6)

        # Controles Inferiores
        ctrl_frame = tk.Frame(self.root, bg="#2b2b2b")
        ctrl_frame.pack(fill=tk.X, padx=16, pady=(6, 12))

        slider_frame = tk.Frame(ctrl_frame, bg="#2b2b2b")
        slider_frame.pack(fill=tk.X, padx=12, pady=(8, 4))

        tk.Label(
            slider_frame,
            text="Sensibilidade do Contorno (Threshold Fundo Preto):",
            fg="#cccccc",
            bg="#2b2b2b",
            font=("Segoe UI", 10),
        ).pack(side=tk.LEFT)

        self.thresh_var = tk.IntVar(value=25)
        self.slider = ttk.Scale(
            slider_frame,
            from_=5,
            to=100,
            variable=self.thresh_var,
            command=self._on_thresh_change,
        )
        self.slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=12)

        self.lbl_thresh = tk.Label(
            slider_frame, text="25", fg="#ffffff", bg="#2b2b2b", width=4, font=("Segoe UI", 10, "bold")
        )
        self.lbl_thresh.pack(side=tk.RIGHT)

        btn_frame = tk.Frame(ctrl_frame, bg="#2b2b2b")
        btn_frame.pack(fill=tk.X, padx=12, pady=(4, 10))

        tk.Button(
            btn_frame,
            text="✕ Rejeitar (← / A)",
            command=self.reject_current,
            bg="#c62828",
            fg="white",
            font=("Segoe UI", 10, "bold"),
            padx=12,
            pady=8,
            relief=tk.FLAT,
        ).pack(side=tk.LEFT, padx=3)

        tk.Button(
            btn_frame,
            text="↩ Desfazer (Z)",
            command=self.undo_last,
            bg="#546e7a",
            fg="white",
            font=("Segoe UI", 10),
            padx=10,
            pady=8,
            relief=tk.FLAT,
        ).pack(side=tk.LEFT, padx=3)

        tk.Button(
            btn_frame,
            text="⇄ Trocar Classe (C)",
            command=self.toggle_class,
            bg="#6a1b9a",
            fg="white",
            font=("Segoe UI", 10),
            padx=10,
            pady=8,
            relief=tk.FLAT,
        ).pack(side=tk.LEFT, padx=3)

        tk.Button(
            btn_frame,
            text="✓ Aprovar + Rotular (→ / D / Enter)",
            command=lambda: self.approve_current(with_label=True),
            bg="#2e7d32",
            fg="white",
            font=("Segoe UI", 10, "bold"),
            padx=12,
            pady=8,
            relief=tk.FLAT,
        ).pack(side=tk.RIGHT, padx=3)

        tk.Button(
            btn_frame,
            text="Aprovar sem rótulo (S)",
            command=lambda: self.approve_current(with_label=False),
            bg="#ef6c00",
            fg="white",
            font=("Segoe UI", 10),
            padx=10,
            pady=8,
            relief=tk.FLAT,
        ).pack(side=tk.RIGHT, padx=3)

    def _bind_keys(self):
        def _ignore_if_typing(fn):
            def wrapper(event):
                if self.root.focus_get() == self.entry_query:
                    return
                fn()
            return wrapper

        self.root.bind("<Left>", _ignore_if_typing(self.reject_current))
        self.root.bind("a", _ignore_if_typing(self.reject_current))
        self.root.bind("<BackSpace>", _ignore_if_typing(self.reject_current))

        self.root.bind("<Right>", _ignore_if_typing(lambda: self.approve_current(with_label=True)))
        self.root.bind("d", _ignore_if_typing(lambda: self.approve_current(with_label=True)))
        self.root.bind("<Return>", _ignore_if_typing(lambda: self.approve_current(with_label=True)))

        self.root.bind("s", _ignore_if_typing(lambda: self.approve_current(with_label=False)))
        self.root.bind("z", _ignore_if_typing(self.undo_last))
        self.root.bind("c", _ignore_if_typing(self.toggle_class))

    def _append_log(self, msg: str):
        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, msg + "\n")
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)

    def start_download(self):
        if self.is_downloading:
            return
        self.is_downloading = True
        self.btn_download.config(state=tk.DISABLED, text="Baixando...")
        self.progress.start(12)
        target = TARGET_MAP[self.target_var.get()]
        custom_q = self.entry_query.get().strip()
        try:
            qty = max(1, int(self.qty_var.get()))
        except ValueError:
            qty = 10

        self._append_log(f"--- Iniciando download ({self.target_var.get()}, {qty}/busca) ---")

        class UILogHandler(logging.Handler):
            def __init__(self, emit_fn):
                super().__init__()
                self.emit_fn = emit_fn

            def emit(self, record):
                msg = self.format(record)
                if "Response status code" in msg or "timed out" in msg or "ERROR" in msg:
                    short = msg.split(" - ")[-1][:120]
                    self.emit_fn(f"[!] {short}")

        handler = UILogHandler(lambda m: self.root.after(0, lambda msg=m: self._append_log(msg)))
        dl_logger = logging.getLogger("downloader")
        dl_logger.addHandler(handler)

        def on_progress(msg: str):
            self.root.after(0, lambda m=msg: (self.lbl_dl_status.config(text=m), self._append_log(f"[*] {m}")))

        def worker():
            try:
                org_kept, rec_kept = download_batch(
                    target=target,
                    custom_query=custom_q,
                    samples_per_query=qty,
                    progress_cb=on_progress,
                )
                self.root.after(0, lambda: self._on_download_done(org_kept, rec_kept))
            except Exception as exc:
                self.root.after(0, lambda e=exc: self._on_download_error(str(e)))
            finally:
                dl_logger.removeHandler(handler)

        threading.Thread(target=worker, daemon=True).start()

    def _on_download_done(self, org_kept: int, rec_kept: int):
        self.is_downloading = False
        self.progress.stop()
        self.btn_download.config(state=tk.NORMAL, text="⬇ Baixar mais")
        self.lbl_dl_status.config(text=f"+{org_kept} Bananas, +{rec_kept} Garrafas")
        self._append_log(f"✓ Concluído: +{org_kept} Bananas, +{rec_kept} Garrafas prontas.")
        self._load_queue()
        if self.current_img_bgr is None and self.queue:
            self._show_current()
        elif self.queue:
            img_path, _ = self.queue[0]
            self.lbl_status.config(text=f"Restantes: {len(self.queue)} | Arquivo: {img_path.name}")

    def _on_download_error(self, err: str):
        self.is_downloading = False
        self.progress.stop()
        self.btn_download.config(state=tk.NORMAL, text="⬇ Baixar mais")
        self.lbl_dl_status.config(text="Erro no download")
        self._append_log(f"✕ Erro: {err}")
        messagebox.showerror("Erro ao baixar", err)

    def _on_thresh_change(self, _val=None):
        val = int(self.thresh_var.get())
        self.lbl_thresh.config(text=str(val))
        if self.current_img_bgr is not None:
            self._update_preview()

    def _show_current(self):
        if not self.queue:
            self._show_empty()
            return

        img_path, cls_name = self.queue[0]
        self.current_img_bgr = cv2.imread(str(img_path))
        if self.current_img_bgr is None:
            self.queue.pop(0)
            self._show_current()
            return

        label = CLASS_TO_LABEL[cls_name]
        self.lbl_status.config(text=f"Restantes: {len(self.queue)} | Arquivo: {img_path.name}")
        self.lbl_class.config(text=f"Classe: {cls_name.upper()} ({label})")
        self._update_preview()

    def _update_preview(self):
        if self.current_img_bgr is None:
            return

        thresh = int(self.thresh_var.get())
        self.current_points, self.current_mask = extract_auto_polygon(self.current_img_bgr, thresh)

        vis = self.current_img_bgr.copy()
        if self.current_points:
            pts_np = np.array(self.current_points, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(vis, [pts_np], isClosed=True, color=(0, 255, 0), thickness=2)

        vis_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(vis_rgb)
        pil_img.thumbnail((900, 500), Image.Resampling.LANCZOS)

        self.photo_ref = ImageTk.PhotoImage(pil_img)
        self.canvas.config(image=self.photo_ref, text="")

    def _show_empty(self):
        self.current_img_bgr = None
        self.canvas.config(
            image="",
            text="Nenhuma imagem na fila.\nClique em '⬇ Baixar mais' acima para buscar novas imagens!",
            fg="#81c784",
            font=("Segoe UI", 14, "bold"),
        )
        self.lbl_status.config(text="Restantes: 0")
        self.lbl_class.config(text="")
        generate_catalog(DATASET_DIR)

    def toggle_class(self):
        if not self.queue:
            return
        img_path, cls_name = self.queue[0]
        new_cls = "reciclavel" if cls_name == "organico" else "organico"
        self.queue[0] = (img_path, new_cls)
        label = CLASS_TO_LABEL[new_cls]
        self.lbl_class.config(text=f"Classe: {new_cls.upper()} ({label})")

    def approve_current(self, with_label: bool = True):
        if not self.queue or self.current_img_bgr is None:
            return

        img_path, cls_name = self.queue.pop(0)
        dst_dir = DATASET_DIR / cls_name
        dst_dir.mkdir(parents=True, exist_ok=True)
        mask_dir = MASKS_DIR / cls_name
        mask_dir.mkdir(parents=True, exist_ok=True)

        next_idx = get_next_index(dst_dir)
        new_stem = f"{next_idx:06d}"
        dst_img = dst_dir / f"{new_stem}.jpg"

        cv2.imwrite(str(dst_img), self.current_img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
        h, w = self.current_img_bgr.shape[:2]

        created_files = [dst_img]
        if with_label and self.current_points:
            dst_json = dst_dir / f"{new_stem}.json"
            save_labelme_json(
                dst_json, dst_img.name, CLASS_TO_LABEL[cls_name], self.current_points, h, w
            )
            created_files.append(dst_json)

            if self.current_mask is not None:
                dst_mask = mask_dir / f"{new_stem}.png"
                cv2.imwrite(str(dst_mask), self.current_mask)
                created_files.append(dst_mask)

        backup_path = STAGING_DIR / f".undo_{img_path.name}"
        shutil.move(str(img_path), str(backup_path))

        self.history.append({
            "action": "approve",
            "src": img_path,
            "backup": backup_path,
            "cls": cls_name,
            "created": created_files,
        })
        self._show_current()

    def reject_current(self):
        if not self.queue or self.current_img_bgr is None:
            return

        img_path, cls_name = self.queue.pop(0)
        h = compute_dhash(self.current_img_bgr)
        REJECTED_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(REJECTED_FILE, "a", encoding="utf-8") as f:
            f.write(f"{h}\n")

        backup_path = STAGING_DIR / f".undo_{img_path.name}"
        shutil.move(str(img_path), str(backup_path))

        self.history.append({
            "action": "reject",
            "src": img_path,
            "backup": backup_path,
            "cls": cls_name,
        })
        self._show_current()

    def undo_last(self):
        if not self.history:
            messagebox.showinfo("Desfazer", "Nenhuma ação para desfazer.")
            return

        last = self.history.pop()
        if last["backup"].exists():
            shutil.move(str(last["backup"]), str(last["src"]))
        if last["action"] == "approve":
            for f in last["created"]:
                f.unlink(missing_ok=True)

        self.queue.insert(0, (last["src"], last["cls"]))
        self._show_current()


def main():
    root = tk.Tk()
    app = TinderRotuladorApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
