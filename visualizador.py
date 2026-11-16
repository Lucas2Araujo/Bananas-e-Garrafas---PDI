import tkinter as tk
from PIL import Image, ImageTk


class JanelaZoom(tk.Toplevel):
    """Janela flutuante com suporte a Zoom (roda do mouse) e Pan (arrastar)."""

    def __init__(self, parent, titulo, matriz_img, matriz_rgb=None, matriz_hsv=None):
        super().__init__(parent)
        self.title(f"Inspeção Detalhada - {titulo}")
        self.geometry("950x720")

        self.pil_original = Image.fromarray(matriz_img)
        self.orig_w, self.orig_h = self.pil_original.size
        self.matriz_rgb = matriz_rgb
        self.matriz_hsv = matriz_hsv

        self.escala = 1.0
        self.img_id = None
        self.tk_img = None

        self._criar_barra_ferramentas()
        self._criar_area_canvas()

        # Ajusta automaticamente à janela após renderizar
        self.after(50, self.ajustar_tela)

    def _criar_barra_ferramentas(self):
        barra = tk.Frame(self, bg="#e8e8e8", bd=1, relief=tk.RAISED)
        barra.pack(side=tk.TOP, fill=tk.X)

        tk.Button(barra, text="➕ Zoom In", command=lambda: self.aplicar_zoom(1.25), padx=8).pack(side=tk.LEFT, padx=4, pady=4)
        tk.Button(barra, text="➖ Zoom Out", command=lambda: self.aplicar_zoom(0.8), padx=8).pack(side=tk.LEFT, padx=4, pady=4)
        tk.Button(barra, text="1:1 (100%)", command=self.tamanho_real, padx=8).pack(side=tk.LEFT, padx=4, pady=4)
        tk.Button(barra, text="⛶ Ajustar à Tela", command=self.ajustar_tela, padx=8).pack(side=tk.LEFT, padx=4, pady=4)

        self.lbl_zoom = tk.Label(barra, text="Zoom: 100%", font=("Consolas", 10, "bold"), bg="#e8e8e8")
        self.lbl_zoom.pack(side=tk.RIGHT, padx=12)

        self.lbl_pixel = tk.Label(
            self,
            text="Use a roda do mouse para dar Zoom e clique/arraste para mover a imagem.",
            bd=1,
            relief=tk.SUNKEN,
            anchor=tk.W,
            font=("Consolas", 10),
            bg="#f4f4f4",
            padx=10,
            pady=4,
        )
        self.lbl_pixel.pack(side=tk.BOTTOM, fill=tk.X)

    def _criar_area_canvas(self):
        frame_canvas = tk.Frame(self)
        frame_canvas.pack(expand=True, fill=tk.BOTH)

        self.canvas = tk.Canvas(frame_canvas, bg="#222222", cursor="fleur")
        self.canvas.pack(expand=True, fill=tk.BOTH)

        # Eventos de Zoom (Windows/macOS e Linux) e Pan
        self.canvas.bind("<MouseWheel>", self._on_mousewheel)
        self.canvas.bind("<Button-4>", lambda e: self.aplicar_zoom(1.2, e.x, e.y))
        self.canvas.bind("<Button-5>", lambda e: self.aplicar_zoom(0.8, e.x, e.y))

        self.canvas.bind("<ButtonPress-1>", self._iniciar_pan)
        self.canvas.bind("<B1-Motion>", self._mover_pan)
        self.canvas.bind("<Motion>", self._inspecionar_pixel)

    def _renderizar(self):
        nova_w = max(1, int(self.orig_w * self.escala))
        nova_h = max(1, int(self.orig_h * self.escala))

        # Usa NEAREST quando muito ampliado para ver os pixels individuais com nitidez
        filtro = Image.Resampling.NEAREST if self.escala > 3.0 else Image.Resampling.BILINEAR
        redimensionada = self.pil_original.resize((nova_w, nova_h), filtro)
        self.tk_img = ImageTk.PhotoImage(redimensionada)

        if self.img_id is None:
            self.img_id = self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_img)
        else:
            self.canvas.itemconfig(self.img_id, image=self.tk_img)

        self.lbl_zoom.config(text=f"Zoom: {int(self.escala * 100)}% ({self.orig_w}x{self.orig_h})")

    def aplicar_zoom(self, fator, cx=None, cy=None):
        nova_escala = max(0.05, min(25.0, self.escala * fator))
        if nova_escala == self.escala:
            return

        if cx is None or cy is None:
            cx = self.canvas.winfo_width() / 2
            cy = self.canvas.winfo_height() / 2

        coords = self.canvas.coords(self.img_id) if self.img_id else [0, 0]
        raz = nova_escala / self.escala

        novo_x = cx - (cx - coords[0]) * raz
        novo_y = cy - (cy - coords[1]) * raz

        self.escala = nova_escala
        self._renderizar()
        self.canvas.coords(self.img_id, novo_x, novo_y)

    def tamanho_real(self):
        self.escala = 1.0
        self._renderizar()
        cx = max(0, (self.canvas.winfo_width() - self.orig_w) // 2)
        cy = max(0, (self.canvas.winfo_height() - self.orig_h) // 2)
        self.canvas.coords(self.img_id, cx, cy)

    def ajustar_tela(self):
        cw = max(100, self.canvas.winfo_width() - 20)
        ch = max(100, self.canvas.winfo_height() - 20)
        self.escala = min(cw / self.orig_w, ch / self.orig_h)
        self._renderizar()

        disp_w = int(self.orig_w * self.escala)
        disp_h = int(self.orig_h * self.escala)
        self.canvas.coords(self.img_id, (self.canvas.winfo_width() - disp_w) // 2, (self.canvas.winfo_height() - disp_h) // 2)

    def _on_mousewheel(self, event):
        fator = 1.2 if event.delta > 0 else 0.8
        self.aplicar_zoom(fator, event.x, event.y)

    def _iniciar_pan(self, event):
        self.canvas.scan_mark(event.x, event.y)
        self._pan_start = (event.x, event.y)

    def _mover_pan(self, event):
        dx = event.x - self._pan_start[0]
        dy = event.y - self._pan_start[1]
        self.canvas.move(self.img_id, dx, dy)
        self._pan_start = (event.x, event.y)

    def _inspecionar_pixel(self, event):
        if self.matriz_rgb is None or self.img_id is None:
            return

        img_x, img_y = self.canvas.coords(self.img_id)
        real_x = int((event.x - img_x) / self.escala)
        real_y = int((event.y - img_y) / self.escala)

        if 0 <= real_x < self.orig_w and 0 <= real_y < self.orig_h:
            r, g, b = self.matriz_rgb[real_y, real_x]
            h, s, v = self.matriz_hsv[real_y, real_x]
            self.lbl_pixel.config(
                text=f"Pixel (X={real_x}, Y={real_y})  |  RGB: ({r:3d}, {g:3d}, {b:3d})  |  HSV: (H={h:3d}°, S={s:3d}, V={v:3d})"
            )