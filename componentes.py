import tkinter as tk
from PIL import Image, ImageTk


class PainelImagem(tk.Frame):
    """Card reutilizável com cabeçalho, botão de Zoom, botão de Download e inspeção de pixel."""

    def __init__(self, parent, titulo, chave_origem, chave_var, on_zoom, on_download, on_hover):
        super().__init__(parent, bd=1, relief=tk.GROOVE, bg="#fafafa")
        self.titulo = titulo
        self.chave_origem = chave_origem
        self.chave_var = chave_var
        self.display_size = None

        # Cabeçalho
        header = tk.Frame(self, bg="#e8e8e8")
        header.pack(fill="x")

        tk.Label(
            header, text=titulo, font=("Arial", 8, "bold"), bg="#e8e8e8", anchor="w"
        ).pack(side=tk.LEFT, padx=5, pady=2)

        # Botão Download Individual
        tk.Button(
            header,
            text="💾",
            command=lambda: on_download(chave_origem, chave_var),
            font=("Arial", 8),
            bg="#ffffff",
            relief=tk.FLAT,
            cursor="hand2",
            padx=4,
        ).pack(side=tk.RIGHT, padx=2, pady=1)

        # Botão Zoom
        tk.Button(
            header,
            text="🔍",
            command=lambda: on_zoom(chave_origem, chave_var, titulo),
            font=("Arial", 8),
            bg="#ffffff",
            relief=tk.FLAT,
            cursor="hand2",
            padx=4,
        ).pack(side=tk.RIGHT, padx=2, pady=1)

        # Área da imagem
        self.lbl_img = tk.Label(
            self, text="Aguardando...", fg="gray", bg="#fafafa", cursor="crosshair"
        )
        self.lbl_img.pack(expand=True, fill="both", padx=3, pady=3)

        # Duplo clique também abre o zoom
        self.lbl_img.bind("<Double-Button-1>", lambda e: on_zoom(chave_origem, chave_var, titulo))

        if not chave_var.startswith("hist_"):
            self.lbl_img.bind("<Motion>", lambda e: on_hover(e, chave_origem, self))

    def atualizar_miniatura(self, matriz_img, tamanho_thumb):
        pil_img = Image.fromarray(matriz_img)
        pil_img.thumbnail(tamanho_thumb, Image.Resampling.LANCZOS)

        tk_img = ImageTk.PhotoImage(pil_img)
        self.lbl_img.config(image=tk_img, text="")
        self.lbl_img.image = tk_img
        self.display_size = pil_img.size