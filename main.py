import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import cv2

from componentes import PainelImagem
from processamento import processar_imagem
from visualizador import JanelaZoom


class AnalisadorCoresApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Analisador e Comparador de Espaços de Cor (RGB & HSV)")
        self.root.geometry("1440x860")

        self.dados_imagens = {"ind": {}, "comp_a": {}, "comp_b": {}}

        # Variáveis de controle dos Toggles
        self.modo_cinza_rgb = tk.BooleanVar(value=False)
        self.modo_foco_hsv = tk.BooleanVar(value=False)

        self.notebook = ttk.Notebook(root)
        self.notebook.pack(expand=True, fill="both", padx=10, pady=(10, 0))

        self.tab_individual = tk.Frame(self.notebook)
        self.notebook.add(self.tab_individual, text="  Análise Individual (RGB + HSV + Histogramas)  ")
        self._setup_aba_individual()

        self.tab_comparativo = tk.Frame(self.notebook)
        self.notebook.add(self.tab_comparativo, text="  Modo Comparativo (Imagem A vs Imagem B)  ")
        self._setup_aba_comparativo()

        self.lbl_status = tk.Label(
            root,
            text="Passe o mouse sobre uma imagem para inspecionar pixels ou clique em 🔍 (ou duplo clique) para dar Zoom.",
            bd=1,
            relief=tk.SUNKEN,
            anchor=tk.W,
            font=("Consolas", 10),
            bg="#f0f0f0",
            padx=10,
            pady=5,
        )
        self.lbl_status.pack(side=tk.BOTTOM, fill=tk.X)

    # =========================================================================
    # SETUP DAS ABAS
    # =========================================================================
    def _setup_aba_individual(self):
        topo = tk.Frame(self.tab_individual)
        topo.pack(pady=10)

        tk.Button(
            topo,
            text="📂 Selecionar Imagem",
            command=lambda: self.carregar_imagem("ind"),
            font=("Arial", 11, "bold"),
            bg="#4CAF50",
            fg="white",
            padx=15,
            pady=5,
        ).pack(side=tk.LEFT, padx=8)

        tk.Button(
            topo,
            text="📦 Baixar Visíveis (Lote)",
            command=lambda: self.baixar_lote(["ind"]),
            font=("Arial", 11, "bold"),
            bg="#607D8B",
            fg="white",
            padx=15,
            pady=5,
        ).pack(side=tk.LEFT, padx=8)

        tk.Checkbutton(
            topo,
            text="Canais R, G, B em Tons de Cinza",
            variable=self.modo_cinza_rgb,
            command=self._alternar_modo_canais_rgb,
            font=("Arial", 10, "bold"),
            bg="#e0e0e0",
            activebackground="#d5d5d5",
            padx=10,
            pady=5,
            relief=tk.RAISED,
            cursor="hand2",
        ).pack(side=tk.LEFT, padx=10)

        tk.Checkbutton(
            topo,
            text="🎯 Foco: Original + Canais HSV",
            variable=self.modo_foco_hsv,
            command=self._alternar_modo_foco_hsv,
            font=("Arial", 10, "bold"),
            bg="#e3f2fd",
            activebackground="#bbdefb",
            padx=10,
            pady=5,
            relief=tk.RAISED,
            cursor="hand2",
        ).pack(side=tk.LEFT, padx=10)

        self.grid_ind = tk.Frame(self.tab_individual)
        self.grid_ind.pack(expand=True, fill="both", padx=12, pady=5)

        itens = [
            ("rgb", "Original (RGB)"),
            ("r", "Canal R (Red)"),
            ("g", "Canal G (Green)"),
            ("b", "Canal B (Blue)"),
            ("hist_rgb", "Histograma RGB"),
            ("hsv", "HSV Completo"),
            ("h", "Canal H (Matiz)"),
            ("s", "Canal S (Saturação)"),
            ("v", "Canal V (Brilho)"),
            ("hist_hsv", "Histograma HSV"),
        ]

        self.paineis_ind = {}
        for chave, titulo in itens:
            self.paineis_ind[chave] = PainelImagem(
                self.grid_ind, titulo, "ind", chave, self.abrir_zoom, self.baixar_individual, self.inspecionar_pixel
            )

        self._aplicar_layout_individual()

    def _setup_aba_comparativo(self):
        topo = tk.Frame(self.tab_comparativo)
        topo.pack(pady=8)

        tk.Button(
            topo,
            text="📂 Carregar Imagem A",
            command=lambda: self.carregar_imagem("comp_a"),
            font=("Arial", 10, "bold"),
            bg="#2196F3",
            fg="white",
            padx=12,
            pady=5,
        ).pack(side=tk.LEFT, padx=8)

        tk.Button(
            topo,
            text="📂 Carregar Imagem B",
            command=lambda: self.carregar_imagem("comp_b"),
            font=("Arial", 10, "bold"),
            bg="#FF9800",
            fg="white",
            padx=12,
            pady=5,
        ).pack(side=tk.LEFT, padx=8)

        tk.Button(
            topo,
            text="📦 Baixar Visíveis (Lote)",
            command=lambda: self.baixar_lote(["comp_a", "comp_b"]),
            font=("Arial", 10, "bold"),
            bg="#607D8B",
            fg="white",
            padx=12,
            pady=5,
        ).pack(side=tk.LEFT, padx=10)

        tk.Checkbutton(
            topo,
            text="🎯 Foco: Original + Canais HSV",
            variable=self.modo_foco_hsv,
            command=self._alternar_modo_foco_hsv,
            font=("Arial", 10, "bold"),
            bg="#e3f2fd",
            activebackground="#bbdefb",
            padx=10,
            pady=5,
            relief=tk.RAISED,
            cursor="hand2",
        ).pack(side=tk.LEFT, padx=10)

        canvas = tk.Canvas(self.tab_comparativo)
        scroll_y = ttk.Scrollbar(self.tab_comparativo, orient="vertical", command=canvas.yview)
        scroll_x = ttk.Scrollbar(self.tab_comparativo, orient="horizontal", command=canvas.xview)
        self.scroll_frame_comp = tk.Frame(canvas)

        self.scroll_frame_comp.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=self.scroll_frame_comp, anchor="nw")
        canvas.configure(yscrollcommand=scroll_y.set, xscrollcommand=scroll_x.set)

        scroll_y.pack(side="right", fill="y")
        scroll_x.pack(side="bottom", fill="x")
        canvas.pack(side="left", fill="both", expand=True, padx=8)

        colunas = [
            ("rgb", "RGB"),
            ("hist_rgb", "Hist. RGB"),
            ("hsv", "HSV"),
            ("h", "Canal H"),
            ("s", "Canal S"),
            ("v", "Canal V"),
            ("hist_hsv", "Hist. HSV"),
        ]

        self.paineis_comp_a = {}
        self.paineis_comp_b = {}

        for chave, nome in colunas:
            self.paineis_comp_a[chave] = PainelImagem(
                self.scroll_frame_comp, f"Img A - {nome}", "comp_a", chave, self.abrir_zoom, self.baixar_individual, self.inspecionar_pixel
            )
            self.paineis_comp_b[chave] = PainelImagem(
                self.scroll_frame_comp, f"Img B - {nome}", "comp_b", chave, self.abrir_zoom, self.baixar_individual, self.inspecionar_pixel
            )

        self._aplicar_layout_comparativo()

    # =========================================================================
    # CONTROLE DINÂMICO DE LAYOUT (MODO COMPLETO vs FOCO HSV)
    # =========================================================================
    def _aplicar_layout_individual(self):
        for painel in self.paineis_ind.values():
            painel.grid_forget()

        for c in range(5):
            self.grid_ind.columnconfigure(c, weight=0)
        for r in range(2):
            self.grid_ind.rowconfigure(r, weight=0)

        if self.modo_foco_hsv.get():
            chaves_visiveis = ["rgb", "h", "s", "v"]
            for c, chave in enumerate(chaves_visiveis):
                self.grid_ind.columnconfigure(c, weight=1)
                self.paineis_ind[chave].grid(row=0, column=c, padx=6, pady=6, sticky="nsew")
            self.grid_ind.rowconfigure(0, weight=1)
        else:
            layout_completo = [
                ("rgb", 0, 0), ("r", 0, 1), ("g", 0, 2), ("b", 0, 3), ("hist_rgb", 0, 4),
                ("hsv", 1, 0), ("h", 1, 1), ("s", 1, 2), ("v", 1, 3), ("hist_hsv", 1, 4),
            ]
            for c in range(5):
                self.grid_ind.columnconfigure(c, weight=1)
            for r in range(2):
                self.grid_ind.rowconfigure(r, weight=1)
            for chave, r, c in layout_completo:
                self.paineis_ind[chave].grid(row=r, column=c, padx=4, pady=4, sticky="nsew")

    def _aplicar_layout_comparativo(self):
        for painel in list(self.paineis_comp_a.values()) + list(self.paineis_comp_b.values()):
            painel.grid_forget()

        for c in range(7):
            self.scroll_frame_comp.columnconfigure(c, weight=0)

        chaves_visiveis = ["rgb", "h", "s", "v"] if self.modo_foco_hsv.get() else [
            "rgb", "hist_rgb", "hsv", "h", "s", "v", "hist_hsv"
        ]

        for col_idx, chave in enumerate(chaves_visiveis):
            self.scroll_frame_comp.columnconfigure(col_idx, weight=1)
            self.paineis_comp_a[chave].grid(row=0, column=col_idx, padx=4, pady=4, sticky="nsew")
            self.paineis_comp_b[chave].grid(row=1, column=col_idx, padx=4, pady=4, sticky="nsew")

    def _alternar_modo_foco_hsv(self):
        self._aplicar_layout_individual()
        self._aplicar_layout_comparativo()

        # Re-renderiza miniaturas com tamanho maior quando no modo focado
        for destino in ("ind", "comp_a", "comp_b"):
            if self.dados_imagens[destino]:
                self._atualizar_paineis(destino)

    def _obter_tamanho_thumb(self, destino):
        if self.modo_foco_hsv.get():
            return (330, 420) if destino == "ind" else (310, 260)
        return (245, 195) if destino == "ind" else (180, 145)

    # =========================================================================
    # AÇÕES E GERENCIAMENTO DE DADOS
    # =========================================================================
    def carregar_imagem(self, destino):
        caminho = filedialog.askopenfilename(
            title="Escolha uma imagem",
            filetypes=[("Imagens", "*.png *.jpg *.jpeg *.bmp *.tiff *.webp")],
        )
        if not caminho:
            return

        variacoes = processar_imagem(caminho)
        if variacoes is None:
            messagebox.showerror("Erro", "Não foi possível abrir a imagem.")
            return

        self.dados_imagens[destino] = {
            "nome_base": os.path.splitext(os.path.basename(caminho))[0],
            "shape": variacoes["rgb"].shape[:2],
            "variacoes": variacoes,
        }
        self._atualizar_paineis(destino)

    def _alternar_modo_canais_rgb(self):
        if self.dados_imagens["ind"]:
            self._atualizar_paineis("ind")

    def _obter_matriz(self, chave_origem, chave_var):
        variacoes = self.dados_imagens[chave_origem]["variacoes"]
        if chave_var in ("r", "g", "b"):
            sufixo = "_gray" if self.modo_cinza_rgb.get() else "_color"
            return variacoes[f"{chave_var}{sufixo}"]
        return variacoes[chave_var]

    def _atualizar_paineis(self, destino):
        tamanho_thumb = self._obter_tamanho_thumb(destino)
        paineis = (
            self.paineis_ind
            if destino == "ind"
            else (self.paineis_comp_a if destino == "comp_a" else self.paineis_comp_b)
        )
        for chave, painel in paineis.items():
            painel.atualizar_miniatura(self._obter_matriz(destino, chave), tamanho_thumb)

    def abrir_zoom(self, chave_origem, chave_var, titulo):
        if not self.dados_imagens[chave_origem]:
            messagebox.showwarning("Aviso", "Carregue uma imagem primeiro.")
            return

        matriz = self._obter_matriz(chave_origem, chave_var)
        is_hist = chave_var.startswith("hist_")
        m_rgb = None if is_hist else self.dados_imagens[chave_origem]["variacoes"]["rgb"]
        m_hsv = None if is_hist else self.dados_imagens[chave_origem]["variacoes"]["hsv"]

        JanelaZoom(self.root, titulo, matriz, m_rgb, m_hsv)

    # =========================================================================
    # DOWNLOADS E INSPEÇÃO
    # =========================================================================
    def baixar_individual(self, chave_origem, chave_var):
        if not self.dados_imagens[chave_origem]:
            messagebox.showwarning("Aviso", "Nenhuma imagem carregada neste painel.")
            return

        nome_base = self.dados_imagens[chave_origem]["nome_base"]
        prefixo = {"ind": "", "comp_a": "ImgA_", "comp_b": "ImgB_"}[chave_origem]
        sufixo_modo = ("_cinza" if self.modo_cinza_rgb.get() else "_colorido") if chave_var in ("r", "g", "b") else ""

        caminho_salvar = filedialog.asksaveasfilename(
            title="Salvar Variação",
            initialfile=f"{prefixo}{nome_base}_{chave_var}{sufixo_modo}.png",
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg"), ("BMP", "*.bmp")],
        )
        if not caminho_salvar:
            return

        self._salvar_disco(self._obter_matriz(chave_origem, chave_var), caminho_salvar)
        messagebox.showinfo("Sucesso", f"Imagem salva em:\n{caminho_salvar}")

    def baixar_lote(self, lista_origens):
        origens_validas = [orig for orig in lista_origens if self.dados_imagens[orig]]
        if not origens_validas:
            messagebox.showwarning("Aviso", "Carregue pelo menos uma imagem antes de baixar em lote.")
            return

        pasta_destino = filedialog.askdirectory(title="Selecione a pasta para salvar as variações visíveis")
        if not pasta_destino:
            return

        if self.modo_foco_hsv.get():
            chaves_aba = ["rgb", "h", "s", "v"]
        else:
            chaves_aba = list(self.paineis_ind.keys()) if lista_origens == ["ind"] else list(self.paineis_comp_a.keys())

        total_salvos = 0
        for orig in origens_validas:
            nome_base = self.dados_imagens[orig]["nome_base"]
            prefixo = {"ind": "", "comp_a": "ImgA_", "comp_b": "ImgB_"}[orig]

            for chave in chaves_aba:
                sufixo_modo = ("_cinza" if self.modo_cinza_rgb.get() else "_colorido") if chave in ("r", "g", "b") else ""
                caminho = os.path.join(pasta_destino, f"{prefixo}{nome_base}_{chave}{sufixo_modo}.png")
                self._salvar_disco(self._obter_matriz(orig, chave), caminho)
                total_salvos += 1

        messagebox.showinfo("Concluído", f"{total_salvos} arquivos salvos em:\n{pasta_destino}")

    def _salvar_disco(self, matriz, caminho):
        matriz_bgr = cv2.cvtColor(matriz, cv2.COLOR_RGB2BGR) if len(matriz.shape) == 3 else matriz
        cv2.imwrite(caminho, matriz_bgr)

    def inspecionar_pixel(self, event, chave_origem, painel):
        if not self.dados_imagens[chave_origem] or not painel.display_size:
            return

        disp_w, disp_h = painel.display_size
        orig_h, orig_w = self.dados_imagens[chave_origem]["shape"]

        offset_x = (painel.lbl_img.winfo_width() - disp_w) // 2
        offset_y = (painel.lbl_img.winfo_height() - disp_h) // 2

        mouse_x = event.x - offset_x
        mouse_y = event.y - offset_y

        if 0 <= mouse_x < disp_w and 0 <= mouse_y < disp_h:
            real_x = int(mouse_x * (orig_w / disp_w))
            real_y = int(mouse_y * (orig_h / disp_h))

            r, g, b = self.dados_imagens[chave_origem]["variacoes"]["rgb"][real_y, real_x]
            h, s, v = self.dados_imagens[chave_origem]["variacoes"]["hsv"][real_y, real_x]

            nome_img = {"ind": "Imagem", "comp_a": "Imagem A", "comp_b": "Imagem B"}[chave_origem]
            self.lbl_status.config(
                text=f"[{nome_img}] Posição (X={real_x}, Y={real_y})  |  "
                     f"RGB: ({r:3d}, {g:3d}, {b:3d})  |  "
                     f"HSV (OpenCV): (H={h:3d}°, S={s:3d}, V={v:3d})"
            )


if __name__ == "__main__":
    janela = tk.Tk()
    app = AnalisadorCoresApp(janela)
    janela.mainloop()