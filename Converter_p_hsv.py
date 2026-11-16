import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import cv2
import numpy as np
from PIL import Image, ImageTk


class AnalisadorCoresApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Analisador e Comparador de Espaços de Cor (RGB & HSV + Histogramas)")
        self.root.geometry("1420x850")

        # Armazena as matrizes em resolução original
        self.dados_imagens = {"ind": {}, "comp_a": {}, "comp_b": {}}

        # Variável de controle do Toggle (False = RGB Colorido por padrão | True = Tons de Cinza)
        self.modo_cinza_rgb = tk.BooleanVar(value=False)

        # Sistema de Abas
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(expand=True, fill="both", padx=10, pady=(10, 0))

        # --- ABA 1: Análise Individual ---
        self.tab_individual = tk.Frame(self.notebook)
        self.notebook.add(self.tab_individual, text="  Análise Individual (RGB + HSV + Histogramas)  ")
        self.setup_aba_individual()

        # --- ABA 2: Modo Comparativo ---
        self.tab_comparativo = tk.Frame(self.notebook)
        self.notebook.add(self.tab_comparativo, text="  Modo Comparativo (Imagem A vs Imagem B)  ")
        self.setup_aba_comparativo()

        # --- Barra de Status Inferior ---
        self.lbl_status = tk.Label(
            root,
            text="Passe o mouse sobre qualquer imagem para inspecionar os valores exatos de RGB e HSV do pixel.",
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
    # CONFIGURAÇÃO DA ABA 1: ANÁLISE INDIVIDUAL (2 LINHAS x 5 COLUNAS)
    # =========================================================================
    def setup_aba_individual(self):
        topo = tk.Frame(self.tab_individual)
        topo.pack(pady=10)

        btn_carregar = tk.Button(
            topo,
            text="📂 Selecionar Imagem",
            command=lambda: self.carregar_imagem("ind", (245, 195)),
            font=("Arial", 11, "bold"),
            bg="#4CAF50",
            fg="white",
            padx=15,
            pady=5,
        )
        btn_carregar.pack(side=tk.LEFT, padx=10)

        btn_lote_ind = tk.Button(
            topo,
            text="📦 Baixar Tudo desta Tela (Lote)",
            command=lambda: self.baixar_lote(["ind"]),
            font=("Arial", 11, "bold"),
            bg="#607D8B",
            fg="white",
            padx=15,
            pady=5,
        )
        btn_lote_ind.pack(side=tk.LEFT, padx=10)

        # Botão Toggle para alternar canais R, G, B entre Colorido e Tons de Cinza
        chk_cinza = tk.Checkbutton(
            topo,
            text="Canais R, G, B em Tons de Cinza",
            variable=self.modo_cinza_rgb,
            command=self.alternar_modo_canais_rgb,
            font=("Arial", 10, "bold"),
            bg="#e0e0e0",
            activebackground="#d5d5d5",
            indicatoron=True,
            padx=10,
            pady=5,
            relief=tk.RAISED,
            cursor="hand2",
        )
        chk_cinza.pack(side=tk.LEFT, padx=15)

        # Grid 2 linhas x 5 colunas
        grid_frame = tk.Frame(self.tab_individual)
        grid_frame.pack(expand=True, fill="both", padx=12, pady=5)

        for col in range(5):
            grid_frame.columnconfigure(col, weight=1)
        for row in range(2):
            grid_frame.rowconfigure(row, weight=1)

        self.paineis_ind = {
            # Linha 0: RGB + R, G, B + Histograma RGB
            "rgb": self.criar_painel(grid_frame, "Original (RGB)", 0, 0, "ind", "rgb"),
            "r": self.criar_painel(grid_frame, "Canal R (Red)", 0, 1, "ind", "r"),
            "g": self.criar_painel(grid_frame, "Canal G (Green)", 0, 2, "ind", "g"),
            "b": self.criar_painel(grid_frame, "Canal B (Blue)", 0, 3, "ind", "b"),
            "hist_rgb": self.criar_painel(grid_frame, "Histograma RGB", 0, 4, "ind", "hist_rgb"),
            # Linha 1: HSV + H, S, V + Histograma HSV
            "hsv": self.criar_painel(grid_frame, "HSV Completo", 1, 0, "ind", "hsv"),
            "h": self.criar_painel(grid_frame, "Canal H (Matiz)", 1, 1, "ind", "h"),
            "s": self.criar_painel(grid_frame, "Canal S (Saturação)", 1, 2, "ind", "s"),
            "v": self.criar_painel(grid_frame, "Canal V (Brilho)", 1, 3, "ind", "v"),
            "hist_hsv": self.criar_painel(grid_frame, "Histograma HSV", 1, 4, "ind", "hist_hsv"),
        }

    # =========================================================================
    # CONFIGURAÇÃO DA ABA 2: MODO COMPARATIVO (2 LINHAS x 7 COLUNAS)
    # =========================================================================
    def setup_aba_comparativo(self):
        topo = tk.Frame(self.tab_comparativo)
        topo.pack(pady=8)

        btn_img_a = tk.Button(
            topo,
            text="📂 Carregar Imagem A",
            command=lambda: self.carregar_imagem("comp_a", (180, 145)),
            font=("Arial", 10, "bold"),
            bg="#2196F3",
            fg="white",
            padx=12,
            pady=5,
        )
        btn_img_a.pack(side=tk.LEFT, padx=8)

        btn_img_b = tk.Button(
            topo,
            text="📂 Carregar Imagem B",
            command=lambda: self.carregar_imagem("comp_b", (180, 145)),
            font=("Arial", 10, "bold"),
            bg="#FF9800",
            fg="white",
            padx=12,
            pady=5,
        )
        btn_img_b.pack(side=tk.LEFT, padx=8)

        btn_lote_comp = tk.Button(
            topo,
            text="📦 Baixar Tudo desta Tela (Lote)",
            command=lambda: self.baixar_lote(["comp_a", "comp_b"]),
            font=("Arial", 10, "bold"),
            bg="#607D8B",
            fg="white",
            padx=12,
            pady=5,
        )
        btn_lote_comp.pack(side=tk.LEFT, padx=15)

        canvas = tk.Canvas(self.tab_comparativo)
        scroll_y = ttk.Scrollbar(self.tab_comparativo, orient="vertical", command=canvas.yview)
        scroll_x = ttk.Scrollbar(self.tab_comparativo, orient="horizontal", command=canvas.xview)
        self.scroll_frame = tk.Frame(canvas)

        self.scroll_frame.bind(
            "<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        canvas.create_window((0, 0), window=self.scroll_frame, anchor="nw")
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

        for col_idx in range(len(colunas)):
            self.scroll_frame.columnconfigure(col_idx, weight=1)

        self.paineis_comp_a = {}
        self.paineis_comp_b = {}

        for col_idx, (chave, nome) in enumerate(colunas):
            self.paineis_comp_a[chave] = self.criar_painel(
                self.scroll_frame, f"Img A - {nome}", 0, col_idx, "comp_a", chave
            )
            self.paineis_comp_b[chave] = self.criar_painel(
                self.scroll_frame, f"Img B - {nome}", 1, col_idx, "comp_b", chave
            )

    # =========================================================================
    # CRIAÇÃO DE PAINÉIS COM BOTÃO INDIVIDUAL DE DOWNLOAD
    # =========================================================================
    def criar_painel(self, parent, titulo, linha, coluna, chave_origem, chave_var):
        container = tk.Frame(parent, bd=1, relief=tk.GROOVE, bg="#fafafa")
        container.grid(row=linha, column=coluna, padx=4, pady=4, sticky="nsew")

        header = tk.Frame(container, bg="#e8e8e8")
        header.pack(fill="x")

        lbl_titulo = tk.Label(
            header, text=titulo, font=("Arial", 8, "bold"), bg="#e8e8e8", anchor="w"
        )
        lbl_titulo.pack(side=tk.LEFT, padx=5, pady=2)

        btn_dl = tk.Button(
            header,
            text="💾",
            command=lambda orig=chave_origem, var=chave_var: self.baixar_individual(orig, var),
            font=("Arial", 8),
            bg="#ffffff",
            relief=tk.FLAT,
            cursor="hand2",
            padx=4,
        )
        btn_dl.pack(side=tk.RIGHT, padx=2, pady=1)

        lbl_img = tk.Label(container, text="Aguardando...", fg="gray", bg="#fafafa", cursor="crosshair")
        lbl_img.pack(expand=True, fill="both", padx=3, pady=3)

        if not chave_var.startswith("hist_"):
            lbl_img.bind(
                "<Motion>",
                lambda event, orig=chave_origem, lbl=lbl_img: self.inspecionar_pixel(event, orig, lbl),
            )
        return lbl_img

    # =========================================================================
    # GERAÇÃO DE HISTOGRAMAS (CORRIGIDO COM .flatten())
    # =========================================================================
    def gerar_histograma_rgb(self, img_rgb):
        """Gera imagem RGB (512x320) com as curvas dos canais R, G e B."""
        largura, altura = 512, 320
        margem_inf = 30
        area_plot_h = altura - margem_inf
        canvas = np.full((altura, largura, 3), 255, dtype=np.uint8)

        for x in range(0, largura, 64):
            cv2.line(canvas, (x, 0), (x, area_plot_h), (235, 235, 235), 1)
        for y in range(0, area_plot_h, 58):
            cv2.line(canvas, (0, y), (largura, y), (235, 235, 235), 1)

        cores = [(220, 30, 30), (30, 180, 30), (30, 80, 230)]  # R, G, B
        nomes = ["R", "G", "B"]

        for idx, (cor, nome) in enumerate(zip(cores, nomes)):
            hist = cv2.calcHist([img_rgb], [idx], None, [256], [0, 256])
            cv2.normalize(hist, hist, 0, area_plot_h - 15, cv2.NORM_MINMAX)
            hist_1d = hist.flatten()  # Garante vetor 1D independente da versão do NumPy/OpenCV

            pontos = []
            for x in range(256):
                px = x * 2
                py = area_plot_h - int(hist_1d[x])
                pontos.append((px, py))

            cv2.polylines(canvas, [np.array(pontos, dtype=np.int32)], False, cor, 2, cv2.LINE_AA)
            cv2.putText(canvas, nome, (15 + idx * 35, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, cor, 2)

        cv2.line(canvas, (0, area_plot_h), (largura, area_plot_h), (120, 120, 120), 1)
        cv2.putText(canvas, "0", (5, altura - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 80, 80), 1)
        cv2.putText(canvas, "128", (245, altura - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 80, 80), 1)
        cv2.putText(canvas, "255", (475, altura - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 80, 80), 1)

        return canvas

    def gerar_histograma_hsv(self, img_hsv):
        """Gera imagem RGB (512x320) com as curvas de Matiz (H), Saturação (S) e Valor (V)."""
        largura, altura = 512, 320
        margem_inf = 30
        area_plot_h = altura - margem_inf
        canvas = np.full((altura, largura, 3), 255, dtype=np.uint8)

        for x in range(0, largura, 64):
            cv2.line(canvas, (x, 0), (x, area_plot_h), (235, 235, 235), 1)
        for y in range(0, area_plot_h, 58):
            cv2.line(canvas, (0, y), (largura, y), (235, 235, 235), 1)

        configs = [
            (0, 180, (210, 0, 210), "H (0-179)"),
            (1, 256, (0, 165, 180), "S (0-255)"),
            (2, 256, (50, 50, 50), "V (0-255)"),
        ]

        for idx, bins, cor, legenda in configs:
            hist = cv2.calcHist([img_hsv], [idx], None, [bins], [0, bins])
            cv2.normalize(hist, hist, 0, area_plot_h - 25, cv2.NORM_MINMAX)
            hist_1d = hist.flatten()

            pontos = []
            escala_x = largura / bins
            for x in range(bins):
                px = int(x * escala_x)
                py = area_plot_h - int(hist_1d[x])
                pontos.append((px, py))

            cv2.polylines(canvas, [np.array(pontos, dtype=np.int32)], False, cor, 2, cv2.LINE_AA)
            cv2.putText(canvas, legenda, (12 + idx * 115, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, cor, 2)

        cv2.line(canvas, (0, area_plot_h), (largura, area_plot_h), (120, 120, 120), 1)
        cv2.putText(canvas, "Min (0)", (5, altura - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 80, 80), 1)
        cv2.putText(canvas, "Max (H:179 | S,V:255)", (340, altura - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 80, 80), 1)

        return canvas

    # =========================================================================
    # CARREGAMENTO, EXIBIÇÃO E TOGGLE DE CORES RGB
    # =========================================================================
    def carregar_imagem(self, destino, tamanho_thumb):
        caminho = filedialog.askopenfilename(
            title="Escolha uma imagem",
            filetypes=[("Imagens", "*.png *.jpg *.jpeg *.bmp *.tiff *.webp")],
        )
        if not caminho:
            return

        img_bgr = cv2.imread(caminho)
        if img_bgr is None:
            messagebox.showerror("Erro", "Não foi possível abrir a imagem.")
            return

        nome_arquivo = os.path.splitext(os.path.basename(caminho))[0]

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

        # Canais em tons de cinza (1 canal)
        r_gray, g_gray, b_gray = cv2.split(img_rgb)
        h, s, v = cv2.split(img_hsv)

        # Canais RGB coloridos (3 canais isolados com zeros nos demais)
        zeros = np.zeros_like(r_gray)
        r_color = cv2.merge([r_gray, zeros, zeros])
        g_color = cv2.merge([zeros, g_gray, zeros])
        b_color = cv2.merge([zeros, zeros, b_gray])

        hist_rgb = self.gerar_histograma_rgb(img_rgb)
        hist_hsv = self.gerar_histograma_hsv(img_hsv)

        self.dados_imagens[destino] = {
            "nome_base": nome_arquivo,
            "shape": img_rgb.shape[:2],
            "thumb_size": tamanho_thumb,
            "variacoes": {
                "rgb": img_rgb,
                "r_color": r_color,
                "g_color": g_color,
                "b_color": b_color,
                "r_gray": r_gray,
                "g_gray": g_gray,
                "b_gray": b_gray,
                "hist_rgb": hist_rgb,
                "hsv": img_hsv,
                "h": h,
                "s": s,
                "v": v,
                "hist_hsv": hist_hsv,
            },
        }

        self.atualizar_paineis(destino)

    def alternar_modo_canais_rgb(self):
        """Chamado ao clicar no checkbox para alternar entre colorido e tons de cinza."""
        if self.dados_imagens["ind"]:
            self.atualizar_paineis("ind")

    def obter_matriz_exibicao(self, chave_origem, chave_var):
        """Retorna a matriz correta respeitando o estado do toggle para R, G e B."""
        variacoes = self.dados_imagens[chave_origem]["variacoes"]
        if chave_var in ("r", "g", "b"):
            sufixo = "_gray" if self.modo_cinza_rgb.get() else "_color"
            return variacoes[f"{chave_var}{sufixo}"]
        return variacoes[chave_var]

    def atualizar_paineis(self, destino):
        tamanho_thumb = self.dados_imagens[destino]["thumb_size"]
        paineis = (
            self.paineis_ind
            if destino == "ind"
            else (self.paineis_comp_a if destino == "comp_a" else self.paineis_comp_b)
        )

        for chave, label in paineis.items():
            matriz = self.obter_matriz_exibicao(destino, chave)
            self.exibir_imagem(matriz, label, tamanho_thumb)

    def exibir_imagem(self, matriz_img, label, tamanho_thumb):
        pil_img = Image.fromarray(matriz_img)
        pil_img.thumbnail(tamanho_thumb, Image.Resampling.LANCZOS)

        tk_img = ImageTk.PhotoImage(pil_img)
        label.config(image=tk_img, text="")
        label.image = tk_img
        label.display_size = pil_img.size

    # =========================================================================
    # DOWNLOADS (INDIVIDUAL E EM LOTE)
    # =========================================================================
    def baixar_individual(self, chave_origem, chave_var):
        if not self.dados_imagens[chave_origem]:
            messagebox.showwarning("Aviso", "Nenhuma imagem carregada neste painel.")
            return

        nome_base = self.dados_imagens[chave_origem]["nome_base"]
        prefixo = {"ind": "", "comp_a": "ImgA_", "comp_b": "ImgB_"}[chave_origem]

        # Indica no nome do arquivo se o canal RGB foi salvo em cinza ou colorido
        sufixo_modo = ""
        if chave_var in ("r", "g", "b"):
            sufixo_modo = "_cinza" if self.modo_cinza_rgb.get() else "_colorido"

        sugestao_nome = f"{prefixo}{nome_base}_{chave_var}{sufixo_modo}.png"

        caminho_salvar = filedialog.asksaveasfilename(
            title="Salvar Variação",
            initialfile=sugestao_nome,
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg"), ("BMP", "*.bmp")],
        )
        if not caminho_salvar:
            return

        matriz = self.obter_matriz_exibicao(chave_origem, chave_var)
        self._salvar_matriz_disco(matriz, caminho_salvar)
        messagebox.showinfo("Sucesso", f"Imagem salva em:\n{caminho_salvar}")

    def baixar_lote(self, lista_origens):
        origens_validas = [orig for orig in lista_origens if self.dados_imagens[orig]]
        if not origens_validas:
            messagebox.showwarning("Aviso", "Carregue pelo menos uma imagem antes de baixar em lote.")
            return

        pasta_destino = filedialog.askdirectory(title="Selecione a pasta para salvar todas as variações")
        if not pasta_destino:
            return

        chaves_aba = (
            list(self.paineis_ind.keys())
            if lista_origens == ["ind"]
            else list(self.paineis_comp_a.keys())
        )

        total_salvos = 0
        for orig in origens_validas:
            nome_base = self.dados_imagens[orig]["nome_base"]
            prefixo = {"ind": "", "comp_a": "ImgA_", "comp_b": "ImgB_"}[orig]

            for chave in chaves_aba:
                sufixo_modo = ""
                if chave in ("r", "g", "b"):
                    sufixo_modo = "_cinza" if self.modo_cinza_rgb.get() else "_colorido"

                matriz = self.obter_matriz_exibicao(orig, chave)
                nome_arquivo = f"{prefixo}{nome_base}_{chave}{sufixo_modo}.png"
                caminho_completo = os.path.join(pasta_destino, nome_arquivo)
                self._salvar_matriz_disco(matriz, caminho_completo)
                total_salvos += 1

        messagebox.showinfo(
            "Download em Lote Concluído",
            f"{total_salvos} arquivos foram salvos com sucesso na pasta:\n{pasta_destino}",
        )

    def _salvar_matriz_disco(self, matriz, caminho):
        if len(matriz.shape) == 3:
            matriz_bgr = cv2.cvtColor(matriz, cv2.COLOR_RGB2BGR)
        else:
            matriz_bgr = matriz
        cv2.imwrite(caminho, matriz_bgr)

    # =========================================================================
    # INSPEÇÃO DE PIXEL
    # =========================================================================
    def inspecionar_pixel(self, event, chave_origem, label):
        if not self.dados_imagens[chave_origem] or not hasattr(label, "display_size"):
            return

        disp_w, disp_h = label.display_size
        orig_h, orig_w = self.dados_imagens[chave_origem]["shape"]

        offset_x = (label.winfo_width() - disp_w) // 2
        offset_y = (label.winfo_height() - disp_h) // 2

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