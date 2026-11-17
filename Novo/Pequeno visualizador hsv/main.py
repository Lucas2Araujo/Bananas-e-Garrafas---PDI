import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import cv2
from PIL import Image, ImageTk

from componentes import PainelImagem
from processamento import gerar_histograma_canal, processar_imagem
from segmentacao import aplicar_grabcut, aplicar_segmentacao, detectar_roi_yolo, segmentar_roi
from visualizador import JanelaZoom


class AnalisadorCoresApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Estúdio PDI - RGB, HSV, Segmentação e YOLO + GrabCut")
        self.root.geometry("1440x860")

        self.dados_imagens = {"ind": {}, "comp_b": {}}
        self.workspace_ativo = False

        # Controles de visualização RGB/HSV
        self.modo_cinza_rgb = tk.BooleanVar(value=False)
        self.modo_foco_hsv = tk.BooleanVar(value=False)
        self.modo_comparar = tk.BooleanVar(value=False)

        # Estado da Segmentação Clássica
        self.seg_img_fundo_bgr = None
        self.seg_img_objeto_bgr = None
        self.seg_resultados = {}
        self.seg_paineis = {}

        # Estado do YOLO + GrabCut
        self.grabcut_img_bgr = None
        self.grabcut_rect = None
        self.grabcut_label = "ROI"
        self.grabcut_resultados = {}
        self.grabcut_paineis = {}

        self._setup_tela_inicial()
        self._setup_workspace()
        self._configurar_atalhos()

    # =========================================================================
    # TELA INICIAL E ATALHOS
    # =========================================================================
    def _setup_tela_inicial(self):
        self.frame_boas_vindas = tk.Frame(self.root, bg="#f4f6f9")
        self.frame_boas_vindas.pack(expand=True, fill="both")

        card = tk.Frame(self.frame_boas_vindas, bg="white", bd=1, relief=tk.SOLID, padx=45, pady=40)
        card.place(relx=0.5, rely=0.5, anchor="center")

        tk.Label(
            card, text="Estúdio de Processamento de Imagens (PDI)",
            font=("Arial", 18, "bold"), bg="white", fg="#1f2937",
        ).pack(pady=(0, 8))

        tk.Label(
            card,
            text="Importe uma imagem para explorar canais RGB/HSV, histogramas, segmentação clássica e YOLO + GrabCut.",
            font=("Arial", 11), bg="white", fg="#4b5563",
        ).pack(pady=(0, 24))

        tk.Button(
            card,
            text="📂 Importar Imagem Principal (Ctrl+O)",
            command=lambda: self.carregar_imagem("ind"),
            font=("Arial", 12, "bold"),
            bg="#4CAF50",
            fg="white",
            padx=22,
            pady=10,
            cursor="hand2",
        ).pack(pady=(0, 20))

        atalhos = (
            "Atalhos rápidos após importar:\n"
            "• Ctrl+O  —  Importar / trocar imagem principal\n"
            "• Ctrl+B  —  Ativar/desativar comparação com 2ª imagem\n"
            "• Ctrl+1 / Ctrl+2 / Ctrl+3  —  Alternar entre RGB/HSV, Segmentação e YOLO + GrabCut"
        )
        tk.Label(
            card, text=atalhos, font=("Consolas", 10), bg="#f9fafb", fg="#374151",
            justify=tk.LEFT, padx=16, pady=12, bd=1, relief=tk.GROOVE,
        ).pack(fill="x")

    def _setup_workspace(self):
        self.frame_workspace = tk.Frame(self.root)

        self.notebook = ttk.Notebook(self.frame_workspace)
        self.notebook.pack(expand=True, fill="both", padx=10, pady=(10, 0))

        self.tab_espacos = tk.Frame(self.notebook)
        self.notebook.add(self.tab_espacos, text="  1. Espaços RGB & HSV  ")
        self._setup_aba_espacos()

        self.tab_segmentacao = tk.Frame(self.notebook)
        self.notebook.add(self.tab_segmentacao, text="  2. Segmentação (Clássica)  ")
        self._setup_aba_segmentacao()

        self.tab_grabcut = tk.Frame(self.notebook)
        self.notebook.add(self.tab_grabcut, text="  3. YOLO ROI + GrabCut  ")
        self._setup_aba_grabcut()

        self.comp_roi_paineis = {}
        self.comp_roi_resultados = {}
        self.tab_comp_roi = tk.Frame(self.notebook)
        self.notebook.add(self.tab_comp_roi, text="  4. Comparativo: Sem ROI vs Pós-ROI  ")
        self._setup_aba_comp_roi()

        self.lbl_status = tk.Label(
            self.frame_workspace,
            text="Passe o mouse sobre uma imagem para inspecionar pixels, 🔍 para Zoom, 📊 para Histograma ou ✂️ para Segmentar.",
            bd=1, relief=tk.SUNKEN, anchor=tk.W, font=("Consolas", 10), bg="#f0f0f0", padx=10, pady=5,
        )
        self.lbl_status.pack(side=tk.BOTTOM, fill=tk.X)

    def _entrar_workspace(self):
        if not self.workspace_ativo:
            self.frame_boas_vindas.pack_forget()
            self.frame_workspace.pack(expand=True, fill="both")
            self.workspace_ativo = True

    def _configurar_atalhos(self):
        self.root.bind("<Control-o>", lambda _e: self.carregar_imagem("ind"))
        self.root.bind("<Control-O>", lambda _e: self.carregar_imagem("ind"))
        self.root.bind("<Control-b>", lambda _e: self._alternar_comparacao(atalho=True))
        self.root.bind("<Control-B>", lambda _e: self._alternar_comparacao(atalho=True))
        self.root.bind("<Control-Key-1>", lambda _e: self.workspace_ativo and self.notebook.select(0))
        self.root.bind("<Control-Key-2>", lambda _e: self.workspace_ativo and self.notebook.select(1))
        self.root.bind("<Control-Key-3>", lambda _e: self.workspace_ativo and self.notebook.select(2))
        self.root.bind("<Control-Key-4>", lambda _e: self.workspace_ativo and self.notebook.select(3))

    # =========================================================================
    # ABA 1: ESPAÇOS RGB & HSV (+ COMPARAÇÃO INTEGRADA)
    # =========================================================================
    def _setup_aba_espacos(self):
        topo = tk.Frame(self.tab_espacos)
        topo.pack(pady=8, fill="x", padx=10)

        tk.Button(
            topo, text="📂 Trocar Imagem (Ctrl+O)", command=lambda: self.carregar_imagem("ind"),
            font=("Arial", 10, "bold"), bg="#4CAF50", fg="white", padx=12, pady=5, cursor="hand2",
        ).pack(side=tk.LEFT, padx=4)

        tk.Checkbutton(
            topo, text="⚖️ Comparar 2ª Imagem (Ctrl+B)", variable=self.modo_comparar,
            command=self._alternar_comparacao, font=("Arial", 10, "bold"),
            bg="#fff3e0", activebackground="#ffe0b2", padx=10, pady=5, relief=tk.RAISED, cursor="hand2",
        ).pack(side=tk.LEFT, padx=4)

        self.btn_carregar_b = tk.Button(
            topo, text="📂 Escolher 2ª Imagem", command=lambda: self.carregar_imagem("comp_b"),
            font=("Arial", 10, "bold"), bg="#FF9800", fg="white", padx=10, pady=5, cursor="hand2",
        )

        tk.Button(
            topo, text="📦 Baixar Visíveis", command=self._baixar_aba_espacos,
            font=("Arial", 10, "bold"), bg="#607D8B", fg="white", padx=10, pady=5, cursor="hand2",
        ).pack(side=tk.LEFT, padx=4)

        tk.Checkbutton(
            topo, text="RGB em Cinza", variable=self.modo_cinza_rgb,
            command=self._alternar_modo_canais_rgb, font=("Arial", 9, "bold"),
            bg="#e0e0e0", padx=8, pady=5, relief=tk.RAISED, cursor="hand2",
        ).pack(side=tk.LEFT, padx=4)

        tk.Checkbutton(
            topo, text="🎯 Foco HSV", variable=self.modo_foco_hsv,
            command=self._alternar_modo_foco_hsv, font=("Arial", 9, "bold"),
            bg="#e3f2fd", padx=8, pady=5, relief=tk.RAISED, cursor="hand2",
        ).pack(side=tk.LEFT, padx=4)

        # Container único
        self.grid_ind = tk.Frame(self.tab_espacos)
        self.grid_ind.pack(expand=True, fill="both", padx=12, pady=5)

        itens = [
            ("rgb", "Original (RGB)"), ("r", "Canal R (Red)"), ("g", "Canal G (Green)"),
            ("b", "Canal B (Blue)"), ("hist_rgb", "Histograma RGB"),
            ("hsv", "HSV Completo"), ("h", "Canal H (Matiz)"), ("s", "Canal S (Saturação)"),
            ("v", "Canal V (Brilho)"), ("hist_hsv", "Histograma HSV"),
        ]

        self.paineis_ind = {}
        for chave, titulo in itens:
            eh_hist = chave.startswith("hist_")
            self.paineis_ind[chave] = PainelImagem(
                self.grid_ind, titulo, "ind", chave,
                self.abrir_zoom, self.baixar_individual, self.inspecionar_pixel,
                on_hist=None if eh_hist else self.abrir_histograma_canal,
                on_segmentar=None if eh_hist else self.enviar_para_segmentacao,
            )

        # Container comparativo (Img A = ind, Img B = comp_b)
        self.frame_comp = tk.Frame(self.tab_espacos)
        canvas = tk.Canvas(self.frame_comp)
        scroll_y = ttk.Scrollbar(self.frame_comp, orient="vertical", command=canvas.yview)
        scroll_x = ttk.Scrollbar(self.frame_comp, orient="horizontal", command=canvas.xview)
        self.scroll_frame_comp = tk.Frame(canvas)

        self.scroll_frame_comp.bind("<Configure>", lambda _e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=self.scroll_frame_comp, anchor="nw")
        canvas.configure(yscrollcommand=scroll_y.set, xscrollcommand=scroll_x.set)
        scroll_y.pack(side="right", fill="y")
        scroll_x.pack(side="bottom", fill="x")
        canvas.pack(side="left", fill="both", expand=True, padx=8)

        colunas = [
            ("rgb", "RGB"), ("hist_rgb", "Hist. RGB"), ("hsv", "HSV"),
            ("h", "Canal H"), ("s", "Canal S"), ("v", "Canal V"), ("hist_hsv", "Hist. HSV"),
        ]
        self.paineis_comp_a = {}
        self.paineis_comp_b = {}
        for chave, nome in colunas:
            eh_hist = chave.startswith("hist_")
            self.paineis_comp_a[chave] = PainelImagem(
                self.scroll_frame_comp, f"Img A - {nome}", "ind", chave,
                self.abrir_zoom, self.baixar_individual, self.inspecionar_pixel,
                on_hist=None if eh_hist else self.abrir_histograma_canal,
                on_segmentar=None if eh_hist else self.enviar_para_segmentacao,
            )
            self.paineis_comp_b[chave] = PainelImagem(
                self.scroll_frame_comp, f"Img B - {nome}", "comp_b", chave,
                self.abrir_zoom, self.baixar_individual, self.inspecionar_pixel,
                on_hist=None if eh_hist else self.abrir_histograma_canal,
                on_segmentar=None if eh_hist else self.enviar_para_segmentacao,
            )

        self._aplicar_layout_individual()
        self._aplicar_layout_comparativo()

    def _alternar_comparacao(self, atalho=False):
        if not self.workspace_ativo:
            return
        if atalho:
            self.modo_comparar.set(not self.modo_comparar.get())

        if self.modo_comparar.get():
            self.grid_ind.pack_forget()
            self.frame_comp.pack(expand=True, fill="both", padx=4, pady=4)
            self.btn_carregar_b.pack(side=tk.LEFT, padx=4)
            if not self.dados_imagens["comp_b"]:
                self.carregar_imagem("comp_b")
            else:
                self._atualizar_paineis("comp_b")
            if self.dados_imagens["ind"]:
                self._atualizar_paineis("ind")
        else:
            self.frame_comp.pack_forget()
            self.btn_carregar_b.pack_forget()
            self.grid_ind.pack(expand=True, fill="both", padx=12, pady=5)
            if self.dados_imagens["ind"]:
                self._atualizar_paineis("ind")

    def _aplicar_layout_individual(self):
        for painel in self.paineis_ind.values():
            painel.grid_forget()
        for c in range(5):
            self.grid_ind.columnconfigure(c, weight=0)
        for r in range(2):
            self.grid_ind.rowconfigure(r, weight=0)

        if self.modo_foco_hsv.get():
            for c, chave in enumerate(["rgb", "h", "s", "v"]):
                self.grid_ind.columnconfigure(c, weight=1)
                self.paineis_ind[chave].grid(row=0, column=c, padx=6, pady=6, sticky="nsew")
            self.grid_ind.rowconfigure(0, weight=1)
        else:
            layout = [
                ("rgb", 0, 0), ("r", 0, 1), ("g", 0, 2), ("b", 0, 3), ("hist_rgb", 0, 4),
                ("hsv", 1, 0), ("h", 1, 1), ("s", 1, 2), ("v", 1, 3), ("hist_hsv", 1, 4),
            ]
            for c in range(5):
                self.grid_ind.columnconfigure(c, weight=1)
            for r in range(2):
                self.grid_ind.rowconfigure(r, weight=1)
            for chave, r, c in layout:
                self.paineis_ind[chave].grid(row=r, column=c, padx=4, pady=4, sticky="nsew")

    def _aplicar_layout_comparativo(self):
        for painel in list(self.paineis_comp_a.values()) + list(self.paineis_comp_b.values()):
            painel.grid_forget()
        for c in range(7):
            self.scroll_frame_comp.columnconfigure(c, weight=0)

        chaves = ["rgb", "h", "s", "v"] if self.modo_foco_hsv.get() else [
            "rgb", "hist_rgb", "hsv", "h", "s", "v", "hist_hsv"
        ]
        for col_idx, chave in enumerate(chaves):
            self.scroll_frame_comp.columnconfigure(col_idx, weight=1)
            self.paineis_comp_a[chave].grid(row=0, column=col_idx, padx=4, pady=4, sticky="nsew")
            self.paineis_comp_b[chave].grid(row=1, column=col_idx, padx=4, pady=4, sticky="nsew")

    def _alternar_modo_foco_hsv(self):
        self._aplicar_layout_individual()
        self._aplicar_layout_comparativo()
        for destino in ("ind", "comp_b"):
            if self.dados_imagens[destino]:
                self._atualizar_paineis(destino)

    def _obter_tamanho_thumb(self, destino):
        if self.modo_comparar.get():
            return (310, 260) if self.modo_foco_hsv.get() else (180, 145)
        return (330, 420) if self.modo_foco_hsv.get() else (245, 195)

    # =========================================================================
    # GERENCIAMENTO DE IMAGENS, HISTOGRAMA POR CANAL E INSPEÇÃO
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

        self._entrar_workspace()
        self.dados_imagens[destino] = {
            "nome_base": os.path.splitext(os.path.basename(caminho))[0],
            "shape": variacoes["rgb"].shape[:2],
            "variacoes": variacoes,
        }
        self._atualizar_paineis(destino)

        if destino == "ind":
            self.seg_img_objeto_bgr = variacoes["bgr"]
            self.grabcut_img_bgr = variacoes["bgr"]
            self.grabcut_rect = None
            self.grabcut_resultados = {}
            self._atualizar_segmentacao()
            self.lbl_status.config(text=f"Imagem principal carregada: {os.path.basename(caminho)}")

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
        if destino == "ind":
            alvos = self.paineis_comp_a if self.modo_comparar.get() else self.paineis_ind
        else:
            alvos = self.paineis_comp_b

        for chave, painel in alvos.items():
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

    def abrir_histograma_canal(self, chave_origem, chave_var, titulo):
        if not self.dados_imagens[chave_origem]:
            messagebox.showwarning("Aviso", "Carregue uma imagem primeiro.")
            return

        variacoes = self.dados_imagens[chave_origem]["variacoes"]
        if chave_var == "rgb":
            JanelaZoom(self.root, f"Histograma - {titulo}", variacoes["hist_rgb"])
            return
        if chave_var == "hsv":
            JanelaZoom(self.root, f"Histograma - {titulo}", variacoes["hist_hsv"])
            return

        cfg = {
            "r": ("r_gray", "Canal R", (220, 40, 40), 256),
            "g": ("g_gray", "Canal G", (30, 175, 50), 256),
            "b": ("b_gray", "Canal B", (35, 90, 230), 256),
            "h": ("h", "Canal H (Matiz)", (200, 20, 200), 180),
            "s": ("s", "Canal S (Saturacao)", (0, 160, 185), 256),
            "v": ("v", "Canal V (Brilho)", (55, 55, 55), 256),
        }[chave_var]
        chave_matriz, nome, cor, max_val = cfg
        hist_img = gerar_histograma_canal(variacoes[chave_matriz], nome_canal=nome, cor=cor, max_val=max_val)
        JanelaZoom(self.root, f"Histograma Individual - {titulo}", hist_img)

    def enviar_para_segmentacao(self, chave_origem, chave_var, _titulo):
        if not self.dados_imagens[chave_origem]:
            messagebox.showwarning("Aviso", "Carregue uma imagem primeiro.")
            return

        mapa_canal = {"r": "R", "g": "G", "b": "B", "h": "H", "s": "S", "v": "V"}
        self.seg_img_objeto_bgr = self.dados_imagens[chave_origem]["variacoes"]["bgr"]
        self.seg_canal_var.set(mapa_canal.get(chave_var, "GRAY"))
        self.notebook.select(self.tab_segmentacao)
        self._atualizar_segmentacao()
        self.lbl_status.config(text=f"[Segmentação] Canal '{self.seg_canal_var.get()}' selecionado a partir da moldura.")

    def baixar_individual(self, chave_origem, chave_var):
        if not self.dados_imagens[chave_origem]:
            messagebox.showwarning("Aviso", "Nenhuma imagem carregada neste painel.")
            return

        nome_base = self.dados_imagens[chave_origem]["nome_base"]
        prefixo = "ImgB_" if chave_origem == "comp_b" else ""
        sufixo_modo = ("_cinza" if self.modo_cinza_rgb.get() else "_colorido") if chave_var in ("r", "g", "b") else ""
        caminho_salvar = filedialog.asksaveasfilename(
            title="Salvar Variação",
            initialfile=f"{prefixo}{nome_base}_{chave_var}{sufixo_modo}.png",
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg"), ("BMP", "*.bmp")],
        )
        if caminho_salvar:
            self._salvar_disco(self._obter_matriz(chave_origem, chave_var), caminho_salvar)
            messagebox.showinfo("Sucesso", f"Imagem salva em:\n{caminho_salvar}")

    def _baixar_aba_espacos(self):
        origens = ["ind", "comp_b"] if self.modo_comparar.get() else ["ind"]
        origens_validas = [orig for orig in origens if self.dados_imagens[orig]]
        if not origens_validas:
            messagebox.showwarning("Aviso", "Carregue uma imagem antes de baixar.")
            return

        pasta = filedialog.askdirectory(title="Selecione a pasta para salvar as variações visíveis")
        if not pasta:
            return

        chaves = ["rgb", "h", "s", "v"] if self.modo_foco_hsv.get() else (
            ["rgb", "hist_rgb", "hsv", "h", "s", "v", "hist_hsv"] if self.modo_comparar.get() else list(self.paineis_ind.keys())
        )
        total = 0
        for orig in origens_validas:
            nome_base = self.dados_imagens[orig]["nome_base"]
            prefixo = "ImgB_" if orig == "comp_b" else ""
            for chave in chaves:
                sufixo = ("_cinza" if self.modo_cinza_rgb.get() else "_colorido") if chave in ("r", "g", "b") else ""
                self._salvar_disco(self._obter_matriz(orig, chave), os.path.join(pasta, f"{prefixo}{nome_base}_{chave}{sufixo}.png"))
                total += 1
        messagebox.showinfo("Concluído", f"{total} arquivos salvos em:\n{pasta}")

    def _salvar_disco(self, matriz, caminho):
        matriz_bgr = cv2.cvtColor(matriz, cv2.COLOR_RGB2BGR) if len(matriz.shape) == 3 else matriz
        cv2.imwrite(caminho, matriz_bgr)

    def inspecionar_pixel(self, event, chave_origem, painel):
        if not self.dados_imagens[chave_origem] or not painel.display_size:
            return
        disp_w, disp_h = painel.display_size
        orig_h, orig_w = self.dados_imagens[chave_origem]["shape"]
        mouse_x = event.x - (painel.lbl_img.winfo_width() - disp_w) // 2
        mouse_y = event.y - (painel.lbl_img.winfo_height() - disp_h) // 2

        if 0 <= mouse_x < disp_w and 0 <= mouse_y < disp_h:
            rx = int(mouse_x * (orig_w / disp_w))
            ry = int(mouse_y * (orig_h / disp_h))
            r, g, b = self.dados_imagens[chave_origem]["variacoes"]["rgb"][ry, rx]
            h, s, v = self.dados_imagens[chave_origem]["variacoes"]["hsv"][ry, rx]
            nome_img = "Imagem B" if chave_origem == "comp_b" else "Imagem A"
            self.lbl_status.config(
                text=f"[{nome_img}] (X={rx}, Y={ry}) | RGB: ({r:3d}, {g:3d}, {b:3d}) | HSV: (H={h:3d}°, S={s:3d}, V={v:3d})"
            )

    # =========================================================================
    # ABA 2: SEGMENTAÇÃO CLÁSSICA (POR CANAL + ESCOLHA DE TÉCNICA)
    # =========================================================================
    def _setup_aba_segmentacao(self):
        topo = tk.Frame(self.tab_segmentacao)
        topo.pack(pady=6, fill=tk.X, padx=10)

        tk.Button(
            topo, text="🖼️ Carregar Fundo (opcional p/ Método 1)",
            command=lambda: self._carregar_img_segmentacao("fundo"),
            font=("Arial", 9, "bold"), bg="#607D8B", fg="white", padx=10, pady=4,
        ).pack(side=tk.LEFT, padx=4)

        tk.Button(
            topo, text="📦 Trocar Objeto",
            command=lambda: self._carregar_img_segmentacao("objeto"),
            font=("Arial", 9, "bold"), bg="#4CAF50", fg="white", padx=10, pady=4,
        ).pack(side=tk.LEFT, padx=4)

        self.seg_canal_var = tk.StringVar(value="GRAY")
        self.seg_metodo_var = tk.StringVar(value="Todos (Comparar 3)")
        self.seg_usar_blur_var = tk.BooleanVar(value=True)
        self.seg_blur_var = tk.IntVar(value=5)
        self.seg_absdiff_var = tk.IntVar(value=30)
        self.seg_bloco_var = tk.IntVar(value=21)
        self.seg_c_var = tk.IntVar(value=5)
        self.seg_inverter_var = tk.BooleanVar(value=True)

        tk.Label(topo, text="Canal:", font=("Arial", 9, "bold")).pack(side=tk.LEFT, padx=(10, 2))
        for texto, valor in [("Cinza", "GRAY"), ("R", "R"), ("G", "G"), ("B", "B"), ("H", "H"), ("S", "S"), ("V", "V")]:
            tk.Radiobutton(
                topo, text=texto, variable=self.seg_canal_var, value=valor,
                command=self._atualizar_segmentacao, font=("Arial", 9), cursor="hand2",
            ).pack(side=tk.LEFT, padx=2)

        tk.Label(topo, text="Técnica:", font=("Arial", 9, "bold")).pack(side=tk.LEFT, padx=(10, 4))
        combo_metodo = ttk.Combobox(
            topo, textvariable=self.seg_metodo_var, state="readonly", width=22,
            values=["Todos (Comparar 3)", "1. Subtração (absdiff)", "2. Otsu", "3. Adaptativa", "4. GrabCut (Sem ROI)"],
        )
        combo_metodo.pack(side=tk.LEFT, padx=2)
        combo_metodo.bind("<<ComboboxSelected>>", lambda _e: self._ao_mudar_metodo_seg())

        self.seg_grabcut_no_2_var = tk.BooleanVar(value=False)
        tk.Checkbutton(
            topo, text="GrabCut no 2º quadro", variable=self.seg_grabcut_no_2_var,
            command=self._ao_mudar_metodo_seg, font=("Arial", 9, "bold"), cursor="hand2",
        ).pack(side=tk.LEFT, padx=6)

        controles = tk.Frame(self.tab_segmentacao, bg="#f5f5f5", bd=1, relief=tk.GROOVE)
        controles.pack(fill=tk.X, padx=10, pady=(0, 6))

        def _criar_slider(parent, texto, var, de, ate, passo=1):
            frame = tk.Frame(parent, bg="#f5f5f5")
            frame.pack(side=tk.LEFT, padx=8, pady=4)
            tk.Label(frame, text=texto, font=("Arial", 9, "bold"), bg="#f5f5f5").pack()
            tk.Scale(
                frame, from_=de, to=ate, resolution=passo, orient=tk.HORIZONTAL,
                variable=var, command=lambda _v: self._atualizar_segmentacao(), length=130, bg="#f5f5f5",
            ).pack()

        tk.Checkbutton(
            controles, text="Usar Filtro Gaussiano", variable=self.seg_usar_blur_var,
            command=self._atualizar_segmentacao, font=("Arial", 9, "bold"), bg="#f5f5f5",
        ).pack(side=tk.LEFT, padx=8)

        _criar_slider(controles, "Gaussian Blur (ímpar)", self.seg_blur_var, 1, 15, 2)
        _criar_slider(controles, "Limiar absdiff", self.seg_absdiff_var, 1, 255, 1)
        _criar_slider(controles, "Bloco Adaptativa (ímpar)", self.seg_bloco_var, 3, 51, 2)
        _criar_slider(controles, "Constante C", self.seg_c_var, -20, 30, 1)

        tk.Checkbutton(
            controles, text="Inverter Limiar (BINARY_INV)", variable=self.seg_inverter_var,
            command=self._atualizar_segmentacao, font=("Arial", 9, "bold"), bg="#f5f5f5",
        ).pack(side=tk.LEFT, padx=8)

        self.seg_grid_frame = tk.Frame(self.tab_segmentacao)
        self.seg_grid_frame.pack(expand=True, fill="both", padx=10, pady=5)

        paineis_cfg = [
            ("objeto_gray", "Entrada (Canal / Blur)"),
            ("mask_subtracao", "1. Subtração de Fundo (absdiff)"),
            ("mask_otsu", "2. Limiarização de Otsu"),
            ("mask_adaptativa", "3. Limiarização Adaptativa"),
            ("mask_grabcut", "4. GrabCut (Sem ROI)"),
        ]
        for chave, titulo in paineis_cfg:
            self.seg_paineis[chave] = PainelImagem(
                self.seg_grid_frame, titulo=titulo, chave_origem="seg", chave_var=chave,
                on_zoom=lambda _orig, ch, t=titulo: self._abrir_zoom_seg(ch, t),
                on_download=lambda _orig, ch: self._salvar_seg_individual(ch),
            )
        self._aplicar_layout_segmentacao()

    def _ao_mudar_metodo_seg(self):
        self._aplicar_layout_segmentacao()
        self._atualizar_segmentacao()

    def _aplicar_layout_segmentacao(self):
        for p in self.seg_paineis.values():
            p.grid_forget()
        for i in range(2):
            self.seg_grid_frame.columnconfigure(i, weight=0)
            self.seg_grid_frame.rowconfigure(i, weight=0)

        metodo = self.seg_metodo_var.get()
        mapa = {
            "1. Subtração (absdiff)": "mask_subtracao",
            "2. Otsu": "mask_otsu",
            "3. Adaptativa": "mask_adaptativa",
            "4. GrabCut (Sem ROI)": "mask_grabcut",
        }
        if metodo in mapa:
            self.seg_grid_frame.rowconfigure(0, weight=1)
            for col, ch in enumerate(["objeto_gray", mapa[metodo]]):
                self.seg_grid_frame.columnconfigure(col, weight=1)
                self.seg_paineis[ch].grid(row=0, column=col, sticky="nsew", padx=6, pady=6)
        else:
            segundo = "mask_grabcut" if self.seg_grabcut_no_2_var.get() else "mask_subtracao"
            posicoes = [("objeto_gray", 0, 0), (segundo, 0, 1), ("mask_otsu", 1, 0), ("mask_adaptativa", 1, 1)]
            for i in range(2):
                self.seg_grid_frame.columnconfigure(i, weight=1)
                self.seg_grid_frame.rowconfigure(i, weight=1)
            for ch, r, c in posicoes:
                self.seg_paineis[ch].grid(row=r, column=c, sticky="nsew", padx=5, pady=5)

    def _carregar_img_segmentacao(self, tipo):
        caminho = filedialog.askopenfilename(
            title=f"Selecionar imagem de {tipo}",
            filetypes=[("Imagens", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"), ("Todos", "*.*")],
        )
        if not caminho:
            return
        img = cv2.imread(caminho)
        if img is None:
            messagebox.showerror("Erro", f"Não foi possível carregar a imagem:\n{caminho}")
            return
        if tipo == "fundo":
            self.seg_img_fundo_bgr = img
            self.lbl_status.config(text=f"[Segmentação] Fundo carregado: {os.path.basename(caminho)}")
        else:
            self.seg_img_objeto_bgr = img
            self.lbl_status.config(text=f"[Segmentação] Objeto carregado: {os.path.basename(caminho)}")
        self._atualizar_segmentacao()

    def _atualizar_segmentacao(self):
        if self.seg_img_objeto_bgr is None:
            return

        incluir_gc = (
            self.seg_metodo_var.get() == "4. GrabCut (Sem ROI)"
            or self.seg_grabcut_no_2_var.get()
        )
        self.seg_resultados = aplicar_segmentacao(
            self.seg_img_objeto_bgr,
            img_fundo_bgr=self.seg_img_fundo_bgr,
            canal=self.seg_canal_var.get(),
            usar_blur=self.seg_usar_blur_var.get(),
            blur_ksize=self.seg_blur_var.get(),
            limiar_absdiff=self.seg_absdiff_var.get(),
            bloco_adaptativo=self.seg_bloco_var.get(),
            constante_c=self.seg_c_var.get(),
            inverter_limiar=self.seg_inverter_var.get(),
            incluir_grabcut=incluir_gc,
        )
        thumb = (380, 260) if self.seg_metodo_var.get() == "Todos (Comparar 3)" else (560, 460)
        for chave, painel in self.seg_paineis.items():
            matriz = self.seg_resultados.get(chave)
            if matriz is not None:
                painel.atualizar_miniatura(matriz, thumb)

    def _abrir_zoom_seg(self, chave, titulo):
        matriz = self.seg_resultados.get(chave)
        if matriz is None:
            messagebox.showwarning("Aviso", "Nenhuma imagem disponível para zoom neste painel.")
            return
        JanelaZoom(self.root, titulo, matriz)

    def _salvar_seg_individual(self, chave):
        matriz = self.seg_resultados.get(chave)
        if matriz is None:
            messagebox.showwarning("Aviso", "Nenhuma imagem disponível para salvar neste painel.")
            return
        caminho = filedialog.asksaveasfilename(
            defaultextension=".png", initialfile=f"segmentacao_{chave}.png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg")],
        )
        if caminho:
            self._salvar_disco(matriz, caminho)
            messagebox.showinfo("Sucesso", f"Imagem salva em:\n{caminho}")

    # =========================================================================
    # ABA 3: YOLO ROI + GRABCUT (IMAGENS ESTÁTICAS)
    # =========================================================================
    def _setup_aba_grabcut(self):
        topo = tk.Frame(self.tab_grabcut)
        topo.pack(pady=8, fill=tk.X, padx=10)

        tk.Button(
            topo, text="🤖 Detectar ROI (YOLO)", command=self._executar_yolo_grabcut,
            font=("Arial", 10, "bold"), bg="#2196F3", fg="white", padx=12, pady=5, cursor="hand2",
        ).pack(side=tk.LEFT, padx=5)

        tk.Button(
            topo, text="✏️ Desenhar ROI Manual", command=self._abrir_seletor_roi_manual,
            font=("Arial", 10, "bold"), bg="#9C27B0", fg="white", padx=12, pady=5, cursor="hand2",
        ).pack(side=tk.LEFT, padx=5)

        tk.Button(
            topo, text="📂 Trocar Imagem", command=self._carregar_img_grabcut,
            font=("Arial", 10, "bold"), bg="#4CAF50", fg="white", padx=12, pady=5, cursor="hand2",
        ).pack(side=tk.LEFT, padx=5)

        self.gc_pad_var = tk.IntVar(value=10)
        self.gc_iter_var = tk.IntVar(value=5)
        self.gc_metodo_var = tk.StringVar(value="GrabCut")
        self.gc_espaco_yolo_var = tk.StringVar(value="BGR")
        self.gc_comparar_3_var = tk.BooleanVar(value=False)
        self.gc_usar_gauss_var = tk.BooleanVar(value=False)
        self.gc_usar_mediana_var = tk.BooleanVar(value=False)
        self.gc_usar_bilateral_var = tk.BooleanVar(value=False)
        self.gc_blur_var = tk.IntVar(value=5)

        tk.Label(topo, text="YOLO:", font=("Arial", 9, "bold")).pack(side=tk.LEFT, padx=(4, 2))
        ttk.Combobox(
            topo, textvariable=self.gc_espaco_yolo_var, state="readonly", width=6,
            values=["BGR", "HSV", "GRAY"],
        ).pack(side=tk.LEFT, padx=2)

        tk.Checkbutton(
            topo, text="Comparar 3 Técnicas", variable=self.gc_comparar_3_var,
            command=self._atualizar_grabcut, font=("Arial", 9, "bold"),
            bg="#fff3e0", padx=8, pady=4, relief=tk.RAISED, cursor="hand2",
        ).pack(side=tk.LEFT, padx=6)

        tk.Label(topo, text="Técnica ROI:", font=("Arial", 9, "bold")).pack(side=tk.LEFT, padx=(6, 4))
        combo_gc = ttk.Combobox(
            topo, textvariable=self.gc_metodo_var, state="readonly", width=11,
            values=["GrabCut", "Otsu", "Adaptativa"],
        )
        combo_gc.pack(side=tk.LEFT, padx=2)
        combo_gc.bind("<<ComboboxSelected>>", lambda _e: self._atualizar_grabcut())

        for txt, var in [("Gauss", self.gc_usar_gauss_var), ("Mediana", self.gc_usar_mediana_var), ("Bilateral", self.gc_usar_bilateral_var)]:
            tk.Checkbutton(
                topo, text=txt, variable=var, command=self._atualizar_grabcut,
                font=("Arial", 9, "bold"), cursor="hand2",
            ).pack(side=tk.LEFT, padx=3)

        blur_box = tk.Frame(topo)
        blur_box.pack(side=tk.LEFT, padx=6)
        tk.Label(blur_box, text="Blur ROI (ímpar)", font=("Arial", 9, "bold")).pack()
        tk.Scale(
            blur_box, from_=1, to=15, resolution=2, orient=tk.HORIZONTAL, variable=self.gc_blur_var,
            command=lambda _v: self._atualizar_grabcut(), length=110,
        ).pack()

        pad_box = tk.Frame(topo)
        pad_box.pack(side=tk.LEFT, padx=8)
        tk.Label(pad_box, text="Padding BBox (%)", font=("Arial", 9, "bold")).pack()
        tk.Scale(
            pad_box, from_=0, to=25, orient=tk.HORIZONTAL, variable=self.gc_pad_var,
            command=lambda _v: self._ao_mudar_padding_grabcut(), length=120,
        ).pack()

        iter_box = tk.Frame(topo)
        iter_box.pack(side=tk.LEFT, padx=8)
        tk.Label(iter_box, text="Iterações GrabCut", font=("Arial", 9, "bold")).pack()
        tk.Scale(
            iter_box, from_=1, to=10, orient=tk.HORIZONTAL, variable=self.gc_iter_var,
            command=lambda _v: self._atualizar_grabcut(), length=120,
        ).pack()

        grid = tk.Frame(self.tab_grabcut)
        grid.pack(expand=True, fill="both", padx=10, pady=5)
        for i in range(2):
            grid.columnconfigure(i, weight=1)
            grid.rowconfigure(i, weight=1)

        paineis_gc = [
            ("bbox_vis", "1. Detecção YOLO / ROI", 0, 0),
            ("roi_crop", "2. Recorte da ROI", 0, 1),
            ("mask_resultado", "3. Máscara (GrabCut / Otsu / Adaptativa)", 1, 0),
            ("objeto_isolado", "4. Objeto Isolado (Fundo Removido)", 1, 1),
        ]
        for chave, titulo, r, c in paineis_gc:
            painel = PainelImagem(
                grid, titulo=titulo, chave_origem="gc", chave_var=chave,
                on_zoom=lambda _o, ch, t=titulo: self._abrir_zoom_gc(ch, t),
                on_download=lambda _o, ch: self._salvar_gc_individual(ch),
            )
            painel.grid(row=r, column=c, sticky="nsew", padx=5, pady=5)
            self.grabcut_paineis[chave] = painel

    def _carregar_img_grabcut(self):
        caminho = filedialog.askopenfilename(
            title="Selecionar imagem para YOLO + GrabCut",
            filetypes=[("Imagens", "*.png *.jpg *.jpeg *.bmp *.tiff *.webp")],
        )
        if not caminho:
            return
        img = cv2.imread(caminho)
        if img is None:
            messagebox.showerror("Erro", "Não foi possível carregar a imagem.")
            return
        self.grabcut_img_bgr = img
        self.grabcut_rect = None
        self._executar_yolo_grabcut()

    def _executar_yolo_grabcut(self):
        if self.grabcut_img_bgr is None:
            messagebox.showwarning("Aviso", "Importe uma imagem primeiro.")
            return
        try:
            self.lbl_status.config(text="[YOLO + GrabCut] Executando detecção YOLO...")
            self.root.update_idletasks()
            rect, texto = detectar_roi_yolo(
                self.grabcut_img_bgr,
                padding_pct=self.gc_pad_var.get(),
                espaco_yolo=self.gc_espaco_yolo_var.get(),
            )
            self.grabcut_rect = rect
            self.grabcut_label = texto
            self._atualizar_grabcut()
        except Exception as exc:
            messagebox.showerror("Erro no YOLO/GrabCut", str(exc))

    def _ao_mudar_padding_grabcut(self):
        if self.grabcut_img_bgr is not None and self.grabcut_rect is not None and self.grabcut_label != "ROI Manual":
            self._executar_yolo_grabcut()

    def _atualizar_grabcut(self):
        if self.grabcut_img_bgr is None or self.grabcut_rect is None:
            return
        ug = self.gc_usar_gauss_var.get()
        k = self.gc_blur_var.get()
        um = self.gc_usar_mediana_var.get()
        ub = self.gc_usar_bilateral_var.get()
        it = self.gc_iter_var.get()

        if self.gc_comparar_3_var.get():
            r_gc = segmentar_roi(self.grabcut_img_bgr, self.grabcut_rect, "GrabCut", it, ug, k, um, ub, self.grabcut_label)
            r_ot = segmentar_roi(self.grabcut_img_bgr, self.grabcut_rect, "Otsu", it, ug, k, um, ub, self.grabcut_label)
            r_ad = segmentar_roi(self.grabcut_img_bgr, self.grabcut_rect, "Adaptativa", it, ug, k, um, ub, self.grabcut_label)
            self.grabcut_resultados = {
                "bbox_vis": r_gc["roi_crop"],
                "roi_crop": r_gc["mask_resultado"],
                "mask_resultado": r_ot["mask_resultado"],
                "objeto_isolado": r_ad["mask_resultado"],
            }
            self.lbl_status.config(text=f"[YOLO + Comparativo 3 Técnicas] {self.grabcut_label} | ROI={self.grabcut_rect}")
        else:
            metodo = self.gc_metodo_var.get()
            self.grabcut_resultados = segmentar_roi(
                self.grabcut_img_bgr, self.grabcut_rect, metodo=metodo, iteracoes=it,
                usar_gauss=ug, gauss_ksize=k, usar_mediana=um, usar_bilateral=ub, label=self.grabcut_label,
            )
            self.lbl_status.config(text=f"[YOLO + {metodo}] {self.grabcut_label} | ROI={self.grabcut_rect}")

        for chave, painel in self.grabcut_paineis.items():
            matriz = self.grabcut_resultados.get(chave)
            if matriz is not None:
                painel.atualizar_miniatura(matriz, (380, 260))

    def _abrir_seletor_roi_manual(self):
        if self.grabcut_img_bgr is None:
            messagebox.showwarning("Aviso", "Importe uma imagem primeiro.")
            return

        img_rgb = cv2.cvtColor(self.grabcut_img_bgr, cv2.COLOR_BGR2RGB)
        h_orig, w_orig = img_rgb.shape[:2]
        escala = min(800.0 / w_orig, 600.0 / h_orig, 1.0)
        disp_w, disp_h = int(w_orig * escala), int(h_orig * escala)

        win = tk.Toplevel(self.root)
        win.title("Arraste com o mouse para definir a ROI do GrabCut")
        canvas = tk.Canvas(win, width=disp_w, height=disp_h, cursor="cross")
        canvas.pack()

        pil = Image.fromarray(img_rgb).resize((disp_w, disp_h), Image.Resampling.LANCZOS)
        tk_img = ImageTk.PhotoImage(pil)
        canvas.create_image(0, 0, anchor=tk.NW, image=tk_img)
        canvas.image = tk_img

        estado = {"x0": 0, "y0": 0, "rect_id": None}

        def _press(e):
            estado["x0"], estado["y0"] = e.x, e.y
            if estado["rect_id"]:
                canvas.delete(estado["rect_id"])
            estado["rect_id"] = canvas.create_rectangle(e.x, e.y, e.x, e.y, outline="#00ff00", width=2)

        def _drag(e):
            canvas.coords(estado["rect_id"], estado["x0"], estado["y0"], e.x, e.y)

        def _release(e):
            x1, x2 = sorted([max(0, estado["x0"]), min(disp_w, e.x)])
            y1, y2 = sorted([max(0, estado["y0"]), min(disp_h, e.y)])
            if x2 - x1 > 5 and y2 - y1 > 5:
                self.grabcut_rect = (
                    int(x1 / escala), int(y1 / escala),
                    int((x2 - x1) / escala), int((y2 - y1) / escala),
                )
                self.grabcut_label = "ROI Manual"
                win.destroy()
                self._atualizar_grabcut()

        canvas.bind("<ButtonPress-1>", _press)
        canvas.bind("<B1-Motion>", _drag)
        canvas.bind("<ButtonRelease-1>", _release)

    def _abrir_zoom_gc(self, chave, titulo):
        matriz = self.grabcut_resultados.get(chave)
        if matriz is None:
            messagebox.showwarning("Aviso", "Execute a detecção YOLO ou desenhe a ROI primeiro.")
            return
        JanelaZoom(self.root, titulo, matriz)

    def _salvar_gc_individual(self, chave):
        matriz = self.grabcut_resultados.get(chave)
        if matriz is None:
            messagebox.showwarning("Aviso", "Nenhuma imagem disponível para salvar neste painel.")
            return
        caminho = filedialog.asksaveasfilename(
            defaultextension=".png", initialfile=f"grabcut_{chave}.png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg")],
        )
        if caminho:
            self._salvar_disco(matriz, caminho)
            messagebox.showinfo("Sucesso", f"Imagem salva em:\n{caminho}")

    # =========================================================================
    # ABA 4: COMPARATIVO SEM ROI vs PÓS-ROI (1 OU 2 IMAGENS - ATÉ 16 QUADROS)
    # =========================================================================
    def _setup_aba_comp_roi(self):
        topo = tk.Frame(self.tab_comp_roi)
        topo.pack(pady=8, fill=tk.X, padx=10)

        tk.Button(
            topo, text="🔄 Gerar / Atualizar Comparativo", command=self._atualizar_comp_roi,
            font=("Arial", 10, "bold"), bg="#2196F3", fg="white", padx=12, pady=5, cursor="hand2",
        ).pack(side=tk.LEFT, padx=6)

        self.comp_roi_2img_var = tk.BooleanVar(value=False)
        tk.Checkbutton(
            topo, text="⚖️ Comparar 2 Imagens (16 quadros)", variable=self.comp_roi_2img_var,
            command=self._ao_alternar_comp_roi_2img, font=("Arial", 10, "bold"),
            bg="#fff3e0", padx=10, pady=5, relief=tk.RAISED, cursor="hand2",
        ).pack(side=tk.LEFT, padx=6)

        self.comp_vis_vars = {
            "sem": tk.BooleanVar(value=True),
            "pos": tk.BooleanVar(value=True),
            "grabcut": tk.BooleanVar(value=True),
            "otsu": tk.BooleanVar(value=True),
            "adaptativa": tk.BooleanVar(value=True),
        }
        for chave_v, rotulo in [("sem", "Sem ROI"), ("pos", "Pós-ROI"), ("grabcut", "GrabCut"), ("otsu", "Otsu"), ("adaptativa", "Adaptativa")]:
            tk.Checkbutton(
                topo, text=rotulo, variable=self.comp_vis_vars[chave_v],
                command=self._reorganizar_comp_roi, font=("Arial", 9, "bold"), cursor="hand2",
            ).pack(side=tk.LEFT, padx=3)

        canvas = tk.Canvas(self.tab_comp_roi)
        scroll_y = ttk.Scrollbar(self.tab_comp_roi, orient="vertical", command=canvas.yview)
        self.comp_roi_grid = tk.Frame(canvas)
        self.comp_roi_grid.bind("<Configure>", lambda _e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=self.comp_roi_grid, anchor="nw")
        canvas.configure(yscrollcommand=scroll_y.set)
        scroll_y.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True, padx=8)

        for c in range(4):
            self.comp_roi_grid.columnconfigure(c, weight=1)

        sufixos = [
            ("sem_entrada", "Sem ROI - Entrada", 0, 0),
            ("sem_grabcut", "Sem ROI - GrabCut", 0, 1),
            ("sem_otsu", "Sem ROI - Otsu", 0, 2),
            ("sem_adaptativa", "Sem ROI - Adaptativa", 0, 3),
            ("pos_roi", "Pós-ROI - Recorte", 1, 0),
            ("pos_grabcut", "Pós-ROI - GrabCut", 1, 1),
            ("pos_otsu", "Pós-ROI - Otsu", 1, 2),
            ("pos_adaptativa", "Pós-ROI - Adaptativa", 1, 3),
        ]
        for prefixo, row_offset in [("a", 0), ("b", 2)]:
            for suf, titulo, r, c in sufixos:
                chave = f"{prefixo}_{suf}"
                t_full = f"{prefixo.upper()} {titulo}"
                p = PainelImagem(
                    self.comp_roi_grid, titulo=t_full, chave_origem="comp_roi", chave_var=chave,
                    on_zoom=lambda _o, ch, t=t_full: self._abrir_zoom_comp_roi(ch, t),
                    on_download=lambda _o, ch: self._salvar_comp_roi_individual(ch),
                )
                if prefixo == "a":
                    p.grid(row=r + row_offset, column=c, sticky="nsew", padx=4, pady=4)
                self.comp_roi_paineis[chave] = (p, r + row_offset, c)

    def _ao_alternar_comp_roi_2img(self):
        if self.comp_roi_2img_var.get() and not self.dados_imagens["comp_b"]:
            self.carregar_imagem("comp_b")
        self._reorganizar_comp_roi()
        self._atualizar_comp_roi()

    def _reorganizar_comp_roi(self):
        mostrar_b = self.comp_roi_2img_var.get()
        mostrar_sem = self.comp_vis_vars["sem"].get()
        mostrar_pos = self.comp_vis_vars["pos"].get()
        cols_ativas = {"entrada", "roi"}
        for k in ("grabcut", "otsu", "adaptativa"):
            if self.comp_vis_vars[k].get():
                cols_ativas.add(k)

        for chave, (p, r, c) in self.comp_roi_paineis.items():
            eh_b = chave.startswith("b_")
            eh_sem = "_sem_" in chave
            eh_pos = "_pos_" in chave
            suf_col = chave.split("_")[-1]
            visivel = (not eh_b or mostrar_b) and ((eh_sem and mostrar_sem) or (eh_pos and mostrar_pos)) and (suf_col in cols_ativas)
            if visivel:
                p.grid(row=r, column=c, sticky="nsew", padx=4, pady=4)
            else:
                p.grid_forget()

    def _descrever_filtros_roi(self):
        partes = []
        if self.gc_usar_gauss_var.get():
            partes.append(f"Gauss k={self.gc_blur_var.get()}")
        if self.gc_usar_mediana_var.get():
            partes.append("Med k=5")
        if self.gc_usar_bilateral_var.get():
            partes.append("Bilat")
        return "+".join(partes) if partes else "Sem filtro"

    def _atualizar_comp_roi(self):
        img_a = self.grabcut_img_bgr if self.grabcut_img_bgr is not None else self.seg_img_objeto_bgr
        if img_a is None:
            messagebox.showwarning("Aviso", "Importe uma imagem primeiro.")
            return

        alvos = [("a", img_a, self.grabcut_rect, self.grabcut_label)]
        if self.comp_roi_2img_var.get() and self.dados_imagens["comp_b"]:
            alvos.append(("b", self.dados_imagens["comp_b"]["variacoes"]["bgr"], None, "ROI"))

        pad = self.gc_pad_var.get()
        it = self.gc_iter_var.get()
        ug, k = self.gc_usar_gauss_var.get(), self.gc_blur_var.get()
        um, ub = self.gc_usar_mediana_var.get(), self.gc_usar_bilateral_var.get()
        desc_f = self._descrever_filtros_roi()

        self.comp_roi_resultados = {}
        for pref, img_bgr, rect, lbl in alvos:
            if rect is None:
                rect, lbl = detectar_roi_yolo(img_bgr, padding_pct=pad, espaco_yolo=self.gc_espaco_yolo_var.get())
                if pref == "a":
                    self.grabcut_rect, self.grabcut_label = rect, lbl

            sem = aplicar_segmentacao(img_bgr, canal="GRAY", usar_blur=True, blur_ksize=5, incluir_grabcut=True)
            pos_gc = segmentar_roi(img_bgr, rect, "GrabCut", it, ug, k, um, ub, lbl)
            pos_ot = segmentar_roi(img_bgr, rect, "Otsu", it, ug, k, um, ub, lbl)
            pos_ad = segmentar_roi(img_bgr, rect, "Adaptativa", it, ug, k, um, ub, lbl)

            dados = {
                f"{pref}_sem_entrada": (cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), f"{pref.upper()} Sem ROI - Entrada"),
                f"{pref}_sem_grabcut": (sem["mask_grabcut"], f"{pref.upper()} Sem ROI - GrabCut (it=3)"),
                f"{pref}_sem_otsu": (sem["mask_otsu"], f"{pref.upper()} Sem ROI - Otsu (Gauss k=5)"),
                f"{pref}_sem_adaptativa": (sem["mask_adaptativa"], f"{pref.upper()} Sem ROI - Adaptativa (b=21, C=5)"),
                f"{pref}_pos_roi": (pos_gc["roi_crop"], f"{pref.upper()} Pós-ROI ({lbl}, pad={pad}%)"),
                f"{pref}_pos_grabcut": (pos_gc["mask_resultado"], f"{pref.upper()} Pós-ROI GrabCut (it={it}, {desc_f})"),
                f"{pref}_pos_otsu": (pos_ot["mask_resultado"], f"{pref.upper()} Pós-ROI Otsu ({desc_f})"),
                f"{pref}_pos_adaptativa": (pos_ad["mask_resultado"], f"{pref.upper()} Pós-ROI Adaptativa ({desc_f})"),
            }
            for chave, (matriz, titulo) in dados.items():
                self.comp_roi_resultados[chave] = matriz
                painel, _, _ = self.comp_roi_paineis[chave]
                painel.definir_titulo(titulo)
                painel.atualizar_miniatura(matriz, (260, 190))

        self.lbl_status.config(text=f"[Comparativo] {len(alvos)} imagem(ns) | Filtros ROI: {desc_f} | pad={pad}% it={it}")

    def _abrir_zoom_comp_roi(self, chave, titulo):
        matriz = self.comp_roi_resultados.get(chave)
        if matriz is None:
            messagebox.showwarning("Aviso", "Clique em 'Gerar / Atualizar Comparativo' primeiro.")
            return
        JanelaZoom(self.root, titulo, matriz)

    def _salvar_comp_roi_individual(self, chave):
        matriz = self.comp_roi_resultados.get(chave)
        if matriz is None:
            messagebox.showwarning("Aviso", "Nenhuma imagem disponível para salvar.")
            return
        caminho = filedialog.asksaveasfilename(
            defaultextension=".png", initialfile=f"comparativo_{chave}.png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg")],
        )
        if caminho:
            self._salvar_disco(matriz, caminho)
            messagebox.showinfo("Sucesso", f"Imagem salva em:\n{caminho}")


if __name__ == "__main__":
    janela = tk.Tk()
    app = AnalisadorCoresApp(janela)
    janela.mainloop()
