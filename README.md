# 🍌🍾 Lixeira Inteligente PDI - Classificação de Resíduos em Tempo Real

Projeto acadêmico de **Processamento Digital de Imagens (PDI)** e **Visão Computacional** desenvolvido na Universidade Federal do Maranhão (**UFMA**).

O objetivo do sistema é atuar como uma **Lixeira Inteligente Automatizada**, identificando e separando resíduos descartados em duas classes representativas:
- **Orgânico**: Representado por **Bananas** (frutas inteiras, cascas, pedaços).
- **Reciclável**: Representado por **Garrafas Plásticas PET** (transparentes, amassadas, deitadas, em pé).
- **Indefinido**: Rejeição de fundos vazios, objetos fora de escopo (ex: maçãs, teclados, mãos) ou predições com baixa confiança.

---

## 🏗 Arquitetura do Pipeline Híbrido

O projeto adota uma arquitetura híbrida de alto desempenho que une **Deep Learning** moderno (para localização da Região de Interesse - ROI) a **Técnicas Clássicas de PDI** (para extração determinística e interpretável de características):

```
       Câmera (DroidCam WiFi / USB V4L2)
                       │
                       ▼
    [1. Detecção e Localização de ROI]
    YOLO26n (Nano, NMS-free, imgsz=320, Frame-Skipping)
                       │
       ┌───────────────┴───────────────┐
   Sucesso                         Falha
       │                               │
  BBox YOLO                  [Fallback Central PDI]
       │                               │
       └───────────────┬───────────────┘
                       │
                       ▼
          [2. Pré-processamento PDI]
    - Redimensionamento proporcional (256px max)
    - Filtro Gaussiano 5x5 (redução de ruído)
    - Limiarização de Otsu Duplo (Canal V + Canal S)
    - Morfologia Matemática (Abertura + Fechamento)
    - Máscara Sólida via Maior Contorno e HUD didático (Convex Hull e minAreaRect)
                       │
                       ▼
        [3. Extração de 20 Características]
    • Forma (10): 7 Momentos Invariantes de Hu +
                  Circularidade + Razão de Aspecto Invariante (minAreaRect) +
                  Solidez via Fecho Convexo (Convex Hull)
    • Textura (4): GLCM (Contraste, Dissimilaridade, Homogeneidade, Energia)
    • Cor (6): Médias e Desvios Padrão no Espaço HSV
                       │
                       ▼
        [4. Classificador Machine Learning]
        Support Vector Machine (SVM) com Kernel RBF
        + StandardScaler + Calibração de Probabilidade
                       │
                       ▼
   Decisão Final & Acionamento Serial (ESP32)
   ['O' = Orgânico | 'R' = Reciclável | 'I' = Indefinido]
```

---

## 🚀 Funcionalidades Principais

1. **Detector Ultralytics YOLO26n**:
   - Versão Nano mais recente, otimizada para borda (edge AI) com inferência nativa sem NMS.
   - Localiza garrafas (classe 39) e bananas (classe 46) com alta velocidade.

2. **Descritores Invariantes à Rotação**:
   - Momentos de Hu e Razão de Aspecto calculada via Retângulo Mínimo Delimitador (`cv2.minAreaRect`).
   - Bananas e garrafas são reconhecidas com a mesma precisão em qualquer orientação (horizontal, vertical ou inclinada).

3. **Fallback Inteligente com Rejeição de Falsos Positivos**:
   - Caso o YOLO falhe, o sistema analisa o centro da lixeira.
   - **Área Mínima de Contorno**: Se a lixeira estiver vazia ou com ruído inferior a 1500 pixels, o sistema não força classificação e exibe *"Lixeira Vazia"*.
   - **Validação de Assinatura HSV**: Objetos com cores totalmente fora do escopo (ex: maçãs vermelhas ou fundos escuros) são marcados como *"Indefinido"*.
   - **Limiar de Confiança Estrito (70%)**: Predições ambíguas são classificadas como *"Indefinido"*.

4. **Interface Gráfica de Controle Leve (Tkinter)**:
   - **Suporte Duplo de Câmera**: Conexão com webcam USB local ou celular via DroidCam WiFi (`http://IP:4747/video`) com thread assíncrona protegida contra travamentos.
   - **Resolução de Exibição Configurável**: Opções para 800x450 (padrão notebook), 640x360, 960x540, 1280x720 e Nativo, evitando que feeds de 1080p do celular tomem a tela inteira.
   - **Janelas Modulares X11 com HUD Didático**: Abertura independente dos filtros de PDI (*ROI*, *Otsu*, *Morfologia*, *Máscara com HUD* exibindo contorno, Convex Hull e minAreaRect) com detecção automática de fechamento pelo botão `[X]` (sem reabertura em loop).
   - **Modo Imagem Estática**: Botão `📂 Testar Imagem` para carregar fotos do dataset ou testes mesmo sem câmera conectada.
   - **Coleta e Treino em 1 Clique**: Botões com atalhos de teclado para salvar amostras (`[O]`, `[R]`) e disparar retreinamento em segundo plano (`[T]`) com hot-reload na memória.

---

## 📦 Instalação e Requisitos

### Pré-requisitos
- Python 3.10+ (testado no Linux com X11/Wayland)
- Câmera USB ou celular com aplicativo DroidCam instalado

### Passo a Passo

1. **Clone o repositório:**
   ```bash
   git clone https://github.com/seu-usuario/bananas-e-garrafas-pdi.git
   cd "Bananas e Garrafas - PDI"
   ```

2. **Crie e ative o ambiente virtual:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Instale as dependências:**
   ```bash
   pip install -r requirements.txt
   ```

---

## 🎮 Como Usar

### 1. Iniciar a Aplicação em Tempo Real
```bash
python main_realtime.py
```
- A interface abrirá com a câmera desligada.
- Selecione **"Dispositivo Local"** ou **"DroidCam WiFi / IP"** (inserindo a URL informada no app do celular).
- Clique em **"▶ Conectar Câmera"**.
- Ajuste a resolução da janela de vídeo se necessário (padrão 800x450).

### 2. Teclas de Atalho Rápidas
| Tecla | Ação |
| :---: | :--- |
| **`O`** | Salva o frame atual como amostra **Orgânica** (`dataset/organico/`) |
| **`R`** | Salva o frame atual como amostra **Reciclável** (`dataset/reciclavel/`) |
| **`T`** | Dispara o **treinamento do classificador** em segundo plano |
| **`Espaço`** | Simula o **gatilho ultrassônico** (envia decisão via Serial para o ESP32) |
| **`Esc` / `Q`** | Fecha a janela de vídeo ativa |

### 3. Treinamento Manual pelo Terminal (Opcional)
```bash
python train_model.py
```
Executa a validação cruzada estratificada (5-Fold Stratified), exibe matriz de confusão e métricas (Precision, Recall, F1-Score) e salva o modelo em `models/classifier_pipeline.pkl`.

### 4. Download / Expansão do Dataset (Opcional)
```bash
python download_datasets.py
```
Baixa imagens de estúdio em fundo preto e higieniza arquivos corrompidos automaticamente.

---

## 📊 Estrutura de Arquivos do Projeto

```
├── config.py             # Configurações globais (portas de câmera, parâmetros YOLO, classes, serial)
├── pipeline.py           # Pipeline híbrido: ROI YOLO26n + Pré-processamento PDI + 20 Features
├── main_realtime.py      # Aplicação principal (GUI Tkinter + OpenCV com Janelas Modulares)
├── train_model.py        # Pipeline de treinamento (StandardScaler + SVM RBF / KNN + StratifiedKFold)
├── download_datasets.py  # Script de aquisição e higienização automática de imagens de treino
├── requirements.txt      # Dependências do projeto
├── dataset/              # Imagens organizadas em organico/ e reciclavel/ + features.csv
├── models/               # Modelos treinados (.pkl)
└── testes/               # Casos de teste e capturas de avaliação de falsos positivos
```

---

## 🎓 Autoria
Desenvolvido como projeto prático da disciplina de **Processamento Digital de Imagens (PDI)** — **Universidade Federal do Maranhão (UFMA)**.
