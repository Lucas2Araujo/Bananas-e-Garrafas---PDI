# Contexto do Repositório (repo_context.md)

Este documento foi elaborado para fornecer aos agentes de IA e desenvolvedores uma compreensão técnica aprofundada da arquitetura, decisões de design, pipelines e convenções do repositório **Bananas e Garrafas - PDI**.

---

## 1. Visão Geral e Propósito do Sistema

O projeto é uma bancada de visão computacional para automação de uma **Lixeira Inteligente** capaz de segregar resíduos em tempo real.
- **Classe 0 (Orgânico)**: Banana (fruta, casca, fatias).
- **Classe 1 (Reciclável)**: Garrafa PET de plástico (transparente, com ou sem rótulo, amassada ou íntegra).
- **Estado Indefinido**: Rejeição de ruídos, fundos vazios e objetos fora do escopo (ex: maçãs, copos, teclados, mãos) quando a confiança estatística for insuficiente ou a assinatura de cor não for compatível.

---

## 2. Mapa dos Componentes e Arquivos

### [config.py](file:///home/loko/Documentos/ufMA/Bananas%20e%20Garrafas%20-%20PDI/config.py)
- **Função**: Central de constantes e configurações do sistema.
- **Destaques**:
  - `scan_available_cameras()`: Varredura instantânea via `/sys/class/video4linux` no Linux, evitando que loops de V4L2 travem o processo.
  - `YOLO_MODEL_PATH = "yolo26n.pt"`: Aponta para a rede neural YOLO26 Nano.
  - `YOLO_SKIP_FRAMES = 3`, `YOLO_IMGSZ = 320`: Otimizações para FPS elevado em hardware modesto ou acelerador OpenVINO.
  - `CLASS_NAMES = {0: "Organico (Banana)", 1: "Reciclavel (Garrafa)", 2: "Indefinido"}`.
  - Comunicação serial com ESP32 via `ENABLE_SERIAL = False` (porta `/dev/ttyUSB0` @ 115200 bps).

### [pipeline.py](file:///home/loko/Documentos/ufMA/Bananas%20e%20Garrafas%20-%20PDI/pipeline.py) - Classe `HybridPDIPipeline`
- **Função**: Núcleo do processamento de imagem e extração determinística de características.
- **Métodos Principais**:
  - `extract_roi(frame, force_infer=False, allow_fallback=True)`:
    - Executa o detector YOLO26n nas classes COCO 39 (*bottle*) e 46 (*banana*).
    - Aplica padding de 10% na bounding box detectada.
    - Se o YOLO falhar e o fallback estiver ativo (`allow_fallback=True`), retorna o recorte central (`FALLBACK_CENTER`, 70% central da imagem). Se `allow_fallback=False`, retorna `None`.
  - `segment_and_extract(roi)`:
    - Redimensiona proporcionalmente mantendo a razão de aspecto (`max_dim=256`).
    - Aplica Filtro Gaussiano 5x5 e converte para HSV.
    - **Otsu Duplo com Inversão de Borda**: Combina binarização do canal V (brilho/reflexos) com o canal S (saturação). Inverte o canal V automaticamente se a borda da ROI for clara (mesa/fundo iluminado).
    - **Morfologia Matemática**: Abertura (eliminação de ruído) com elipse 5x5, seguida de Fechamento (união de paredes e reflexos) com elipse 19x19.
    - Preenchimento do maior contorno para obter a máscara sólida com HUD didático visual (Contorno, Convex Hull e minAreaRect).
    - **Validação de Área Mínima (`min_contour_area = 1500px`)**: Se a área do maior contorno for inferior a 1500 px, a imagem é descartada como vazia (`return None, debug_views`).
    - **Validação de Assinatura HSV**: Verifica se a média de cor bate com o escopo de banana ou garrafa PET no fallback, marcando `last_color_valid`.
    - **Vetor de 20 Features**:
      - 7 Momentos de Hu log-transformados ($Hu_1 \dots Hu_7$) — Invariantes a translação, rotação e escala.
      - Circularidade: $\frac{4 \pi \cdot \text{Área}}{\text{Perímetro}^2}$.
      - Razão de Aspecto Invariante via `cv2.minAreaRect`: $\frac{\text{Eixo Menor}}{\text{Eixo Maior}}$.
      - Solidez via Fecho Convexo (`cv2.convexHull`): $\frac{\text{Área}}{\text{Área do Convex Hull}}$ (diferencia corpos compactos de cascas abertas/tiras com reentrâncias).
      - 4 métricas de Textura GLCM (0°, 45°, 90°, 135° restrito ao objeto): Contraste, Dissimilaridade, Homogeneidade, Energia.
      - 6 métricas de Cor HSV: Médias e Desvios Padrão dos canais H, S e V.

### [main_realtime.py](file:///home/loko/Documentos/ufMA/Bananas%20e%20Garrafas%20-%20PDI/main_realtime.py) - Classe `PDIApplication`
- **Função**: Ponto de entrada com Interface Gráfica Tkinter desacoplada e gerenciador de janelas OpenCV.
- **Destaques de Engenharia**:
  - **Inicialização Desconectada**: O app abre sem travar esperando sinal de vídeo.
  - **Thread Assíncrona `CameraThread`**: Conecta via URL IP do DroidCam ou V4L2 em background com timeout, sem congelar a interface Tkinter.
  - **Tamanho de Janela de Vídeo Configurável**: Resoluções de exibição (800x450, 640x360, 960x540, 1280x720 e Nativo) via `cv2.namedWindow(..., cv2.WINDOW_NORMAL)` e `cv2.resize`.
  - **Gerenciamento Inteligente de Janelas X11**: Monitoramento contínuo de `cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1` para detectar quando o usuário clica no `[X]` da janela, sincronizando o checkbox na UI sem reabrir em loop.
  - **Filtros sob demanda**: Se um filtro não estiver ativado, sua janela não é criada nem desenhada, economizando recursos.
  - **Decisão Estrita PDI**: O YOLO26n atua como localizador da ROI; a classificação final decorre 100% da extração das 20 características de PDI avaliadas pelo classificador SVM.
  - **Hot-Reload de Treinamento**: Treina em thread separada com feedback em console e recarrega `clf_model` em tempo de execução.

### [train_model.py](file:///home/loko/Documentos/ufMA/Bananas%20e%20Garrafas%20-%20PDI/train_model.py)
- **Função**: Treinamento, validação e serialização do classificador scikit-learn.
- **Pipeline scikit-learn**:
  - `StandardScaler()`: Normalização de variância unitária (essencial para Hu, GLCM e HSV conviverem).
  - `CalibratedClassifierCV(SVC(kernel='rbf', C=1.0))`: Suporte a probabilidades calibradas sem avisos de depreciação do scikit-learn moderno.
- **Validação Rigorosa e Comparativo**: Avaliação Out-of-Fold honesta (`cross_val_predict`), matriz de confusão e comparativo com KNN e Random Forest no mesmo StratifiedKFold (sem data leakage).
- **Exportação**:
  - `models/classifier_pipeline.pkl`: Pipeline serializado via joblib.
  - `dataset/features.csv`: Tabela com as 20 features e rótulo para auditoria acadêmica.

### [download_datasets.py](file:///home/loko/Documentos/ufMA/Bananas%20e%20Garrafas%20-%20PDI/download_datasets.py)
- **Função**: Utilitário para raspagem automatizada via Bing Image Crawler (`icrawler`).
- Focado em imagens de estúdio com fundo preto isolado e diferentes poses (verticais, penduradas, deitadas, amassadas), acompanhado de rotina de limpeza de imagens corrompidas ou minúsculas.

---

## 3. Decisões de Design Importantes

1. **Por que SVM (RBF) e não KNN?**
   - O vetor possui 20 dimensões contínuas com magnitudes muito discrepantes. O SVM com kernel RBF cria margens de decisão ótimas e generaliza com alta robustez a variações de iluminação, enquanto o KNN sofre de sensibilidade a outliers e degradação pela distância euclidiana em 20 dimensões.

2. **Por que Razão de Aspecto via `minAreaRect`?**
   - A razão cartesiana clássica $\frac{W_{box}}{H_{box}}$ variava drasticamente se uma banana estava em pé ($\approx 0.3$) ou deitada ($\approx 2.5$). Usando os eixos menor e maior da elipse/retângulo mínimo orientado, o valor representa a excentricidade intrínseca da geometria do objeto, sendo 100% invariante ao ângulo.

3. **Por que o estado "Indefinido"?**
   - Um classificador fechado em 2 classes sempre forçaria uma atribuição (com $P > 0.5$). O estado "Indefinido" atua como mecanismo de rejeição (Open-Set Recognition), barrando predições com probabilidade $< 0.70$ ou objetos em Fallback com assinaturas de cor destoantes (ex: maçãs, mesas, mãos).
