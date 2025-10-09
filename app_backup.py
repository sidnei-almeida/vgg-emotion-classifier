import os
import io
import time
import platform
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import tensorflow as tf
from tensorflow import keras
import plotly.graph_objects as go
from streamlit_option_menu import option_menu
import cv2
from streamlit_image_select import image_select
import json

# Base do app
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
TRAINING_DIR = os.path.join(BASE_DIR, "training")

# Configuração das emoções
EMOTIONS = {
    0: "angry",
    1: "disgust",
    2: "fear",
    3: "happy",
    4: "neutral",
    5: "sad",
    6: "surprise"
}

# Mensagens criativas para cada emoção
EMOTION_MESSAGES = {
    "angry": "😠 Parece que alguém acordou com o pé esquerdo hoje! Mantenha a calma!",
    "disgust": "🤢 Eca! Algo te deixou enojado. Vamos melhorar esse astral?",
    "fear": "😨 Medo? Não tema! Você é mais forte do que pensa!",
    "happy": "😄 Alegria pura! Continue espalhando esse sorriso contagiante!",
    "neutral": "😐 Neutro como uma segunda-feira. Vamos adicionar um pouco de cor?",
    "sad": "😢 Tristeza no ar... Lembre-se que depois da chuva vem o arco-íris!",
    "surprise": "😲 Uau! Que surpresa! O mundo está cheio de coisas inesperadas!"
}

st.set_page_config(
    page_title="Facial Emotion Classifier • AI",
    page_icon="😊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Estilo premium dark - Tema de Emoções com acentos coloridos
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
:root {
  --primary: #8b5cf6;
  --accent: #06b6d4;
  --success: #10b981;
  --danger: #ef4444;
  --warning: #f59e0b;
  --happy: #fbbf24;
  --sad: #6b7280;
  --angry: #dc2626;
  --fear: #7c3aed;
  --surprise: #06b6d4;
  --disgust: #84cc16;
  --neutral: #6b7280;
  --dark-bg: #0f172a;
  --card-bg: #1e293b;
  --sidebar-bg: #1e293b;
  --text: #f1f5f9;
  --text-secondary: #cbd5e1;
  --muted: #64748b;
  --border: #334155;
  --shadow: 0 4px 16px rgba(0,0,0,0.3);
}
.stApp { 
  background: var(--dark-bg); 
  color: var(--text); 
}
.main .block-container { 
  max-width: none !important; 
  padding-left: 1.5rem; 
  padding-right: 1.5rem; 
}

h1, h2, h3, h4, h5 { 
  font-family: 'Inter', sans-serif; 
  color: var(--text);
  font-weight: 600;
}

h1 { font-size: 1.75rem; }
h2 { font-size: 1.5rem; }
h3 { font-size: 1.25rem; }
h4 { font-size: 1rem; }

/* Hero Title */
.main-hero {
  font-size: 2rem;
  font-weight: 700;
  margin: 0.75rem 0 1rem;
  display: flex;
  align-items: center;
  gap: 0.875rem;
}
.title-gradient {
  background: linear-gradient(135deg, var(--primary), var(--accent));
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
}
.subtitle {
  color: var(--text-secondary);
  margin-bottom: 1.5rem;
  font-size: 0.938rem;
}

/* Emotion icon */
.emotion-icon {
  width: 54px;
  height: 54px;
  background: linear-gradient(135deg, var(--primary), var(--accent));
  border-radius: 50%;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  font-size: 1.5rem;
  box-shadow: 0 4px 12px rgba(139, 92, 246, 0.3);
}

/* Cards */
.card { 
  background: var(--card-bg); 
  border: 1px solid var(--border); 
  border-radius: 8px; 
  padding: 1rem; 
  box-shadow: var(--shadow); 
  margin-bottom: 0.875rem;
  font-size: 0.875rem;
}
.metric-card { 
  background: linear-gradient(135deg, rgba(255,107,53,0.05), rgba(247,147,30,0.05)); 
  border: 1px solid rgba(255,107,53,0.15); 
  border-radius: 8px; 
  padding: 0.875rem; 
  text-align: center;
}
.metric-value {
  font-size: 1.5rem;
  font-weight: 700;
  color: var(--primary);
  margin: 0.375rem 0;
}
.metric-label {
  font-size: 0.688rem;
  text-transform: uppercase;
  letter-spacing: 0.5px;
  color: var(--muted);
  font-weight: 600;
}

/* Badge */
.badge { 
  display: inline-block; 
  padding: 0.25rem 0.625rem; 
  border-radius: 4px; 
  font-size: 0.75rem; 
  font-weight: 600;
  border: 1px solid;
}
.badge-primary {
  background: rgba(255,107,53,0.12); 
  border-color: rgba(255,107,53,0.25); 
  color: var(--primary);
}
.badge-success {
  background: rgba(82,183,136,0.12); 
  border-color: rgba(82,183,136,0.25); 
  color: var(--success);
}
.badge-danger {
  background: rgba(230,57,70,0.12); 
  border-color: rgba(230,57,70,0.25); 
  color: var(--danger);
}

/* Emotion detection result */
.emotion-result {
  background: linear-gradient(135deg, var(--card-bg), rgba(139, 92, 246, 0.1));
  border: 2px solid var(--primary);
  border-radius: 12px;
  padding: 1.5rem;
  text-align: center;
  margin: 1rem 0;
  box-shadow: 0 8px 24px rgba(139, 92, 246, 0.2);
}

.emotion-emoji {
  font-size: 3rem;
  margin-bottom: 0.5rem;
  display: block;
}

.emotion-text {
  font-size: 1.25rem;
  font-weight: 600;
  color: var(--text);
  margin-bottom: 0.5rem;
}

.emotion-confidence {
  font-size: 0.875rem;
  color: var(--text-secondary);
}

hr { 
  border-top: 1px solid var(--border); 
  margin: 1.5rem 0;
}

/* Custom alerts */
.stAlert {
  background: var(--card-bg) !important;
  border: 1px solid var(--border) !important;
  border-radius: 8px !important;
  padding: 0.875rem !important;
  font-size: 0.875rem !important;
}

.stAlert > div {
  gap: 0.5rem !important;
}

/* Success alert */
div[data-baseweb="notification"][kind="success"] {
  background: rgba(82,183,136,0.08) !important;
  border-left: 3px solid var(--success) !important;
  color: var(--text) !important;
}

/* Error alert */
div[data-baseweb="notification"][kind="error"] {
  background: rgba(230,57,70,0.08) !important;
  border-left: 3px solid var(--danger) !important;
  color: var(--text) !important;
}

/* Warning alert */
div[data-baseweb="notification"][kind="warning"] {
  background: rgba(255,165,0,0.08) !important;
  border-left: 3px solid var(--warning) !important;
  color: var(--text) !important;
}

/* Info alert */
div[data-baseweb="notification"][kind="info"] {
  background: rgba(255,107,53,0.08) !important;
  border-left: 3px solid var(--primary) !important;
  color: var(--text) !important;
}

/* Streamlit buttons */
.stButton > button {
  background: linear-gradient(135deg, rgba(255,107,53,0.12), rgba(247,147,30,0.12)) !important;
  border: 1px solid rgba(255,107,53,0.25) !important;
  border-radius: 6px !important;
  color: var(--text) !important;
  font-size: 0.875rem !important;
  font-weight: 600 !important;
  padding: 0.5rem 1rem !important;
  transition: all 0.2s ease !important;
}

.stButton > button:hover {
  background: linear-gradient(135deg, rgba(255,107,53,0.2), rgba(247,147,30,0.2)) !important;
  border-color: rgba(255,107,53,0.4) !important;
  transform: translateY(-1px);
}

.stButton > button[kind="primary"] {
  background: linear-gradient(135deg, var(--primary), var(--accent)) !important;
  border: 1px solid var(--primary) !important;
  color: white !important;
}

.stButton > button[kind="primary"]:hover {
  background: linear-gradient(135deg, var(--accent), var(--primary)) !important;
  box-shadow: 0 4px 12px rgba(255,107,53,0.3);
}

/* Sliders */
.stSlider {
  padding: 0.5rem 0;
}

.stSlider > div > div > div {
  background: var(--border) !important;
}

.stSlider > div > div > div > div {
  background: var(--primary) !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
  gap: 0.5rem;
  background: transparent;
}

.stTabs [data-baseweb="tab"] {
  background: var(--card-bg);
  border: 1px solid var(--border);
  border-radius: 6px 6px 0 0;
  color: var(--text-secondary);
  font-size: 0.875rem;
  font-weight: 600;
  padding: 0.625rem 1.25rem;
}

.stTabs [data-baseweb="tab"]:hover {
  background: var(--sidebar-bg);
  color: var(--text);
}

.stTabs [aria-selected="true"] {
  background: linear-gradient(135deg, rgba(255,107,53,0.12), rgba(247,147,30,0.12));
  border-bottom-color: var(--primary);
  color: var(--primary);
}

/* Download button */
.stDownloadButton > button {
  background: linear-gradient(135deg, var(--success), rgba(82,183,136,0.8)) !important;
  border: 1px solid var(--success) !important;
  color: white !important;
}

.stDownloadButton > button:hover {
  background: linear-gradient(135deg, rgba(82,183,136,0.8), var(--success)) !important;
  box-shadow: 0 4px 12px rgba(82,183,136,0.3);
}

/* File uploader */
.stFileUploader {
  background: var(--card-bg);
  border: 1px dashed var(--border);
  border-radius: 8px;
  padding: 1rem;
}

.stFileUploader:hover {
  border-color: var(--primary);
  background: rgba(255,107,53,0.03);
}

/* Image selector do streamlit-image-select */
/* Container principal do image_select */
div[data-testid="column"] > div > div {
  display: flex !important;
  flex-wrap: wrap !important;
  gap: 0.75rem !important;
  width: 100% !important;
}

/* Cada imagem individual no image_select - 5 por linha com espaço distribuído */
div[data-testid="column"] > div > div > div {
  flex: 1 1 calc(20% - 0.6rem) !important;
  min-width: 150px !important;
  margin: 0 !important;
}

/* Imagens dentro do image_select - altura proporcional */
div[data-testid="column"] > div > div > div img {
  width: 100% !important;
  height: auto !important;
  aspect-ratio: 4/3 !important;
  object-fit: cover !important;
  border: 2px solid var(--border) !important;
  border-radius: 8px !important;
  cursor: pointer !important;
  transition: all 0.2s ease !important;
}

/* Hover nas imagens do image_select */
div[data-testid="column"] > div > div > div img:hover {
  transform: translateY(-2px) !important;
  border-color: var(--primary) !important;
  box-shadow: 0 6px 16px rgba(255,107,53,0.25) !important;
}

/* Legenda das imagens */
div[data-testid="column"] > div > div > div p {
  font-size: 0.75rem !important;
  color: var(--text-secondary) !important;
  text-align: center !important;
  margin-top: 0.5rem !important;
}

/* Imagens normais do Streamlit (Analytics, etc) - manter tamanho normal */
div[data-testid="stImage"]:not(div[data-testid="column"] div[data-testid="stImage"]) img {
  border: 2px solid var(--border);
  border-radius: 8px;
  transition: all 0.2s ease;
}

div[data-testid="stImage"]:not(div[data-testid="column"] div[data-testid="stImage"]) img:hover {
  transform: translateY(-2px);
  border-color: var(--primary);
  box-shadow: 0 6px 16px rgba(255,107,53,0.25);
}

/* Responsive */
@media (max-width: 768px) {
  .main-hero { font-size: 1.5rem; }
  .card { padding: 0.875rem; }
  .metric-card { padding: 0.75rem; }
  
  /* Image select mobile - 2 por linha */
  div[data-testid="column"] > div > div > div {
    flex: 1 1 calc(50% - 0.375rem) !important;
    max-width: calc(50% - 0.375rem) !important;
    min-width: 120px !important;
  }
}
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def load_emotion_model():
    """Carrega o modelo de classificação de emoções faciais"""
    model_path = os.path.join(MODELS_DIR, "emotion_model.keras")
    if not os.path.exists(model_path):
        st.error(f"Modelo de emoções não encontrado em {model_path}")
        return None

    try:
        model = keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Erro ao carregar modelo de emoções: {str(e)}")
        return None

@st.cache_resource(show_spinner=False)
def load_face_cascade():
    """Carrega o classificador Haar Cascade para detecção de rostos"""
    cascade_path = os.path.join(BASE_DIR, "haarcascade_frontalface_default.xml")
    if not os.path.exists(cascade_path):
        st.error(f"Arquivo Haar Cascade não encontrado em {cascade_path}")
        return None

    try:
        face_cascade = cv2.CascadeClassifier(cascade_path)
        return face_cascade
    except Exception as e:
        st.error(f"Erro ao carregar Haar Cascade: {str(e)}")
        return None


@st.cache_data(show_spinner=False)
def load_training_data():
    """Carrega dados de treinamento do modelo de emoções"""
    summary_path = os.path.join(TRAINING_DIR, "training_summary.json")

    summary = None

    if os.path.exists(summary_path):
        with open(summary_path, "r") as f:
            summary = json.load(f)

    return summary


def get_env_status():
    """Retorna informações do ambiente"""
    # TensorFlow info
    tf_version = tf.__version__
    physical_devices = tf.config.list_physical_devices()

    gpu_devices = [d for d in physical_devices if d.device_type == 'GPU']
    if gpu_devices:
        device_name = "GPU Available"
    else:
        device_name = platform.processor() or "CPU"

    return {
        "device": "GPU" if gpu_devices else "CPU",
        "device_name": device_name,
        "tensorflow": tf_version,
        "python": platform.python_version(),
    }


def preprocess_face(image: Image.Image, face_cascade):
    """Pré-processa uma imagem para classificação de emoções"""
    # Converter PIL para numpy array
    img_np = np.array(image.convert("RGB"))

    # Converter para escala de cinza para detecção de rosto
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

    # Detectar rostos
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) == 0:
        return None, None, "Nenhum rosto detectado na imagem"

    # Usar o primeiro rosto detectado
    (x, y, w, h) = faces[0]

    # Recortar o rosto
    face_roi = gray[y:y+h, x:x+w]

    # Redimensionar para 48x48 (tamanho esperado pelo modelo)
    resized_face = cv2.resize(face_roi, (48, 48), interpolation=cv2.INTER_AREA)

    # Normalizar pixels
    normalized_face = resized_face / 255.0

    # Expandir dimensões para formato do modelo (1, 48, 48, 1)
    reshaped_face = np.reshape(normalized_face, (1, 48, 48, 1))

    return reshaped_face, (x, y, w, h), None


def predict_emotion(model, processed_face):
    """Faz a predição da emoção"""
    if model is None or processed_face is None:
        return None

    try:
        predictions = model.predict(processed_face, verbose=False)
        predicted_class = np.argmax(predictions[0])
        confidence = np.max(predictions[0])

        return {
            "emotion": EMOTIONS[predicted_class],
            "confidence": float(confidence),
            "probabilities": {EMOTIONS[i]: float(prob) for i, prob in enumerate(predictions[0])}
        }
    except Exception as e:
        st.error(f"Erro na predição: {str(e)}")
        return None


def page_home(model, summary, results_df):
    """Página inicial"""
    st.markdown(
        '<div class="main-hero">\
          <div class="plate-icon">ABC1D23</div>\
          <span class="title-gradient">Brazilian License Plate Recognition</span>\
        </div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="subtitle">Sistema ALPR com YOLOv8 para detecção de placas Mercosul</div>', unsafe_allow_html=True)
    
    # Status cards
    col1, col2, col3 = st.columns(3)
    with col1:
        status = "Carregado" if model else "Erro"
        badge_class = "badge-success" if model else "badge-danger"
        st.markdown(f"""
<div class="metric-card">
  <p class="metric-label">Modelo YOLO</p>
  <span class="badge {badge_class}">{status}</span>
</div>
""", unsafe_allow_html=True)
    
    with col2:
        num_images = len(_gather_test_images())
        st.markdown(f"""
<div class="metric-card">
  <p class="metric-label">Imagens de Teste</p>
  <div class="metric-value">{num_images}</div>
</div>
""", unsafe_allow_html=True)
    
    with col3:
        if summary and 'best_model_metrics' in summary:
            map_val = summary['best_model_metrics'].get('mAP50-95(B)', 0)
            map_text = f"{map_val:.1%}"
        else:
            map_text = "N/A"
        st.markdown(f"""
<div class="metric-card">
  <p class="metric-label">mAP@50-95</p>
  <div class="metric-value">{map_text}</div>
</div>
""", unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Destaques
    if summary and 'best_model_metrics' in summary:
        st.markdown('<h3 style="color: var(--text); font-size: 1.125rem; margin: 1.5rem 0 0.75rem 0;">Model Performance</h3>', unsafe_allow_html=True)
        metrics = summary['best_model_metrics']
        
        col1, col2, col3, col4 = st.columns(4)
        metrics_display = [
            (col1, "Precision", metrics.get('precision(B)', 0), "--primary"),
            (col2, "Recall", metrics.get('recall(B)', 0), "--success"),
            (col3, "mAP@50", metrics.get('mAP50(B)', 0), "--primary"),
            (col4, "mAP@50-95", metrics.get('mAP50-95(B)', 0), "--accent"),
        ]
        
        for col, label, value, color in metrics_display:
            with col:
                st.markdown(f"""
<div class="card" style="text-align: center;">
  <p style="font-size: 0.75rem; color: var(--muted); margin: 0;">{label}</p>
  <p style="font-size: 1.75rem; font-weight: 700; color: var({color}); margin: 0.5rem 0;">{value:.1%}</p>
</div>
""", unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Visualizações
    st.markdown('<h3 style="color: var(--text); font-size: 1.125rem; margin: 1.5rem 0 0.75rem 0;">Training Results</h3>', unsafe_allow_html=True)
    _, _, _, images = load_training_data()
    
    col1, col2 = st.columns(2)
    with col1:
        if os.path.exists(images["results"]):
            st.image(images["results"], caption="Training Evolution", use_container_width=True)
    with col2:
        if os.path.exists(images["confusion_norm"]):
            st.image(images["confusion_norm"], caption="Confusion Matrix", use_container_width=True)


def page_detect(model):
    """Página de detecção"""
    st.markdown('<h2 style="color: var(--primary); font-size: 1.5rem; margin-bottom: 0.5rem;">License Plate Detector</h2>', unsafe_allow_html=True)
    st.markdown('<p style="color: var(--text-secondary); font-size: 0.938rem; margin-bottom: 1.5rem;">Upload an image or select a test example to detect Brazilian license plates</p>', unsafe_allow_html=True)
    
    if model is None:
        st.error("Model not loaded. Please check the weights directory.")
        return
    
    # Presets
    st.markdown('<h3 style="color: var(--text); font-size: 1.125rem; margin-bottom: 0.75rem;">Detection Settings</h3>', unsafe_allow_html=True)
    cols_p = st.columns(3)
    preset = st.session_state.get("preset", "Balanced")
    with cols_p[0]:
        if st.button("Fast", use_container_width=True):
            preset = "Fast"
    with cols_p[1]:
        if st.button("Balanced", use_container_width=True):
            preset = "Balanced"
    with cols_p[2]:
        if st.button("Precise", use_container_width=True):
            preset = "Precise"
    st.session_state["preset"] = preset
    
    # Valores por preset
    if preset == "Fast":
        conf_default, iou_default, size_default = 0.30, 0.45, 640
    elif preset == "Precise":
        conf_default, iou_default, size_default = 0.15, 0.55, 960
    else:
        conf_default, iou_default, size_default = 0.25, 0.50, 768
    
    col1, col2, col3 = st.columns(3)
    with col1:
        conf = st.slider("Confidence", 0.05, 0.95, conf_default, 0.05)
    with col2:
        iou = st.slider("IoU", 0.1, 0.9, iou_default, 0.05)
    with col3:
        imgsz = st.select_slider("Image Size", options=[640, 768, 896, 960, 1024], value=size_default)
    
    # Tabs
    tab_upload, tab_examples = st.tabs(["Upload", "Examples"])
    
    def run_detection(pil_img: Image.Image, key_prefix: str = "single"):
        """Executa detecção e mostra resultados"""
        start = time.time()
        results = yolo_predict(model, pil_img, conf, iou, imgsz)
        latency = (time.time() - start) * 1000
        
        if results:
            r0 = results[0]
            xyxy = r0.boxes.xyxy.cpu().numpy() if r0.boxes is not None else np.empty((0, 4))
            confs = r0.boxes.conf.cpu().numpy() if r0.boxes is not None else np.empty((0,))
            
            img_np = np.array(pil_img.convert("RGB"))
            annotated = _draw_boxes(img_np, xyxy, confs)
            
            col_res1, col_res2 = st.columns(2)
            with col_res1:
                st.image(pil_img, caption="Original Image", use_container_width=True)
            with col_res2:
                st.image(annotated, caption=f"Detection Results ({latency:.1f} ms)", use_container_width=True)
            
            # Detalhes
            st.markdown('<h3 style="color: var(--text); font-size: 1.125rem; margin: 1.5rem 0 0.75rem 0;">Detection Details</h3>', unsafe_allow_html=True)
            if len(xyxy) > 0:
                st.markdown(f"""
<div style="background: rgba(82,183,136,0.08); border-left: 3px solid var(--success); padding: 0.875rem; border-radius: 6px; margin-bottom: 1rem;">
  <span style="color: var(--success); font-weight: 600;">✓ {len(xyxy)} plate(s) detected</span>
</div>
""", unsafe_allow_html=True)
                for i, (box, conf_val) in enumerate(zip(xyxy, confs)):
                    x1, y1, x2, y2 = box.astype(int)
                    st.markdown(f"""
<div class="card">
  <b>Plate {i+1}</b><br>
  Confidence: <span class="badge badge-primary">{conf_val:.1%}</span><br>
  Position: ({x1}, {y1}) → ({x2}, {y2})
</div>
""", unsafe_allow_html=True)
            else:
                st.markdown("""
<div style="background: rgba(255,165,0,0.08); border-left: 3px solid var(--warning); padding: 0.875rem; border-radius: 6px;">
  <span style="color: var(--warning); font-weight: 600;">⚠ No plates detected in this image</span>
</div>
""", unsafe_allow_html=True)
            
            # Download
            buf = io.BytesIO()
            Image.fromarray(annotated).save(buf, format="PNG")
            st.download_button(
                "Download Annotated Image",
                data=buf.getvalue(),
                file_name=f"detection_{key_prefix}.png",
                mime="image/png",
                use_container_width=True
            )
    
    with tab_upload:
        uploaded = st.file_uploader("Select an image (PNG/JPG)", type=["png", "jpg", "jpeg"])
        if uploaded is not None:
            image = Image.open(uploaded)
            run_detection(image, key_prefix="upload")
    
    with tab_examples:
        examples = _gather_test_images()
        if examples:
            # Carregar imagens (do GitHub ou local)
            loaded_images = []
            captions = []
            
            with st.spinner("Loading example images..."):
                for i, img_data in enumerate(examples):
                    if img_data["source"] == "github":
                        img = _load_image_from_url(img_data["url"])
                    else:
                        img = Image.open(img_data["url"])
                    
                    if img is not None:
                        loaded_images.append(img)
                        captions.append(f"Image {i+1}")
            
            if loaded_images:
                # Container com scroll horizontal
                st.markdown('<p style="font-size: 0.875rem; color: var(--text-secondary); margin-bottom: 0.75rem;">Choose a test image:</p>', unsafe_allow_html=True)
                
                selected_idx = image_select(
                    "",
                    images=loaded_images,
                    captions=captions,
                    use_container_width=True,
                    return_value="index"
                )
                
                if selected_idx is not None:
                    if st.button("Detect Plates", type="primary", use_container_width=True):
                        run_detection(loaded_images[selected_idx], key_prefix="example")
            else:
                st.warning("Could not load example images from GitHub.")
        else:
            st.markdown("""
<div style="background: rgba(255,107,53,0.08); border-left: 3px solid var(--primary); padding: 0.875rem; border-radius: 6px;">
  <span style="color: var(--primary); font-weight: 600;">ℹ No test images found</span>
</div>
""", unsafe_allow_html=True)


def page_training():
    """Página de análise de treinamento"""
    st.markdown('<h2 style="color: var(--primary); font-size: 1.5rem; margin-bottom: 1rem;">Training Analytics</h2>', unsafe_allow_html=True)
    
    summary, results_df, args, images = load_training_data()
    
    # Matriz de confusão e gráficos
    col1, col2 = st.columns(2)
    with col1:
        if os.path.exists(images["confusion"]):
            st.image(images["confusion"], caption="Confusion Matrix", use_container_width=True)
    with col2:
        if os.path.exists(images["pr_curve"]):
            st.image(images["pr_curve"], caption="Precision-Recall Curve", use_container_width=True)
    
    st.markdown("---")
    
    # Gráficos de evolução
    if results_df is not None:
        st.markdown('<h3 style="color: var(--text); font-size: 1.125rem; margin: 1.5rem 0 0.75rem 0;">Training Evolution</h3>', unsafe_allow_html=True)
        
        metric_cols = [
            c for c in results_df.columns
            if any(k in c for k in ["precision", "recall", "mAP50", "box_loss", "cls_loss"])
        ]
        
        if metric_cols:
            # Gráfico de métricas
            fig = go.Figure()
            epochs = results_df.get("epoch", pd.Series(range(len(results_df))))
            
            for c in metric_cols:
                if "metrics" in c:
                    fig.add_trace(go.Scatter(
                        x=epochs, 
                        y=results_df[c], 
                        mode="lines+markers", 
                        name=c.replace("metrics/", "").replace("(B)", "")
                    ))
            
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#e4e4e7',
                xaxis_title="Epoch",
                yaxis_title="Value",
                legend_title="Metric",
                height=500,
                colorway=['#ff6b35', '#52b788', '#f7931e', '#e63946']  # Paleta quente
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.markdown("""
<div style="background: rgba(255,165,0,0.08); border-left: 3px solid var(--warning); padding: 0.875rem; border-radius: 6px;">
  <span style="color: var(--warning); font-weight: 600;">⚠ Training results not found</span>
</div>
""", unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Hiperparâmetros
    if args:
        st.markdown('<h3 style="color: var(--text); font-size: 1.125rem; margin-bottom: 0.75rem;">Hyperparameters</h3>', unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
<div class="card">
  <h4>Basic</h4>
  <p>Epochs: <span class="badge badge-primary">{}</span></p>
  <p>Batch: <span class="badge badge-primary">{}</span></p>
  <p>Image Size: <span class="badge badge-primary">{}</span></p>
</div>
""".format(args.get('epochs', 'N/A'), args.get('batch', 'N/A'), args.get('imgsz', 'N/A')), unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
<div class="card">
  <h4>Optimization</h4>
  <p>LR: <span class="badge badge-success">{}</span></p>
  <p>Momentum: <span class="badge badge-success">{}</span></p>
  <p>Weight Decay: <span class="badge badge-success">{}</span></p>
</div>
""".format(args.get('lr0', 'N/A'), args.get('momentum', 'N/A'), args.get('weight_decay', 'N/A')), unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
<div class="card">
  <h4>Augmentation</h4>
  <p>Flip: <span class="badge badge-primary">{}</span></p>
  <p>Mosaic: <span class="badge badge-primary">{}</span></p>
  <p>MixUp: <span class="badge badge-primary">{}</span></p>
</div>
""".format(args.get('fliplr', 'N/A'), args.get('mosaic', 'N/A'), args.get('mixup', 'N/A')), unsafe_allow_html=True)


def page_about():
    """Página sobre"""
    st.markdown('<h2 style="color: var(--primary); font-size: 1.5rem; margin-bottom: 1rem;">About</h2>', unsafe_allow_html=True)
    
    st.markdown("""
<div class="card">
<h3>Brazilian License Plate Recognition System</h3>
<p>Professional ALPR system using YOLOv8 for detecting Brazilian Mercosul standard license plates.</p>

<h4>Features</h4>
<ul>
  <li>Real-time license plate detection</li>
  <li>Support for Mercosul standard plates</li>
  <li>Adjustable confidence and IoU thresholds</li>
  <li>Multiple detection presets (Fast, Balanced, Precise)</li>
  <li>Training analytics and performance metrics</li>
  <li>Export annotated images</li>
</ul>

<h4>Technical Stack</h4>
<ul>
  <li><b>Model:</b> YOLOv8s</li>
  <li><b>Framework:</b> PyTorch + Ultralytics</li>
  <li><b>Interface:</b> Streamlit</li>
  <li><b>Visualization:</b> Plotly</li>
</ul>

<h4>👨‍💻 Author</h4>
<p>
  <b>Sidnei Almeida</b><br>
  <a href="https://github.com/sidnei-almeida" target="_blank" style="color: var(--primary); text-decoration: none;">
    <span style="margin-right: 0.5rem;">🔗 GitHub: @sidnei-almeida</span>
  </a><br>
  <a href="https://www.linkedin.com/in/saaelmeida93/" target="_blank" style="color: var(--primary); text-decoration: none;">
    <span>💼 LinkedIn: @saaelmeida93</span>
  </a>
</p>

<h4>📞 Contact & Support</h4>
<p>
  • Open an issue on <a href="https://github.com/sidnei-almeida/brazilian-license-plate-recognition/issues" target="_blank" style="color: var(--primary);">GitHub</a><br>
  • Connect on <a href="https://www.linkedin.com/in/saaelmeida93/" target="_blank" style="color: var(--primary);">LinkedIn</a>
</p>
</div>
""", unsafe_allow_html=True)


def main():
    """Função principal"""
    with st.sidebar:
        st.markdown("<h3 style='color:#ff6b35; margin-bottom: 0.875rem; font-size: 1.125rem;'>Navigation</h3>", unsafe_allow_html=True)
        
        selected = option_menu(
            menu_title=None,
            options=["Home", "Detector", "Analytics", "About"],
            icons=["house", "search", "graph-up", "info-circle"],
            default_index=0,
            styles={
                "container": {"padding": "0", "background": "transparent"},
                "icon": {"color": "#ff6b35", "font-size": "14px"},
                "nav-link": {
                    "color": "#e4e4e7",
                    "font-size": "0.875rem",
                    "padding": "0.625rem 0.875rem",
                    "border-radius": "6px",
                    "margin": "0.2rem 0"
                },
                "nav-link-selected": {
                    "background": "linear-gradient(135deg, rgba(255,107,53,0.12), rgba(247,147,30,0.12))",
                    "color": "#ff6b35",
                    "border-left": "3px solid #ff6b35",
                    "font-weight": "600"
                },
            },
        )
        
        # Status do ambiente
        st.markdown("---")
        st.markdown("<h4 style='margin-bottom:0.625rem; font-size: 0.938rem;'>System Info</h4>", unsafe_allow_html=True)
        env = get_env_status()
        
        device_badge = "badge-success" if env['device'] == "GPU" else "badge-primary"
        st.markdown(f"""
<div class="card" style="padding: 0.875rem;">
  <p style="margin: 0.2rem 0; font-size: 0.813rem;"><b>Device:</b> <span class="badge {device_badge}">{env['device']}</span></p>
  <p style="margin: 0.2rem 0; font-size: 0.688rem; color: var(--muted);">{env['device_name'][:30]}</p>
  <hr style="margin: 0.625rem 0;">
  <p style="margin: 0.2rem 0; font-size: 0.75rem;"><b>Python:</b> <span style="color: var(--text-secondary);">{env['python']}</span></p>
  <p style="margin: 0.2rem 0; font-size: 0.75rem;"><b>PyTorch:</b> <span style="color: var(--text-secondary);">{env['torch']}</span></p>
  <p style="margin: 0.2rem 0; font-size: 0.75rem;"><b>YOLO:</b> <span style="color: var(--text-secondary);">{env['ultralytics']}</span></p>
</div>
""", unsafe_allow_html=True)
    
    # Carrega recursos
    model = load_model()
    summary, results_df, _, _ = load_training_data()
    
    # Roteamento de páginas
    if selected == "Home":
        page_home(model, summary, results_df)
    elif selected == "Detector":
        page_detect(model)
    elif selected == "Analytics":
        page_training()
    else:
        page_about()


if __name__ == "__main__":
    main()
