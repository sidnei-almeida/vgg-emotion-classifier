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
import urllib.request
import zipfile

# Base do app
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
TRAINING_DIR = os.path.join(BASE_DIR, "training")

# GitHub repository info
GITHUB_USER = "sidnei-almeida"
GITHUB_REPO = "cnn-emotion-classifier"
GITHUB_RAW_BASE = f"https://raw.githubusercontent.com/{GITHUB_USER}/{GITHUB_REPO}/main/"

def ensure_file_exists(file_path, github_url, description=""):
    """Garante que um arquivo existe, baixando do GitHub se necessário"""
    if not os.path.exists(file_path):
        st.info(f"📥 Baixando {description} do repositório...")

        try:
            # Cria diretório se não existir
            os.makedirs(os.path.dirname(file_path), exist_ok=True)

            # Baixa o arquivo
            urllib.request.urlretrieve(github_url, file_path)
            st.success(f"✅ {description} baixado com sucesso!")
            return True
        except Exception as e:
            st.error(f"❌ Erro ao baixar {description}: {str(e)}")
            return False
    return True

def ensure_model_files():
    """Garante que todos os arquivos necessários estão disponíveis"""
    files_ok = True

    # Modelo VGG16 - baixa do repositório vgg-emotion-classifier (GitHub LFS)
    model_path = os.path.join(MODELS_DIR, "emotion_model_vgg_finetuned_stage2.keras")
    # URL especial para arquivos GitHub LFS - usa media.githubusercontent.com
    model_url = "https://media.githubusercontent.com/media/sidnei-almeida/vgg-emotion-classifier/main/models/emotion_model_vgg_finetuned_stage2.keras"
    if not ensure_file_exists(model_path, model_url, "Modelo VGG16 do GitHub LFS (pode levar alguns minutos, ~169MB)"):
        files_ok = False

    # Dados de treinamento do modelo VGG16
    training_path = os.path.join(TRAINING_DIR, "training_summary_vgg_finetuned.json")
    training_url = f"{GITHUB_RAW_BASE}models/training_summary_vgg_finetuned.json"
    if not ensure_file_exists(training_path, training_url, "Dados de Treinamento VGG16"):
        files_ok = False

    # Haar Cascade (já deve estar presente, mas verifica)
    cascade_path = os.path.join(BASE_DIR, "haarcascade_frontalface_default.xml")
    cascade_url = f"{GITHUB_RAW_BASE}haarcascade_frontalface_default.xml"
    if not ensure_file_exists(cascade_path, cascade_url, "Detector de Faces"):
        files_ok = False

    return files_ok

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

/* Responsive */
@media (max-width: 768px) {
  .main-hero { font-size: 1.5rem; }
  .card { padding: 0.875rem; }
  .metric-card { padding: 0.75rem; }
}
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def load_emotion_model():
    """Carrega o modelo de classificação de emoções faciais"""
    # Garante que os arquivos necessários estão disponíveis
    if not ensure_model_files():
        st.error("❌ Não foi possível baixar os arquivos necessários do repositório.")
        return None
    
    # Usa o modelo VGG16 com fine-tuning (70.2% de acurácia)
    model_path = os.path.join(MODELS_DIR, "emotion_model_vgg_finetuned_stage2.keras")
    
    try:
        model = keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"❌ Erro ao carregar modelo de emoções: {str(e)}")
        return None


@st.cache_resource(show_spinner=False)
def load_face_cascade():
    """Carrega o classificador Haar Cascade para detecção de rostos"""
    # Garante que os arquivos necessários estão disponíveis
    if not ensure_model_files():
        st.error("❌ Não foi possível baixar os arquivos necessários do repositório.")
        return None

    cascade_path = os.path.join(BASE_DIR, "haarcascade_frontalface_default.xml")

    try:
        face_cascade = cv2.CascadeClassifier(cascade_path)
        if face_cascade.empty():
            st.error("❌ Erro: Haar Cascade não foi carregado corretamente.")
            return None
        return face_cascade
    except Exception as e:
        st.error(f"❌ Erro ao carregar Haar Cascade: {str(e)}")
        return None


@st.cache_data(show_spinner=False)
def load_training_data():
    """Carrega dados de treinamento do modelo VGG16 de emoções"""
    # Garante que os arquivos necessários estão disponíveis
    if not ensure_model_files():
        st.error("❌ Não foi possível baixar os arquivos necessários do repositório.")
        return None

    summary_path = os.path.join(TRAINING_DIR, "training_summary_vgg_finetuned.json")
    summary = None
    
    if os.path.exists(summary_path):
        try:
            with open(summary_path, "r") as f:
                summary = json.load(f)
        except Exception as e:
            st.error(f"❌ Erro ao carregar dados de treinamento: {str(e)}")

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
    """Pré-processa uma imagem para classificação de emoções usando VGG16"""
    # Converter PIL para numpy array BGR (formato esperado pelo OpenCV)
    img_bgr = np.array(image.convert("RGB"))
    img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_RGB2BGR)

    # Usar a função atualizada do image_pre_processing.py para VGG16
    from image_pre_processing import preprocess_for_prediction

    try:
        processed_faces, coords = preprocess_for_prediction(img_bgr)

        if len(processed_faces) == 0:
            return None, None, "Nenhum rosto detectado na imagem"

        # Usar o primeiro rosto detectado
        return processed_faces[0], coords[0], None

    except Exception as e:
        return None, None, f"Erro no pré-processamento: {str(e)}"


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


def page_Home(model, face_cascade, summary):
    """Página inicial"""
    st.markdown(
        '<div class="main-hero">\
          <div class="emotion-icon">😊</div>\
          <span class="title-gradient">Facial Emotion Classifier</span>\
        </div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="subtitle">Classificador de Emoções Faciais usando CNN e TensorFlow</div>', unsafe_allow_html=True)

    # Mensagem sobre download automático de arquivos
    if not model or not face_cascade:
        st.info("💡 **Nota:** Os arquivos necessários (modelo, dados de treinamento e detector de faces) serão baixados automaticamente do repositório GitHub na primeira execução.")
    
    # Status cards
    col1, col2, col3 = st.columns(3)
    with col1:
        model_status = "Carregado" if model else "Erro"
        badge_class = "badge-success" if model else "badge-danger"
        st.markdown(f"""
<div class="metric-card">
  <p class="metric-label">Modelo CNN</p>
  <span class="badge {badge_class}">{model_status}</span>
</div>
""", unsafe_allow_html=True)
    
    with col2:
        face_status = "Carregado" if face_cascade else "Erro"
        badge_class = "badge-success" if face_cascade else "badge-danger"
        st.markdown(f"""
<div class="metric-card">
  <p class="metric-label">Detector de Rosto</p>
  <span class="badge {badge_class}">{face_status}</span>
</div>
""", unsafe_allow_html=True)
    
    with col3:
        if summary and 'final_metrics' in summary:
            accuracy = summary['final_metrics'].get('best_validation_accuracy', 0)
            accuracy_text = f"{accuracy:.1%}"
        else:
            accuracy_text = "72.4%"  # Acurácia do modelo VGG16 Final
        st.markdown(f"""
<div class="metric-card">
  <p class="metric-label">Acurácia (Validação)</p>
  <div class="metric-value">{accuracy_text}</div>
</div>
""", unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Informações sobre o modelo
    if summary:
        st.markdown('<h3 style="color: var(--text); font-size: 1.125rem; margin: 1.5rem 0 0.75rem 0;">Sobre o Modelo</h3>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
<div class="card">
  <h4>Arquitetura</h4>
  <p>VGG16 com Fine-Tuning</p>
  <p>Transfer Learning do ImageNet</p>
  <p>Camadas finais treinadas para emoções</p>
  <p>Input: 96x96 pixels coloridos</p>
</div>
""", unsafe_allow_html=True)

        with col2:
            st.markdown("""
<div class="card">
  <h4>Emoções Detectadas</h4>
  <div style="display: flex; flex-wrap: wrap; gap: 0.5rem; margin-top: 0.5rem;">
    <span class="badge" style="background: rgba(220,38,38,0.12); border-color: rgba(220,38,38,0.25); color: var(--angry);">😠 Raiva</span>
    <span class="badge" style="background: rgba(132,204,22,0.12); border-color: rgba(132,204,22,0.25); color: var(--disgust);">🤢 Nojo</span>
    <span class="badge" style="background: rgba(124,58,237,0.12); border-color: rgba(124,58,237,0.25); color: var(--fear);">😨 Medo</span>
    <span class="badge" style="background: rgba(251,191,36,0.12); border-color: rgba(251,191,36,0.25); color: var(--happy);">😄 Feliz</span>
    <span class="badge" style="background: rgba(107,114,128,0.12); border-color: rgba(107,114,128,0.25); color: var(--neutral);">😐 Neutro</span>
    <span class="badge" style="background: rgba(107,114,128,0.12); border-color: rgba(107,114,128,0.25); color: var(--sad);">😢 Triste</span>
    <span class="badge" style="background: rgba(6,182,212,0.12); border-color: rgba(6,182,212,0.25); color: var(--surprise);">😲 Surpresa</span>
  </div>
</div>
""", unsafe_allow_html=True)
    
    # Como funciona
    st.markdown("---")
    st.markdown('<h3 style="color: var(--text); font-size: 1.125rem; margin: 1.5rem 0 0.75rem 0;">Como Funciona</h3>', unsafe_allow_html=True)

    st.markdown("""
<div class="card">
  <ol style="color: var(--text-secondary); line-height: 1.6;">
    <li><b>📷 Capture uma foto</b> usando a câmera ou faça upload de uma imagem</li>
    <li><b>👤 Detectamos o rosto</b> usando OpenCV e Haar Cascade</li>
    <li><b>🧠 Classificamos a emoção</b> com nossa CNN treinada</li>
    <li><b>🎭 Mostramos o resultado</b> com mensagem personalizada e confiança</li>
  </ol>
</div>
""", unsafe_allow_html=True)


def page_emotion_detector(model, face_cascade):
    """Página do detector de emoções"""
    st.markdown('<h2 style="color: var(--primary); font-size: 1.5rem; margin-bottom: 0.5rem;">Detector de Emoções</h2>', unsafe_allow_html=True)
    st.markdown('<p style="color: var(--text-secondary); font-size: 0.938rem; margin-bottom: 1.5rem;">Capture uma foto com sua câmera ou faça upload de uma imagem para detectar emoções faciais</p>', unsafe_allow_html=True)

    if model is None or face_cascade is None:
        st.error("Modelo ou detector de rosto não carregado. Verifique os arquivos necessários.")
        return
    
    # Abas para diferentes métodos de input
    tab_camera, tab_upload = st.tabs(["📷 Câmera", "📁 Upload"])

    with tab_camera:
        st.markdown('<h3 style="color: var(--text); font-size: 1.125rem; margin-bottom: 0.75rem;">Capture com a Câmera</h3>', unsafe_allow_html=True)

        # Controle de estado para mostrar/esconder botão inicial
        if 'show_camera' not in st.session_state:
            st.session_state.show_camera = False

        # Botão inicial - desaparece quando a câmera está ativa
        if not st.session_state.show_camera:
            if st.button("📸 Iniciar Câmera", type="primary", use_container_width=True):
                st.session_state.show_camera = True
                st.rerun()  # Força recarregamento para mostrar a câmera

        # Quando a câmera estiver ativa, mostra apenas ela
        if st.session_state.show_camera:
            st.markdown('<p style="color: var(--text-secondary); font-size: 0.875rem; margin-bottom: 1rem;">🎯 Aponte para seu rosto e clique no botão abaixo para capturar</p>', unsafe_allow_html=True)

            # Câmera - quando tirar foto, processa automaticamente
            img_file_buffer = st.camera_input("Tire uma foto do seu rosto")

            if img_file_buffer is not None:
                # Processar a imagem capturada automaticamente
                image = Image.open(img_file_buffer)

                with st.spinner("🔍 Detectando emoção..."):
                    processed_face, face_coords, error_msg = preprocess_face(image, face_cascade)

                    if error_msg:
                        st.error(f"❌ {error_msg}")
                        # Mostra botão para tentar novamente
                        if st.button("🔄 Tentar Novamente", type="secondary"):
                            st.session_state.show_camera = False
                            st.rerun()
                    else:
                        # Fazer predição
                        prediction = predict_emotion(model, processed_face)

                        if prediction:
                            emotion = prediction["emotion"]
                            confidence = prediction["confidence"]

                            # Mostrar resultado
                            col1, col2 = st.columns(2)
                            with col1:
                                st.image(image, caption="📸 Foto Capturada", use_container_width=True)
                            with col2:
                                st.markdown(f"""
<div class="emotion-result">
  <span class="emotion-emoji">{EMOTION_MESSAGES[emotion].split()[0]}</span>
  <div class="emotion-text">{emotion.title()}</div>
  <div class="emotion-confidence">🎯 Confiança: {confidence:.1%}</div>
  <p style="margin-top: 1rem; color: var(--text-secondary);">{EMOTION_MESSAGES[emotion]}</p>
</div>
""", unsafe_allow_html=True)
            
                                # Gráfico de probabilidades
                                prob_df = pd.DataFrame(list(prediction["probabilities"].items()),
                                                     columns=["Emoção", "Probabilidade"])
                                prob_df = prob_df.sort_values("Probabilidade", ascending=False)

                                fig = go.Figure(data=[
                                    go.Bar(
                                        x=prob_df["Probabilidade"] * 100,
                                        y=prob_df["Emoção"],
                                        orientation='h',
                                        marker_color=['var(--primary)' if emo == emotion else 'var(--muted)' for emo in prob_df["Emoção"]]
                                    )
                                ])
                                fig.update_layout(
                                    plot_bgcolor='rgba(0,0,0,0)',
                                    paper_bgcolor='rgba(0,0,0,0)',
                                    font_color='#f1f5f9',
                                    xaxis_title="Probabilidade (%)",
                                    yaxis_title="Emoção",
                                    height=300,
                                    margin=dict(l=0, r=0, t=20, b=0)
                                )
                                st.plotly_chart(fig, use_container_width=True)

                                # Botão para tirar outra foto
                                col1, col2, col3 = st.columns(3)
                                with col2:
                                    if st.button("📸 Tirar Outra Foto", type="primary", use_container_width=True):
                                        st.session_state.show_camera = False
                                        st.rerun()
    
    with tab_upload:
        st.markdown('<h3 style="color: var(--text); font-size: 1.125rem; margin-bottom: 0.75rem;">Upload de Imagem</h3>', unsafe_allow_html=True)

        uploaded = st.file_uploader("Selecione uma imagem (PNG/JPG)", type=["png", "jpg", "jpeg"])

        if uploaded is not None:
            image = Image.open(uploaded)

            if st.button("🔍 Detectar Emoção", type="primary", use_container_width=True):
                with st.spinner("Detectando emoção..."):
                    processed_face, face_coords, error_msg = preprocess_face(image, face_cascade)

                    if error_msg:
                        st.error(error_msg)
                    else:
                        # Fazer predição
                        prediction = predict_emotion(model, processed_face)

                        if prediction:
                            emotion = prediction["emotion"]
                            confidence = prediction["confidence"]

                            # Mostrar resultado
                            col1, col2 = st.columns(2)
                            with col1:
                                st.image(image, caption="Imagem Enviada", use_container_width=True)
                            with col2:
                                st.markdown(f"""
<div class="emotion-result">
  <span class="emotion-emoji">{EMOTION_MESSAGES[emotion].split()[0]}</span>
  <div class="emotion-text">{emotion.title()}</div>
  <div class="emotion-confidence">Confiança: {confidence:.1%}</div>
  <p style="margin-top: 1rem; color: var(--text-secondary);">{EMOTION_MESSAGES[emotion]}</p>
</div>
""", unsafe_allow_html=True)

                                # Gráfico de probabilidades
                                prob_df = pd.DataFrame(list(prediction["probabilities"].items()),
                                                     columns=["Emoção", "Probabilidade"])
                                prob_df = prob_df.sort_values("Probabilidade", ascending=False)

                                fig = go.Figure(data=[
                                    go.Bar(
                                        x=prob_df["Probabilidade"] * 100,
                                        y=prob_df["Emoção"],
                                        orientation='h',
                                        marker_color=['var(--primary)' if emo == emotion else 'var(--muted)' for emo in prob_df["Emoção"]]
                                    )
                                ])
                                fig.update_layout(
                                    plot_bgcolor='rgba(0,0,0,0)',
                                    paper_bgcolor='rgba(0,0,0,0)',
                                    font_color='#f1f5f9',
                                    xaxis_title="Probabilidade (%)",
                                    yaxis_title="Emoção",
                                    height=300,
                                    margin=dict(l=0, r=0, t=20, b=0)
                                )
                                st.plotly_chart(fig, use_container_width=True)


def page_about():
    """Página sobre"""
    st.markdown('<h2 style="color: var(--primary); font-size: 1.5rem; margin-bottom: 1rem;">Sobre</h2>', unsafe_allow_html=True)
    
    st.markdown("""
<div class="card">
<h3>Facial Emotion Classifier</h3>
<p>Sistema de classificação de emoções faciais usando Convolutional Neural Networks (CNN) e TensorFlow.</p>

<h4>Características</h4>
<ul>
  <li>Detecção facial usando OpenCV e Haar Cascade</li>
  <li>Classificação de 7 emoções básicas</li>
  <li>Interface interativa com câmera ao vivo</li>
  <li>Upload de imagens personalizado</li>
  <li>Visualizações detalhadas de probabilidades</li>
  <li>Mensagens personalizadas para cada emoção</li>
</ul>

<h4>Emoções Detectadas</h4>
<div style="display: flex; flex-wrap: wrap; gap: 0.5rem; margin: 0.5rem 0;">
  <span class="badge" style="background: rgba(220,38,38,0.12); border-color: rgba(220,38,38,0.25); color: var(--angry);">😠 Raiva</span>
  <span class="badge" style="background: rgba(132,204,22,0.12); border-color: rgba(132,204,22,0.25); color: var(--disgust);">🤢 Nojo</span>
  <span class="badge" style="background: rgba(124,58,237,0.12); border-color: rgba(124,58,237,0.25); color: var(--fear);">😨 Medo</span>
  <span class="badge" style="background: rgba(251,191,36,0.12); border-color: rgba(251,191,36,0.25); color: var(--happy);">😄 Feliz</span>
  <span class="badge" style="background: rgba(107,114,128,0.12); border-color: rgba(107,114,128,0.25); color: var(--neutral);">😐 Neutro</span>
  <span class="badge" style="background: rgba(107,114,128,0.12); border-color: rgba(107,114,128,0.25); color: var(--sad);">😢 Triste</span>
  <span class="badge" style="background: rgba(6,182,212,0.12); border-color: rgba(6,182,212,0.25); color: var(--surprise);">😲 Surpresa</span>
</div>

<h4>Tecnologias Utilizadas</h4>
<ul>
  <li><b>Modelo:</b> CNN personalizada com TensorFlow/Keras</li>
  <li><b>Detecção de Rosto:</b> OpenCV + Haar Cascade</li>
  <li><b>Interface:</b> Streamlit</li>
  <li><b>Visualização:</b> Plotly</li>
  <li><b>Processamento:</b> NumPy + PIL</li>
</ul>

<h4>👨‍💻 Autor</h4>
<p>
  <b>Sidnei Almeida</b><br>
  Desenvolvido como projeto de demonstração de CNN para classificação de emoções faciais.
</p>
</div>
""", unsafe_allow_html=True)


def main():
    """Função principal"""
    with st.sidebar:
        st.markdown("<h3 style='color:#8b5cf6; margin-bottom: 0.875rem; font-size: 1.125rem;'>Navegação</h3>", unsafe_allow_html=True)
        
        selected = option_menu(
            menu_title=None,
            options=["Início", "Detector", "Sobre"],
            icons=["house", "camera", "info-circle"],
            default_index=0,
            styles={
                "container": {"padding": "0", "background": "transparent"},
                "icon": {"color": "#8b5cf6", "font-size": "14px"},
                "nav-link": {
                    "color": "#f1f5f9",
                    "font-size": "0.875rem",
                    "padding": "0.625rem 0.875rem",
                    "border-radius": "6px",
                    "margin": "0.2rem 0"
                },
                "nav-link-selected": {
                    "background": "linear-gradient(135deg, rgba(139,92,246,0.12), rgba(6,182,212,0.12))",
                    "color": "#8b5cf6",
                    "border-left": "3px solid #8b5cf6",
                    "font-weight": "600"
                },
            },
        )
        
        # Status do ambiente
        st.markdown("---")
        st.markdown("<h4 style='margin-bottom:0.625rem; font-size: 0.938rem;'>Sistema</h4>", unsafe_allow_html=True)
        env = get_env_status()
        
        device_badge = "badge-success" if env['device'] == "GPU" else "badge-primary"
        st.markdown(f"""
<div class="card" style="padding: 0.875rem;">
  <p style="margin: 0.2rem 0; font-size: 0.813rem;"><b>Dispositivo:</b> <span class="badge {device_badge}">{env['device']}</span></p>
  <p style="margin: 0.2rem 0; font-size: 0.688rem; color: var(--muted);">{env['device_name'][:30]}</p>
  <hr style="margin: 0.625rem 0;">
  <p style="margin: 0.2rem 0; font-size: 0.75rem;"><b>Python:</b> <span style="color: var(--text-secondary);">{env['python']}</span></p>
  <p style="margin: 0.2rem 0; font-size: 0.75rem;"><b>TensorFlow:</b> <span style="color: var(--text-secondary);">{env['tensorflow']}</span></p>
</div>
""", unsafe_allow_html=True)
    
    # Carrega recursos
    model = load_emotion_model()
    face_cascade = load_face_cascade()
    summary = load_training_data()
    
    # Roteamento de páginas
    if selected == "Início":
        page_Home(model, face_cascade, summary)
    elif selected == "Detector":
        page_emotion_detector(model, face_cascade)
    else:
        page_about()


if __name__ == "__main__":
    main()
