import cv2
import numpy as np

# O arquivo .xml deve estar no mesmo diretório do seu app Streamlit,
# ou o caminho completo deve ser fornecido aqui.
CASCADE_PATH = 'haarcascade_frontalface_default.xml'

def preprocess_for_prediction(img_bgr, model=None):
    """
    Detecta um ou mais rostos em uma imagem, pré-processa cada um e os prepara
    para o modelo VGG16 de emoções.

    Args:
        img_bgr (np.array): A imagem no formato BGR, vinda do OpenCV (upload ou webcam).
        model: O modelo VGG16 (não usado, mas mantido para compatibilidade).

    Returns:
        tuple: Uma tupla contendo (lista_de_rostos_processados, lista_de_coordenadas).
               Retorna ([], []) se nenhum rosto for detectado.
    """
    try:
        face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
    except Exception:
        raise FileNotFoundError(
            f"Erro ao carregar o arquivo Haar Cascade. "
            f"Verifique se '{CASCADE_PATH}' está no diretório correto."
        )

    # A detecção de rosto funciona melhor em escala de cinza
    gray_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    # Detecta múltiplos rostos na imagem
    faces = face_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) == 0:
        return [], []  # Nenhum rosto encontrado

    processed_faces = []
    coords = []

    for (x, y, w, h) in faces:
        # Pré-processamento para VGG16 (96x96x3)
        # 1. Recorta a região do rosto da imagem BGR original (colorida)
        face_roi = img_bgr[y:y+h, x:x+w]
        
        # 2. Converte o rosto recortado de BGR para RGB
        face_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
        
        # 3. Redimensiona para o tamanho que o modelo VGG16 espera (96x96)
        resized_face = cv2.resize(face_rgb, (96, 96), interpolation=cv2.INTER_AREA)
        
        # 4. Normaliza os valores dos pixels para o intervalo [0, 1]
        normalized_face = resized_face / 255.0
        
        # 5. Expande as dimensões para o formato que o Keras espera: (1, 96, 96, 3)
        reshaped_face = np.expand_dims(normalized_face, axis=0)
        
        processed_faces.append(reshaped_face)
        coords.append((x, y, w, h))
    
    return processed_faces, coords


