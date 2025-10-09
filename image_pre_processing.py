# image_processor.py

import cv2
import numpy as np

# O arquivo .xml deve estar no mesmo diretório ou o caminho completo deve ser fornecido.
CASCADE_PATH = 'haarcascade_frontalface_default.xml'

def preprocess_for_prediction(img_bgr):
    """
    Detecta um rosto em uma imagem, pré-processa e o prepara para o modelo de emoções.

    Args:
        img_bgr (np.array): A imagem no formato BGR, geralmente vinda do OpenCV.

    Returns:
        tuple: Uma tupla contendo (rosto_processado, (x, y, w, h))
               onde rosto_processado é a imagem pronta para o model.predict().
               Retorna (None, None) se nenhum rosto for detectado.
    """
    try:
        face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
    except Exception:
        raise FileNotFoundError(f"Erro ao carregar o arquivo Haar Cascade. Verifique se '{CASCADE_PATH}' está no diretório correto.")

    # A detecção de rosto funciona melhor em escala de cinza
    gray_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) == 0:
        return None, None  # Nenhum rosto encontrado

    # Pega as coordenadas do primeiro rosto encontrado
    (x, y, w, h) = faces[0]

    # Recorta a região do rosto da imagem em escala de cinza
    face_roi = gray_img[y:y+h, x:x+w]
    
    # Redimensiona para o tamanho que o modelo espera (48x48)
    resized_face = cv2.resize(face_roi, (48, 48), interpolation=cv2.INTER_AREA)
    
    # Normaliza os valores dos pixels para o intervalo [0, 1]
    normalized_face = resized_face / 255.0
    
    # Expande as dimensões para o formato que o Keras espera: (1, 48, 48, 1)
    reshaped_face = np.reshape(normalized_face, (1, 48, 48, 1))
    
    return reshaped_face, (x, y, w, h)
