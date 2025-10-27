import cv2
import numpy as np

# The .xml file should be in the same directory as your Streamlit app,
# or the full path must be provided here.
CASCADE_PATH = 'haarcascade_frontalface_default.xml'

def preprocess_for_prediction(img_bgr, model=None):
    """
    Detects one or more faces in an image, preprocesses each one and prepares
    them for the VGG16 emotion model.

    Args:
        img_bgr (np.array): The image in BGR format, from OpenCV (upload or webcam).
        model: The VGG16 model (not used, but kept for compatibility).

    Returns:
        tuple: A tuple containing (list_of_processed_faces, list_of_coordinates).
               Returns ([], []) if no face is detected.
    """
    try:
        face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
    except Exception:
        raise FileNotFoundError(
            f"Error loading Haar Cascade file. "
            f"Verify that '{CASCADE_PATH}' is in the correct directory."
        )

    # Face detection works better in grayscale
    gray_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    # Detect multiple faces in the image
    faces = face_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) == 0:
        return [], []  # No face found

    processed_faces = []
    coords = []

    for (x, y, w, h) in faces:
        # Preprocessing for VGG16 (96x96x3)
        # 1. Crop the face region from the original BGR image (colored)
        face_roi = img_bgr[y:y+h, x:x+w]
        
        # 2. Convert the cropped face from BGR to RGB
        face_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
        
        # 3. Resize to the size that the VGG16 model expects (96x96)
        resized_face = cv2.resize(face_rgb, (96, 96), interpolation=cv2.INTER_AREA)
        
        # 4. Normalize pixel values to the range [0, 1]
        normalized_face = resized_face / 255.0
        
        # 5. Expand dimensions to the format that Keras expects: (1, 96, 96, 3)
        reshaped_face = np.expand_dims(normalized_face, axis=0)
        
        processed_faces.append(reshaped_face)
        coords.append((x, y, w, h))
    
    return processed_faces, coords

