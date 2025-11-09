"""
FastAPI API for Facial Emotion Classification
Deploy-ready for Hugging Face Spaces
"""

import os
import io
import json
import logging
from typing import Optional, List
import numpy as np
from PIL import Image
import cv2
import tensorflow as tf
from tensorflow import keras
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# App configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
TRAINING_DIR = os.path.join(BASE_DIR, "training")
CASCADE_PATH = os.path.join(BASE_DIR, "haarcascade_frontalface_default.xml")

# Emotion mapping
EMOTIONS = {
    0: "angry",
    1: "disgust",
    2: "fear",
    3: "happy",
    4: "neutral",
    5: "sad",
    6: "surprise"
}

EMOTION_MESSAGES = {
    "angry": "😠 Looks like someone woke up on the wrong side of bed today! Stay calm!",
    "disgust": "🤢 Eww! Something left you disgusted. Let's improve that mood?",
    "fear": "😨 Fear? Don't be afraid! You're stronger than you think!",
    "happy": "😄 Pure joy! Keep spreading that contagious smile!",
    "neutral": "😐 Neutral like Ironically. Let's add some color?",
    "sad": "😢 Sadness in the air... Remember that after rain comes the rainbow!",
    "surprise": "😲 Wow! What a surprise! The world is full of unexpected things!"
}

# Initialize FastAPI app
app = FastAPI(
    title="Facial Emotion Classifier API",
    description="API for facial emotion classification using VGG16 deep learning model",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure according to your needs
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for loaded models
model = None
face_cascade = None


# Response models
class EmotionPrediction(BaseModel):
    emotion: str
    confidence: float
    probabilities: dict[str, float]
    message: str


class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    cascade_loaded: bool


# Load models
def load_emotion_model():
    """Loads the VGG16 model for facial emotion classification"""
    global model
    
    if model is not None:
        return model
    
    model_paths = [
        os.path.join(MODELS_DIR, "emotion_model_final_vgg.h5"),
        os.path.join(MODELS_DIR, "emotion_model_vgg_finetuned_stage2.h5"),
        os.path.join(MODELS_DIR, "emotion_model_vgg_finetuned_stage2.keras")
    ]
    
    for model_path in model_paths:
        if os.path.exists(model_path):
            try:
                logger.info(f"Attempting to load model from: {model_path}")
                # Try different loading methods
                try:
                    model = keras.models.load_model(model_path)
                    logger.info("Model loaded successfully (standard method)")
                    return model
                except Exception as e1:
                    logger.warning(f"Standard loading failed: {e1}")
                    try:
                        model = keras.models.load_model(model_path, compile=False)
                        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
                        logger.info("Model loaded successfully (compile=False method)")
                        return model
                    except Exception as e2:
                        logger.warning(f"Compile=False loading failed: {e2}")
                        try:
                            model = keras.models.load_model(model_path, safe_mode=False)
                            logger.info("Model loaded successfully (safe_mode=False method)")
                            return model
                        except Exception as e3:
                            logger.warning(f"Safe_mode=False loading failed: {e3}")
                            try:
                                custom_objects = {
                                    'Flatten': keras.layers.Flatten,
                                    'Dense': keras.layers.Dense,
                                    'Dropout': keras.layers.Dropout,
                                    'GlobalAveragePooling2D': keras.layers.GlobalAveragePooling2D,
                                    'GlobalMaxPooling2D': keras.layers.GlobalMaxPooling2D,
                                    'BatchNormalization': keras.layers.BatchNormalization,
                                    'ReLU': keras.layers.ReLU,
                                    'Softmax': keras.layers.Softmax,
                                }
                                model = keras.models.load_model(
                                    model_path, 
                                    custom_objects=custom_objects, 
                                    compile=False
                                )
                                model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
                                logger.info("Model loaded successfully (custom_objects method)")
                                return model
                            except Exception as e4:
                                logger.error(f"All loading methods failed: {e4}")
                                continue
            
            except Exception as e:
                logger.error(f"Error loading model from {model_path}: {e}")
                continue
    
    logger.error("Failed to load model from any available path")
    return None


def load_face_cascade():
    """Loads the Haar Cascade classifier for face detection"""
    global face_cascade
    
    if face_cascade is not None:
        return face_cascade
    
    if not os.path.exists(CASCADE_PATH):
        logger.error(f"Cascade file not found: {CASCADE_PATH}")
        return None
    
    try:
        face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
        if face_cascade.empty():
            logger.error("Cascade classifier is empty")
            return None
        logger.info("Face cascade loaded successfully")
        return face_cascade
    except Exception as e:
        logger.error(f"Error loading cascade: {e}")
        return None


def preprocess_for_prediction(img_bgr, model=None):
    """
    Detects one or more faces in an image and preprocesses them for the VGG16 emotion model.
    """
    if face_cascade is None:
        raise ValueError("Face cascade not loaded")
    
    # Face detection works better in grayscale
    gray_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    # Detect multiple faces in the image
    faces = face_cascade.detectMultiScale(
        gray_img, 
        scaleFactor=1.1, 
        minNeighbors=5, 
        minSize=(30, 30)
    )
    
    if len(faces) == 0:
        return [], []
    
    processed_faces = []
    coords = []
    
    for (x, y, w, h) in faces:
        # Crop the face region from the original BGR image
        face_roi = img_bgr[y:y+h, x:x+w]
        
        # Convert BGR to RGB
        face_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
        
        # Resize to 96x96 (VGG16 input size)
        resized_face = cv2.resize(face_rgb, (96, 96), interpolation=cv2.INTER_AREA)
        
        # Normalize pixel values to [0, 1]
        normalized_face = resized_face / 255.0
        
        # Expand dimensions: (1, 96, 96, 3)
        reshaped_face = np.expand_dims(normalized_face, axis=0)
        
        processed_faces.append(reshaped_face)
        coords.append({"x": int(x), "y": int(y), "width": int(w), "height": int(h)})
    
    return processed_faces, coords


def predict_emotion(processed_face):
    """Makes emotion prediction"""
    if model is None:
        raise ValueError("Model not loaded")
    
    if processed_face is None:
        raise ValueError("Processed face is None")
    
    try:
        predictions = model.predict(processed_face, verbose=False)
        predicted_class = np.argmax(predictions[0])
        confidence = float(np.max(predictions[0]))
        
        probabilities = {
            EMOTIONS[i]: float(prob) 
            for i, prob in enumerate(predictions[0])
        }
        
        emotion = EMOTIONS[predicted_class]
        
        return {
            "emotion": emotion,
            "confidence": confidence,
            "probabilities": probabilities,
            "message": EMOTION_MESSAGES.get(emotion, "")
        }
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise ValueError(f"Prediction error: {str(e)}")


# Startup event
@app.on_event("startup")
async def startup_event():
    """Load models on startup"""
    logger.info("Starting up API...")
    load_emotion_model()
    load_face_cascade()
    logger.info("API startup complete")


# Endpoints
@app.get("/", tags=["General"])
async def root():
    """Root endpoint"""
    return {
        "message": "Facial Emotion Classifier API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "cascade_loaded": face_cascade is not None
    }


@app.post("/predict", response_model=EmotionPrediction, tags=["Prediction"])
async def predict_emotion_from_image(file: UploadFile = File(...)):
    """
    Predict emotion from an uploaded image.
    
    Accepts image files (PNG, JPG, JPEG) and returns emotion prediction.
    """
    # Validate model and cascade are loaded
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please check server logs."
        )
    
    if face_cascade is None:
        raise HTTPException(
            status_code=503,
            detail="Face cascade not loaded. Please check server logs."
        )
    
    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="File must be an image (PNG, JPG, JPEG)"
        )
    
    try:
        # Read image
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert PIL to numpy array BGR (format expected by OpenCV)
        img_bgr = np.array(image.convert("RGB"))
        img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_RGB2BGR)
        
        # Preprocess image
        processed_faces, coords = preprocess_for_prediction(img_bgr)
        
        if len(processed_faces) == 0:
            raise HTTPException(
                status_code=400,
                detail="No face detected in the image"
            )
        
        # Use the first detected face
        processed_face = processed_faces[0]
        
        # Predict emotion
        result = predict_emotion(processed_face)
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing image: {str(e)}"
        )


@app.post("/predict/batch", tags=["Prediction"])
async def predict_emotion_batch(files: List[UploadFile] = File(...)):
    """
    Predict emotions from multiple images.
    
    Returns predictions for all faces detected in all images.
    """
    if model is None or face_cascade is None:
        raise HTTPException(
            status_code=503,
            detail="Model or cascade not loaded"
        )
    
    results = []
    
    for file in files:
        if not file.content_type.startswith("image/"):
            results.append({
                "filename": file.filename,
                "error": "File must be an image"
            })
            continue
        
        try:
            image_bytes = await file.read()
            image = Image.open(io.BytesIO(image_bytes))
            
            img_bgr = np.array(image.convert("RGB"))
            img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_RGB2BGR)
            
            processed_faces, coords = preprocess_for_prediction(img_bgr)
            
            if len(processed_faces) == 0:
                results.append({
                    "filename": file.filename,
                    "error": "No face detected"
                })
                continue
            
            # Predict for all faces in the image
            face_predictions = []
            for i, processed_face in enumerate(processed_faces):
                result = predict_emotion(processed_face)
                result["face_coordinates"] = coords[i]
                face_predictions.append(result)
            
            results.append({
                "filename": file.filename,
                "faces": face_predictions
            })
            
        except Exception as e:
            logger.error(f"Error processing {file.filename}: {e}")
            results.append({
                "filename": file.filename,
                "error": str(e)
            })
    
    return {"results": results}


@app.get("/emotions", tags=["General"])
async def get_emotions():
    """Get list of available emotions"""
    return {
        "emotions": list(EMOTIONS.values()),
        "emotion_messages": EMOTION_MESSAGES
    }


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))  # Hugging Face Spaces uses port 7860
    uvicorn.run(app, host="0.0.0.0", port=port)

