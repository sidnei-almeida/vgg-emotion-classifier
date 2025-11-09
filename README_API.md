# Facial Emotion Classifier API

REST API for facial emotion classification using VGG16 Fine-Tuning model.

## 🚀 Deploy to Hugging Face Spaces

This API is configured for deployment on Hugging Face Spaces using Docker.

### Prerequisites

1. Hugging Face account
2. Trained model in the `models/` folder
3. `haarcascade_frontalface_default.xml` file in the project root

### Required File Structure

```
.
├── api.py                 # Main FastAPI application
├── Dockerfile            # Docker configuration
├── requirements-api.txt  # Python dependencies
├── app.yaml              # Hugging Face Spaces configuration
├── image_pre_processing.py
├── haarcascade_frontalface_default.xml
└── models/
    └── emotion_model_final_vgg.h5 (or .keras)
```

### Deployment

1. **Create a new Space on Hugging Face:**
   - Visit https://huggingface.co/spaces
   - Click "Create new Space"
   - Choose "Docker" as SDK
   - Configure name and visibility

2. **Upload files:**
   ```bash
   git clone https://huggingface.co/spaces/your-username/your-space
   cd your-space
   # Copy all necessary files
   git add .
   git commit -m "Initial API deployment"
   git push
   ```

3. **Wait for build:**
   - Hugging Face will build the Docker image automatically
   - The process may take a few minutes on the first run

## 📡 API Endpoints

### `GET /`
Returns basic API information.

**Response:**
```json
{
  "message": "Facial Emotion Classifier API",
  "version": "1.0.0",
  "docs": "/docs",
  "health": "/health"
}
```

### `GET /health`
Checks API status and whether models are loaded.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "cascade_loaded": true
}
```

### `POST /predict`
Classifies emotion from an uploaded image.

**Request:**
- Content-Type: `multipart/form-data`
- Body: image file (PNG, JPG, JPEG)

**Response:**
```json
{
  "emotion": "happy",
  "confidence": 0.95,
  "probabilities": {
    "angry": 0.01,
    "disgust": 0.01,
    "fear": 0.01,
    "happy": 0.95,
    "neutral": 0.01,
    "sad": 0.005,
    "surprise": 0.005
  },
  "message": "😄 Pure joy! Keep spreading that contagious smile!"
}
```

**Errors:**
- `400`: No face detected or invalid file
- `503`: Model not loaded

### `POST /predict/batch`
Classifies emotions from multiple images.

**Request:**
- Content-Type: `multipart/form-data`
- Body: multiple image files

**Response:**
```json
{
  "results": [
    {
      "filename": "image1.jpg",
      "faces": [
        {
          "emotion": "happy",
          "confidence": 0.95,
          "probabilities": {...},
          "message": "...",
          "face_coordinates": {"x": 100, "y": 150, "width": 200, "height": 200}
        }
      ]
    }
  ]
}
```

### `GET /emotions`
Returns list of available emotions.

**Response:**
```json
{
  "emotions": ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"],
  "emotion_messages": {
    "angry": "😠 Looks like someone woke up on the wrong side of bed today! Stay calm!",
    ...
  }
}
```

## 🔧 Local Development

### Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements-api.txt
```

### Run Locally

```bash
# Development
uvicorn api:app --reload --host 0.0.0.0 --port 8000

# Production
uvicorn api:app --host 0.0.0.0 --port 8000
```

### Test the API

```bash
# Health check
curl http://localhost:8000/health

# Prediction
curl -X POST "http://localhost:8000/predict" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@path/to/image.jpg"
```

### Interactive Documentation

Visit `http://localhost:8000/docs` to see the interactive Swagger documentation.

## 🐳 Docker

### Build Image

```bash
docker build -t emotion-classifier-api .
```

### Run Container

```bash
docker run -p 8000:8000 emotion-classifier-api
```

### With Volumes for Models

```bash
docker run -p 8000:8000 \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/haarcascade_frontalface_default.xml:/app/haarcascade_frontalface_default.xml \
  emotion-classifier-api
```

## 📝 Notes

- The API supports multiple model formats (`.h5`, `.keras`)
- The model is loaded once on startup
- Face detection uses OpenCV Haar Cascade
- The VGG16 model expects 96x96 pixel images
- The API returns probabilities for all 7 emotions

## 🔒 Security

For production, consider:
- Adding authentication (API keys, JWT)
- Limiting file size
- Rate limiting
- More rigorous image validation
- Properly configured CORS

## 📄 License

This project is provided as-is, for educational and demonstration purposes.
