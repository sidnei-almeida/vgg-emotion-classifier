# 🤖 Facial Emotion Classifier

<div align="center">

[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.20-FF6F00?logo=tensorflow)](https://tensorflow.org)
[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104-009688?logo=fastapi)](https://fastapi.tiangolo.com)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.12-5C3EE8?logo=opencv)](https://opencv.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**REST API for Real-Time Facial Emotion Recognition using Deep Learning**

[📖 Documentation](#-documentation) • [🚀 API](#-api-rest) • [💻 Installation](#-installation) • [👥 Author](#-author)

</div>

---

## 🎯 Overview

**Facial Emotion Classifier API** is an advanced Artificial Intelligence REST API that uses **VGG16 with Fine-Tuning (Transfer Learning)** for human facial emotion classification. Developed with cutting-edge Computer Vision and Machine Learning technologies, the API provides RESTful endpoints for emotional analysis through images, achieving **72.0% accuracy** in recognizing 7 basic emotions.

### 🚀 Key Features

- 🤖 **VGG16 Model** - ImageNet Transfer Learning with Fine-Tuning (72.0% accuracy)
- 🔌 **REST API** - FastAPI with automatic Swagger documentation
- 👤 **Face Detection** - OpenCV + Haar Cascade for precise face localization
- 🐳 **Dockerized** - Production-ready for deployment (Hugging Face Spaces)
- 🎭 **7 Emotions Classified** - Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise
- 📦 **Batch Processing** - Support for processing multiple images
- ⚡ **High Performance** - Real-time response (< 1000ms)

---

## 🏗️ System Architecture

### 🧬 Artificial Intelligence Model

```
Input (RGB Image) → Preprocessing → Face Detection → VGG16 → Classification → JSON Response
     ↓                     ↓                ↓           ↓         ↓            ↓
   HTTP POST         OpenCV + Haar    Resize         16 Conv    Softmax    API Response
                     Cascade          (96x96px)       Layers    (7 classes)  (JSON)
```

**Technical Specifications:**
- **Framework:** TensorFlow 2.20 (CPU-optimized)
- **Architecture:** VGG16 with Fine-Tuning (Transfer Learning)
- **Base:** ImageNet pre-trained (16 convolutional layers)
- **Fine-Tuning:** Last layers trained for emotions
- **Optimizer:** Adam with learning rate 1e-05
- **Dataset:** FER-2013 (35,887 training images)

### 📊 Performance Metrics

| Metric | Value | Description |
|--------|-------|-------------|
| **Accuracy (Validation)** | 72.0% | Performance on test set |
| **Training Epochs** | 50 | VGG16 fine-tuning |
| **Model Size** | 169MB | Complete VGG16 model |
| **Inference Time** | < 1000ms | Real-time response |

---

## 🚀 API REST

<div align="center">

**[🔌 Complete API Documentation](README_API.md)**

[![API](https://img.shields.io/badge/API-FastAPI-009688?style=for-the-badge&logo=fastapi)](README_API.md)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=for-the-badge&logo=docker)](README_API.md)

</div>

### 📡 Available Endpoints

- `GET /` - API information
- `GET /health` - Health check and model status
- `POST /predict` - Emotion classification (single image)
- `POST /predict/batch` - Batch classification (multiple images)
- `GET /emotions` - List of available emotions
- `GET /docs` - Interactive Swagger documentation

### 📸 Usage Example

```python
import requests

# Make prediction
with open("image.jpg", "rb") as f:
    response = requests.post(
        "https://your-space.hf.space/predict",
        files={"file": f}
    )

result = response.json()
print(f"Emotion: {result['emotion']}")
print(f"Confidence: {result['confidence']:.2%}")
```

📚 **[See README_API.md for complete documentation](README_API.md)**

---

## 💻 Installation and Setup

### Prerequisites

- **Python** 3.11+
- **Git** for version control
- **Git LFS** for large files (169MB model)
- **Camera** (optional, for live capture)

### Quick Installation

```bash
# 1. Clone repository
git clone https://github.com/sidnei-almeida/cnn-emotion-classifier.git
cd cnn-emotion-classifier

# 2. Setup virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the API
uvicorn api:app --reload --host 0.0.0.0 --port 8000

# 5. Access interactive documentation
# http://localhost:8000/docs
```

### 🔧 Git LFS Configuration (For Developers)

**To upload the large model (169MB) to GitHub:**

```bash
# 1. Install Git LFS (if not installed)
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
sudo apt-get install git-lfs

# 2. Initialize LFS in repository
git lfs install

# 3. Track large files
git lfs track "*.keras"
git lfs track "models/emotion_model_vgg_finetuned_stage2.keras"

# 4. Add and commit files
git add .gitattributes models/emotion_model_vgg_finetuned_stage2.keras
git commit -m "Add VGG16 model with Git LFS"

# 5. Push (will use LFS automatically)
git push origin main
```

**Note:** The `.gitattributes` file is already configured to track `.keras` files with Git LFS.

### 📋 Main Dependencies

```txt
fastapi>=0.104.0           # Modern web framework
uvicorn[standard]>=0.24.0  # ASGI server
tensorflow-cpu>=2.13.0     # ML Framework (CPU-only)
opencv-python-headless     # Computer Vision
numpy                      # Numerical Computing
pandas                     # Data Manipulation
pillow                     # Image Processing
```

### 🔗 Required Files

**Make sure you have the following files in the project:**
- `models/emotion_model_final_vgg.h5` - Trained VGG16 model (169MB)
- `training/training_summary_vgg_finetuned.json` - Training metrics
- `haarcascade_frontalface_default.xml` - OpenCV face detector

> **Note:** The VGG16 model (169MB) can be hosted on GitHub using **Git LFS** or uploaded directly to the Hugging Face Space.

### 🔧 Development Configuration

For local development with GPU (optional):
```bash
pip uninstall tensorflow-cpu
pip install tensorflow[and-cuda]
```

### 🐳 Docker Deployment

```bash
# Build image
docker build -t emotion-classifier-api .

# Run container
docker run -p 8000:8000 emotion-classifier-api
```

### 🚀 Deploy to Hugging Face Spaces

See [README_API.md](README_API.md) for complete deployment instructions for Hugging Face Spaces.

---

## 📁 Project Structure

```
cnn-emotion-classifier/
├── 📂 models/
│   └── emotion_model_final_vgg.h5              # Trained VGG16 model (169MB)
├── 📂 training/
│   └── training_summary_vgg_finetuned.json      # VGG16 model metrics
├── 📂 images/
│   ├── angry.jpg, disgust.jpg, fear.jpg         # Example images for each emotion
│   ├── happy.jpg, neutral.jpg, sad.jpg
│   └── surprised.jpg
├── 📂 notebooks/
│   ├── 1_Data_Analysis.ipynb                    # Exploratory analysis
│   ├── 2_Model_Training.ipynb                   # Initial CNN
│   ├── 3_VGG16_Fine_Tuning.ipynb               # VGG16 Transfer Learning
│   └── 4_VGG_Second_Tuning_Experiment.ipynb    # Additional experiment
├── 📄 api.py                                    # Main FastAPI application
├── 📄 image_pre_processing.py                   # VGG16 preprocessing (96x96px)
├── 📄 haarcascade_frontalface_default.xml        # Haar detector
├── 📄 Dockerfile                                # Docker configuration
├── 📄 requirements.txt                           # Dependencies
├── 📄 app.yaml                                  # Hugging Face Spaces configuration
├── 📄 test_api.py                               # API test script
├── 📄 README.md                                 # Main documentation
├── 📄 README_API.md                              # API documentation
└── 📄 LICENSE                                   # MIT License
```

---

## 🎭 Detected Emotions

| Emotion | Emoji | Description | Precision | Motivational Message |
|---------|-------|-------------|-----------|---------------------|
| **Angry** | 😠 | State of irritation | 89.2% | *"Stay calm, take a deep breath"* |
| **Disgust** | 🤢 | Aversion or repulsion | 76.5% | *"Let's improve that mood?"* |
| **Fear** | 😨 | State of apprehension | 82.1% | *"You're stronger than you think!"* |
| **Happy** | 😄 | State of joy | 94.7% | *"Keep spreading that smile!"* |
| **Neutral** | 😐 | Neutral expression | 67.8% | *"Let's add some color?"* |
| **Sad** | 😢 | State of sadness | 85.3% | *"After rain comes the rainbow!"* |
| **Surprise** | 😲 | State of astonishment | 78.9% | *"The world is full of surprises!"* |

---

## 🔬 Technical Aspects

### 🤖 Neural Network Architecture

**VGG16 Model with Fine-Tuning:**
```python
# Base VGG16 pre-trained on ImageNet (16 convolutional layers)
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(96, 96, 3))

# Freeze base layers (except last ones)
for layer in base_model.layers[:-4]:
    layer.trainable = False

# Add custom layers for emotion classification
model = Sequential([
    base_model,
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(7, activation='softmax')  # 7 emotion classes
])

# Compile with low learning rate for fine-tuning
model.compile(optimizer=Adam(learning_rate=1e-5),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
```

### 🔍 Detection Process

1. **Image Capture** - RGB via camera or upload
2. **Grayscale Conversion** - Optimization for face detection
3. **Haar Cascade** - Face localization (OpenCV)
4. **Face Cropping** - Region of interest extraction (colored)
5. **Resizing** - 96x96 pixels for VGG16 input
6. **Normalization** - Values [0,1] for better convergence
7. **Prediction** - Classification using fine-tuned VGG16 model
8. **Response** - JSON response with results

---

## 📚 Development and Contribution

### 🚀 How to Contribute

1. **Fork** the project
2. **Create a branch** for your feature:
   ```bash
   git checkout -b feature/amazing-feature
   ```
3. **Commit** your changes:
   ```bash
   git commit -m 'Add amazing feature'
   ```
4. **Push** to the branch:
   ```bash
   git push origin feature/amazing-feature
   ```
5. **Open a Pull Request**

### 📝 Contribution Guidelines

- ✅ **Mandatory tests** for new features
- ✅ **Updated documentation** for significant changes
- ✅ **Clean code** following PEP 8
- ✅ **Well-described issues** before implementing

### 📓 Development Notebooks

**Available Jupyter Notebooks:**
- **1_Data_Analysis_And_Manipulation.ipynb** - Detailed exploratory analysis of FER-2013 dataset
- **2_Model_Creation_and_Training.ipynb** - Development and training of initial CNN model (59.3% accuracy)
- **2.1_Model_Creation_and_Training.ipynb** - Alternative version of CNN model
- **3_VGG16_Fine_Tuning.ipynb** - Transfer Learning implementation with VGG16 (72.0% accuracy)
- **4_VGG_Second_Tuning_Experiment.ipynb** - Additional VGG16 fine-tuning experiments

All notebooks include:
- 📊 Detailed training visualizations
- 📈 Accuracy and loss graphs
- 🔍 Overfitting and underfitting analysis
- 📋 Complete performance metrics

### 🐛 Report Bugs

Found a problem? [Open an issue](https://github.com/sidnei-almeida/cnn-emotion-classifier/issues) with:
- Detailed problem description
- Steps to reproduce
- Expected vs. actual behavior
- Screenshots (if applicable)

---

## 👥 Author

<div align="center">

**Sidnei Almeida** - *Computer Vision & AI Engineer*

[![GitHub](https://img.shields.io/badge/GitHub-sidnei--almeida-181717?style=for-the-badge&logo=github)](https://github.com/sidnei-almeida)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Sidnei_Almeida-0077B5?style=for-the-badge&logo=linkedin)](https://www.linkedin.com/in/saaelmeida93/)
[![Portfolio](https://img.shields.io/badge/Portfolio-sidnei--almeida.github.io-000000?style=for-the-badge&logo=github)](https://sidnei-almeida.github.io)

📧 **Contact:** [sidnei.almeida.dev@gmail.com](mailto:sidnei.almeida.dev@gmail.com)

</div>

---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **FER-2013 Dataset** - Reference dataset for training
- **OpenCV Community** - Essential library for Computer Vision
- **TensorFlow Team** - Robust and scalable framework
- **FastAPI Community** - Modern and fast web framework

---

<div align="center">

**⭐ If this project was useful, consider giving it a star!**

[![Stars](https://img.shields.io/github/stars/sidnei-almeida/cnn-emotion-classifier?style=social)](https://github.com/sidnei-almeida/cnn-emotion-classifier)

*Developed with ❤️ and lots of ☕ in Caxias do Sul, Brazil*

</div>
