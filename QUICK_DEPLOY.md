# ⚡ Quick Deploy - Hugging Face Spaces

## 🎯 Deploy in 3 Steps

### 1️⃣ Clone the Space

```bash
git clone https://huggingface.co/spaces/salmeida/vgg16-emotion-classifier
cd vgg16-emotion-classifier
```

### 2️⃣ Copy Files

From the project directory (`vgg-emotion-classifier`) to the Space directory:

```bash
# Essential files
cp ../vgg-emotion-classifier/api.py .
cp ../vgg-emotion-classifier/Dockerfile .
cp ../vgg-emotion-classifier/requirements-api.txt requirements.txt
cp ../vgg-emotion-classifier/app.yaml .
cp ../vgg-emotion-classifier/image_pre_processing.py .
cp ../vgg-emotion-classifier/haarcascade_frontalface_default.xml .

# Models (IMPORTANT!)
cp -r ../vgg-emotion-classifier/models .
```

### 3️⃣ Commit and Push

```bash
git add .
git commit -m "Deploy FastAPI API"
git push
```

**When prompted:**
- **Username**: `salmeida`
- **Password**: Use your Hugging Face access token (get it from https://huggingface.co/settings/tokens)

## ✅ Done!

Wait a few minutes for the build and your API will be at:
**https://salmeida-vgg16-emotion-classifier.hf.space**

Interactive documentation: **https://salmeida-vgg16-emotion-classifier.hf.space/docs**
