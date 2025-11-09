# 🚀 Deployment Guide for Hugging Face Spaces

This guide shows how to deploy the API to your Space: [vgg16-emotion-classifier](https://huggingface.co/spaces/salmeida/vgg16-emotion-classifier)

## 📋 Prerequisites

1. Git installed
2. Hugging Face access token (provided below)
3. Project files ready

## 🔑 Access Token

⚠️ **IMPORTANT**: You need a Hugging Face access token to deploy.

**Get your token:**
1. Go to https://huggingface.co/settings/tokens
2. Create a new token with write permissions
3. Copy the token (starts with `hf_`)

**Use the token:**
- When prompted during `git push`, use your token as the password
- Or set it as an environment variable: `export HF_TOKEN=your_token_here`
- Or configure git credential helper to store it securely

## 🚀 Method 1: Automatic Deployment (Recommended)

```bash
# Run the deployment script
./deploy_hf.sh
```

The script will:
- Check required files
- Configure Hugging Face remote
- Commit and push files
- Provide instructions for using the token

## 🚀 Method 2: Manual Deployment

### Step 1: Clone the Space Repository

```bash
git clone https://huggingface.co/spaces/salmeida/vgg16-emotion-classifier
cd vgg16-emotion-classifier
```

### Step 2: Configure Git Credential Helper (Optional)

To avoid typing the token every time:

```bash
git config credential.helper store
```

When prompted, use the token as password.

### Step 3: Copy Required Files

From the project directory to the Space directory:

```bash
# Main files
cp ../vgg-emotion-classifier/api.py .
cp ../vgg-emotion-classifier/Dockerfile .
cp ../vgg-emotion-classifier/requirements-api.txt requirements.txt
cp ../vgg-emotion-classifier/app.yaml .
cp ../vgg-emotion-classifier/image_pre_processing.py .
cp ../vgg-emotion-classifier/haarcascade_frontalface_default.xml .

# Models (if not already in Space)
cp -r ../vgg-emotion-classifier/models .

# Documentation (optional)
cp ../vgg-emotion-classifier/README.md .
cp ../vgg-emotion-classifier/README_API.md .
```

### Step 4: Commit and Push

```bash
git add .
git commit -m "Deploy FastAPI API"
git push
```

When prompted:
- **Username**: `salmeida`
- **Password**: Use your Hugging Face access token (get it from https://huggingface.co/settings/tokens)

## 📁 Required Files in Space

Make sure these files are in the Space:

```
vgg16-emotion-classifier/
├── api.py                          # ✅ FastAPI application
├── Dockerfile                      # ✅ Docker configuration
├── requirements.txt                # ✅ Dependencies (renamed from requirements-api.txt)
├── app.yaml                       # ✅ Hugging Face config
├── image_pre_processing.py        # ✅ Preprocessing
├── haarcascade_frontalface_default.xml  # ✅ Face detector
└── models/                        # ✅ Trained models
    └── emotion_model_final_vgg.h5
```

## 🔍 Post-Deployment Checks

After pushing, verify:

1. **Build Status**: Visit [the Space](https://huggingface.co/spaces/salmeida/vgg16-emotion-classifier) and check if build is running
2. **Logs**: Click "Logs" to see build progress
3. **API Health**: After build, test the `/health` endpoint

## 🧪 Test the API

After successful deployment:

```bash
# Health check
curl https://salmeida-vgg16-emotion-classifier.hf.space/health

# Test prediction
curl -X POST "https://salmeida-vgg16-emotion-classifier.hf.space/predict" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@image.jpg"
```

Or visit the interactive documentation:
```
https://salmeida-vgg16-emotion-classifier.hf.space/docs
```

## ⚠️ Troubleshooting

### Error: "No application file"
- Make sure `api.py` is in the Space root
- Verify that `Dockerfile` is present

### Error: "Port not found"
- Hugging Face Spaces uses port **7860** (already configured in Dockerfile)

### Error: "Model not found"
- Make sure to upload models to the `models/` folder
- The model may be large (169MB), use Git LFS if necessary

### Build failing
- Check logs in the Space
- Make sure all dependencies are in `requirements.txt`
- Verify that Dockerfile is correct

## 📚 Resources

- [Hugging Face Spaces Documentation](https://huggingface.co/docs/hub/spaces)
- [Docker Spaces Documentation](https://huggingface.co/docs/hub/spaces-sdks-docker)
- [API Documentation](README_API.md)

## 🎉 Done!

After successful deployment, your API will be publicly available and ready to be integrated into frontend applications!
