#!/bin/bash

# Deployment script for GitHub with Git LFS
# Usage: ./deploy.sh

echo "🚀 Starting deployment of Facial Emotion Classifier..."

# 1. Check if Git LFS is installed
if ! command -v git-lfs &> /dev/null; then
    echo "❌ Git LFS not found. Installing..."
    curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
    sudo apt-get install git-lfs -y
fi

# 2. Initialize Git LFS
echo "🔧 Initializing Git LFS..."
git lfs install

# 3. Verify large files
echo "📊 Verifying large files..."
git lfs ls-files

# 4. Add files to Git
echo "📁 Adding files to Git..."
git add .

# 5. Make commit
echo "💾 Making commit..."
git commit -m "feat: Update application with VGG16 model (72.4% accuracy)

- VGG16 model with Transfer Learning
- 72.4% accuracy on test set
- Optimized preprocessing for 96x96px
- Functional auto-download system
- Git LFS configuration for large files"

# 6. Push to GitHub
echo "⬆️  Pushing to GitHub..."
git push origin main

echo "✅ Deployment completed!"
echo "🎉 Application available at: https://facial-emotion-classifier.streamlit.app"
echo "📁 Repository: https://github.com/sidnei-almeida/cnn-emotion-classifier"
