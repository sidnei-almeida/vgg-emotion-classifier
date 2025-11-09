#!/bin/bash

# Script to deploy API to Hugging Face Spaces

set -e

echo "🚀 Preparing API deployment to Hugging Face Spaces..."

# Check if in a git repository
if [ ! -d ".git" ]; then
    echo "❌ Error: This directory is not a git repository"
    echo "   Run: git init"
    exit 1
fi

# Check if required files exist
echo "📋 Checking required files..."

REQUIRED_FILES=(
    "api.py"
    "Dockerfile"
    "requirements.txt"
    "app.yaml"
    "image_pre_processing.py"
    "haarcascade_frontalface_default.xml"
)

for file in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "$file" ]; then
        echo "❌ File not found: $file"
        exit 1
    fi
done

# Check if models exist
if [ ! -d "models" ] || [ -z "$(ls -A models/*.h5 models/*.keras 2>/dev/null)" ]; then
    echo "⚠️  Warning: No models found in models/ folder"
    echo "   Make sure to upload models before deployment"
fi

echo "✅ All required files are present"
echo ""
echo "📝 Next steps:"
echo "1. Create a new Space on Hugging Face: https://huggingface.co/spaces"
echo "2. Choose 'Docker' as SDK"
echo "3. Clone the Space repository:"
echo "   git clone https://huggingface.co/spaces/your-username/your-space"
echo "4. Copy files to the Space directory"
echo "5. Commit and push:"
echo "   git add ."
echo "   git commit -m 'Deploy API'"
echo "   git push"
echo ""
echo "📚 See README_API.md for more information"
