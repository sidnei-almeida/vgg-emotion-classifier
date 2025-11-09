#!/bin/bash

# Script to push to Hugging Face Space

set -e

HF_REMOTE="hf"
HF_SPACE="salmeida/vgg16-emotion-classifier"

echo "🚀 Pushing to Hugging Face Space: ${HF_SPACE}"
echo ""

# Check if remote exists
if ! git remote | grep -q "^${HF_REMOTE}$"; then
    echo "❌ Remote '${HF_REMOTE}' not found"
    echo "Adding remote..."
    git remote add ${HF_REMOTE} https://huggingface.co/spaces/${HF_SPACE}
fi

# Check required files
echo "📋 Checking required files..."

REQUIRED_FILES=(
    "api.py"
    "Dockerfile"
    "requirements.txt"
    "app.yaml"
    "image_pre_processing.py"
    "haarcascade_frontalface_default.xml"
)

MISSING=()
for file in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "$file" ]; then
        MISSING+=("$file")
    fi
done

if [ ${#MISSING[@]} -ne 0 ]; then
    echo "❌ Missing files:"
    printf '   - %s\n' "${MISSING[@]}"
    exit 1
fi

echo "✅ All required files present"
echo ""

# Check models
if [ ! -d "models" ] || [ -z "$(ls -A models/*.h5 models/*.keras 2>/dev/null)" ]; then
    echo "⚠️  WARNING: No models found in models/ folder"
    echo "   The API will not work without models"
    read -p "Continue anyway? (y/n): " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo "📦 Pushing to Hugging Face..."
echo ""
echo "⚠️  When prompted:"
echo "   Username: salmeida"
echo "   Password: Use your Hugging Face access token"
echo "   Get token from: https://huggingface.co/settings/tokens"
echo ""

# Push to Hugging Face
git push ${HF_REMOTE} main || {
    echo ""
    echo "❌ Push failed. Trying with current branch name..."
    CURRENT_BRANCH=$(git branch --show-current)
    git push -u ${HF_REMOTE} ${CURRENT_BRANCH} || {
        echo ""
        echo "❌ Push failed. Please check:"
        echo "   1. Your Hugging Face token is correct"
        echo "   2. You have write access to the Space"
        echo "   3. The Space exists at: https://huggingface.co/spaces/${HF_SPACE}"
        exit 1
    }
}

echo ""
echo "✅ Successfully pushed to Hugging Face!"
echo ""
echo "📊 Monitor build at: https://huggingface.co/spaces/${HF_SPACE}"
echo "⏱️  Build may take 5-10 minutes on first run"
echo ""
echo "🔍 After build, your API will be at:"
echo "   https://${HF_SPACE//\//-}.hf.space"
echo ""
echo "📚 API documentation:"
echo "   https://${HF_SPACE//\//-}.hf.space/docs"

