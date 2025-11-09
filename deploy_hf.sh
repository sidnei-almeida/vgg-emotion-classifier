#!/bin/bash

# Script to deploy API to Hugging Face Spaces
# Space: https://huggingface.co/spaces/salmeida/vgg16-emotion-classifier

set -e

SPACE_NAME="salmeida/vgg16-emotion-classifier"
SPACE_URL="https://huggingface.co/spaces/${SPACE_NAME}"
REPO_URL="https://huggingface.co/spaces/${SPACE_NAME}"

echo "🚀 Preparing deployment to Hugging Face Space: ${SPACE_NAME}"
echo ""

# Check if in a git repository
if [ ! -d ".git" ]; then
    echo "📦 Initializing git repository..."
    git init
fi

# Check if Hugging Face remote is configured
if ! git remote | grep -q "hf"; then
    echo "🔗 Configuring Hugging Face remote..."
    git remote add hf ${REPO_URL}
    echo "✅ Remote configured: hf -> ${REPO_URL}"
else
    echo "✅ Hugging Face remote already configured"
fi

# Check required files
echo ""
echo "📋 Checking required files..."

REQUIRED_FILES=(
    "api.py"
    "Dockerfile"
    "requirements.txt"
    "app.yaml"
    "image_pre_processing.py"
    "haarcascade_frontalface_default.xml"
)

MISSING_FILES=()
for file in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "$file" ]; then
        MISSING_FILES+=("$file")
    fi
done

if [ ${#MISSING_FILES[@]} -ne 0 ]; then
    echo "❌ Files not found:"
    printf '   - %s\n' "${MISSING_FILES[@]}"
    exit 1
fi

echo "✅ All required files are present"
echo ""

# Check if models exist
if [ ! -d "models" ] || [ -z "$(ls -A models/*.h5 models/*.keras 2>/dev/null)" ]; then
    echo "⚠️  WARNING: No models found in models/ folder"
    echo "   Make sure to upload models before deployment"
    echo ""
fi

# Add files to git
echo "📝 Adding files to git..."
git add api.py Dockerfile requirements.txt app.yaml image_pre_processing.py
git add haarcascade_frontalface_default.xml

# Add models if they exist
if [ -d "models" ] && [ -n "$(ls -A models/*.h5 models/*.keras 2>/dev/null 2>&1)" ]; then
    echo "📦 Adding models..."
    git add models/
fi

# Add other useful files
if [ -f "README.md" ]; then
    git add README.md
fi
if [ -f "README_API.md" ]; then
    git add README_API.md
fi

# Check if there are changes to commit
if git diff --staged --quiet; then
    echo "ℹ️  No changes to commit"
else
    echo "💾 Committing changes..."
    git commit -m "Deploy FastAPI to Hugging Face Spaces" || echo "⚠️  No changes to commit"
fi

echo ""
echo "🚀 Pushing to Hugging Face..."
echo ""
echo "⚠️  IMPORTANT: When prompted, use your Hugging Face access token as password"
echo "   Get your token from: https://huggingface.co/settings/tokens"
echo ""
echo "   Or set it as environment variable:"
echo "   export HF_TOKEN=your_token_here"
echo ""
echo "   Or configure git credential helper:"
echo "   git config credential.helper store"
echo ""

# Push
git push hf main || git push hf master || {
    echo ""
    echo "❌ Error pushing. Trying to configure branch..."
    CURRENT_BRANCH=$(git branch --show-current)
    if [ -z "$CURRENT_BRANCH" ]; then
        git checkout -b main
        git push -u hf main
    else
        git push -u hf ${CURRENT_BRANCH}
    fi
}

echo ""
echo "✅ Deployment started!"
echo ""
echo "📊 Monitor build progress at: ${SPACE_URL}"
echo "⏱️  Build may take a few minutes on first run"
echo ""
echo "🔍 After build, your API will be available at:"
echo "   ${SPACE_URL}"
echo ""
echo "📚 API documentation will be at:"
echo "   ${SPACE_URL}/docs"
