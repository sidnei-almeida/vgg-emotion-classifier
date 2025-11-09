#!/bin/bash

# Script to resolve merge conflicts by force pushing clean history

set -e

echo "🔧 Resolving merge conflicts"
echo ""
echo "Current situation:"
echo "  - Local: Migrated to FastAPI, removed Streamlit"
echo "  - Remote: Has old Streamlit app.py update"
echo ""
echo "Since we've completely migrated to API, we'll force push our clean history."
echo ""

read -p "Continue with force push? (y/n): " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 1
fi

echo ""
echo "🚀 Force pushing clean history..."
echo ""

# Use --force-with-lease for safety (fails if remote was updated)
git push --force-with-lease origin main || {
    echo ""
    echo "⚠️  Force-with-lease failed. Remote may have been updated."
    echo ""
    read -p "Use regular force push? (y/n): " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        git push --force origin main
        echo "✅ Force push completed!"
    else
        echo "Cancelled. Manual intervention needed."
        exit 1
    fi
}

echo ""
echo "✅ Successfully pushed clean history!"
echo ""
echo "📊 Verify at: https://github.com/sidnei-almeida/vgg-emotion-classifier"

