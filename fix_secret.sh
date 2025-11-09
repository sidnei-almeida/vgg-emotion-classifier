#!/bin/bash

# Script to fix the secret token issue in git history

echo "🔧 Fixing secret token in git history..."
echo ""

# Check if we're in a git repository
if [ ! -d ".git" ]; then
    echo "❌ Error: Not a git repository"
    exit 1
fi

echo "📝 Current commit contains the token. We need to amend it."
echo ""
echo "Option 1: Amend the last commit (recommended if it's the last commit)"
echo "Option 2: Create a new commit that removes the token"
echo ""

# Show the last commit
echo "Last commit:"
git log -1 --oneline
echo ""

read -p "Do you want to amend the last commit? (y/n): " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "🔄 Amending last commit..."
    git add DEPLOY.md QUICK_DEPLOY.md deploy_hf.sh
    git commit --amend --no-edit
    echo "✅ Commit amended!"
    echo ""
    echo "⚠️  IMPORTANT: You need to force push since we rewrote history:"
    echo "   git push --force origin main"
    echo ""
    echo "⚠️  WARNING: Force push rewrites history. Make sure no one else"
    echo "   has pulled the commit with the token!"
else
    echo "📝 Creating new commit to remove token..."
    git add DEPLOY.md QUICK_DEPLOY.md deploy_hf.sh
    git commit -m "Remove hardcoded Hugging Face token for security"
    echo "✅ New commit created!"
    echo ""
    echo "⚠️  NOTE: The token is still in git history in the previous commit."
    echo "   For complete security, consider using git filter-branch or BFG"
    echo "   to remove it from history, or create a new repository."
    echo ""
    echo "Now you can push:"
    echo "   git push origin main"
fi

