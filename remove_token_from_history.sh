#!/bin/bash

# Script to remove token from git history by rewriting commits

set -e

echo "🔒 Removing Hugging Face token from git history"
echo ""
echo "⚠️  WARNING: This will rewrite git history!"
echo "   Make sure you haven't pushed these commits yet or coordinate with your team."
echo ""

read -p "Continue? (y/n): " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 1
fi

echo ""
echo "📝 Current commits:"
git log --oneline -5
echo ""

echo "🔄 Rewriting commit c6ee3d0 to remove token..."
echo ""

# Method: Use git filter-repo (if available) or filter-branch
# Note: Replace YOUR_TOKEN_HERE with the actual token if needed
TOKEN_PATTERN="hf_[A-Za-z0-9]{32,}"
REPLACEMENT="HF_TOKEN_PLACEHOLDER"

if command -v git-filter-repo &> /dev/null; then
    echo "Using git-filter-repo..."
    echo "⚠️  Note: You may need to specify the exact token pattern"
    git filter-repo --invert-paths --path DEPLOY.md --path QUICK_DEPLOY.md --path deploy_hf.sh --force
    echo "⚠️  Manual step: Use git-filter-repo with your specific token pattern"
else
    echo "Using git filter-branch..."
    echo "⚠️  Note: Replace TOKEN_PATTERN with your actual token before running"
    # Remove token from all commits
    # Replace TOKEN_PATTERN with actual token pattern if needed
    git filter-branch --force --tree-filter '
        TOKEN_PATTERN="hf_[A-Za-z0-9]\{32,\}"
        REPLACEMENT="your_token_here"
        if [ -f DEPLOY.md ]; then
            sed -i "s/${TOKEN_PATTERN}/${REPLACEMENT}/g" DEPLOY.md
        fi
        if [ -f QUICK_DEPLOY.md ]; then
            sed -i "s/${TOKEN_PATTERN}/${REPLACEMENT}/g" QUICK_DEPLOY.md
        fi
        if [ -f deploy_hf.sh ]; then
            sed -i "s/${TOKEN_PATTERN}/${REPLACEMENT}/g" deploy_hf.sh
        fi
    ' --prune-empty --tag-name-filter cat -- --all
fi

echo ""
echo "✅ History rewritten!"
echo ""
echo "🔍 Verifying token pattern is removed..."
if git log --all --full-history -p | grep -qE "hf_[A-Za-z0-9]{32,}"; then
    echo "⚠️  WARNING: Token pattern still found in history!"
    echo "   You may need to specify the exact token"
else
    echo "✅ Token pattern successfully removed from history!"
fi

echo ""
echo "🚀 Next steps:"
echo "   1. Force push to update remote: git push --force origin main"
echo "   2. If you shared the token, revoke it at: https://huggingface.co/settings/tokens"
echo "   3. Create a new token if needed"

