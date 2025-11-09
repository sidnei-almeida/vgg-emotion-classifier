# 🔒 Security Fix: Removing Token from Git History

The Hugging Face token was accidentally committed to git history. Here's how to fix it:

## ✅ Step 1: Token Removed from Files

The token has been removed from:
- `DEPLOY.md`
- `QUICK_DEPLOY.md`
- `deploy_hf.sh`

A new commit has been created to remove it.

## ⚠️ Step 2: Remove Token from Git History

The token is still in commit `c6ee3d0`. You have two options:

### Option A: Amend the Previous Commit (Recommended)

```bash
# Reset to before the commit with the token
git reset --soft HEAD~2

# Stage all changes (including token removal)
git add .

# Create a new commit without the token
git commit -m "Migrate from Streamlit to FastAPI and remove hardcoded tokens"

# Force push (required since we rewrote history)
git push --force origin main
```

### Option B: Use git filter-branch (Advanced)

```bash
# Remove token from entire git history
git filter-branch --force --index-filter \
  "git rm --cached --ignore-unmatch DEPLOY.md QUICK_DEPLOY.md deploy_hf.sh" \
  --prune-empty --tag-name-filter cat -- --all

# Force push
git push --force --all origin
```

## 🔑 Step 3: Get a New Token (If Needed)

If the exposed token needs to be revoked:

1. Go to https://huggingface.co/settings/tokens
2. Revoke the old token
3. Create a new token
4. Store it securely (environment variable, password manager, etc.)

## 📝 Step 4: Use Token Securely Going Forward

**Never commit tokens to git!** Instead:

1. Use environment variables:
   ```bash
   export HF_TOKEN=your_token_here
   ```

2. Use git credential helper:
   ```bash
   git config credential.helper store
   # Enter token once when prompted
   ```

3. Use `.env` file (add to `.gitignore`):
   ```bash
   echo "HF_TOKEN=your_token_here" > .env
   echo ".env" >> .gitignore
   ```

## ✅ Verification

After fixing, verify the token is gone:

```bash
# Search for token pattern in git history
git log -p | grep -iE "hf_[A-Za-z0-9]{32,}"

# Should return nothing if successfully removed
# Replace pattern with your specific token if needed
```

## 🚀 After Fix

Once the token is removed from history, you can push normally:

```bash
git push origin main
```

