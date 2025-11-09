# 🔧 Fix Merge Conflicts

## Current Situation

Your local branch has diverged from `origin/main`:
- **Local**: Migrated to FastAPI, removed Streamlit files
- **Remote**: Has old Streamlit `app.py` update

Since we've completely migrated to API, we need to force push the clean history.

## ✅ Solution: Force Push Clean History

### Option 1: Force Push with Lease (Safer)

```bash
git push --force-with-lease origin main
```

This will:
- ✅ Push your clean API migration
- ✅ Overwrite the old Streamlit commit
- ✅ Fail safely if someone else pushed changes

### Option 2: Regular Force Push

If `--force-with-lease` fails and you're sure no one else pushed:

```bash
git push --force origin main
```

⚠️ **Warning**: This overwrites remote history. Only use if you're certain.

## 🔐 Authentication

When prompted:
- **Username**: `sidnei-almeida`
- **Password**: Use a GitHub Personal Access Token (not your password)
  - Get token from: https://github.com/settings/tokens
  - Token needs `repo` permissions

Or configure credential helper:
```bash
git config credential.helper store
# Enter credentials once, they'll be saved
```

## 📊 Verify After Push

1. Check GitHub: https://github.com/sidnei-almeida/vgg-emotion-classifier
2. Verify `app.py` is gone (Streamlit removed)
3. Verify `api.py` exists (FastAPI added)
4. Check that token is not in any files

## ✅ Expected Result

After push:
- ✅ All Streamlit files removed
- ✅ FastAPI files present
- ✅ No hardcoded tokens
- ✅ Clean git history

## 🚨 If Push Still Fails

If you still get token/secret errors:

1. Check for any remaining tokens:
   ```bash
   git log -p | grep -i "hf_"
   ```

2. If tokens found, use the removal script:
   ```bash
   ./remove_token_from_history.sh
   ```

3. Then force push again

