# Step-by-Step GitHub Push Guide

Follow these exact steps to push this project to your GitHub account:

---

## 📋 Prerequisites

1. Have a GitHub account at [github.com](https://github.com).
2. If you don't have Git installed on your Windows machine, download it from [git-scm.com/download/win](https://git-scm.com/download/win) (or use **GitHub Desktop** from [desktop.github.com](https://desktop.github.com)).

---

## 🚀 Method 1: Using Terminal / PowerShell (Recommended)

### Step 1: Create a New GitHub Repository
1. Go to [https://github.com/new](https://github.com/new).
2. Set **Repository name** to: `restaurant-recommendation-system` (or any name you like).
3. Set visibility to **Public** or **Private**.
4. **Do NOT** check "Add a README file", "Add .gitignore", or "Choose a license" (our project already includes all of them).
5. Click **Create repository**.
6. Copy your repository URL (e.g., `https://github.com/YOUR_USERNAME/restaurant-recommendation-system.git`).

---

### Step 2: Open PowerShell in the Project Directory
Open PowerShell and navigate to the project directory:
```powershell
cd "C:\Users\Navya shree\.gemini\antigravity\scratch\restaurant-recommendation-system"
```

---

### Step 3: Initialize Git & Stage All Files
```powershell
# 1. Initialize git repository
git init

# 2. Add all files to staging (our .gitignore automatically excludes caches and secrets)
git add .

# 3. Create your first commit
git commit -m "feat(release): Bengaluru Restaurant Recommendation System complete release"

# 4. Set default branch to main
git branch -M main
```

---

### Step 4: Link Your Remote Repository & Push
Replace `YOUR_USERNAME` and `YOUR_REPO_NAME` with your actual GitHub username and repository name:
```powershell
# 5. Add remote origin URL
git remote add origin https://github.com/navyad-a/bengaluru-restaurant-recommender.git

# 6. Push to GitHub
git push -u origin main
```

*(If prompted, sign in with your GitHub credentials or Personal Access Token).*

---

## 🖥️ Method 2: Using GitHub Desktop (Visual App)

If you prefer a visual application without using the command line:

1. Download and open **GitHub Desktop** ([desktop.github.com](https://desktop.github.com)).
2. Click **File** > **Add Local Repository...** (or press `Ctrl + O`).
3. Click **Choose...** and select:
   `C:\Users\Navya shree\.gemini\antigravity\scratch\restaurant-recommendation-system`
4. If it says "This directory does not appear to be a Git repository", click **create a repository** here.
5. Click **Publish repository** in the top right corner.
6. Choose whether to make it Public, then click **Publish Repository**.
7. Done! Your code is now live on GitHub.

---

## 🔒 Security & Cleanliness Verified
- ✅ `.env` (containing private database passwords) is excluded by `.gitignore`.
- ✅ Only safe template `.env.example` will be pushed.
- ✅ Python cache (`__pycache__`), coverage, and temporary test files are excluded.
- ✅ Full 12,481 venue catalog and pre-trained model artifacts are included.
