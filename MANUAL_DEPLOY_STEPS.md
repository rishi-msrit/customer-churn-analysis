# Deploying to Vercel — Step-by-Step

This guide walks through connecting your GitHub repo to Vercel
for the first time and deploying the Customer Churn Analysis app.

---

## Prerequisites

- GitHub repo is public and contains the committed `public/data/` JSON files
  (run `python analysis/run_analysis.py` locally first, then commit everything)
- A free Vercel account (sign up with your GitHub account at vercel.com)

---

## Step 1 — Push the repo to GitHub

If you haven't already:

```
git init
git add .
git commit -m "initial build"
git remote add origin https://github.com/rishi-msrit/customer-churn-analysis.git
git push -u origin main
```

---

## Step 2 — Connect to Vercel

1. Go to **https://vercel.com** and click **Sign Up** (or **Log In**).
2. Select **Continue with GitHub** and authorise Vercel.
3. On the Vercel dashboard, click **Add New… → Project**.
4. Under "Import Git Repository", find `customer-churn-analysis` and click **Import**.

---

## Step 3 — Configure the project

Vercel will show a configuration screen. Make the following settings:

| Field | Value |
|-------|-------|
| Framework Preset | **Other** (leave as-is — this is a static site) |
| Root Directory | `.` (leave blank / project root) |
| Build Command | *(leave empty)* |
| Output Directory | `public` |
| Install Command | *(leave empty)* |

> **Note**: The `vercel.json` at the project root already sets `"outputDirectory": "public"`,
> so Vercel may pre-fill this for you. Confirm it says `public`, then continue.

---

## Step 4 — Deploy

1. Click **Deploy**.
2. Vercel runs in about 30–60 seconds. Watch the build log.
3. When done, you'll see a green checkmark and a URL like:
   `https://customer-churn-analysis-xyz.vercel.app`
4. Click **Visit** to open your live app.

---

## Step 5 — Set a custom project name (optional)

1. In Vercel dashboard → your project → **Settings → General**.
2. Change the project name to something like `churn-analysis`.
3. Your URL becomes `https://churn-analysis.vercel.app` (if available).

---

## Step 6 — Verify all pages

Open the live URL and check:

- `/` — Overview page loads with KPI values and 4 charts
- `/pages/drivers.html` — 5 driver charts + statistical findings
- `/pages/prediction.html` — sliders update the probability in real time
- `/pages/model-performance.html` — metrics strip + 4 charts
- `/pages/segments.html` — scatter plot + 4 segment cards

---

## Future updates

Every time you push new commits to `main`, Vercel redeploys automatically.
If you re-run the analysis script and regenerate the JSON files, commit those
files too — the deployed app reads directly from `public/data/`.
