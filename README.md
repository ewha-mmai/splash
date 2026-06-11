# Splash — Project Page

Project website for **"Wake up for Touch! Mask-isolated Tactile Alignment Learning in MLLMs."**

## Contents

```
index.html                  # the project page (everything is in here)
static/images/              # figures pulled from the paper
  fig1_forgetting.png
  fig2_framework.png
  fig3a_tvl.png
  fig3b_ssvtp.png
```

## Publish with GitHub Pages

1. Push these files to the root of the `splash` repository (so `index.html` sits at the top level).
2. On GitHub, go to **Settings → Pages**.
3. Under **Source**, choose **Deploy from a branch**, pick the `main` branch and the `/ (root)` folder, then **Save**.
4. After a minute, the site is live at:

   ```
   https://ewha-mmai.github.io/splash/
   ```

## Things to update before sharing

- **Paper / arXiv links** — in `index.html`, the `Paper` and `arXiv` buttons currently point to `#`. Replace the `href` once you have the PDF and arXiv URLs (search for `id="paperLink"` and `id="arxivLink"`).
- **BibTeX** — update the citation block once the official entry is assigned.
- **Author emails / links** — add personal or lab links to the author names if you'd like.

## Editing

Everything (HTML, CSS, and the little parameter-grid animation) lives in a single `index.html`, so you can open it directly in a browser to preview changes — no build step.
