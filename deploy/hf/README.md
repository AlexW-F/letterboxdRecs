---
title: Movienight API
emoji: 🎬
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# movienight — API

FastAPI backend for [movienight](https://movienight.alex-wf.com): group movie-night
recommendations from everyone's Letterboxd taste (SVD + ALS + Tag-Genome re-ranking).

This Space serves the REST API only; the frontend is hosted separately on
Cloudflare Pages. Interactive docs at `/docs`.

**Required Space settings (Settings → Variables and secrets):**
- `CORS_ALLOW_ORIGINS` = `https://movienight.alex-wf.com` (the frontend origin)
- `TMDB_API_KEY` = optional; only needed to enrich raw Letterboxd CSV uploads that
  lack a `tmdb_id` column. The username/RSS flow includes TMDB IDs inline.

The trained models, content features, popularity table, and MovieLens
`movies.csv`/`links.csv` are committed to this Space repo via git-LFS.
