# movienight — frontend

SvelteKit + Tailwind frontend for [movienight](../README.md). Hosts the landing
page hero (3D ALS-latent-space point cloud rendered with Three.js), the
group-recs flow, voting UI, compatibility report, and the personalized `/explore`
viz.

In production this is built with `@sveltejs/adapter-cloudflare` and deployed to
Cloudflare Pages; the API runs separately on Hugging Face Spaces and is
addressed via `VITE_API_BASE`. In docker compose, an nginx sidecar serves the
built app and proxies `/api/*` to the FastAPI container on the internal
network.

## Local dev

```sh
npm install
VITE_API_BASE=http://localhost:8000 VITE_TMDB_KEY=your_tmdb_v3_key npm run dev
# open http://localhost:5173/
```

Both env vars are optional:

- `VITE_API_BASE` — base URL for the FastAPI backend (default
  `http://localhost:8000`). At runtime you can also set
  `window.LBRECS_API` from an injected `<script>` tag to override it without
  rebuilding (handy on Cloudflare Pages → HF Spaces deploys).
- `VITE_TMDB_KEY` — TMDB v3 read key. Without it, posters and
  streaming-availability badges silently degrade. Grab one at
  https://www.themoviedb.org/settings/api.

## Build

```sh
npm run build      # production build
npm run preview    # serve the built app locally
```

See the [project root README](../README.md) for the full stack, dataset
requirements, and architecture overview.
