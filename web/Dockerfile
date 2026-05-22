# Static SvelteKit build served by nginx, with /api/* proxied to the
# FastAPI service over the docker-compose network.

FROM node:22-alpine AS build
WORKDIR /web

COPY package.json package-lock.json* ./
RUN npm ci --no-audit --no-fund

COPY svelte.config.js vite.config.ts tsconfig.json ./
COPY src ./src
COPY static ./static

# At build time we want the SPA to use a relative /api prefix so that
# in production the same origin (nginx) serves both the bundle and the
# proxied API. The dev workflow continues to use VITE_API_BASE=http://localhost:8000.
ENV VITE_API_BASE=/api
# TMDB key is optional; the UI degrades gracefully without posters.
ARG VITE_TMDB_KEY=
ENV VITE_TMDB_KEY=$VITE_TMDB_KEY

RUN npm run build

FROM nginx:1.27-alpine
COPY --from=build /web/build /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
