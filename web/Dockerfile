# EcoArena web demo — static build served by Nginx.
# Build context MUST be the repo root (the sim imports ../configs/*.json):
#   docker build -f web/Dockerfile -t ecoarena-web .

# ---- 1. Build ----
FROM node:22-alpine AS build
WORKDIR /app
RUN corepack enable && corepack prepare pnpm@10 --activate

# Install deps first so this layer is cached between redeploys
COPY web/package.json web/pnpm-lock.yaml web/pnpm-workspace.yaml ./web/
RUN cd web && pnpm install --frozen-lockfile

# Source + the shared world configs from the repo root
COPY configs ./configs
COPY web ./web
RUN cd web && pnpm build

# ---- 2. Serve ----
FROM nginx:1.27-alpine
COPY web/nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/web/dist /usr/share/nginx/html
EXPOSE 80
HEALTHCHECK --interval=30s --timeout=3s CMD wget -qO- http://127.0.0.1/healthz || exit 1
