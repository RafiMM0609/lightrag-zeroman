# Stage 1: Base & Development
FROM oven/bun:1-alpine AS base

WORKDIR /app

# Copy package files first for caching
COPY package.json bun.lock* package-lock.json* ./

# Install dependencies
RUN bun install

# Copy application source code
COPY . .

# Expose Vite dev server port
EXPOSE 5173

# Run Vite dev server bound to 0.0.0.0 so it's accessible from host
CMD ["bun", "run", "dev", "--host"]


# Stage 2: Production Builder
FROM base AS builder
# Build the application. The output goes to /app/lightrag/api/webui
RUN bun run build


# Stage 3: Production Server (Nginx)
FROM nginx:alpine AS production

# Copy custom nginx configuration for SPA routing
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Copy build artifacts to Nginx html directory
COPY --from=builder /app/lightrag/api/webui /usr/share/nginx/html

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
