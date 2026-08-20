# Stage 1: Base image
FROM node:24-bookworm-slim AS base
WORKDIR /app

# Stage 2: Install dependencies
FROM base AS deps
COPY package.json package-lock.json ./
RUN npm ci

# Stage 3: Build application
FROM base AS builder
COPY --from=deps /app/node_modules ./node_modules
COPY . .
ENV NEXT_TELEMETRY_DISABLED=1
RUN npm run build

# Stage 4: Production runner
FROM node:24-bookworm-slim AS runner
WORKDIR /app

ENV NODE_ENV=production \
    NEXT_TELEMETRY_DISABLED=1 \
    PORT=3000 \
    HOSTNAME="0.0.0.0"

# Create non-root user (UID/GID 10001:10001)
RUN groupadd -g 10001 nodejs && \
    useradd -u 10001 -g nodejs -s /bin/bash nextjs

# Copy standalone output and static assets
COPY --from=builder --chown=nextjs:nodejs /app/.next/standalone ./
COPY --from=builder --chown=nextjs:nodejs /app/.next/static ./.next/static

USER nextjs:nodejs

EXPOSE 3000

# Container health check probe
HEALTHCHECK --interval=15s --timeout=5s --start-period=10s --retries=3 \
    CMD node -e "require('http').get('http://localhost:3000', (res) => process.exit(res.statusCode < 400 ? 0 : 1)).on('error', () => process.exit(1))"

CMD ["node", "server.js"]
