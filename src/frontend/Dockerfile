# =============================================================================
# Dockerfile - Acne Advisor AI Frontend (React/Vite)
# Deploy như Railway service RIÊNG, root directory = src/frontend
# =============================================================================

FROM node:20-alpine AS build
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci
COPY . .

ARG VITE_API_URL
ENV VITE_API_URL=$VITE_API_URL
RUN npm run build

FROM node:20-alpine
WORKDIR /app
RUN npm install -g serve
COPY --from=build /app/dist ./dist
EXPOSE 3000
CMD ["sh", "-c", "serve -s dist -l ${PORT:-3000}"]
