FROM node:20-slim AS development-dependencies-env
COPY . /app
WORKDIR /app
RUN npm ci

FROM node:20-slim AS production-dependencies-env
COPY ./package.json package-lock.json /app/
WORKDIR /app
RUN npm ci --omit=dev

FROM node:20-slim AS build-env
COPY . /app/
COPY --from=development-dependencies-env /app/node_modules /app/node_modules
WORKDIR /app
RUN npm run build

FROM node:20-slim
# Install Python and build dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    g++ \
    make \
    && rm -rf /var/lib/apt/lists/*

# Set up Python environment
ENV PYTHON_BINARY=python3
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Install Python dependencies
COPY requirements-docker.txt /app/requirements-docker.txt
RUN pip install --no-cache-dir -r /app/requirements-docker.txt
RUN playwright install --with-deps chromium

COPY ./package.json package-lock.json /app/
COPY --from=production-dependencies-env /app/node_modules /app/node_modules
COPY --from=build-env /app/build /app/build
COPY ./Image-to-Code-UI-Generation /app/Image-to-Code-UI-Generation
COPY server.js /app/server.js

WORKDIR /app

# Verify directory exists (User Request)
RUN ls -R /app/Image-to-Code-UI-Generation

CMD ["npm", "run", "start"]