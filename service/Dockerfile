# Multi-stage build. Stage 1 compiles the C++ engine into a wheel; stage 2 is a slim runtime
# image with just the wheel, the web stack and the app (no compiler, no build tree).
# Build context: the repo root (see docker-compose.yml).

FROM python:3.12-slim AS build
RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential cmake ninja-build \
 && rm -rf /var/lib/apt/lists/*
WORKDIR /src
COPY CMakeLists.txt CMakePresets.json pyproject.toml README.md ./
COPY include include
COPY src src
COPY python python
RUN pip wheel --no-deps --wheel-dir /wheels .

FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
COPY service/requirements.txt /tmp/requirements.txt
COPY --from=build /wheels /wheels
RUN pip install --no-cache-dir -r /tmp/requirements.txt /wheels/*.whl && rm -rf /wheels
WORKDIR /srv
COPY service/app app
RUN useradd --system --no-create-home vecsearch
USER vecsearch
EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=3s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
