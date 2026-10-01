# Development / benchmark environment: GCC, Clang, CMake, Python toolchain.
FROM ubuntu:24.04
ENV DEBIAN_FRONTEND=noninteractive PIP_BREAK_SYSTEM_PACKAGES=1
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential g++ clang clang-format cmake ninja-build git ca-certificates curl wget \
      python3 python3-dev python3-pip python3-venv linux-tools-generic valgrind \
    && rm -rf /var/lib/apt/lists/*
RUN pip3 install --no-cache-dir numpy h5py matplotlib pytest scikit-build-core pybind11 \
      hnswlib faiss-cpu psutil fastapi "uvicorn[standard]" pydantic httpx locust
WORKDIR /work
