FROM python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_LINK_MODE=copy \
    MUJOCO_GL=egl \
    PYOPENGL_PLATFORM=egl \
    MPLBACKEND=Agg

WORKDIR /app

# System libraries for building Python wheels, MuJoCo headless rendering, and video output.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    cmake \
    curl \
    ffmpeg \
    git \
    libavcodec-dev \
    libavformat-dev \
    libegl1 \
    libegl-dev \
    libffi-dev \
    libglew-dev \
    libglfw3 \
    libglfw3-dev \
    libgl1 \
    libgl1-mesa-dev \
    libglib2.0-0 \
    libgomp1 \
    libgles2 \
    libosmesa6 \
    libosmesa6-dev \
    libsm6 \
    libstdc++6 \
    libswscale-dev \
    libx11-6 \
    libxcursor1 \
    libxext6 \
    libxfixes3 \
    libxi6 \
    libxinerama1 \
    libxrandr2 \
    libxrender1 \
    libxxf86vm1 \
    patchelf \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

COPY pyproject.toml README.md ./

RUN uv pip install --system "cython==0.29.36"

RUN uv sync --no-dev --no-install-project

COPY . .

CMD ["uv", "run", "python", "train.py", "env=walker-walk"]
