# hash:sha256:41540ce033e8c94defd0106245f0277833e6e193f44a959b06b76499af96ed0a
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/pytorch:2.4.0-cuda12.4.0-mambaforge24.5.0-0-python3.12.4-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG GIT_ASKPASS
COPY git-ask-pass /

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        gcc=4:11.2.0-1ubuntu1 \
        libasound2=1.2.6.1-1ubuntu1.1 \
        libatk-bridge2.0-0=2.38.0-3 \
        libatk1.0-0=2.36.0-3build1 \
        libatspi2.0-0=2.44.0-3 \
        libcairo2=1.16.0-5ubuntu2.1 \
        libcups2=2.4.1op1-1ubuntu4.16 \
        libdrm2=2.4.113-2~ubuntu0.22.04.1 \
        libgbm1=23.2.1-1ubuntu3.1~22.04.3 \
        libgl1-mesa-glx=23.0.4-0ubuntu1~22.04.1 \
        libglib2.0-0=2.72.4-0ubuntu2.9 \
        libnspr4=2:4.35-0ubuntu0.22.04.1 \
        libnss3=2:3.98-0ubuntu0.22.04.3 \
        libpango-1.0-0=1.50.6+ds-2ubuntu1 \
        libxcomposite1=1:0.4.5-1build2 \
        libxdamage1=1:1.1.5-2build2 \
        libxfixes3=1:6.0.0-1 \
        libxkbcommon0=1.4.0-1 \
        libxrandr2=2:1.5.2-1build1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install -U --no-cache-dir \
    accelerate==1.13.0 \
    ai2-olmo-core==2.4.0 \
    bitsandbytes==0.49.2 \
    decord2==3.3.0 \
    einops==0.8.2 \
    httpx==0.28.1 \
    imageio[ffmpeg]==2.37.3 \
    molmo-utils==0.0.1 \
    pillow==12.2.0 \
    playwright==1.44.0 \
    s3fs==2026.3.0 \
    scikit-image==0.26.0 \
    transformers==4.57.6 \
    zarr==3.1.6

ADD "https://github.com/coder/code-server/releases/download/v4.100.3/code-server-4.100.3-linux-amd64.tar.gz" /.code-server/code-server.tar.gz

RUN cd /.code-server \
    && tar -xvf code-server.tar.gz \
    && rm code-server.tar.gz \
    && ln -s /.code-server/code-server-4.100.3-linux-amd64/bin/code-server /usr/bin/code-server \
    && mkdir -p /.vscode/extensions

COPY postInstall /
RUN --mount=type=secret,id=secrets . /run/secrets/secrets \
    && /postInstall
