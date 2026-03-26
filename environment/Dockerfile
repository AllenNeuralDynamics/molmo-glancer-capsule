# hash:sha256:2769ca3003d587e6c98dcf1111fa41770b409aa1174501afcc02556ff52410a5
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/pytorch:2.4.0-cuda12.4.0-mambaforge24.5.0-0-python3.12.4-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG GIT_ASKPASS
COPY git-ask-pass /

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        apt=2.0.6 \
        base-passwd=3.5.47 \
        bash=5.0-6ubuntu1.2 \
        bsdutils=1:2.34-0.1ubuntu9.3 \
        bzip2=1.0.8-2 \
        ca-certificates=20210119~20.04.2 \
        coreutils=8.30-3ubuntu2 \
        cuda-compat-11-7=515.43.04-1 \
        cuda-libraries-11-7=11.7.0-1 \
        cuda-nvtx-11-7=11.7.50-1 \
        curl=7.68.0-1ubuntu2.11 \
        dash=0.5.10.2-6 \
        debconf=1.5.73 \
        diffutils=1:3.7-3 \
        e2fsprogs=1.45.5-2ubuntu1 \
        fdisk=2.34-0.1ubuntu9.3 \
        findutils=4.7.0-1ubuntu1 \
        git=1:2.25.1-1ubuntu3.4 \
        gnupg2=2.2.19-3ubuntu2.1 \
        grep=3.4-1 \
        gzip=1.10-0ubuntu4.1 \
        hostname=3.23 \
        less=551-1ubuntu0.1 \
        libacl1=2.2.53-6 \
        libattr1=1:2.4.48-5 \
        libcudnn8=8.4.1.50-1+cuda11.6 \
        libext2fs2=1.45.5-2ubuntu1 \
        libpam-modules-bin=1.3.1-5ubuntu4.3 \
        libpam-runtime=1.3.1-5ubuntu4.3 \
        libpcre3=2:8.39-12build1 \
        libss2=1.45.5-2ubuntu1 \
        locales=2.31-0ubuntu9.9 \
        mawk=1.3.4.20200120-2 \
        mount=2.34-0.1ubuntu9.3 \
        nano=4.8-1ubuntu1 \
        ncurses-base=6.2-0ubuntu2 \
        ncurses-bin=6.2-0ubuntu2 \
        procps=2:3.3.16-1ubuntu2.3 \
        sed=4.7-1 \
        sensible-utils=0.0.12+nmu1 \
        sysvinit-utils=2.96-2.1ubuntu1 \
        unzip=6.0-25ubuntu1 \
    && rm -rf /var/lib/apt/lists/*

RUN conda install -y \
        conda==4.9.2 \
    && conda clean -ya
RUN conda install -y \
        python==3.8.5 \
    && conda clean -ya

ADD "https://github.com/coder/code-server/releases/download/v4.100.3/code-server-4.100.3-linux-amd64.tar.gz" /.code-server/code-server.tar.gz

RUN cd /.code-server \
    && tar -xvf code-server.tar.gz \
    && rm code-server.tar.gz \
    && ln -s /.code-server/code-server-4.100.3-linux-amd64/bin/code-server /usr/bin/code-server \
    && mkdir -p /.vscode/extensions

RUN code-server --extensions-dir="/.vscode/extensions" --install-extension anthropic.claude-code@2.1.84 \
    && code-server --extensions-dir="/.vscode/extensions" --install-extension ms-python.python@2025.14.0 \
    && code-server --extensions-dir="/.vscode/extensions" --install-extension ms-toolsai.jupyter@2025.4.1 \
    && code-server --extensions-dir="/.vscode/extensions" --install-extension reditorsupport.r@2.8.6 \
    && code-server --extensions-dir="/.vscode/extensions" --install-extension saoudrizwan.claude-dev@3.35.1
