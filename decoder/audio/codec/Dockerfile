FROM nvcr.io/nvidia/pytorch:22.12-py3

ARG CONDA_ENV=audiollm
ENV CONDA_ENV=$CONDA_ENV
WORKDIR /app

# 필수 패키지 설치 및 캐시 정리
RUN apt-get update && \
    apt-get install -y wget bzip2 ca-certificates libsndfile-dev openjdk-11-jdk && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Optionally set JAVA_HOME (if TorchServe specifically needs it)
ENV JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64

# Miniconda 설치 및 초기화
ENV CONDA_HOME=/opt/conda
ENV PATH=/opt/conda/bin:$PATH
RUN ARCH=$(uname -m) && \
    wget "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-${ARCH}.sh" -O /tmp/miniforge.sh && \
    bash /tmp/miniforge.sh -b -p $CONDA_HOME && \
    rm /tmp/miniforge.sh

# Create your environment
RUN conda create -y -n $CONDA_ENV python=3.11

# Install packages into that environment (conda or pip)
RUN conda run -n $CONDA_ENV conda install -y -c conda-forge ffmpeg libsndfile==1.2.2

COPY requirements.txt /app/requirements.txt
RUN conda run -n $CONDA_ENV pip install --no-cache-dir -r /app/requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cu118

COPY src /app/src
COPY entrypoint.sh /app/entrypoint.sh
COPY setup.py /app/setup.py
RUN conda run -n $CONDA_ENV pip install -e /app

RUN chmod +x /app/entrypoint.sh
ENTRYPOINT ["/app/entrypoint.sh"]
