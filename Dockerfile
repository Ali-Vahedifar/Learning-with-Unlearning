# Environment for the LwU experiments.
#
# SCOPE: this image pins the software environment. It does not contain
# datasets, trained checkpoints, or results. Running the container gives a
# working environment in which the reproduction scripts can be executed; the
# runs themselves require the datasets to be mounted and take GPU time.
#
#   docker build -t lwu:neurips .
#   docker run --gpus all -it \
#       -v /path/to/data:/data \
#       -v $(pwd)/results:/workspace/results \
#       lwu:neurips \
#       bash scripts/reproduce_table1.sh
#
# Run `pytest -q` inside the container to confirm the environment is sane
# before committing to a long run.

FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        wget \
        unzip \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY requirements.txt /workspace/requirements.txt
RUN pip install --no-cache-dir -r /workspace/requirements.txt

COPY . /workspace
RUN pip install --no-cache-dir -e .

# Datasets are mounted, never baked in.
ENV LWU_DATA_ROOT=/data

CMD ["bash"]
