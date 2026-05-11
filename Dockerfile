# Runtime image for the cafaeval-protea CAFA-evaluator speedup fork.
#
# Ships the cafaeval CLI (Fmax / Smin / coverage / AuPRC) plus the fast
# PyArrow-backed prediction parser. PROTEA invokes this binary from the
# run_cafa_evaluation operation; the image is therefore primarily a
# CLI-shaped artifact, not a library.

FROM python:3.12-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml setup.py README.md LICENCE.md NOTICE ./
COPY src/ ./src/

# Install with the `fast` extra so PyArrow is available and the
# vectorised prediction parser is selected at runtime.
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir ".[fast]"

FROM python:3.12-slim

# libgomp1 for numpy / pyarrow at runtime.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /work

COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

ENV PYTHONUNBUFFERED=1

# Default to the CLI; users mount their input directory and run e.g.:
#   docker run --rm -v $PWD:/work cafaeval-protea \
#     <obo_file> <pred_dir> <gt_file> -out_dir results
ENTRYPOINT ["cafaeval"]
CMD ["--help"]
