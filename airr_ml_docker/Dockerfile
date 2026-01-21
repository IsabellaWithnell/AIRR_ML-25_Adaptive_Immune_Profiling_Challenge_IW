# =============================================================================
# AIRR-ML Competition Docker Image
# =============================================================================
# Unified Pipeline v9 + Parallel Task 2 Scoring
#
# Build:
#   docker build -t airr-ml:v9 .
#
# Run single dataset:
#   docker run -v /path/to/data:/data -v /path/to/output:/output airr-ml:v9 \
#       --train_dir /data/train_dataset_1 \
#       --test_dirs /data/test_dataset_1 \
#       --out_dir /output \
#       --n_jobs 16
#
# Run with environment variables:
#   docker run -v /path/to/data:/data -v /path/to/output:/output \
#       -e AIRR_MODELS="vj,kmer4,gapped" \
#       -e TASK2_MODEL_OVERRIDE="BEST" \
#       -e TASK2_USE_TEST_DATA=1 \
#       airr-ml:v9 \
#       --train_dir /data/train_dataset_1 \
#       --test_dirs /data/test_dataset_1 \
#       --out_dir /output
# =============================================================================

FROM python:3.10-slim

LABEL maintainer="AIRR-ML Pipeline"
LABEL version="9.0"
LABEL description="AIRR-ML Competition Submission with Parallel Task 2 Scoring"

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Create working directory
WORKDIR /app

# Copy requirements first (for better caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the main script
COPY airr_ml_submission_v9_parallel_task2.py .

# Create directories for data mounting
RUN mkdir -p /data /output

# Set default environment variables (can be overridden at runtime)
ENV AIRR_MODELS=""
ENV TASK2_MODEL_OVERRIDE="BEST"
ENV TASK2_USE_TEST_DATA="0"
ENV AIRR_ENSEMBLE_MODE="standard"

# Default number of jobs (can be overridden)
ENV N_JOBS="16"

# Entry point
ENTRYPOINT ["python3", "airr_ml_submission_v9_parallel_task2.py"]

# Default arguments (can be overridden)
CMD ["--help"]
