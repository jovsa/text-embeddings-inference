# TEI Server Setup Guide

This guide walks you through setting up a Text Embeddings Inference (TEI) server using different methods.

## Table of Contents

1. [Quick Start with Docker](#quick-start-with-docker)
2. [Local Installation (CPU)](#local-installation-cpu)
3. [Local Installation (GPU/CUDA)](#local-installation-gpucuda)
4. [Verifying the Installation](#verifying-the-installation)
5. [Running the Benchmark](#running-the-benchmark)
6. [Troubleshooting](#troubleshooting)

---

## Quick Start with Docker

The fastest way to get started is using Docker. This method requires Docker to be installed and running.

### Prerequisites

- Docker installed and running
- For GPU support: [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) installed
- NVIDIA drivers compatible with CUDA 12.2+ (for GPU support)

### Step 1: Pull and Run the Docker Container

```bash
# Set your model (example: BAAI/bge-small-en-v1.5)
model=BAAI/bge-small-en-v1.5

# Optional: Create a volume to cache model weights
volume=$PWD/data

# Run the container
docker run --gpus all -p 8080:80 -v $volume:/data --pull always \
  ghcr.io/huggingface/text-embeddings-inference:1.8 \
  --model-id $model
```

**Notes:**
- `--gpus all`: Enable GPU support (remove if CPU-only)
- `-p 8080:80`: Map container port 80 to host port 8080
- `-v $volume:/data`: Mount a volume to cache model weights (optional but recommended)
- `--pull always`: Always pull the latest image

### Step 2: Verify the Server is Running

```bash
# Check health endpoint
curl http://127.0.0.1:8080/health

# Get model info
curl http://127.0.0.1:8080/info
```

---

## Local Installation (CPU)

Building from source gives you more control and doesn't require Docker.

### Prerequisites

- Linux, macOS, or Windows with WSL2
- Rust toolchain (installed via rustup)
- Build dependencies

### Step 1: Install Rust

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source "$HOME/.cargo/env"
```

### Step 2: Install Build Dependencies

**On Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install -y libssl-dev gcc pkg-config
```

**On macOS:**
```bash
brew install openssl pkg-config
```

**On Fedora/RHEL:**
```bash
sudo dnf install openssl-devel gcc pkg-config
```

### Step 3: Build TEI

Choose the appropriate backend for your system:

```bash
cd /path/to/text-embeddings-inference

# For x86 systems with ONNX backend (recommended)
cargo install --path router -F ort

# For x86 systems with Intel MKL backend
cargo install --path router -F mkl

# For Apple M1/M2/M3 (Metal support)
cargo install --path router -F metal
```

**Build Time:** Expect 15-30 minutes depending on your system.

### Step 4: Start the Server

```bash
# Set your model
model=BAAI/bge-small-en-v1.5

# Start the server
text-embeddings-router --model-id $model --port 8080
```

---

## Local Installation (GPU/CUDA)

For GPU acceleration, you'll need CUDA support.

### Prerequisites

- NVIDIA GPU with CUDA compute capability ≥ 7.5 (T4, RTX 2000+, A100, etc.)
- CUDA 12.2+ installed
- NVIDIA drivers compatible with CUDA 12.2+
- Rust toolchain (see CPU installation above)

### Step 1: Set Up CUDA Environment

```bash
# Add CUDA to PATH
export PATH=$PATH:/usr/local/cuda/bin

# Verify CUDA installation
nvcc --version
```

### Step 2: Build TEI with CUDA Support

```bash
cd /path/to/text-embeddings-inference

# For Turing GPUs (T4, RTX 2000 series)
cargo install --path router -F candle-cuda-turing

# For Ampere and Hopper GPUs (A100, H100, RTX 3000+, RTX 4000+)
cargo install --path router -F candle-cuda
```

**Build Time:** Expect 30-60 minutes as CUDA kernels need to be compiled.

### Step 3: Start the Server

```bash
model=BAAI/bge-small-en-v1.5
text-embeddings-router --model-id $model --port 8080
```

---

## Verifying the Installation

### 1. Check Server Health

```bash
curl http://127.0.0.1:8080/health
```

Expected response: `{"status":"ok"}`

### 2. Get Model Information

```bash
curl http://127.0.0.1:8080/info | jq
```

This returns model details including:
- Model ID
- Max input length
- Tokenization workers
- Supported features

### 3. Test Embedding Generation

```bash
curl http://127.0.0.1:8080/embed \
  -X POST \
  -H "Content-Type: application/json" \
  -d '{"inputs": "What is Deep Learning?"}'
```

You should receive a JSON response with embedding vectors.

---

## Running the Benchmark

### Component Benchmark

Once your server is running, you can benchmark its performance:

```bash
cd /path/to/text-embeddings-inference/examples
python3 benchmark_components.py --url http://127.0.0.1:8080
```

The benchmark script will:
- Test different text lengths (short, medium, long)
- Measure component timings (tokenization, queue, inference)
- Test concurrency and batching performance
- Display detailed statistics

**Example Output:**
```
📦 Model: BAAI/bge-small-en-v1.5
   Max input length: 512 tokens
   Tokenization workers: 8

📈 Results (20 samples):
  Total              : mean=   12.45ms, median=   12.30ms
  Tokenization       : mean=    0.15ms, median=    0.14ms
  Queue              : mean=    0.05ms, median=    0.03ms
  Inference          : mean=   12.25ms, median=   12.10ms
```

### Backend Comparison Benchmark

Compare different backends (Candle, ONNX Runtime, Python) by running multiple server instances:

**Step 1: Start Multiple Backend Servers**

You need separate TEI server instances with different backends on different ports (each binary must be built with the corresponding feature):

```bash
# Quick start script (starts all three on 8080/8081/8082; edit ports/binaries as needed)
cd /path/to/text-embeddings-inference/examples
bash start_all_backends.sh

# Terminal 1: ONNX Runtime backend (CPU-optimized; built with -F ort)
text-embeddings-router --model-id BAAI/bge-small-en-v1.5 --port 8080

# Terminal 2: Candle backend (GPU-optimized; built with -F candle-cuda or candle-cuda-turing)
text-embeddings-router --model-id BAAI/bge-small-en-v1.5 --port 8081

# Terminal 3: Python backend (PyTorch; built with -F python, python-text-embeddings-server installed)
text-embeddings-router --model-id BAAI/bge-small-en-v1.5 --port 8082 --dtype float32
```

**Step 2: Run Backend Benchmark**

```bash
cd /path/to/text-embeddings-inference/examples
python3 benchmark_backends.py \
    --ort-url http://127.0.0.1:8080 \
    --candle-url http://127.0.0.1:8081 \
    --python-url http://127.0.0.1:8082 \
    --iterations 50 \
    --concurrency 8
```

**What the Benchmark Tests:**
- Sequential latency (single-request performance)
- Concurrent throughput (batching performance)
- Component breakdown (tokenization, queue, inference times)
- Relative performance comparison

**Example Output:**
```
📊 Concurrent Throughput
════════════════════════════════════════════════════════════════════════════════════════════════════════
Backend         Latency (ms)       Throughput (req/s)   Inference (ms)     Success Rate
────────────────────────────────────────────────────────────────────────────────────────────────────────
Candle          35.01              207.52               11.22              100.0%
ONNX Runtime    37.89              195.46               14.60              100.0%

📈 Relative Performance (vs fastest backend):
  Candle          Throughput: 100.0% | Latency: 100.0%
  ONNX Runtime    Throughput:  94.2% | Latency:  92.4%
```

**Note:** You can benchmark any combination of backends. For example, to compare only Candle and ONNX Runtime:

```bash
python3 benchmark_backends.py \
    --candle-url http://127.0.0.1:8081 \
    --ort-url http://127.0.0.1:8080 \
    --iterations 30
```

---

## Troubleshooting

### Docker Issues

**Problem:** `Cannot connect to Docker daemon`
- **Solution:** Ensure Docker is running: `sudo systemctl start docker` (Linux) or start Docker Desktop (macOS/Windows)

**Problem:** `NVIDIA Container Toolkit not found`
- **Solution:** Install the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

**Problem:** `Permission denied` when accessing Docker socket
- **Solution:** Add your user to the docker group: `sudo usermod -aG docker $USER` (then log out and back in)

### Build Issues

**Problem:** `openssl-sys` build fails
- **Solution:** Install OpenSSL development packages:
  - Ubuntu/Debian: `sudo apt-get install libssl-dev pkg-config`
  - macOS: `brew install openssl pkg-config`
  - Fedora: `sudo dnf install openssl-devel pkg-config`

**Problem:** `pkg-config` not found
- **Solution:** Install pkg-config:
  - Ubuntu/Debian: `sudo apt-get install pkg-config`
  - macOS: `brew install pkg-config`

**Problem:** Build takes too long
- **Solution:** This is normal! First-time builds compile many dependencies. Subsequent builds are faster due to caching.

### Runtime Issues

**Problem:** Server fails to start with "model not found"
- **Solution:**
  - Check internet connection (model downloads from HuggingFace)
  - Verify model ID is correct: `https://huggingface.co/MODEL_ID`
  - For private models, set `HF_TOKEN` environment variable

**Problem:** Port 8080 already in use
- **Solution:** Use a different port: `--port 8081`

**Problem:** Out of memory errors
- **Solution:**
  - Use a smaller model (e.g., `bge-small-en-v1.5` instead of `bge-large-en-v1.5`)
  - Reduce `--max-concurrent-requests`
  - Ensure sufficient RAM/VRAM

### Performance Issues

**Problem:** Slow inference on CPU
- **Solution:**
  - Use GPU if available
  - Try ONNX backend (`-F ort`) for better CPU performance
  - Increase `--tokenization-workers` for parallel tokenization

**Problem:** Low throughput
- **Solution:**
  - Increase concurrency (send multiple requests in parallel)
  - Adjust `--max-batch-tokens` and `--max-batch-requests`
  - Use GPU acceleration

---

## Next Steps

- Read the [walkthrough guide](walkthrough.md) for architecture details
- Explore [supported models](https://huggingface.co/models?library=sentence-transformers)
- Check the [API documentation](https://huggingface.github.io/text-embeddings-inference)
- Review [CLI arguments](https://huggingface.github.io/text-embeddings-inference/cli_arguments) for advanced configuration

---

## Quick Reference

### Common Commands

```bash
# Docker (CPU)
docker run -p 8080:80 ghcr.io/huggingface/text-embeddings-inference:1.8 \
  --model-id BAAI/bge-small-en-v1.5

# Docker (GPU)
docker run --gpus all -p 8080:80 ghcr.io/huggingface/text-embeddings-inference:1.8 \
  --model-id BAAI/bge-small-en-v1.5

# Local (CPU)
text-embeddings-router --model-id BAAI/bge-small-en-v1.5 --port 8080

# Local (GPU)
text-embeddings-router --model-id BAAI/bge-small-en-v1.5 --port 8080

# Test endpoint
curl http://127.0.0.1:8080/embed \
  -X POST \
  -H "Content-Type: application/json" \
  -d '{"inputs": "Hello, world!"}'
```

### Popular Models

- **Small & Fast:** `BAAI/bge-small-en-v1.5` (33M parameters)
- **Balanced:** `BAAI/bge-base-en-v1.5` (110M parameters)
- **High Quality:** `BAAI/bge-large-en-v1.5` (335M parameters)
- **Multilingual:** `intfloat/multilingual-e5-large-instruct` (560M parameters)
- **Latest:** `Qwen/Qwen3-Embedding-0.6B` (600M parameters)

---

**Need Help?** Check the [official documentation](https://huggingface.github.io/text-embeddings-inference) or open an issue on [GitHub](https://github.com/huggingface/text-embeddings-inference).

