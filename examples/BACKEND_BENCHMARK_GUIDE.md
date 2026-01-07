# Backend Benchmarking Guide

This guide explains how to benchmark and compare the three TEI backends: **Candle**, **ONNX Runtime (ORT)**, and **Python**.

## Overview

The `benchmark_backends.py` script compares the performance of different TEI backends by:
- Measuring **latency** (sequential requests)
- Measuring **throughput** (concurrent requests)
- Comparing **inference times** across backends
- Generating detailed statistics and comparison tables

## Prerequisites

1. **Multiple TEI server instances** - Since backends are compile-time features, you need separate server instances
2. **Same model** - All backends should use the same model for fair comparison
3. **Python 3.7+** with `requests` library

## Setup Instructions

### Option 1: Build Separate Binaries

Build TEI with different backend features:

```bash
# Build Candle backend (GPU)
cargo build --release --bin text-embeddings-router --features candle-cuda

# Build ONNX Runtime backend (CPU)
cargo build --release --bin text-embeddings-router --features ort

# Build Python backend
cargo build --release --bin text-embeddings-router --features python
```

Then run them on different ports:

```bash
# Terminal 1: Candle backend
./target/release/text-embeddings-router --model-id BAAI/bge-small-en-v1.5 --port 8080

# Terminal 2: ONNX Runtime backend
./target/release/text-embeddings-router --model-id BAAI/bge-small-en-v1.5 --port 8081

# Terminal 3: Python backend
./target/release/text-embeddings-router --model-id BAAI/bge-small-en-v1.5 --port 8082
```

### Option 2: Use Docker Images

Use different Docker images that target different backends:

```bash
# Terminal 1: Candle backend (GPU)
docker run --gpus all -p 8080:80 \
  ghcr.io/huggingface/text-embeddings-inference:1.8-gpu \
  --model-id BAAI/bge-small-en-v1.5

# Terminal 2: ONNX Runtime backend (CPU)
docker run -p 8081:80 \
  ghcr.io/huggingface/text-embeddings-inference:1.8 \
  --model-id BAAI/bge-small-en-v1.5

# Terminal 3: Python backend (if available)
docker run --gpus all -p 8082:80 \
  ghcr.io/huggingface/text-embeddings-inference:1.8-python \
  --model-id BAAI/bge-small-en-v1.5
```

### Option 3: Install with Different Features

Install TEI multiple times with different features:

```bash
# Install Candle backend
cargo install --path router -F candle-cuda --force

# Rename binary
mv ~/.cargo/bin/text-embeddings-router ~/.cargo/bin/text-embeddings-router-candle

# Install ONNX Runtime backend
cargo install --path router -F ort --force

# Rename binary
mv ~/.cargo/bin/text-embeddings-router ~/.cargo/bin/text-embeddings-router-ort

# Install Python backend
cargo install --path router -F python --force

# Rename binary
mv ~/.cargo/bin/text-embeddings-router ~/.cargo/bin/text-embeddings-router-python
```

Then run them:

```bash
# Terminal 1
text-embeddings-router-candle --model-id BAAI/bge-small-en-v1.5 --port 8080

# Terminal 2
text-embeddings-router-ort --model-id BAAI/bge-small-en-v1.5 --port 8081

# Terminal 3
text-embeddings-router-python --model-id BAAI/bge-small-en-v1.5 --port 8082
```

## Running the Benchmark

### Basic Usage

```bash
python benchmark_backends.py \
    --candle-url http://127.0.0.1:8080 \
    --ort-url http://127.0.0.1:8081 \
    --python-url http://127.0.0.1:8082 \
    --iterations 50
```

### Benchmark Only Specific Backends

```bash
# Compare only Candle and ONNX Runtime
python benchmark_backends.py \
    --candle-url http://127.0.0.1:8080 \
    --ort-url http://127.0.0.1:8081 \
    --iterations 30

# Compare only ONNX Runtime and Python
python benchmark_backends.py \
    --ort-url http://127.0.0.1:8081 \
    --python-url http://127.0.0.1:8082 \
    --iterations 30
```

### Advanced Options

```bash
python benchmark_backends.py \
    --candle-url http://127.0.0.1:8080 \
    --ort-url http://127.0.0.1:8081 \
    --python-url http://127.0.0.1:8082 \
    --iterations 100 \
    --concurrency 16 \
    --output results.json
```

**Options:**
- `--iterations`: Number of requests per test (default: 30)
- `--concurrency`: Number of concurrent requests for throughput test (default: 8)
- `--output`: Save results to JSON file

## Understanding the Results

### Sequential Latency Test

Measures **latency** (time per request) with sequential requests:
- Lower is better
- Shows single-request performance
- No batching effects

### Concurrent Throughput Test

Measures **throughput** (requests per second) with concurrent requests:
- Higher is better
- Shows batching performance
- More realistic for production workloads

### Metrics Explained

- **Latency (ms)**: Average time per request
- **Throughput (req/s)**: Requests processed per second
- **Inference (ms)**: Time spent in the neural network
- **Success Rate**: Percentage of successful requests

### Example Output

```
📊 Concurrent Throughput
════════════════════════════════════════════════════════════════════════════════════════════════════════
Backend         Latency (ms)      Throughput (req/s)   Inference (ms)     Success Rate
────────────────────────────────────────────────────────────────────────────────────────────────────────
Candle         15.23             523.45               14.50              100.0%
ONNX Runtime   18.67             428.12               17.20              100.0%
Python         22.34             357.89               20.10              100.0%

📈 Relative Performance (vs fastest backend):
  Candle         Throughput: 100.0% | Latency: 100.0%
  ONNX Runtime   Throughput:  81.8% | Latency:  81.6%
  Python         Throughput:  68.4% | Latency:  68.2%
```

## Interpreting Results

### For GPU Servers

**Expected Results:**
- **Candle**: Best performance (GPU-accelerated, Flash Attention)
- **ONNX Runtime**: Not applicable (CPU-only)
- **Python**: Good performance (PyTorch CUDA)

**Recommendation**: Use **Candle** backend for GPU deployments

### For CPU Servers (x86)

**Expected Results:**
- **Candle**: Good performance (MKL optimizations)
- **ONNX Runtime**: Best performance (highly optimized CPU inference)
- **Python**: Moderate performance (PyTorch CPU)

**Recommendation**: Use **ONNX Runtime** backend for CPU deployments

### For Apple Silicon

**Expected Results:**
- **Candle**: Best performance (Metal acceleration)
- **ONNX Runtime**: Good performance
- **Python**: Moderate performance

**Recommendation**: Use **Candle** backend with Metal support

## Troubleshooting

### Backend Not Responding

```bash
# Check if server is running
curl http://127.0.0.1:8080/health

# Check server logs for errors
# Look for backend-specific error messages
```

### Different Model Versions

**Problem**: Backends show different results because they're using different models.

**Solution**: Ensure all backends use the same model:
```bash
--model-id BAAI/bge-small-en-v1.5  # Same for all
```

### Port Conflicts

**Problem**: Port already in use.

**Solution**: Use different ports:
```bash
--port 8080  # Candle
--port 8081  # ONNX Runtime
--port 8082  # Python
```

### Python Backend Not Available

**Problem**: Python backend requires Python/PyTorch setup.

**Solution**:
- Install Python dependencies
- Or skip Python backend: `--candle-url ... --ort-url ...` (no `--python-url`)

## Best Practices

1. **Use the same model** for all backends
2. **Run multiple iterations** (50+) for reliable results
3. **Test on production-like hardware** (same CPU/GPU as production)
4. **Warm up backends** before benchmarking (first requests are slower)
5. **Monitor system resources** (CPU, GPU, memory) during tests
6. **Save results** with `--output` for later analysis

## Example Workflow

```bash
# 1. Start all three backends
# (In separate terminals)

# 2. Wait for models to load
sleep 30

# 3. Run benchmark
python benchmark_backends.py \
    --candle-url http://127.0.0.1:8080 \
    --ort-url http://127.0.0.1:8081 \
    --python-url http://127.0.0.1:8082 \
    --iterations 100 \
    --concurrency 16 \
    --output backend_comparison.json

# 4. Analyze results
cat backend_comparison.json | jq '.results.concurrent_throughput'
```

## Additional Notes

- **Backend selection is compile-time**: You can't switch backends at runtime
- **Hardware matters**: Results vary significantly based on CPU/GPU
- **Model matters**: Different models may favor different backends
- **Workload matters**: Sequential vs concurrent shows different patterns

For more information about backends, see the main README or backend documentation.

