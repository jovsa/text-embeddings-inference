# Python Backend (PyTorch) Guide

The Python backend uses PyTorch to run inference, providing maximum flexibility and compatibility with the full PyTorch ecosystem.

## Overview

The Python backend is a **separate gRPC server** written in Python that runs PyTorch models. The main TEI router (Rust) communicates with it via gRPC over a Unix Domain Socket (UDS).

```
┌─────────────────┐         gRPC          ┌──────────────────┐
│  TEI Router     │ ◄──────────────────► │  Python Server   │
│  (Rust)         │    (Unix Socket)      │  (PyTorch)       │
└─────────────────┘                       └──────────────────┘
```

## Architecture

### Components

1. **Rust Wrapper** (`backends/python/src/lib.rs`)
   - Manages the Python process lifecycle
   - Communicates via gRPC client
   - Handles batch conversion

2. **Python gRPC Server** (`backends/python/server/`)
   - Runs PyTorch models
   - Handles inference requests
   - Supports Flash Attention, HPU, etc.

### Communication Flow

```
Client Request → TEI Router → Tokenization → Queue → Python Backend
                                                         ↓
                                              Python gRPC Server
                                                         ↓
                                              PyTorch Model Inference
                                                         ↓
                                              Results → TEI Router → Client
```

## Features

### ✅ Advantages

- **Full PyTorch Ecosystem**: Use any PyTorch model
- **Flash Attention**: Supports Flash Attention v1 and v2
- **HPU Support**: Habana Gaudi accelerator support
- **Easy Debugging**: Python stack traces
- **Flexible**: Easy to modify and extend
- **Model Compatibility**: Works with models not supported by Candle/ORT

### ⚠️ Limitations

- **Higher Memory Usage**: Python overhead
- **Slower Startup**: Python initialization time
- **No Raw Embeddings**: Only pooled embeddings supported
- **Additional Dependencies**: Requires Python and PyTorch

## Installation

### Prerequisites

- Python 3.9 - 3.12
- PyTorch (CPU or CUDA)
- Poetry (for dependency management)

### Step 1: Install Python Dependencies

```bash
cd backends/python/server
poetry install
# Or with pip:
pip install -r requirements.txt
```

### Step 2: Build TEI with Python Backend

```bash
# From project root
cargo build --release --bin text-embeddings-router --features python
```

Or install:

```bash
cargo install --path router -F python
```

### Step 3: Verify Installation

```bash
# Check that python-text-embeddings-server is available
which python-text-embeddings-server
```

## Usage

### Basic Usage

```bash
# Start TEI with Python backend
text-embeddings-router \
    --model-id BAAI/bge-small-en-v1.5 \
    --port 8080
```

The Python backend will automatically:
1. Start the Python gRPC server as a subprocess
2. Load the model using PyTorch
3. Handle inference requests

### With Flash Attention

```bash
# Install Flash Attention first
pip install flash-attn --no-build-isolation

# Start TEI (will auto-detect Flash Attention)
text-embeddings-router \
    --model-id BAAI/bge-small-en-v1.5 \
    --port 8080
```

### With HPU (Habana Gaudi)

```bash
# Set HPU device
export PYTORCH_HPU_DEVICE=0

# Start TEI
text-embeddings-router \
    --model-id BAAI/bge-small-en-v1.5 \
    --port 8080
```

## Supported Models

The Python backend supports a wide range of models:

- **BERT-based**: BERT, RoBERTa, DistilBERT, CamemBERT, XLM-RoBERTa
- **Flash Models**: FlashBERT, FlashMistral, FlashQwen3
- **Specialized**: JinaBERT, ModernBERT, MPNet
- **Custom Models**: Any PyTorch model via transformers

### Model Detection

The Python backend automatically detects model type from the config:

```python
# From backends/python/server/text_embeddings_server/models/__init__.py
- FlashBERT (if Flash Attention available)
- FlashMistral
- FlashQwen3
- JinaBERT
- DefaultModel (fallback for most models)
```

## Configuration

### Environment Variables

```bash
# Data type (float32, float16, bfloat16)
export DTYPE=float32

# Trust remote code (for custom models)
export TRUST_REMOTE_CODE=true

# Disable tensor cache (for HPU)
export DISABLE_TENSOR_CACHE=false

# Intel optimizations
export USE_IPEX=true  # Requires Intel Extension for PyTorch
```

### Data Types

Supported data types:
- **float32**: Default, best compatibility
- **float16**: Faster, lower memory (requires CUDA)
- **bfloat16**: Better numerical stability (requires CUDA)

## Performance Considerations

### CPU Performance

- **Baseline**: Standard PyTorch CPU inference
- **Intel Optimizations**: Use `USE_IPEX=true` for Intel CPUs
- **Expected**: Similar to standard PyTorch performance

### GPU Performance

- **CUDA**: Full PyTorch CUDA support
- **Flash Attention**: Significant speedup for long sequences
- **Expected**: Good performance, may be slower than Candle due to Python overhead

### Memory Usage

- **Higher than Candle/ORT**: Python interpreter overhead
- **Model Loading**: PyTorch model loading uses more memory
- **Batch Processing**: Efficient batching via gRPC

## Comparison with Other Backends

| Feature | Python Backend | Candle Backend | ONNX Runtime |
|---------|---------------|----------------|--------------|
| **Language** | Python | Rust | Rust (wrapper) |
| **GPU Support** | ✅ CUDA | ✅ CUDA, Metal | ❌ |
| **Flash Attention** | ✅ v1, v2 | ✅ | ❌ |
| **HPU Support** | ✅ | ❌ | ❌ |
| **Model Support** | Extensive | Limited | Limited |
| **Startup Time** | Slower | Fast | Fast |
| **Memory Usage** | Higher | Lower | Lower |
| **Performance** | Good | Best | Best (CPU) |
| **Flexibility** | High | Medium | Low |

## Troubleshooting

### Python Server Not Starting

```bash
# Check Python installation
python --version  # Should be 3.9-3.12

# Check PyTorch installation
python -c "import torch; print(torch.__version__)"

# Check if python-text-embeddings-server is installed
which python-text-embeddings-server
```

### Flash Attention Not Working

```bash
# Check if Flash Attention is installed
python -c "import flash_attn; print('OK')"

# Install if missing
pip install flash-attn --no-build-isolation
```

### Out of Memory

```bash
# Reduce batch size
export MAX_BATCH_SIZE=8

# Use float16 instead of float32
export DTYPE=float16
```

### gRPC Connection Errors

```bash
# Check Unix socket path
ls -la /tmp/text-embeddings-*.sock

# Check Python server logs
# Look for errors in TEI server output
```

## Advanced Usage

### Custom Model Loading

The Python backend uses `sentence-transformers` and `transformers` libraries:

```python
# Models are loaded via:
from sentence_transformers import SentenceTransformer
model = SentenceTransformer(model_path)
```

### Debugging

Enable verbose logging:

```bash
export LOGGER_LEVEL=DEBUG
text-embeddings-router --model-id BAAI/bge-small-en-v1.5 --port 8080
```

### Direct Python Server Usage

You can run the Python server directly (for testing):

```bash
cd backends/python/server
python-text-embeddings-server \
    /path/to/model \
    --dtype float32 \
    --uds-path /tmp/tei-python.sock \
    --pool mean
```

## Code Structure

```
backends/python/
├── src/
│   ├── lib.rs              # Rust wrapper, gRPC client
│   ├── management.rs        # Python process management
│   └── logging.rs          # Log handling
└── server/
    ├── text_embeddings_server/
    │   ├── cli.py           # CLI entry point
    │   ├── server.py        # gRPC server
    │   ├── models/          # Model implementations
    │   │   ├── default_model.py
    │   │   ├── flash_bert.py
    │   │   ├── flash_mistral.py
    │   │   └── ...
    │   └── utils/           # Utilities
    │       ├── device.py    # Device detection
    │       └── flash_attn.py
    └── requirements.txt     # Python dependencies
```

## When to Use Python Backend

### ✅ Use Python Backend When:

- You need **Flash Attention** support
- You're using **Habana Gaudi (HPU)** hardware
- Your model isn't supported by Candle/ORT
- You need to **customize model behavior**
- You're doing **research/experimentation**
- You need **easy debugging** (Python stack traces)

### ❌ Don't Use Python Backend When:

- You need **maximum performance** (use Candle)
- You're on **CPU-only** (use ONNX Runtime)
- You need **low memory usage**
- You need **fast startup time**
- You're in **production** without specific Python requirements

## Example: Benchmarking Python Backend

```bash
# Start Python backend server
text-embeddings-router --model-id BAAI/bge-small-en-v1.5 --port 8080

# Run benchmark
python examples/benchmark_backends.py \
    --python-url http://127.0.0.1:8080 \
    --iterations 50
```

## Summary

The Python backend provides:
- **Maximum flexibility** with full PyTorch support
- **Flash Attention** for efficient attention computation
- **HPU support** for Habana Gaudi accelerators
- **Easy extensibility** for custom models

Trade-offs:
- **Higher memory usage** than Rust backends
- **Slower startup** due to Python initialization
- **Good but not best** performance compared to optimized Rust backends

Choose Python backend when you need its specific features (Flash Attention, HPU, custom models) rather than maximum raw performance.

