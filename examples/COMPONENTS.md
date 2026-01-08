# Text Embeddings Inference - Main Components

This document outlines the main components and architecture of Text Embeddings Inference (TEI).

## 📋 Table of Contents

- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Main Components](#main-components)
  - [1. Router](#1-router)
  - [2. Core](#2-core)
  - [3. Backends](#3-backends)
- [Data Flow](#data-flow)
- [Component Interactions](#component-interactions)
- [Key Features](#key-features)

---

## Overview

Text Embeddings Inference (TEI) is a high-performance inference server for text embeddings models. It consists of three main layers:

1. **Router Layer** - HTTP/gRPC API server that handles client requests
2. **Core Layer** - Tokenization, queuing, and batching logic
3. **Backend Layer** - Inference engines (Candle, ONNX Runtime, Python)

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Client Requests                        │
│                    (HTTP/gRPC)                               │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│                    ROUTER LAYER                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ HTTP Server  │  │ gRPC Server  │  │  Prometheus  │      │
│  │   (Axum)     │  │   (Tonic)    │  │   Metrics    │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│                      CORE LAYER                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │Tokenization │  │    Queue     │  │   Inference  │      │
│  │  Workers    │  │  (Batching)   │  │   Manager    │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│                    BACKEND LAYER                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │   Candle     │  │ ONNX Runtime │  │   Python     │      │
│  │  (Rust/GPU)  │  │   (CPU/GPU)  │  │  (PyTorch)   │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────────────────────────────────────────┘
```

---

## Main Components

### 1. Router

**Location:** `router/`

The Router layer provides the API interface for clients to interact with TEI.

#### Components

- **HTTP Server** (`router/src/http/server.rs`)
  - Built with [Axum](https://github.com/tokio-rs/axum)
  - RESTful API endpoints:
    - `POST /embed` - Generate embeddings
    - `POST /embed_all` - Generate embeddings with all pooling methods
    - `POST /predict` - Sequence classification
    - `POST /rerank` - Reranking
    - `GET /health` - Health check
    - `GET /info` - Model information
    - `GET /metrics` - Prometheus metrics
  - OpenAPI/Swagger documentation
  - CORS support
  - Request validation and error handling

- **gRPC Server** (`router/src/grpc/server.rs`)
  - Built with [Tonic](https://github.com/hyperium/tonic)
  - Protocol buffer definitions in `proto/tei.proto`
  - Streaming support for batch processing
  - Reflection support for service discovery

- **Prometheus Metrics** (`router/src/prometheus.rs`)
  - Request counts and latencies
  - Queue size and batch statistics
  - Tokenization, queue, and inference timings
  - Error counters by type

- **Logging** (`router/src/logging.rs`)
  - Structured logging with tracing
  - OpenTelemetry integration for distributed tracing
  - JSON log format support

#### Key Responsibilities

- Accept HTTP/gRPC requests from clients
- Validate request payloads
- Route requests to the Core layer
- Format and return responses
- Export metrics and traces

---

### 2. Core

**Location:** `core/`

The Core layer contains the business logic for tokenization, queuing, and batching.

#### Components

- **Tokenization** (`core/src/tokenization.rs`)
  - Converts text input to token IDs
  - Uses [tokenizers](https://github.com/huggingface/tokenizers) library
  - Supports multiple tokenization workers (parallel processing)
  - Handles truncation and padding
  - Very fast: < 1ms typically

- **Queue** (`core/src/queue.rs`)
  - Dynamic batching system
  - Background thread manages queue state
  - Forms batches based on:
    - `max_batch_tokens` - Maximum total tokens per batch
    - `max_batch_requests` - Maximum requests per batch
  - Handles both padded and non-padded models
  - Tracks queue time for each request

- **Inference Manager** (`core/src/infer.rs`)
  - Coordinates tokenization, queuing, and backend inference
  - Manages concurrent request limits (semaphore)
  - Spawns background tasks:
    - `batching_task` - Forms batches from queue
    - `backend_task` - Sends batches to backend and handles responses
  - Tracks timing metrics (tokenization, queue, inference)

#### Key Responsibilities

- Tokenize incoming text requests
- Queue requests for batching
- Form optimal batches
- Coordinate with backend for inference
- Return results to clients

#### Queue Batching Logic

The queue uses a smart batching algorithm:

```rust
// Pseudo-code for batch formation
while queue.has_entries() {
    entry = queue.pop_front()

    // Check if adding this entry would exceed token limit
    if total_tokens + entry.tokens > max_batch_tokens {
        queue.push_front(entry)  // Put it back
        break  // Batch is full
    }

    // Check if we've reached max requests
    if batch_size >= max_batch_requests {
        break
    }

    // Add to batch
    batch.add(entry)
}
```

---

### 3. Backends

**Location:** `backends/`

The Backend layer provides the actual inference engines. TEI supports multiple backends, each optimized for different use cases.

#### 3.1 Candle Backend

**Location:** `backends/candle/`

- **Language:** Rust
- **Best For:** GPU inference, production deployments
- **Features:**
  - CUDA support for NVIDIA GPUs
  - Metal support for Apple Silicon
  - Flash Attention support
  - Optimized with cuBLASLt
  - Safetensors weight loading
- **Performance:** Highest throughput, lowest latency (with GPU)
- **Models Supported:** BERT, RoBERTa, GTE, Qwen, Mistral, Nomic, Jina, ModernBERT, Gemma3, and more

#### 3.2 ONNX Runtime Backend

**Location:** `backends/ort/`

- **Language:** Rust (wraps ONNX Runtime C++ library)
- **Best For:** CPU inference, x86 servers
- **Features:**
  - Intel MKL-DNN optimizations
  - CPU-optimized inference
  - ONNX model format
- **Performance:** Best CPU performance, 2-3x faster than Candle on CPU
- **Models Supported:** Models converted to ONNX format

#### 3.3 Python Backend

**Location:** `backends/python/`

- **Language:** Python (PyTorch)
- **Best For:** Flash Attention, HPU (Habana Gaudi), custom models
- **Architecture:**
  - Separate gRPC server process
  - Communicates with Rust router via Unix Domain Socket
  - Full PyTorch ecosystem support
- **Features:**
  - Flash Attention v1/v2
  - HPU (Habana Gaudi) support
  - Intel Extension for PyTorch (IPEX)
  - Maximum flexibility for custom models
- **Performance:** Good but slower than Rust backends due to Python overhead
- **Models Supported:** Any PyTorch model

#### Backend Selection

Backends are compile-time features. You build TEI with specific backend support:

```bash
# Candle backend
cargo build --release -F candle-cuda

# ONNX Runtime backend
cargo build --release -F ort

# Python backend
cargo build --release -F python

# Multiple backends
cargo build --release -F candle-cuda,ort,python
```

---

## Data Flow

### Request Flow

```
1. Client Request
   ↓
2. Router (HTTP/gRPC Server)
   - Validates request
   - Extracts text input
   ↓
3. Tokenization
   - Converts text → token IDs
   - Handles truncation/padding
   ↓
4. Queue
   - Adds request to queue
   - Tracks queue_time start
   ↓
5. Batching Task
   - Forms batch from queued requests
   - Respects max_batch_tokens and max_batch_requests
   ↓
6. Backend Task
   - Sends batch to backend (Candle/ORT/Python)
   - Backend runs inference
   ↓
7. Results
   - Backend returns embeddings
   - Results distributed to individual requests
   ↓
8. Response
   - Router formats response
   - Returns to client
```

### Timing Breakdown

Each request tracks three key timings:

1. **Tokenization Time** - Time to convert text to tokens (< 1ms)
2. **Queue Time** - Time waiting in queue for batching (varies, 0-50ms+)
3. **Inference Time** - Time for model inference (2-33ms+ depending on model)

Total latency = Tokenization + Queue + Inference + Network overhead

---

## Component Interactions

### Request Lifecycle

```rust
// 1. Router receives request
let request = http_request.body();

// 2. Tokenization
let encoding = infer.tokenize(inputs).await?;

// 3. Create queue entry
let entry = Entry {
    encoding,
    metadata: Metadata {
        response_tx,           // Channel to send result back
        tokenization: duration,
        queue_time: Instant::now(),
        prompt_tokens,
        pooling,
    },
};

// 4. Add to queue
queue.append(entry);
notify_batching_task.notify_one();

// 5. Batching task forms batch
let batch = queue.next_batch().await;

// 6. Backend processes batch
let results = backend.embed(batch).await?;

// 7. Results sent back via response_tx
response_tx.send(Ok(result));
```

### Background Tasks

TEI uses several background tasks for async processing:

1. **Queue Task** (`queue_blocking_task`)
   - Runs in a blocking thread
   - Manages queue state
   - Forms batches on demand

2. **Batching Task** (`batching_task`)
   - Async task
   - Waits for queue notifications
   - Requests batches from queue
   - Sends batches to backend

3. **Backend Task** (`backend_task`)
   - Async task
   - Receives batches
   - Calls backend inference
   - Distributes results to requests

---

## Key Features

### 1. Dynamic Batching

- Automatically groups requests into batches
- Maximizes GPU/CPU utilization
- Reduces per-request latency in high-concurrency scenarios
- Configurable batch size limits

### 2. Multiple Backend Support

- Choose the best backend for your use case:
  - **GPU available?** → Use Candle
  - **CPU only?** → Use ONNX Runtime
  - **Need Flash Attention?** → Use Python backend

### 3. Production Ready

- **Metrics:** Prometheus-compatible metrics
- **Tracing:** OpenTelemetry distributed tracing
- **Health Checks:** `/health` endpoint
- **Error Handling:** Comprehensive error types and responses
- **Rate Limiting:** Configurable concurrent request limits

### 4. High Performance

- Optimized inference engines
- Parallel tokenization
- Efficient memory management
- Fast startup times

### 5. Model Support

- Supports 50+ popular embedding models
- Automatic model detection
- Handles different model architectures (BERT, RoBERTa, GTE, Qwen, etc.)
- Supports various pooling methods (CLS, mean, max, etc.)

---

## Configuration

### Key Parameters

- `--model-id` - Model to load (HuggingFace ID or local path)
- `--max-batch-tokens` - Maximum tokens per batch (default: 2048)
- `--max-batch-requests` - Maximum requests per batch (optional)
- `--max-concurrent-requests` - Maximum concurrent requests (default: 512)
- `--tokenization-workers` - Number of tokenization workers (default: 8)
- `--port` - Server port (default: 80)

### Environment Variables

- `MODEL_ID` - Model identifier
- `REVISION` - Model revision/commit
- `DTYPE` - Data type (float16, float32, bfloat16)
- `MAX_BATCH_TOKENS` - Batch size limit
- `MAX_CONCURRENT_REQUESTS` - Concurrency limit

---

## Summary

TEI is architected as a three-layer system:

1. **Router** - API layer (HTTP/gRPC)
2. **Core** - Business logic (tokenization, queuing, batching)
3. **Backends** - Inference engines (Candle, ONNX Runtime, Python)

This separation of concerns allows for:
- Easy API changes without touching inference logic
- Multiple backend support with a unified interface
- Optimized performance at each layer
- Production-ready features (metrics, tracing, health checks)

The dynamic batching system is the key innovation that enables high throughput by efficiently grouping requests and maximizing hardware utilization.


