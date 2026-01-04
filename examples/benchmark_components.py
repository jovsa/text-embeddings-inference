#!/usr/bin/env python3
"""
TEI Component Benchmarking Script
==================================

This script demonstrates the full request flow through Text Embeddings Inference
and benchmarks each component. The request flows through:

    ┌──────────────────────────────────────────────────────────────────────┐
    │                     Full Request Flow                                 │
    │                                                                       │
    │   Client Request                                                      │
    │        │                                                              │
    │        ▼                                                              │
    │   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐              │
    │   │ Tokenization│───▶│   Queue     │───▶│  Backend    │              │
    │   │  (HF Tok.)  │    │ (Batching)  │    │ (Inference) │              │
    │   └─────────────┘    └─────────────┘    └─────────────┘              │
    │        │                   │                  │                      │
    │        │                   │                  │                      │
    │   X-Tokenization-Time  X-Queue-Time    X-Inference-Time              │
    │                                                                       │
    │        ◀────────────────────────────────────────┘                    │
    │                                                                       │
    │   X-Total-Time (end-to-end latency)                                  │
    └──────────────────────────────────────────────────────────────────────┘

Each component's timing is returned in HTTP headers:
- X-Tokenization-Time: Time to convert text → token IDs
- X-Queue-Time: Time waiting in the batching queue  
- X-Inference-Time: Time running the neural network
- X-Total-Time: End-to-end request latency

Usage:
    python benchmark_components.py [--url http://127.0.0.1:8080]
"""

import requests
import time
import statistics
import argparse
from typing import NamedTuple


class ComponentTimes(NamedTuple):
    """Parsed timing information from response headers."""
    total_ms: float
    tokenization_ms: float
    queue_ms: float
    inference_ms: float


def parse_timing_headers(headers: dict) -> ComponentTimes:
    """Extract timing information from TEI response headers."""
    return ComponentTimes(
        total_ms=float(headers.get("x-total-time", 0)),
        tokenization_ms=float(headers.get("x-tokenization-time", 0)),
        queue_ms=float(headers.get("x-queue-time", 0)),
        inference_ms=float(headers.get("x-inference-time", 0)),
    )


def benchmark_concurrent(url: str, text: str, concurrency: int = 1, iterations: int = 20) -> tuple[float, list[ComponentTimes]]:
    """Run concurrent embedding requests and measure throughput."""
    import concurrent.futures
    endpoint = f"{url}/embed"
    payload = {"inputs": text}
    headers = {"Content-Type": "application/json"}
    
    times_list = []
    
    print(f"\n🚀 Running {iterations} iterations with concurrency={concurrency}...")
    
    start_time = time.time()
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        # Submit all tasks
        futures = [
            executor.submit(requests.post, endpoint, json=payload, headers=headers)
            for _ in range(iterations)
        ]
        
        # Collect results
        for future in concurrent.futures.as_completed(futures):
            try:
                response = future.result()
                if response.status_code == 200:
                    times_list.append(parse_timing_headers(response.headers))
                else:
                    print(f"  ❌ Request failed: {response.status_code}")
            except Exception as e:
                print(f"  ❌ Request exception: {e}")

    total_duration = time.time() - start_time
    throughput = len(times_list) / total_duration if total_duration > 0 else 0
    
    return throughput, times_list

def benchmark_embed(url: str, text: str, iterations: int = 10) -> list[ComponentTimes]:
    """Run multiple embedding requests and collect timing data."""
    results = []
    endpoint = f"{url}/embed"
    
    print(f"\n📊 Running {iterations} iterations...")
    
    for i in range(iterations):
        response = requests.post(
            endpoint,
            json={"inputs": text},
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code != 200:
            print(f"  ❌ Request {i+1} failed: {response.status_code}")
            continue
            
        times = parse_timing_headers(response.headers)
        results.append(times)
        

        if (i + 1) % 5 == 0 or i == 0:
            print(f"  ✓ Iteration {i+1}/{iterations} - Total: {times.total_ms:.2f}ms")
    
    return results


def print_statistics(results: list[ComponentTimes], component: str, values: list[float]):
    """Print statistics for a component."""
    if not values:
        return
    
    mean = statistics.mean(values)
    median = statistics.median(values)
    stdev = statistics.stdev(values) if len(values) > 1 else 0
    min_val = min(values)
    max_val = max(values)
    
    print(f"  {component:20s}: mean={mean:8.2f}ms, median={median:8.2f}ms, "
          f"std={stdev:7.2f}ms, min={min_val:8.2f}ms, max={max_val:8.2f}ms")


def print_component_explanation():
    """Print explanation of each component."""
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                        TEI COMPONENT BREAKDOWN                               ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  TOKENIZATION (X-Tokenization-Time)                                          ║
║  ─────────────────────────────────                                           ║
║  • Converts text string → token IDs using HuggingFace tokenizers             ║
║  • Code: core/src/tokenization.rs                                            ║
║  • Workers run in parallel (default: num_cpus)                               ║
║  • Includes validation and optional truncation                               ║
║                                                                              ║
║  QUEUE (X-Queue-Time)                                                        ║
║  ─────────────────────                                                       ║
║  • Dynamic batching for optimal GPU utilization                              ║
║  • Code: core/src/queue.rs                                                   ║
║  • Groups requests to maximize throughput while minimizing latency           ║
║  • Respects max_batch_tokens and max_batch_requests limits                   ║
║                                                                              ║
║  INFERENCE (X-Inference-Time)                                                ║
║  ───────────────────────────                                                 ║
║  • Actual neural network forward pass                                        ║
║  • Code: backends/src/lib.rs → candle/ort/python backends                    ║
║  • Includes: embedding computation, pooling, normalization                   ║
║  • GPU-accelerated when available (Flash Attention, cuBLAS)                  ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")


def main():
    parser = argparse.ArgumentParser(description="Benchmark TEI components")
    parser.add_argument("--url", default="http://127.0.0.1:8080", 
                        help="TEI server URL")
    parser.add_argument("--iterations", type=int, default=20,
                        help="Number of iterations per test")
    args = parser.parse_args()
    
    print("=" * 80)
    print("           TEXT EMBEDDINGS INFERENCE - COMPONENT BENCHMARK")
    print("=" * 80)
    
    # Check server health
    try:
        health = requests.get(f"{args.url}/health")
        if health.status_code != 200:
            print(f"❌ Server not healthy: {health.status_code}")
            return
    except requests.exceptions.ConnectionError:
        print(f"❌ Cannot connect to server at {args.url}")
        print("   Make sure TEI is running: ./target/release/text-embeddings-router --model-id BAAI/bge-small-en-v1.5 --port 8080")
        return
    
    # Get model info
    info = requests.get(f"{args.url}/info").json()
    print(f"\n📦 Model: {info['model_id']}")
    print(f"   Max input length: {info['max_input_length']} tokens")
    print(f"   Tokenization workers: {info['tokenization_workers']}")
    
    print_component_explanation()
    
    # Test cases with different text lengths
    test_cases = [
        ("Short text (5 words)", "Hello, how are you today?"),
        ("Medium text (50 words)", "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed. It focuses on developing algorithms that can access data and use it to learn for themselves. The process begins with observations or data, such as examples, direct experience, or instruction."),
        ("Long text (150 words)", "Text embeddings are dense vector representations of text that capture semantic meaning. They are fundamental to modern natural language processing applications including semantic search, recommendation systems, and retrieval-augmented generation (RAG). The quality of embeddings directly impacts downstream task performance. Modern embedding models like BGE, GTE, and E5 achieve state-of-the-art results by training on large-scale datasets with contrastive learning objectives. These models learn to place semantically similar texts close together in the embedding space while pushing dissimilar texts apart. The resulting vectors can be compared using cosine similarity or dot product to measure semantic relatedness. Text Embeddings Inference (TEI) provides a highly optimized server for running these models in production, with features like dynamic batching, Flash Attention, and multi-backend support."),
    ]
    
    all_results = {}
    
    for name, text in test_cases:
        print(f"\n{'─' * 80}")
        print(f"🧪 Test: {name}")
        print(f"   Input: {len(text)} chars, ~{len(text.split())} words")
        
        results = benchmark_embed(args.url, text, iterations=args.iterations)
        
        if not results:
            print("   ⚠️  No successful results")
            continue
        
        all_results[name] = results
        
        print(f"\n📈 Results ({len(results)} samples):")
        print_statistics(results, "Total", [r.total_ms for r in results])
        print_statistics(results, "Tokenization", [r.tokenization_ms for r in results])
        print_statistics(results, "Queue", [r.queue_ms for r in results])
        print_statistics(results, "Inference", [r.inference_ms for r in results])
        
        # Show percentage breakdown
        totals = [r.total_ms for r in results]
        tok = [r.tokenization_ms for r in results]
        queue = [r.queue_ms for r in results]
        inf = [r.inference_ms for r in results]
        
        if statistics.mean(totals) > 0:
            print(f"\n   📊 Breakdown (% of total time):")
            print(f"      Tokenization: {100 * statistics.mean(tok) / statistics.mean(totals):5.1f}%")
            print(f"      Queue:        {100 * statistics.mean(queue) / statistics.mean(totals):5.1f}%")
            print(f"      Inference:    {100 * statistics.mean(inf) / statistics.mean(totals):5.1f}%")
    
    # Summary comparison
    if len(all_results) > 1:
        print(f"\n{'═' * 80}")
        print("📋 SUMMARY COMPARISON")
        print("═" * 80)
        print(f"{'Test Case':<25} {'Total (ms)':<15} {'Tokenization':<15} {'Queue':<15} {'Inference':<15}")
        print("-" * 85)
        
        for name, results in all_results.items():
            total = statistics.mean([r.total_ms for r in results])
            tok = statistics.mean([r.tokenization_ms for r in results])
            queue = statistics.mean([r.queue_ms for r in results])
            inf = statistics.mean([r.inference_ms for r in results])
            print(f"{name:<25} {total:<15.2f} {tok:<15.2f} {queue:<15.2f} {inf:<15.2f}")
    
    # Concurrency Test (Demonstrating Batching)
    print(f"\n{'═' * 80}")
    print("🚀 CONCURRENCY & BATCHING TEST")
    print("═" * 80)
    print("Demonstrating how batching improves throughput even if single-request latency is high.")
    
    short_text = test_cases[0][1] # Use short text
    
    # 1. Sequential (Concurrency = 1)
    tps_1, results_1 = benchmark_concurrent(args.url, short_text, concurrency=1, iterations=20)
    avg_lat_1 = statistics.mean([r.total_ms for r in results_1]) if results_1 else 0
    print(f"  Concurrency 1:  TPS = {tps_1:6.2f} req/s | Avg Latency = {avg_lat_1:6.2f} ms")

    # 2. Parallel (Concurrency = 8)
    tps_8, results_8 = benchmark_concurrent(args.url, short_text, concurrency=8, iterations=40)
    avg_lat_8 = statistics.mean([r.total_ms for r in results_8]) if results_8 else 0
    print(f"  Concurrency 8:  TPS = {tps_8:6.2f} req/s | Avg Latency = {avg_lat_8:6.2f} ms")

    # 3. Parallel (Concurrency = 32)
    tps_32, results_32 = benchmark_concurrent(args.url, short_text, concurrency=32, iterations=100)
    avg_lat_32 = statistics.mean([r.total_ms for r in results_32]) if results_32 else 0
    print(f"  Concurrency 32: TPS = {tps_32:6.2f} req/s | Avg Latency = {avg_lat_32:6.2f} ms")

    print("\n💡 OBSERVATION:")
    print("Notice how Throughput (TPS) increases with internal batching")
    print("even if individual latency stays similar or increases slightly.")
    
    # Concurrency Test (Demonstrating Batching)
    print(f"\n{'═' * 80}")
    print("🚀 CONCURRENCY & BATCHING TEST")
    print("═" * 80)
    print("Demonstrating how batching improves throughput even if single-request latency is high.")
    
    short_text = test_cases[0][1] # Use short text
    
    # 1. Sequential (Concurrency = 1)
    tps_1, results_1 = benchmark_concurrent(args.url, short_text, concurrency=1, iterations=20)
    avg_lat_1 = statistics.mean([r.total_ms for r in results_1]) if results_1 else 0
    print(f"  Concurrency 1:  TPS = {tps_1:6.2f} req/s | Avg Latency = {avg_lat_1:6.2f} ms")

    # 2. Parallel (Concurrency = 8)
    tps_8, results_8 = benchmark_concurrent(args.url, short_text, concurrency=8, iterations=40)
    avg_lat_8 = statistics.mean([r.total_ms for r in results_8]) if results_8 else 0
    print(f"  Concurrency 8:  TPS = {tps_8:6.2f} req/s | Avg Latency = {avg_lat_8:6.2f} ms")

    # 3. Parallel (Concurrency = 32)
    tps_32, results_32 = benchmark_concurrent(args.url, short_text, concurrency=32, iterations=100)
    avg_lat_32 = statistics.mean([r.total_ms for r in results_32]) if results_32 else 0
    print(f"  Concurrency 32: TPS = {tps_32:6.2f} req/s | Avg Latency = {avg_lat_32:6.2f} ms")

    print("\n💡 OBSERVATION:")
    print("Notice how Throughput (TPS) increases with internal batching")
    print("even if individual latency stays similar or increases slightly.")
    
    print(f"\n{'═' * 80}")
    print("✅ Benchmark complete!")
    print("═" * 80)


if __name__ == "__main__":
    main()
