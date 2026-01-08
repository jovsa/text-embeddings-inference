#!/usr/bin/env python3
"""
TEI Batching & Queue Benchmarking Script
=========================================

This script specifically benchmarks the queuing and batching components of TEI.
It measures how queue time, batch sizes, and throughput change with different
concurrency levels and request patterns.

Key Metrics:
- Queue Time: Time requests spend waiting in the batching queue
- Batch Size: Number of requests processed together
- Throughput: Requests per second
- Latency: End-to-end request time

Usage:
    # Basic usage
    python benchmark_batching.py --url http://127.0.0.1:8080

    # Test channel capacity impact (requires multiple servers with different capacities)
    # First start servers:
    #   Server 1: text-embeddings-router --model-id BAAI/bge-small-en-v1.5 --port 8080 --batch-channel-capacity 1
    #   Server 2: text-embeddings-router --model-id BAAI/bge-small-en-v1.5 --port 8081 --batch-channel-capacity 4
    #   Server 3: text-embeddings-router --model-id BAAI/bge-small-en-v1.5 --port 8082 --batch-channel-capacity 8
    # Then run:
    python benchmark_batching.py \
        --test-channel-capacity \
        --capacity-urls http://127.0.0.1:8080 http://127.0.0.1:8081 http://127.0.0.1:8082
"""

import requests
import time
import statistics
import argparse
import json
from typing import NamedTuple, List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict


class RequestMetrics(NamedTuple):
    """Metrics for a single request."""
    total_ms: float
    tokenization_ms: float
    queue_ms: float
    inference_ms: float
    timestamp: float  # When request was sent


def parse_timing_headers(headers: dict) -> Dict[str, float]:
    """Extract timing information from TEI response headers."""
    return {
        "total_ms": float(headers.get("x-total-time", 0)),
        "tokenization_ms": float(headers.get("x-tokenization-time", 0)),
        "queue_ms": float(headers.get("x-queue-time", 0)),
        "inference_ms": float(headers.get("x-inference-time", 0)),
    }


def make_request(url: str, text: str, request_id: int = 0) -> tuple[int, RequestMetrics, float]:
    """Make a single embedding request and return metrics."""
    endpoint = f"{url}/embed"
    start_time = time.time()

    try:
        response = requests.post(
            endpoint,
            json={"inputs": text},
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        elapsed = (time.time() - start_time) * 1000  # Convert to ms

        if response.status_code != 200:
            return request_id, None, elapsed

        timings = parse_timing_headers(response.headers)
        metrics = RequestMetrics(
            total_ms=elapsed,
            tokenization_ms=timings["tokenization_ms"],
            queue_ms=timings["queue_ms"],
            inference_ms=timings["inference_ms"],
            timestamp=start_time
        )
        return request_id, metrics, elapsed
    except Exception as e:
        print(f"  ⚠️  Request {request_id} failed: {e}")
        return request_id, None, elapsed


def benchmark_concurrency(
    url: str,
    text: str,
    concurrency: int,
    total_requests: int,
    delay_ms: float = 0
) -> List[RequestMetrics]:
    """Benchmark with specific concurrency level."""
    results = []
    endpoint = f"{url}/embed"

    print(f"  🔄 Concurrency {concurrency:2d}: Sending {total_requests} requests...", end="", flush=True)

    start_time = time.time()

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = []
        for i in range(total_requests):
            if delay_ms > 0 and i > 0:
                time.sleep(delay_ms / 1000.0)
            future = executor.submit(make_request, url, text, i)
            futures.append(future)

        for future in as_completed(futures):
            request_id, metrics, elapsed = future.result()
            if metrics:
                results.append(metrics)

    duration = time.time() - start_time
    throughput = len(results) / duration if duration > 0 else 0

    print(f" {len(results)}/{total_requests} successful | {throughput:.1f} req/s")

    return results


def analyze_batching(results: List[RequestMetrics]) -> Dict:
    """Analyze batching behavior from results."""
    if not results:
        return {}

    # Sort by timestamp to see request order
    sorted_results = sorted(results, key=lambda x: x.timestamp)

    # Group requests by inference time (requests in same batch have same inference time)
    inference_groups = defaultdict(list)
    for r in sorted_results:
        # Round to nearest 0.1ms to group similar inference times
        inference_key = round(r.inference_ms, 1)
        inference_groups[inference_key].append(r)

    # Calculate batch sizes
    batch_sizes = [len(group) for group in inference_groups.values()]

    # Analyze queue times
    queue_times = [r.queue_ms for r in results]

    # Find requests that were likely in the same batch
    # (same inference time, sequential timestamps)
    batches = []
    current_batch = [sorted_results[0]]

    for i in range(1, len(sorted_results)):
        prev = sorted_results[i-1]
        curr = sorted_results[i]

        # If inference time is very similar and timestamps are close, same batch
        if (abs(curr.inference_ms - prev.inference_ms) < 0.5 and
            (curr.timestamp - prev.timestamp) < 0.1):
            current_batch.append(curr)
        else:
            if len(current_batch) > 0:
                batches.append(current_batch)
            current_batch = [curr]

    if len(current_batch) > 0:
        batches.append(current_batch)

    return {
        "total_requests": len(results),
        "unique_batches": len(batches),
        "avg_batch_size": statistics.mean(batch_sizes) if batch_sizes else 0,
        "max_batch_size": max(batch_sizes) if batch_sizes else 0,
        "min_batch_size": min(batch_sizes) if batch_sizes else 0,
        "avg_queue_time": statistics.mean(queue_times),
        "median_queue_time": statistics.median(queue_times),
        "max_queue_time": max(queue_times),
        "min_queue_time": min(queue_times),
        "std_queue_time": statistics.stdev(queue_times) if len(queue_times) > 1 else 0,
        "avg_inference_time": statistics.mean([r.inference_ms for r in results]),
        "avg_total_time": statistics.mean([r.total_ms for r in results]),
    }


def print_batching_analysis(analysis: Dict, concurrency: int):
    """Print detailed batching analysis."""
    print(f"\n  📊 Batching Analysis (Concurrency {concurrency}):")
    print(f"     Total Requests:     {analysis['total_requests']}")
    print(f"     Unique Batches:      {analysis['unique_batches']}")
    print(f"     Avg Batch Size:     {analysis['avg_batch_size']:.1f}")
    print(f"     Batch Size Range:   {analysis['min_batch_size']:.0f} - {analysis['max_batch_size']:.0f}")
    print(f"\n     Queue Time Stats:")
    print(f"       Average:           {analysis['avg_queue_time']:8.2f} ms")
    print(f"       Median:            {analysis['median_queue_time']:8.2f} ms")
    print(f"       Min/Max:           {analysis['min_queue_time']:8.2f} / {analysis['max_queue_time']:8.2f} ms")
    print(f"       Std Dev:         {analysis['std_queue_time']:8.2f} ms")
    print(f"\n     Inference Time:    {analysis['avg_inference_time']:8.2f} ms")
    print(f"     Total Latency:      {analysis['avg_total_time']:8.2f} ms")


def benchmark_queue_behavior(url: str, text: str):
    """Benchmark queue behavior with different concurrency levels."""
    print("\n" + "=" * 100)
    print("🔬 QUEUE & BATCHING BEHAVIOR ANALYSIS")
    print("=" * 100)
    print("\nTesting how queue time and batch sizes change with concurrency...")

    concurrency_levels = [1, 2, 4, 8, 16, 32, 64]
    requests_per_level = 50

    all_results = {}

    for concurrency in concurrency_levels:
        results = benchmark_concurrency(url, text, concurrency, requests_per_level)
        if results:
            analysis = analyze_batching(results)
            all_results[concurrency] = {
                "results": results,
                "analysis": analysis
            }
            print_batching_analysis(analysis, concurrency)

    # Summary comparison
    print("\n" + "=" * 100)
    print("📈 CONCURRENCY COMPARISON")
    print("=" * 100)
    print(f"{'Concurrency':<15} {'Avg Queue':<15} {'Max Queue':<15} {'Avg Batch':<15} {'Max Batch':<15} {'Throughput':<15}")
    print("-" * 100)

    for concurrency in sorted(all_results.keys()):
        analysis = all_results[concurrency]["analysis"]
        results = all_results[concurrency]["results"]
        total_time = max(r.timestamp for r in results) - min(r.timestamp for r in results)
        throughput = len(results) / total_time if total_time > 0 else 0

        print(f"{concurrency:<15} "
              f"{analysis['avg_queue_time']:<15.2f} "
              f"{analysis['max_queue_time']:<15.2f} "
              f"{analysis['avg_batch_size']:<15.1f} "
              f"{analysis['max_batch_size']:<15.0f} "
              f"{throughput:<15.1f}")

    return all_results


def benchmark_burst_patterns(url: str, text: str):
    """Test different request patterns (burst vs steady)."""
    print("\n" + "=" * 100)
    print("💥 BURST vs STEADY REQUEST PATTERNS")
    print("=" * 100)

    total_requests = 100

    # Pattern 1: Burst (all at once)
    print("\n1️⃣  BURST PATTERN: All requests sent simultaneously")
    burst_results = benchmark_concurrency(url, text, concurrency=total_requests, total_requests=total_requests, delay_ms=0)
    if burst_results:
        burst_analysis = analyze_batching(burst_results)
        print_batching_analysis(burst_analysis, total_requests)

    time.sleep(2)  # Let queue clear

    # Pattern 2: Steady stream (small delay between requests)
    print("\n2️⃣  STEADY PATTERN: Requests sent with 10ms delay")
    steady_results = benchmark_concurrency(url, text, concurrency=10, total_requests=total_requests, delay_ms=10)
    if steady_results:
        steady_analysis = analyze_batching(steady_results)
        print_batching_analysis(steady_analysis, 10)

    time.sleep(2)

    # Pattern 3: Medium concurrency
    print("\n3️⃣  MEDIUM CONCURRENCY: 16 concurrent, no delay")
    medium_results = benchmark_concurrency(url, text, concurrency=16, total_requests=total_requests, delay_ms=0)
    if medium_results:
        medium_analysis = analyze_batching(medium_results)
        print_batching_analysis(medium_analysis, 16)

    # Compare patterns
    print("\n" + "=" * 100)
    print("📊 PATTERN COMPARISON")
    print("=" * 100)
    print(f"{'Pattern':<25} {'Avg Queue':<15} {'Max Queue':<15} {'Avg Batch':<15} {'Throughput':<15}")
    print("-" * 100)

    patterns = [
        ("Burst (100 concurrent)", burst_results, burst_analysis if burst_results else None),
        ("Steady (10 concurrent)", steady_results, steady_analysis if steady_results else None),
        ("Medium (16 concurrent)", medium_results, medium_analysis if medium_results else None),
    ]

    for name, results, analysis in patterns:
        if analysis and results:
            total_time = max(r.timestamp for r in results) - min(r.timestamp for r in results)
            throughput = len(results) / total_time if total_time > 0 else 0
            print(f"{name:<25} "
                  f"{analysis['avg_queue_time']:<15.2f} "
                  f"{analysis['max_queue_time']:<15.2f} "
                  f"{analysis['avg_batch_size']:<15.1f} "
                  f"{throughput:<15.1f}")


def benchmark_channel_capacity(urls: Optional[List[str]], text: str, concurrency: int = 32):
    """Test how different channel capacity values affect performance."""
    if not urls or len(urls) < 2:
        print(f"\n{'═' * 100}")
        print("🔧 BATCH CHANNEL CAPACITY IMPACT TEST")
        print("═" * 100)
        print("⚠️  To test channel capacity impact, you need to:")
        print("   1. Start multiple TEI servers with different --batch-channel-capacity values")
        print("   2. Use --capacity-urls to specify their URLs")
        print("\n   Example:")
        print("     # Terminal 1: Capacity 1 (low latency)")
        print("     text-embeddings-router --model-id BAAI/bge-small-en-v1.5 --port 8080 --batch-channel-capacity 1")
        print("     # Terminal 2: Capacity 4 (balanced)")
        print("     text-embeddings-router --model-id BAAI/bge-small-en-v1.5 --port 8081 --batch-channel-capacity 4")
        print("     # Terminal 3: Capacity 8 (high throughput)")
        print("     text-embeddings-router --model-id BAAI/bge-small-en-v1.5 --port 8082 --batch-channel-capacity 8")
        print("\n   Then run:")
        print("     python benchmark_batching.py \\")
        print("       --test-channel-capacity \\")
        print("       --capacity-urls http://127.0.0.1:8080 http://127.0.0.1:8081 http://127.0.0.1:8082")
        return

    print(f"\n{'═' * 100}")
    print("🔧 BATCH CHANNEL CAPACITY IMPACT TEST")
    print("═" * 100)
    print("Testing how different --batch-channel-capacity values affect performance")
    print(f"\n📝 Note: These servers should be started with different --batch-channel-capacity values")
    print(f"   Example:")
    print(f"     Server 1: text-embeddings-router --model-id BAAI/bge-small-en-v1.5 --port 8080 --batch-channel-capacity 1")
    print(f"     Server 2: text-embeddings-router --model-id BAAI/bge-small-en-v1.5 --port 8081 --batch-channel-capacity 4")
    print(f"     Server 3: text-embeddings-router --model-id BAAI/bge-small-en-v1.5 --port 8082 --batch-channel-capacity 8")

    capacity_results = []
    capacity_labels = ["Capacity 1", "Capacity 4", "Capacity 8", "Capacity 16"]

    for i, url in enumerate(urls):
        print(f"\n🔹 Testing server {i+1} at {url}...")
        try:
            health = requests.get(f"{url}/health", timeout=2)
            if health.status_code != 200:
                print(f"  ⚠️  Server not healthy, skipping")
                continue

            # Get model info
            try:
                info = requests.get(f"{url}/info", timeout=5).json()
                print(f"     Model: {info.get('model_id', 'N/A')}")
            except:
                pass

            # Run concurrent benchmark to see capacity impact
            print(f"     Running benchmark with concurrency={concurrency}...")
            results = benchmark_concurrency(url, text, concurrency, total_requests=100, delay_ms=0)

            if results:
                analysis = analyze_batching(results)
                capacity_results.append((url, results, analysis, capacity_labels[i] if i < len(capacity_labels) else f"Server {i+1}"))
                print_batching_analysis(analysis, concurrency)
        except Exception as e:
            print(f"  ❌ Error testing {url}: {e}")
            continue

    if len(capacity_results) >= 2:
        print(f"\n{'═' * 100}")
        print("📊 CHANNEL CAPACITY COMPARISON")
        print("═" * 100)
        print(f"{'Capacity':<15} {'URL':<25} {'Latency (ms)':<18} {'Queue (ms)':<18} {'Throughput (req/s)':<20} {'Inference (ms)':<18}")
        print("-" * 100)

        for url, results, analysis, label in capacity_results:
            total_time = max(r.timestamp for r in results) - min(r.timestamp for r in results)
            throughput = len(results) / total_time if total_time > 0 else 0
            print(f"{label:<15} {url:<25} {analysis['avg_total_time']:<18.2f} {analysis['avg_queue_time']:<18.2f} {throughput:<20.2f} {analysis['avg_inference_time']:<18.2f}")

        # Calculate improvements
        baseline_url, baseline_results, baseline_analysis, baseline_label = capacity_results[0]
        baseline_throughput = len(baseline_results) / (max(r.timestamp for r in baseline_results) - min(r.timestamp for r in baseline_results)) if baseline_results else 0

        print(f"\n📈 Performance Improvement (vs baseline {baseline_label}):")
        for url, results, analysis, label in capacity_results[1:]:
            total_time = max(r.timestamp for r in results) - min(r.timestamp for r in results)
            throughput = len(results) / total_time if total_time > 0 else 0

            latency_improvement = ((baseline_analysis['avg_total_time'] - analysis['avg_total_time']) / baseline_analysis['avg_total_time'] * 100) if baseline_analysis['avg_total_time'] > 0 else 0
            throughput_improvement = ((throughput - baseline_throughput) / baseline_throughput * 100) if baseline_throughput > 0 else 0
            queue_change = analysis['avg_queue_time'] - baseline_analysis['avg_queue_time']

            print(f"\n  {label} ({url}):")
            print(f"    Latency:    {latency_improvement:+.1f}% ({analysis['avg_total_time']:.2f} ms vs {baseline_analysis['avg_total_time']:.2f} ms)")
            print(f"    Throughput: {throughput_improvement:+.1f}% ({throughput:.2f} vs {baseline_throughput:.2f} req/s)")
            print(f"    Queue Time: {queue_change:+.2f} ms ({analysis['avg_queue_time']:.2f} ms vs {baseline_analysis['avg_queue_time']:.2f} ms)")

        print(f"\n💡 Insights:")
        print(f"   • Higher channel capacity typically increases throughput but may increase queue time")
        print(f"   • Optimal capacity depends on your latency/throughput trade-off requirements")
        print(f"   • Capacity 1: Best for low-latency, interactive applications")
        print(f"   • Capacity 4-8: Best for high-throughput, batch processing workloads")
        print(f"   • Pipeline parallelism: Higher capacity allows batch formation to overlap with inference")


def benchmark_queue_under_load(url: str, text: str):
    """Test queue behavior under sustained load."""
    print("\n" + "=" * 100)
    print("⚡ SUSTAINED LOAD TEST")
    print("=" * 100)
    print("Testing queue behavior with continuous requests over time...")

    duration_seconds = 10
    concurrency = 32
    requests_per_second = 50

    print(f"\n  Running for {duration_seconds} seconds with {concurrency} concurrent workers")
    print(f"  Target rate: ~{requests_per_second} requests/second")

    all_results = []
    start_time = time.time()
    request_count = 0

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = []

        while time.time() - start_time < duration_seconds:
            # Submit new requests to maintain target rate
            for _ in range(requests_per_second // 10):  # Submit in chunks
                if time.time() - start_time >= duration_seconds:
                    break
                future = executor.submit(make_request, url, text, request_count)
                futures.append(future)
                request_count += 1
                time.sleep(0.1)  # ~10 requests per second

            # Collect completed requests
            for future in as_completed(futures):
                _, metrics, _ = future.result()
                if metrics:
                    all_results.append(metrics)
                futures.remove(future)

        # Wait for remaining requests
        for future in as_completed(futures):
            _, metrics, _ = future.result()
            if metrics:
                all_results.append(metrics)

    if all_results:
        analysis = analyze_batching(all_results)
        print(f"\n  ✅ Completed {len(all_results)} requests in {duration_seconds} seconds")
        print(f"  📊 Actual throughput: {len(all_results) / duration_seconds:.1f} req/s")
        print_batching_analysis(analysis, concurrency)


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark TEI queuing and batching components",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--url", default="http://127.0.0.1:8080",
                        help="TEI server URL")
    parser.add_argument("--text", default="Hello, how are you today?",
                        help="Text to embed for testing")
    parser.add_argument("--skip-load-test", action="store_true",
                        help="Skip the sustained load test")
    parser.add_argument("--test-channel-capacity", action="store_true",
                        help="Test different channel capacity values (requires multiple server instances)")
    parser.add_argument("--capacity-urls", type=str, nargs="+",
                        help="URLs of servers with different --batch-channel-capacity values (e.g., http://127.0.0.1:8080 http://127.0.0.1:8081 http://127.0.0.1:8082)")

    args = parser.parse_args()

    print("=" * 100)
    print("           TEI QUEUING & BATCHING BENCHMARK")
    print("=" * 100)

    # Channel Capacity Impact Test (skip other tests if this is enabled)
    if args.test_channel_capacity:
        benchmark_channel_capacity(args.capacity_urls, args.text, concurrency=32)
        return

    # Regular benchmarks (only if not testing channel capacity)
    # Check server health
    try:
        health = requests.get(f"{args.url}/health", timeout=2)
        if health.status_code != 200:
            print(f"❌ Server not healthy: {health.status_code}")
            return
    except requests.exceptions.RequestException as e:
        print(f"❌ Cannot connect to server at {args.url}: {e}")
        print("   Make sure TEI is running:")
        print("   ./target/release/text-embeddings-router --model-id BAAI/bge-small-en-v1.5 --port 8080")
        return

    # Get model info
    try:
        info = requests.get(f"{args.url}/info", timeout=5).json()
        print(f"\n📦 Model: {info.get('model_id', 'N/A')}")
        print(f"   Max batch tokens: {info.get('max_batch_tokens', 'N/A')}")
        print(f"   Max concurrent requests: {info.get('max_concurrent_requests', 'N/A')}")
    except Exception as e:
        print(f"  ⚠️  Could not get model info: {e}")

    print(f"\n📝 Test text: {args.text}")
    print(f"   Length: {len(args.text)} chars, ~{len(args.text.split())} words")

    # Run benchmarks
    benchmark_queue_behavior(args.url, args.text)
    benchmark_burst_patterns(args.url, args.text)

    if not args.skip_load_test:
        benchmark_queue_under_load(args.url, args.text)

    print("\n" + "=" * 100)
    print("✅ Batching benchmark complete!")
    print("=" * 100)
    print("\n💡 Key Insights:")
    print("   • Higher concurrency → Larger batches → Better throughput")
    print("   • Queue time increases with concurrency (waiting for batch formation)")
    print("   • Optimal concurrency depends on your latency/throughput trade-off")
    print("   • Burst patterns create larger batches but higher queue times")
    if args.test_channel_capacity:
        print("   • Higher channel capacity improves throughput via pipeline parallelism")
        print("   • Channel capacity 1: Best for low-latency, interactive applications")
        print("   • Channel capacity 4-8: Best for high-throughput, batch processing")


if __name__ == "__main__":
    main()

