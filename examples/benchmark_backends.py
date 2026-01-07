#!/usr/bin/env python3
"""
TEI Backend Benchmarking Script
================================

This script benchmarks different TEI backends (Candle, ONNX Runtime, Python)
and compares their performance.

Since backends are compile-time features, you need to run separate TEI server
instances with different backends. This script connects to multiple servers
and compares their performance.

BACKENDS:
---------
1. Candle Backend (Rust)
   - Best for: GPU inference, production deployments
   - Features: Flash Attention, CUDA, Metal (Apple Silicon)
   - Build: cargo install --path router -F candle-cuda
   - Performance: Highest throughput, lowest latency

2. ONNX Runtime Backend (ORT)
   - Best for: CPU inference, x86 servers
   - Features: Intel MKL-DNN optimizations
   - Build: cargo install --path router -F ort
   - Performance: Best CPU performance, 2-3x faster than Python on CPU

3. Python Backend (PyTorch)
   - Best for: Flash Attention, HPU (Habana Gaudi), custom models
   - Features: Full PyTorch ecosystem, Flash Attention v1/v2
   - Build: cargo install --path router -F python
   - Requirements: Python 3.9-3.12, PyTorch, sentence-transformers
   - Performance: Good but slower than Rust backends due to Python overhead
   - Note: Requires python-text-embeddings-server to be installed

Usage:
    # Start TEI servers with different backends on different ports:
    # Server 1 (Candle): text-embeddings-router --model-id BAAI/bge-small-en-v1.5 --port 8080
    # Server 2 (ORT):    text-embeddings-router --model-id BAAI/bge-small-en-v1.5 --port 8081
    # Server 3 (Python): text-embeddings-router --model-id BAAI/bge-small-en-v1.5 --port 8082

    # Run benchmark:
    python benchmark_backends.py \
        --candle-url http://127.0.0.1:8080 \
        --ort-url http://127.0.0.1:8082 \
        --python-url http://127.0.0.1:8083 \
        --iterations 50

Python Backend Setup:
    # Install Python dependencies
    cd backends/python/server
    poetry install
    # Or: pip install -r requirements.txt

    # Verify installation
    which python-text-embeddings-server

    # Build TEI with Python backend
    cargo install --path router -F python

    # Start server (Python backend will auto-start as subprocess)
    text-embeddings-router --model-id BAAI/bge-small-en-v1.5 --port 8082
"""

import requests
import time
import statistics
import argparse
from typing import NamedTuple, Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import json


class BackendResult(NamedTuple):
    """Performance results for a backend."""
    backend_name: str
    url: str
    total_ms: float
    tokenization_ms: float
    queue_ms: float
    inference_ms: float
    throughput: float  # requests per second
    success_count: int
    error_count: int


class BackendBenchmark:
    """Benchmark a single backend."""

    def __init__(self, name: str, url: str):
        self.name = name
        self.url = url
        self.endpoint = f"{url}/embed"

    def check_health(self) -> bool:
        """Check if the backend server is healthy."""
        try:
            response = requests.get(f"{self.url}/health", timeout=2)
            return response.status_code == 200
        except Exception as e:
            print(f"  ❌ {self.name} health check failed: {e}")
            return False

    def get_info(self) -> Optional[dict]:
        """Get model information from the backend."""
        try:
            response = requests.get(f"{self.url}/info", timeout=5)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            print(f"  ⚠️  {self.name} info request failed: {e}")
        return None

    def single_request(self, text: str) -> Optional[dict]:
        """Make a single embedding request and return timing data."""
        try:
            start_time = time.time()
            response = requests.post(
                self.endpoint,
                json={"inputs": text},
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            elapsed = (time.time() - start_time) * 1000  # Convert to ms

            if response.status_code != 200:
                return None

            headers = response.headers
            return {
                "total_ms": elapsed,
                "tokenization_ms": float(headers.get("x-tokenization-time", 0)),
                "queue_ms": float(headers.get("x-queue-time", 0)),
                "inference_ms": float(headers.get("x-inference-time", 0)),
                "success": True
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def benchmark_sequential(self, text: str, iterations: int = 20) -> BackendResult:
        """Run sequential requests and measure latency."""
        results = []
        success_count = 0
        error_count = 0

        print(f"  Running {iterations} sequential requests...")

        for i in range(iterations):
            result = self.single_request(text)
            if result and result.get("success"):
                results.append(result)
                success_count += 1
            else:
                error_count += 1

            if (i + 1) % 10 == 0:
                print(f"    Completed {i+1}/{iterations} requests")

        if not results:
            return BackendResult(
                backend_name=self.name,
                url=self.url,
                total_ms=0,
                tokenization_ms=0,
                queue_ms=0,
                inference_ms=0,
                throughput=0,
                success_count=0,
                error_count=error_count
            )

        # Calculate statistics
        totals = [r["total_ms"] for r in results]
        tokenizations = [r["tokenization_ms"] for r in results]
        queues = [r["queue_ms"] for r in results]
        inferences = [r["inference_ms"] for r in results]

        total_time = sum(totals) / 1000  # Convert to seconds
        throughput = len(results) / total_time if total_time > 0 else 0

        return BackendResult(
            backend_name=self.name,
            url=self.url,
            total_ms=statistics.mean(totals),
            tokenization_ms=statistics.mean(tokenizations),
            queue_ms=statistics.mean(queues),
            inference_ms=statistics.mean(inferences),
            throughput=throughput,
            success_count=success_count,
            error_count=error_count
        )

    def benchmark_concurrent(self, text: str, concurrency: int = 8, iterations: int = 40) -> BackendResult:
        """Run concurrent requests and measure throughput."""
        results = []
        success_count = 0
        error_count = 0

        print(f"  Running {iterations} requests with concurrency={concurrency}...")

        start_time = time.time()

        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = [
                executor.submit(self.single_request, text)
                for _ in range(iterations)
            ]

            for future in as_completed(futures):
                result = future.result()
                if result and result.get("success"):
                    results.append(result)
                    success_count += 1
                else:
                    error_count += 1

        total_duration = time.time() - start_time
        throughput = len(results) / total_duration if total_duration > 0 else 0

        if not results:
            return BackendResult(
                backend_name=self.name,
                url=self.url,
                total_ms=0,
                tokenization_ms=0,
                queue_ms=0,
                inference_ms=0,
                throughput=0,
                success_count=0,
                error_count=error_count
            )

        # Calculate statistics
        totals = [r["total_ms"] for r in results]
        tokenizations = [r["tokenization_ms"] for r in results]
        queues = [r["queue_ms"] for r in results]
        inferences = [r["inference_ms"] for r in results]

        return BackendResult(
            backend_name=self.name,
            url=self.url,
            total_ms=statistics.mean(totals),
            tokenization_ms=statistics.mean(tokenizations),
            queue_ms=statistics.mean(queues),
            inference_ms=statistics.mean(inferences),
            throughput=throughput,
            success_count=success_count,
            error_count=error_count
        )


def print_comparison_table(results: List[BackendResult], test_name: str):
    """Print a comparison table of backend results."""
    print(f"\n{'═' * 100}")
    print(f"📊 {test_name}")
    print("═" * 100)
    print(f"{'Backend':<15} {'Latency (ms)':<18} {'Throughput (req/s)':<20} {'Inference (ms)':<18} {'Success Rate':<15}")
    print("-" * 100)

    for result in results:
        success_rate = f"{100 * result.success_count / (result.success_count + result.error_count):.1f}%" if (result.success_count + result.error_count) > 0 else "0%"
        print(f"{result.backend_name:<15} {result.total_ms:<18.2f} {result.throughput:<20.2f} {result.inference_ms:<18.2f} {success_rate:<15}")

    # Calculate relative performance
    if len(results) >= 2:
        print("\n📈 Relative Performance (vs fastest backend):")
        fastest_throughput = max(r.throughput for r in results if r.throughput > 0)
        fastest_latency = min(r.total_ms for r in results if r.total_ms > 0)

        for result in results:
            if result.throughput > 0:
                throughput_ratio = (result.throughput / fastest_throughput) * 100
                latency_ratio = (fastest_latency / result.total_ms) * 100 if result.total_ms > 0 else 0
                print(f"  {result.backend_name:<15} Throughput: {throughput_ratio:5.1f}% | Latency: {latency_ratio:5.1f}%")


def print_detailed_stats(results: List[BackendResult]):
    """Print detailed statistics for each backend."""
    print(f"\n{'═' * 100}")
    print("📋 DETAILED STATISTICS")
    print("═" * 100)

    for result in results:
        print(f"\n🔹 {result.backend_name} ({result.url})")
        print(f"   Total Latency:      {result.total_ms:8.2f} ms")
        print(f"   ├─ Tokenization:    {result.tokenization_ms:8.2f} ms ({100*result.tokenization_ms/result.total_ms:.1f}%)" if result.total_ms > 0 else "   ├─ Tokenization:    0.00 ms")
        print(f"   ├─ Queue:           {result.queue_ms:8.2f} ms ({100*result.queue_ms/result.total_ms:.1f}%)" if result.total_ms > 0 else "   ├─ Queue:           0.00 ms")
        print(f"   └─ Inference:       {result.inference_ms:8.2f} ms ({100*result.inference_ms/result.total_ms:.1f}%)" if result.total_ms > 0 else "   └─ Inference:       0.00 ms")
        print(f"   Throughput:         {result.throughput:8.2f} req/s")
        print(f"   Success Rate:       {result.success_count}/{result.success_count + result.error_count} ({100*result.success_count/(result.success_count + result.error_count):.1f}%)" if (result.success_count + result.error_count) > 0 else "   Success Rate:       0/0 (0%)")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark TEI backends (Candle, ONNX Runtime, Python)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Benchmark all three backends
  python benchmark_backends.py \\
      --candle-url http://127.0.0.1:8080 \\
      --ort-url http://127.0.0.1:8081 \\
      --python-url http://127.0.0.1:8082 \\
      --iterations 50

  # Benchmark only Candle and ONNX Runtime
  python benchmark_backends.py \\
      --candle-url http://127.0.0.1:8080 \\
      --ort-url http://127.0.0.1:8081 \\
      --iterations 30

Note: You need to start separate TEI server instances with different backends
on different ports. Backends are compile-time features, so you need to build
separate binaries or use different Docker images.
        """
    )

    parser.add_argument("--candle-url", type=str,
                        help="Candle backend server URL (Rust, GPU-optimized)")
    parser.add_argument("--ort-url", type=str,
                        help="ONNX Runtime backend server URL (CPU-optimized)")
    parser.add_argument("--python-url", type=str,
                        help="Python backend server URL (PyTorch, requires python-text-embeddings-server)")
    parser.add_argument("--iterations", type=int, default=30, help="Number of iterations per test")
    parser.add_argument("--concurrency", type=int, default=8, help="Concurrency level for throughput test")
    parser.add_argument("--output", type=str, help="Output JSON file for results")

    args = parser.parse_args()

    # Collect backends to benchmark
    backends = []
    if args.candle_url:
        backends.append(BackendBenchmark("Candle", args.candle_url))
    if args.ort_url:
        backends.append(BackendBenchmark("ONNX Runtime", args.ort_url))
    if args.python_url:
        backends.append(BackendBenchmark("Python", args.python_url))

    if not backends:
        print("❌ Error: No backends specified. Use --candle-url, --ort-url, or --python-url")
        parser.print_help()
        return

    print("=" * 100)
    print("           TEI BACKEND BENCHMARK")
    print("=" * 100)
    print(f"\n📦 Testing {len(backends)} backend(s)")
    print(f"   Iterations: {args.iterations}")
    print(f"   Concurrency: {args.concurrency}")

    # Health checks
    print("\n🏥 Health Checks:")
    healthy_backends = []
    for backend in backends:
        if backend.check_health():
            print(f"  ✅ {backend.name}: Healthy")
            info = backend.get_info()
            if info:
                print(f"     Model: {info.get('model_id', 'N/A')}")
                print(f"     Max input length: {info.get('max_input_length', 'N/A')} tokens")
                # Show backend-specific info
                if backend.name == "Python":
                    print(f"     Note: Python backend uses PyTorch (may have higher memory usage)")
                elif backend.name == "ONNX Runtime":
                    print(f"     Note: ONNX Runtime optimized for CPU inference")
                elif backend.name == "Candle":
                    print(f"     Note: Candle backend optimized for GPU (if available)")
            healthy_backends.append(backend)
        else:
            print(f"  ❌ {backend.name}: Unhealthy or unreachable")
            if backend.name == "Python":
                print(f"     💡 Tip: Ensure python-text-embeddings-server is installed:")
                print(f"        cd backends/python/server && poetry install")

    if not healthy_backends:
        print("\n❌ No healthy backends found. Exiting.")
        return

    # Test cases
    test_cases = [
        ("Short text (5 words)", "Hello, how are you today?"),
        ("Medium text (50 words)", "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed. It focuses on developing algorithms that can access data and use it to learn for themselves. The process begins with observations or data, such as examples, direct experience, or instruction."),
    ]

    all_results = {}

    # Sequential latency tests
    print(f"\n{'═' * 100}")
    print("⏱️  SEQUENTIAL LATENCY TEST")
    print("═" * 100)
    print("Measuring latency with sequential requests (no batching effects)")

    for name, text in test_cases:
        print(f"\n📝 Test: {name}")
        print(f"   Input: {len(text)} chars, ~{len(text.split())} words")

        results = []
        for backend in healthy_backends:
            print(f"\n🔹 Benchmarking {backend.name}...")
            result = backend.benchmark_sequential(text, iterations=args.iterations)
            results.append(result)

        all_results[f"sequential_{name}"] = results
        print_comparison_table(results, f"Sequential Latency - {name}")

    # Concurrent throughput tests
    print(f"\n{'═' * 100}")
    print("🚀 CONCURRENT THROUGHPUT TEST")
    print("═" * 100)
    print("Measuring throughput with concurrent requests (batching enabled)")

    short_text = test_cases[0][1]
    print(f"\n📝 Using: {test_cases[0][0]}")

    results = []
    for backend in healthy_backends:
        print(f"\n🔹 Benchmarking {backend.name}...")
        result = backend.benchmark_concurrent(
            short_text,
            concurrency=args.concurrency,
            iterations=args.iterations * 2  # More iterations for throughput
        )
        results.append(result)

    all_results["concurrent_throughput"] = results
    print_comparison_table(results, "Concurrent Throughput")

    # Detailed statistics
    print_detailed_stats(results)

    # Summary
    print(f"\n{'═' * 100}")
    print("📊 SUMMARY")
    print("═" * 100)

    if len(results) >= 2:
        fastest = max(results, key=lambda r: r.throughput)
        lowest_latency = min(results, key=lambda r: r.total_ms)

        print(f"\n🏆 Fastest Throughput: {fastest.backend_name} ({fastest.throughput:.2f} req/s)")
        print(f"⚡ Lowest Latency: {lowest_latency.backend_name} ({lowest_latency.total_ms:.2f} ms)")

        if fastest.backend_name != lowest_latency.backend_name:
            print(f"\n💡 Note: Different backends excel at different metrics:")
            print(f"   • {fastest.backend_name} has highest throughput (good for batch processing)")
            print(f"   • {lowest_latency.backend_name} has lowest latency (good for real-time)")

        # Backend-specific recommendations
        print(f"\n📚 Backend Recommendations:")
        python_backend = next((r for r in results if r.backend_name == "Python"), None)
        candle_backend = next((r for r in results if r.backend_name == "Candle"), None)
        ort_backend = next((r for r in results if r.backend_name == "ONNX Runtime"), None)

        if python_backend:
            print(f"   • Python Backend: Use for Flash Attention, HPU, or custom models")
            print(f"     Trade-off: Higher memory usage, slower startup than Rust backends")

        if ort_backend:
            print(f"   • ONNX Runtime: Best for CPU-only deployments (x86)")
            print(f"     Trade-off: CPU-only, no GPU support")

        if candle_backend:
            print(f"   • Candle: Best for GPU deployments, production use")
            print(f"     Trade-off: Limited model support compared to Python backend")

    # Save results to JSON if requested
    if args.output:
        output_data = {
            "test_config": {
                "iterations": args.iterations,
                "concurrency": args.concurrency,
            },
            "results": {
                name: [
                    {
                        "backend": r.backend_name,
                        "url": r.url,
                        "total_ms": r.total_ms,
                        "tokenization_ms": r.tokenization_ms,
                        "queue_ms": r.queue_ms,
                        "inference_ms": r.inference_ms,
                        "throughput": r.throughput,
                        "success_count": r.success_count,
                        "error_count": r.error_count,
                    }
                    for r in results
                ]
                for name, results in all_results.items()
            }
        }
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\n💾 Results saved to {args.output}")

    print(f"\n{'═' * 100}")
    print("✅ Benchmark complete!")
    print("═" * 100)


if __name__ == "__main__":
    main()

