#!/usr/bin/env bash
# Start TEI servers for three backends (ORT, Candle, Python) on different ports
# using locally built binaries (no Docker).
# This script will automatically build missing binaries if Rust/cargo is available.

set -euo pipefail

MODEL_ID="${MODEL_ID:-BAAI/bge-small-en-v1.5}"
LOG_DIR="${LOG_DIR:-/tmp/tei-logs}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Point to per-backend binaries. Defaults assume you built to /tmp/tei-*/bin.
# Override these if you built elsewhere.
ORT_BIN="${ORT_BIN:-/tmp/tei-ort/bin/text-embeddings-router}"
CANDLE_BIN="${CANDLE_BIN:-/tmp/tei-candle/bin/text-embeddings-router}"
PY_BIN="${PY_BIN:-/tmp/tei-python/bin/text-embeddings-router}"

ORT_PORT="${ORT_PORT:-8080}"
CANDLE_PORT="${CANDLE_PORT:-8081}"
PY_PORT="${PY_PORT:-8082}"

# Extra args per backend. Python backend uses float32 to avoid bfloat16 issues.
ORT_ARGS="${ORT_ARGS:-}"
CANDLE_ARGS="${CANDLE_ARGS:-}"
PY_ARGS="${PY_ARGS:---dtype float32}"

mkdir -p "${LOG_DIR}"

# Stop any existing TEI servers first
if [[ -f "$(dirname "${BASH_SOURCE[0]}")/stop_all_backends.sh" ]]; then
  echo "🛑 Stopping any existing TEI servers..."
  bash "$(dirname "${BASH_SOURCE[0]}")/stop_all_backends.sh" || true
  sleep 2
fi

# Check if a port is available
check_port_available() {
  local port="$1"

  # Try multiple methods to check port availability
  if command -v ss >/dev/null 2>&1; then
    if ss -tlnp 2>/dev/null | grep -q ":${port} "; then
      return 1  # Port is in use
    fi
  elif command -v netstat >/dev/null 2>&1; then
    if netstat -tlnp 2>/dev/null | grep -q ":${port} "; then
      return 1  # Port is in use
    fi
  elif command -v lsof >/dev/null 2>&1; then
    if lsof -ti:${port} >/dev/null 2>&1; then
      return 1  # Port is in use
    fi
  fi

  # Fallback: try to connect to the port
  if command -v nc >/dev/null 2>&1; then
    if nc -z 127.0.0.1 ${port} 2>/dev/null; then
      return 1  # Port is in use
    fi
  fi

  return 0  # Port appears to be available
}

# Find an available port starting from the requested port, incrementing by 1
find_available_port() {
  local requested_port="$1"
  local max_attempts=100
  local port="${requested_port}"
  local attempts=0

  while [ $attempts -lt $max_attempts ]; do
    if check_port_available "${port}"; then
      echo "${port}"
      return 0
    fi
    port=$((port + 1))
    attempts=$((attempts + 1))
  done

  echo ""
  echo "❌ Error: Could not find an available port after ${max_attempts} attempts starting from ${requested_port}"
  exit 1
}

# Find available ports for all backends, auto-incrementing if needed
find_all_ports() {
  echo "🔍 Finding available ports..."

  local requested_ort="${ORT_PORT}"
  local requested_candle="${CANDLE_PORT}"
  local requested_py="${PY_PORT}"

  # Find available port for ORT
  local found_ort=$(find_available_port "${requested_ort}")
  if [[ "${found_ort}" != "${requested_ort}" ]]; then
    echo "  ⚠️  Port ${requested_ort} (ORT) is in use, using port ${found_ort} instead"
  else
    echo "  ✅ Port ${found_ort} (ORT) is available"
  fi

  # Find available port for Candle (make sure it's different from ORT)
  local found_candle=$(find_available_port "${requested_candle}")
  while [[ "${found_candle}" == "${found_ort}" ]]; do
    found_candle=$(find_available_port $((found_candle + 1)))
  done
  if [[ "${found_candle}" != "${requested_candle}" ]]; then
    echo "  ⚠️  Port ${requested_candle} (Candle) is in use, using port ${found_candle} instead"
  else
    echo "  ✅ Port ${found_candle} (Candle) is available"
  fi

  # Find available port for Python (make sure it's different from ORT and Candle)
  local found_py="${requested_py}"
  # Start from requested port and increment until we find one that's available and doesn't conflict
  while ! check_port_available "${found_py}" || [[ "${found_py}" == "${found_ort}" ]] || [[ "${found_py}" == "${found_candle}" ]]; do
    if [[ "${found_py}" == "${requested_py}" ]] && ! check_port_available "${found_py}"; then
      echo "  ⚠️  Port ${requested_py} (Python) is in use, searching for alternative..."
    fi
    found_py=$((found_py + 1))
    if [ $found_py -gt $((requested_py + 100)) ]; then
      echo "  ❌ Error: Could not find an available port for Python after 100 attempts"
      exit 1
    fi
  done
  if [[ "${found_py}" != "${requested_py}" ]]; then
    echo "  ⚠️  Port ${requested_py} (Python) is in use, using port ${found_py} instead"
  else
    echo "  ✅ Port ${found_py} (Python) is available"
  fi

  # Update the port variables
  ORT_PORT="${found_ort}"
  CANDLE_PORT="${found_candle}"
  PY_PORT="${found_py}"

  echo ""
}

# Check if Rust/cargo is available
check_rust() {
  if ! command -v cargo >/dev/null 2>&1; then
    echo "⚠️  Rust/cargo not found. Attempting to install Rust..."
    if command -v curl >/dev/null 2>&1; then
      curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
      # Source cargo env if it exists
      if [[ -f "$HOME/.cargo/env" ]]; then
        source "$HOME/.cargo/env"
      fi
    else
      echo "❌ Error: cargo not found and curl not available. Please install Rust manually:"
      echo "   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh"
      exit 1
    fi
  fi

  # Ensure cargo is in PATH
  if [[ -f "$HOME/.cargo/env" ]]; then
    source "$HOME/.cargo/env" || true
  fi

  if ! command -v cargo >/dev/null 2>&1; then
    echo "❌ Error: cargo still not available after installation attempt."
    exit 1
  fi

  echo "✅ Rust/cargo available: $(cargo --version)"
}

# Check and install lld linker if needed (required for Python backend)
check_lld() {
  if ! command -v lld >/dev/null 2>&1; then
    echo "⚠️  lld linker not found. Attempting to install..."
    if command -v apt-get >/dev/null 2>&1; then
      apt-get update -qq && apt-get install -y -qq lld >/dev/null 2>&1 || true
    elif command -v yum >/dev/null 2>&1; then
      yum install -y -q lld >/dev/null 2>&1 || true
    elif command -v brew >/dev/null 2>&1; then
      brew install lld >/dev/null 2>&1 || true
    fi
  fi
}

# Check and install pkg-config if needed (required for OpenSSL)
check_pkg_config() {
  if ! command -v pkg-config >/dev/null 2>&1; then
    echo "⚠️  pkg-config not found. Attempting to install..."
    if command -v apt-get >/dev/null 2>&1; then
      apt-get update -qq && apt-get install -y -qq pkg-config >/dev/null 2>&1 || true
    elif command -v yum >/dev/null 2>&1; then
      yum install -y -q pkgconfig >/dev/null 2>&1 || true
    elif command -v brew >/dev/null 2>&1; then
      brew install pkg-config >/dev/null 2>&1 || true
    fi
  fi
}

# Check and install protobuf compiler if needed (required for Python backend gRPC)
check_protoc() {
  if ! command -v protoc >/dev/null 2>&1; then
    echo "⚠️  protoc (Protocol Buffer compiler) not found. Attempting to install..."
    if command -v apt-get >/dev/null 2>&1; then
      apt-get update -qq && apt-get install -y -qq protobuf-compiler >/dev/null 2>&1 || true
    elif command -v yum >/dev/null 2>&1; then
      yum install -y -q protobuf-compiler >/dev/null 2>&1 || true
    elif command -v brew >/dev/null 2>&1; then
      brew install protobuf >/dev/null 2>&1 || true
    fi
  fi
}

# Build a binary if it doesn't exist
build_binary() {
  local name="$1"
  local bin_path="$2"
  local feature="$3"
  local toolchain="${4:-}"

  local bin_dir="$(dirname "${bin_path}")"
  # cargo install --root creates a 'bin' subdirectory, so use parent as root
  local install_root="$(dirname "${bin_dir}")"
  mkdir -p "${install_root}"

  if [[ -x "${bin_path}" ]]; then
    echo "✅ ${name} binary already exists at ${bin_path}"
    return 0
  fi

  echo "🔨 Building ${name} backend binary (this may take several minutes)..."

  local build_cmd="cargo install --path ${REPO_DIR}/router -F ${feature} --root ${install_root}"

  # Python backend requires stable toolchain and additional dependencies
  if [[ "${name}" == "python" ]]; then
    check_lld
    check_pkg_config
    check_protoc
    # Workaround for rust-lld bus error: force use of system linker (bfd/gold) instead of lld
    # Remove any existing -fuse-ld=lld from RUSTFLAGS and explicitly use bfd
    if [[ -n "${RUSTFLAGS:-}" ]]; then
      export RUSTFLAGS="${RUSTFLAGS//-fuse-ld=lld/} -C link-arg=-fuse-ld=bfd"
    else
      export RUSTFLAGS="-C link-arg=-fuse-ld=bfd"
    fi
    # Also set CARGO_TARGET_* to avoid lld
    export CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_LINKER="gcc"
    if [[ -n "${toolchain}" ]]; then
      build_cmd="RUSTUP_TOOLCHAIN=${toolchain} ${build_cmd}"
    else
      # Try stable toolchain
      if rustup toolchain list | grep -q "^stable"; then
        build_cmd="RUSTUP_TOOLCHAIN=stable ${build_cmd}"
      fi
    fi
  fi

  # Limit parallelism for Candle CUDA builds to avoid OOM
  if [[ "${name}" == "candle" ]] && [[ "${feature}" == *"cuda"* ]]; then
    export CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:-1}"
    export RAYON_NUM_THREADS="${RAYON_NUM_THREADS:-1}"
    # Limit nvcc parallelism (nvcc uses -j flag, but we can limit via MAX_JOBS)
    export MAX_JOBS="${MAX_JOBS:-1}"
    # Limit CUDA compiler memory usage (only if CUDAFLAGS is not set)
    if [[ -z "${CUDAFLAGS:-}" ]]; then
      export CUDAFLAGS="--maxrregcount=64"
    else
      export CUDAFLAGS="${CUDAFLAGS} --maxrregcount=64"
    fi
    echo "  ⚙️  Using ${CARGO_BUILD_JOBS} parallel job for CUDA build (to reduce memory usage)"
    echo "  ⚙️  Limiting nvcc parallelism to ${MAX_JOBS} job(s)"

    # Set CUDA_COMPUTE_CAP if not already set (per Candle installation guide)
    if [[ -z "${CUDA_COMPUTE_CAP:-}" ]]; then
      if command -v nvidia-smi >/dev/null 2>&1; then
        compute_cap=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d '.')
        if [[ -n "${compute_cap}" ]]; then
          export CUDA_COMPUTE_CAP="${compute_cap}"
          echo "  🔧 Setting CUDA_COMPUTE_CAP=${CUDA_COMPUTE_CAP} (detected from nvidia-smi)"
        fi
      fi
    else
      echo "  🔧 Using CUDA_COMPUTE_CAP=${CUDA_COMPUTE_CAP} (from environment)"
    fi
  fi

  if ! eval "${build_cmd}" 2>&1 | tee "${LOG_DIR}/build-${name}.log"; then
    echo "❌ Failed to build ${name} backend. Check ${LOG_DIR}/build-${name}.log for details."
    if [[ "${name}" == "python" ]]; then
      echo "💡 Tip: Python backend requires Rust stable toolchain. Try:"
      echo "   rustup toolchain install stable"
      echo "   rustup default stable"
    fi
    exit 1
  fi

  if [[ -x "${bin_path}" ]]; then
    echo "✅ ${name} binary built successfully at ${bin_path}"
  else
    echo "❌ Binary not found at ${bin_path} after build"
    exit 1
  fi
}

ensure_bin() {
  local name="$1"
  local bin="$2"
  local feature="$3"
  local toolchain="${4:-}"

  if [[ ! -x "${bin}" ]]; then
    echo "📦 Binary for ${name} not found at ${bin}"
    check_rust
    # Check for build dependencies before building
    if [[ "${name}" == "python" ]]; then
      check_pkg_config
      check_protoc
    fi
    build_binary "${name}" "${bin}" "${feature}" "${toolchain}"
  else
    echo "✅ ${name} binary found at ${bin}"

    # For Candle, verify CUDA support in binary
    if [[ "${name}" == "candle" ]]; then
      if command -v ldd >/dev/null 2>&1; then
        if ldd "${bin}" 2>/dev/null | grep -q "libcuda\|libcudart"; then
          echo "   ✅ Binary compiled with CUDA support"
        else
          echo "   ⚠️  Binary NOT compiled with CUDA support (will use CPU)"
        fi
      fi
    fi
  fi
}

# Verify Candle GPU usage
verify_candle_gpu_usage() {
  local bin="$1"
  local log_file="$2"

  echo "  🔍 Verifying Candle GPU usage..."

  # Check 1: Verify binary was built with CUDA support
  if command -v ldd >/dev/null 2>&1; then
    if ldd "${bin}" 2>/dev/null | grep -q "libcuda\|libcudart"; then
      echo "  ✅ Binary has CUDA libraries linked"
    else
      echo "  ⚠️  Binary does NOT have CUDA libraries - using CPU only"
      return
    fi
  fi

  # Check 2: Look for device info in logs
  sleep 2  # Give server time to log device info
  if grep -qi "cuda\|gpu\|device.*cuda" "${log_file}" 2>/dev/null; then
    echo "  ✅ Logs indicate CUDA/GPU usage"
    grep -i "cuda\|gpu\|device" "${log_file}" 2>/dev/null | head -3 | sed 's/^/     /'
  elif grep -qi "using cpu\|device.*cpu" "${log_file}" 2>/dev/null; then
    echo "  ⚠️  Logs indicate CPU usage (GPU not available or not compiled)"
    grep -i "cpu\|device" "${log_file}" 2>/dev/null | head -3 | sed 's/^/     /'
  fi

  # Check 3: Monitor GPU utilization during a test request
  if command -v nvidia-smi >/dev/null 2>&1; then
    echo "  🔍 Testing GPU utilization with a sample request..."
    local port=$(echo "${log_file}" | grep -o "candle" >/dev/null && echo "${CANDLE_PORT:-8081}" || echo "")
    if [[ -n "${port}" ]]; then
      # Get baseline GPU usage
      local baseline=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1 || echo "0")

      # Make a test request
      curl -s -X POST "http://127.0.0.1:${port}/embed" \
        -H "Content-Type: application/json" \
        -d '{"inputs":"test"}' >/dev/null 2>&1 &

      sleep 1
      local gpu_usage=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1 || echo "0")

      if [[ "${gpu_usage}" -gt "${baseline}" ]] && [[ "${gpu_usage}" -gt "5" ]]; then
        echo "  ✅ GPU utilization detected: ${gpu_usage}% (baseline: ${baseline}%)"
      else
        echo "  ⚠️  No significant GPU utilization detected (${gpu_usage}%)"
        echo "     This may indicate CPU-only mode"
      fi
    fi
  fi
}

start_backend() {
  local name="$1"
  local bin="$2"
  local port="$3"
  local extra_args="$4"

  # Double-check port is still available (should be, but verify)
  if ! check_port_available "${port}"; then
    echo "  ❌ Port ${port} is in use. Skipping ${name} backend."
    echo "     To use a different port, set ${name^^}_PORT environment variable"
    return 1
  fi

  echo "🚀 Starting ${name} backend on port ${port}..."
  # shellcheck disable=SC2086
  "${bin}" --model-id "${MODEL_ID}" --port "${port}" ${extra_args} \
    > "${LOG_DIR}/${name}.log" 2>&1 &
  local pid=$!
  echo "${pid}" > "${LOG_DIR}/${name}.pid"
  echo "  ✅ Started (pid=${pid}), log=${LOG_DIR}/${name}.log"

  # Wait longer for server to start (servers need time to load models)
  echo "  ⏳ Waiting for ${name} backend to initialize (this may take 10-30 seconds)..."
  local wait_count=0
  local max_wait=30

  while [ $wait_count -lt $max_wait ]; do
    sleep 1
    wait_count=$((wait_count + 1))

    # Check if process is still running
    if ! kill -0 "${pid}" 2>/dev/null; then
      echo "  ❌ Process died after ${wait_count} seconds. Check ${LOG_DIR}/${name}.log for errors."
      tail -20 "${LOG_DIR}/${name}.log"
      exit 1
    fi

    # Check if health endpoint responds (server is ready)
    if curl -s -f "http://127.0.0.1:${port}/health" >/dev/null 2>&1; then
      echo "  ✅ ${name} backend is ready!"

      # For Candle backend, check if GPU is being used
      if [[ "${name}" == "candle" ]]; then
        verify_candle_gpu_usage "${bin}" "${LOG_DIR}/${name}.log"
      fi

      return 0
    fi

    if [ $((wait_count % 5)) -eq 0 ]; then
      echo "  ⏳ Still waiting... (${wait_count}/${max_wait}s)"
    fi
  done

  echo "  ⚠️  ${name} backend started but health check timed out after ${max_wait}s"
  echo "     Check ${LOG_DIR}/${name}.log - server may still be loading the model"
}

# Find available ports for all backends (auto-increment if needed)
find_all_ports

# Check for CUDA availability to determine Candle feature
# Per Candle installation guide: https://huggingface.github.io/candle/guide/installation.html
# Allow skipping CUDA build if memory is limited
SKIP_CANDLE_CUDA="${SKIP_CANDLE_CUDA:-}"

CANDLE_FEATURE="candle"
if [[ -n "${SKIP_CANDLE_CUDA}" ]]; then
  echo "⚠️  SKIP_CANDLE_CUDA is set - will build Candle for CPU only (to save memory)"
  CANDLE_FEATURE="candle"
elif command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
  echo "✅ CUDA GPU detected - will build Candle with GPU support (candle-cuda)"

  # Verify CUDA installation per Candle guide
  if command -v nvcc >/dev/null 2>&1; then
    echo "  ✅ CUDA compiler (nvcc) found: $(nvcc --version 2>/dev/null | grep -o 'release [0-9.]*' | head -1 || echo 'version check failed')"
  else
    echo "  ⚠️  CUDA compiler (nvcc) not found in PATH"
    echo "     Make sure CUDA is installed and /usr/local/cuda/bin is in PATH"
  fi

  # Check compute capability
  compute_cap=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits 2>/dev/null | head -1)
  if [[ -n "${compute_cap}" ]]; then
    echo "  ✅ GPU compute capability: ${compute_cap}"
  else
    echo "  ⚠️  Could not determine GPU compute capability"
  fi

  CANDLE_FEATURE="candle-cuda"
else
  echo "⚠️  No CUDA GPU detected - will build Candle for CPU only"
fi

# Ensure all binaries exist (build if needed)
# Build sequentially to avoid memory pressure
echo "🔍 Checking for required binaries..."
echo "📦 Building backends sequentially to reduce memory usage..."

# Build ORT first (CPU-only, less memory intensive)
if [[ ! -x "${ORT_BIN}" ]]; then
  echo ""
  echo "🔨 [1/3] Building ORT backend..."
  ensure_bin "ort" "${ORT_BIN}" "ort"
else
  echo "✅ [1/3] ORT binary already exists"
fi

# Build Candle (most memory intensive - build separately)
if [[ ! -x "${CANDLE_BIN}" ]]; then
  echo ""
  echo "🔨 [2/3] Building Candle backend (this is the most memory-intensive step)..."
  echo "   💡 Tip: If this fails, you can skip Candle and use only ORT/Python backends"
  ensure_bin "candle" "${CANDLE_BIN}" "${CANDLE_FEATURE}"
else
  echo "✅ [2/3] Candle binary already exists"
fi

# Build Python last
if [[ ! -x "${PY_BIN}" ]]; then
  echo ""
  echo "🔨 [3/3] Building Python backend..."
  ensure_bin "python" "${PY_BIN}" "python" "stable"
else
  echo "✅ [3/3] Python binary already exists"
fi

# Start all backends
echo ""
echo "🚀 Starting all backends..."
failed=0
start_backend "ort" "${ORT_BIN}" "${ORT_PORT}" "${ORT_ARGS}" || failed=$((failed + 1))
start_backend "candle" "${CANDLE_BIN}" "${CANDLE_PORT}" "${CANDLE_ARGS}" || failed=$((failed + 1))
start_backend "python" "${PY_BIN}" "${PY_PORT}" "${PY_ARGS}" || failed=$((failed + 1))

if [[ $failed -gt 0 ]]; then
  echo ""
  echo "⚠️  Warning: $failed backend(s) failed to start. Check logs in ${LOG_DIR}/"
fi

echo ""
echo "✅ All backends started! Check health endpoints:"
echo "  curl -s http://127.0.0.1:${ORT_PORT}/health"
echo "  curl -s http://127.0.0.1:${CANDLE_PORT}/health"
echo "  curl -s http://127.0.0.1:${PY_PORT}/health"
echo ""
echo "📋 Logs are in ${LOG_DIR}/"
echo "🛑 To stop all backends: pkill -f 'text-embeddings-router'"