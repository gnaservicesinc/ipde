#!/usr/bin/env bash
set -euo pipefail

# One isolated GPU workload. Compare the same grid/scope on the same Mac;
# the first update includes compilation, the second reuses frozen features,
# and subsequent updates change the input. No user dataset is modified.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BENCHMARK_DIR="${TEXTURE_STUDIO_BENCHMARK_BUILD:-$ROOT_DIR/build/TrainingThroughputValidation}"
REPORT="${TEXTURE_STUDIO_THROUGHPUT_REPORT:-$ROOT_DIR/out/training-throughput-audit/benchmark.json}"
mkdir -p "$(dirname "$REPORT")"
ARGS=(-project "$ROOT_DIR/src/TextureStudio/TextureStudio.xcodeproj" -scheme TextureStudio
  -configuration Release -destination 'platform=macOS,arch=arm64' -derivedDataPath "$BENCHMARK_DIR"
  -parallel-testing-enabled NO ENABLE_TESTABILITY=YES ENABLE_HARDENED_RUNTIME=NO)
if [[ "${1:-}" != --no-build ]]; then
  xcodebuild "${ARGS[@]}" build-for-testing > "$REPORT.build.log" 2>&1
fi
if [[ -n "${TEXTURE_STUDIO_PROGRAM_CACHE_DIRECTORY:-}" ]]; then
  export TEST_RUNNER_TEXTURE_STUDIO_PROGRAM_CACHE_DIRECTORY="$TEXTURE_STUDIO_PROGRAM_CACHE_DIRECTORY"
fi
export TEST_RUNNER_TEXTURE_STUDIO_DISABLE_PROGRAM_CACHE="${TEXTURE_STUDIO_DISABLE_PROGRAM_CACHE:-0}"
export TEST_RUNNER_TEXTURE_STUDIO_COALESCE_ACTIVE_BLOCKS="${TEXTURE_STUDIO_COALESCE_ACTIVE_BLOCKS:-1}"
export TEST_RUNNER_TEXTURE_STUDIO_THROUGHPUT_RANK="${TEXTURE_STUDIO_THROUGHPUT_RANK:-64}"
export TEST_RUNNER_TEXTURE_STUDIO_THROUGHPUT_ALPHA="${TEXTURE_STUDIO_THROUGHPUT_ALPHA:-16}"
TEST_RUNNER_TEXTURE_STUDIO_THROUGHPUT_BENCHMARK=1 \
TEST_RUNNER_TEXTURE_STUDIO_THROUGHPUT_REPORT="$REPORT" \
TEST_RUNNER_TEXTURE_STUDIO_THROUGHPUT_SIZE="${TEXTURE_STUDIO_THROUGHPUT_SIZE:-1024}" \
TEST_RUNNER_TEXTURE_STUDIO_THROUGHPUT_STEPS="${TEXTURE_STUDIO_THROUGHPUT_STEPS:-3}" \
TEST_RUNNER_TEXTURE_STUDIO_THROUGHPUT_SCOPE="${TEXTURE_STUDIO_THROUGHPUT_SCOPE:-map-decoder}" \
  /usr/bin/time -l xcodebuild "${ARGS[@]}" \
    -only-testing:TextureStudioTests/NativeMaterialModelTests/testInstalledModelTrainingThroughput \
    test-without-building > "$REPORT.log" 2>&1
cat "$REPORT"
