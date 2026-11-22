# CudaVec

**A CUDA-accelerated vector search engine written in C++.**

CudaVec is a vector search engine project focused on implementing and optimizing similarity search using CUDA. The project explores GPU-accelerated vector operations, efficient Top-K retrieval, and performance evaluation against a CPU baseline.

The project is currently under development. The CPU search implementation, benchmark dataset loader, and evaluation pipeline are implemented. CUDA search acceleration is the next development milestone.

## Current Features

### CPU Vector Search

- Flat, row-major vector storage.
- L2 normalization of database vectors.
- Cosine-similarity-based exhaustive search.
- Exact Top-K similarity search.
- Deterministic result ordering, with smaller vector indices preferred when similarities are equal.

### Dataset Support

CudaVec currently supports the GloVe-200 Angular dataset from ANN-Benchmarks.

| Property | Value |
|---|---:|
| Database vectors | 1,183,514 |
| Query vectors | 10,000 |
| Dimensions | 200 |
| Ground-truth neighbours per query | 100 |
| Distance metric | Angular |

The dataset loader handles binary vector files, metadata validation, ground-truth loading, and consistency checks.

### Benchmarking

The CPU benchmark evaluates:

- Recall@K
- Average search latency
- Median search latency
- P95 search latency
- Maximum search latency
- Search throughput (queries per second)

Search execution time is measured separately from query preparation and recall calculation.

## Project Structure

```text
CudaVec/
├── include/
│   ├── vector_dataset.h
│   ├── similarity.h
│   └── utility.h
├── src/
│   ├── vector_dataset.cpp
│   ├── similarity.cpp
│   ├── utility.cpp
│   └── cpu_search.cpp
├── tests/
│   ├── test_vec_db.cpp
│   ├── test_similarity.cpp
│   └── test_benchmark_dataset.cpp
├── data/
│   ├── raw/
│   └── processed/
│       └── glove-200-angular/
├── CMakeLists.txt
└── README.md
```

## Requirements

- C++17-compatible compiler
- CMake 3.18 or newer
- Git

The current CPU benchmark does not require CUDA. The build configuration fetches the nlohmann/json dependency through CMake FetchContent.

## Build

Clone the repository:

```bash
git clone https://github.com/Sobhyz/CudaVec.git
cd CudaVec
```

Configure a Release build:

```bash
cmake -S . -B build-release -DCMAKE_BUILD_TYPE=Release
```

Compile the project:

```bash
cmake --build build-release -j$(nproc)
```

## Dataset Setup

Place the processed GloVe-200 Angular dataset in:

```text
data/processed/glove-200-angular/
```

The processed dataset directory should contain:

```text
glove-200-angular/
├── base.f32
├── queries.f32
├── neighbors.i32
├── distances.f32
└── metadata.json
```

The binary files contain the database vectors, queries, ground-truth neighbour IDs, and ground-truth distances. Dataset metadata describes the dimensions, query count, ground-truth K, and file information.

Dataset files are kept outside version control.

## Run the CPU Benchmark

Run with the default K value of 10:

```bash
./build-release/cpu_search \
    --path data/processed/glove-200-angular
```

Specify a different K:

```bash
./build-release/cpu_search \
    --path data/processed/glove-200-angular \
    -k 10
```

The benchmark performs warm-up searches before collecting measurements. Warm-up executions are excluded from the reported statistics.

## Run Tests

Build all test executables:

```bash
cmake --build build-release -j$(nproc)
```

Run the registered CTest tests:

```bash
ctest --test-dir build-release --output-on-failure
```

Run the benchmark dataset tests with the real GloVe dataset:

```bash
./build-release/test_benchmark_dataset \
    --glove data/processed/glove-200-angular
```

## CPU Benchmark Results

Reference run using the GloVe-200 Angular dataset, with K = 10 and 10,000 queries.

| Metric | Result |
|---|---:|
| Queries evaluated | 10,000 |
| Recall@10 | 1.0 |
| Average latency | 212.257 ms |
| Median latency | 211.612 ms |
| P95 latency | 215.923 ms |
| Maximum latency | 240.330 ms |
| Throughput | 4.71128 queries/s |

These results represent one Release-mode CPU benchmark run. Performance depends on the CPU, compiler, system configuration, and build environment. The hardware specifications for this reference run have not yet been recorded.

Latency measurements cover the `topK_search()` execution itself and exclude dataset loading, query preparation, and recall calculation.

## Roadmap

- [x] Flat vector database implementation
- [x] Vector normalization
- [x] Exact CPU similarity search
- [x] CPU Top-K search
- [x] Benchmark dataset loader
- [x] Ground-truth validation
- [x] Recall@K evaluation
- [x] CPU latency and throughput benchmark
- [ ] CUDA vector similarity kernels
- [ ] GPU-accelerated Top-K search
- [ ] CPU versus GPU performance evaluation
- [ ] CUDA performance optimization

## License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.