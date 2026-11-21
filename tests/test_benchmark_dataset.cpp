// Tests for BenchmarkDataset (include/utility.h).
//
// Synthetic tests use a tiny fixture with the same schema as the HDF5
// converter output: metadata.json + base.f32 + queries.f32 + neighbors.i32
// + distances.f32. Binary arrays are raw, little-endian, C-order data.
//
// By default this executable runs only the synthetic tests and is suitable
// for CTest. To additionally run the real GloVe-200-angular smoke test, pass:
//   test_benchmark_dataset --glove <path-to-processed-dataset-directory>
// Example:
//   ./build/test_benchmark_dataset --glove data/processed/glove-200-angular

#include "utility.h"
#include "similarity.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

int failures = 0;

void check(bool condition, const std::string& message) {
    if (condition) {
        std::cout << "[PASS] " << message << '\n';
    } else {
        std::cerr << "[FAIL] " << message << '\n';
        ++failures;
    }
}

template <typename Function>
void expect_exception(Function&& function, const std::string& message) {
    try {
        function();
        check(false, message);
    } catch (const std::exception&) {
        check(true, message);
    }
}

bool approximately_equal(float a, float b, float tolerance = 1e-5f) {
    return std::abs(a - b) <= tolerance;
}

// --------------------------------------------------
// Synthetic fixture
// --------------------------------------------------
// Five database vectors of dimension 2, three queries, ground-truth K = 2.
// Comparisons use a tolerance because values such as 0.2 are not represented
// exactly in IEEE-754 float32.
//
// Database (raw)      Normalized          cosine with [1,0]
//   [ 3, 4]      ->   [ 0.6, 0.8]          0.6
//   [ 4, 3]      ->   [ 0.8, 0.6]          0.8
//   [ 0, 2]      ->   [ 0,   1  ]          0.0
//   [-1, 0]      ->   [-1,   0  ]         -1.0
//   [ 1, 0]      ->   [ 1,   0  ]          1.0
//
// Queries: [1,0], [0,1], [2,0]
// Ground truth (angular distance = 1 - cosine, ascending):
//   query 0 -> neighbors {4, 1}, distances {0.0, 0.2}
//   query 1 -> neighbors {2, 0}, distances {0.0, 0.2}
//   query 2 -> neighbors {4, 1}, distances {0.0, 0.2}

const std::size_t FIXTURE_DIMENSIONS = 2;
const std::size_t FIXTURE_NUM_QUERIES = 3;
const std::size_t FIXTURE_K = 2;
const std::size_t FIXTURE_NUM_VECTORS = 5;

const std::vector<float> FIXTURE_DATABASE = {
     3.0f, 4.0f,
     4.0f, 3.0f,
     0.0f, 2.0f,
    -1.0f, 0.0f,
     1.0f, 0.0f
};

const std::vector<float> FIXTURE_QUERIES = {
    1.0f, 0.0f,
    0.0f, 1.0f,
    2.0f, 0.0f
};

const std::vector<std::int32_t> FIXTURE_NEIGHBORS = {
    4, 1,
    2, 0,
    4, 1
};

const std::vector<float> FIXTURE_DISTANCES = {
    0.0f, 0.2f,
    0.0f, 0.2f,
    0.0f, 0.2f
};

template <typename T>
void write_binary(
    const std::filesystem::path& path,
    const std::vector<T>& values
) {
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    if (!file.is_open()) {
        throw std::runtime_error("test fixture: cannot write binary file: " + path.string());
    }

    file.write(
        reinterpret_cast<const char*>(values.data()),
        static_cast<std::streamsize>(values.size() * sizeof(T))
    );

    if (!file) {
        throw std::runtime_error("test fixture: failed writing binary file: " + path.string());
    }
}

std::filesystem::path fixture_directory() {
    return std::filesystem::temp_directory_path() / "cudavec_test_fixture";
}

// Writes the fixture files in the converter's schema.
// Options allow tests to create malformed metadata or invalid neighbor IDs.
void write_fixture(
    const std::filesystem::path& directory,
    bool truncate_database = false,
    std::size_t metadata_query_count = FIXTURE_NUM_QUERIES,
    bool invalid_neighbor_index = false
) {
    std::filesystem::remove_all(directory);
    std::filesystem::create_directories(directory);

    std::vector<float> database = FIXTURE_DATABASE;
    if (truncate_database) {
        database.pop_back();
    }

    std::vector<std::int32_t> neighbors = FIXTURE_NEIGHBORS;
    if (invalid_neighbor_index) {
        // There are five vectors, so index 5 is out of range.
        neighbors.back() = static_cast<std::int32_t>(FIXTURE_NUM_VECTORS);
    }

    write_binary(directory / "base.f32", database);
    write_binary(directory / "queries.f32", FIXTURE_QUERIES);
    write_binary(directory / "neighbors.i32", neighbors);
    write_binary(directory / "distances.f32", FIXTURE_DISTANCES);

    std::string metadata = R"({
    "dataset": "synthetic-test-fixture",
    "database_size": 5,
    "query_count": 3,
    "dimensions": 2,
    "ground_truth_k": 2,
    "files": {
        "train": {
            "filename": "base.f32",
            "shape": [5, 2],
            "dtype": "float32",
            "byte_order": "little-endian",
            "order": "C",
            "elements": 10,
            "size_bytes": 40
        },
        "test": {
            "filename": "queries.f32",
            "shape": [3, 2],
            "dtype": "float32",
            "byte_order": "little-endian",
            "order": "C",
            "elements": 6,
            "size_bytes": 24
        },
        "neighbors": {
            "filename": "neighbors.i32",
            "shape": [3, 2],
            "dtype": "int32",
            "byte_order": "little-endian",
            "order": "C",
            "elements": 6,
            "size_bytes": 24
        },
        "distances": {
            "filename": "distances.f32",
            "shape": [3, 2],
            "dtype": "float32",
            "byte_order": "little-endian",
            "order": "C",
            "elements": 6,
            "size_bytes": 24
        }
    }
})";

    // Substitute the metadata query count without depending on a JSON writer
    // in the test fixture itself.
    const std::string query_count_marker = "\"query_count\": 3";
    const std::size_t marker_position = metadata.find(query_count_marker);
    if (marker_position == std::string::npos) {
        throw std::runtime_error("test fixture: query_count marker not found");
    }
    metadata.replace(
        marker_position,
        query_count_marker.size(),
        "\"query_count\": " + std::to_string(metadata_query_count)
    );

    std::ofstream file(directory / "metadata.json", std::ios::trunc);
    if (!file.is_open()) {
        throw std::runtime_error("test fixture: cannot write metadata");
    }
    file << metadata;
    if (!file) {
        throw std::runtime_error("test fixture: failed writing metadata");
    }
}

void remove_fixture(const std::filesystem::path& directory) {
    std::error_code ignored;
    std::filesystem::remove_all(directory, ignored);
}

// --------------------------------------------------
// Constructor and metadata
// --------------------------------------------------

void test_constructor_rejects_missing_directory() {
    expect_exception(
        [] { BenchmarkDataset dataset("/nonexistent/cudavec/path"); },
        "Constructor rejects a directory that does not exist"
    );
}

void test_constructor_rejects_file_path() {
    const auto directory = fixture_directory();
    write_fixture(directory);

    expect_exception(
        [&] {
            BenchmarkDataset dataset((directory / "metadata.json").string());
        },
        "Constructor rejects a path that is a file, not a directory"
    );

    remove_fixture(directory);
}

void test_constructor_rejects_missing_metadata() {
    const auto directory = fixture_directory();
    std::filesystem::remove_all(directory);
    std::filesystem::create_directories(directory);

    expect_exception(
        [&] { BenchmarkDataset dataset(directory.string()); },
        "Constructor rejects a directory without metadata.json"
    );

    remove_fixture(directory);
}

void test_constructor_accepts_valid_fixture() {
    const auto directory = fixture_directory();
    write_fixture(directory);

    bool constructed = true;
    try {
        BenchmarkDataset dataset(directory.string());
    } catch (const std::exception&) {
        constructed = false;
    }

    check(constructed, "Constructor accepts a valid dataset directory");
    remove_fixture(directory);
}

void test_inconsistent_query_count_rejected() {
    const auto directory = fixture_directory();
    // The query file still contains 3 queries, but metadata claims 4.
    write_fixture(directory, false, FIXTURE_NUM_QUERIES + 1);

    expect_exception(
        [&] { BenchmarkDataset dataset(directory.string()); },
        "Loader rejects metadata query_count inconsistent with queries.f32"
    );

    remove_fixture(directory);
}

// --------------------------------------------------
// Getter basics and bounds
// --------------------------------------------------

void test_scalar_getters() {
    const auto directory = fixture_directory();
    write_fixture(directory);

    BenchmarkDataset dataset(directory.string());

    check(
        dataset.get_dimensions() == FIXTURE_DIMENSIONS,
        "get_dimensions returns the metadata dimension count"
    );
    check(
        dataset.get_num_queries() == FIXTURE_NUM_QUERIES,
        "get_num_queries returns the metadata query count"
    );
    check(
        dataset.get_ground_truth_k() == FIXTURE_K,
        "get_ground_truth_k returns the metadata ground truth K"
    );

    remove_fixture(directory);
}

void test_query_out_of_bounds_rejected() {
    const auto directory = fixture_directory();
    write_fixture(directory);
    BenchmarkDataset dataset(directory.string());

    expect_exception(
        [&] { (void)dataset.get_query(FIXTURE_NUM_QUERIES); },
        "get_query rejects index equal to query count"
    );

    remove_fixture(directory);
}

void test_ground_truth_out_of_bounds_rejected() {
    const auto directory = fixture_directory();
    write_fixture(directory);
    BenchmarkDataset dataset(directory.string());

    expect_exception(
        [&] { (void)dataset.get_ground_truth(FIXTURE_NUM_QUERIES); },
        "get_ground_truth rejects index equal to query count"
    );

    remove_fixture(directory);
}

void test_distance_ground_truth_out_of_bounds_rejected() {
    const auto directory = fixture_directory();
    write_fixture(directory);
    BenchmarkDataset dataset(directory.string());

    expect_exception(
        [&] { (void)dataset.get_distance_ground_truth(FIXTURE_NUM_QUERIES); },
        "get_distance_ground_truth rejects index equal to query count"
    );

    remove_fixture(directory);
}

// --------------------------------------------------
// Database loading
// --------------------------------------------------

void test_database_size_and_dimensions() {
    const auto directory = fixture_directory();
    write_fixture(directory);

    BenchmarkDataset dataset(directory.string());

    check(
        dataset.get_database().size() == FIXTURE_NUM_VECTORS,
        "Loaded database contains every vector from base.f32"
    );
    check(
        dataset.get_database().get_dimensions() == FIXTURE_DIMENSIONS,
        "Loaded database has the fixture dimensions"
    );

    remove_fixture(directory);
}

void test_database_vectors_normalized() {
    const auto directory = fixture_directory();
    write_fixture(directory);

    BenchmarkDataset dataset(directory.string());
    const auto& data = dataset.get_database().data();

    check(
        approximately_equal(data[0], 0.6f) &&
        approximately_equal(data[1], 0.8f),
        "First database vector is normalized"
    );
    check(
        approximately_equal(data[8], 1.0f) &&
        approximately_equal(data[9], 0.0f),
        "Last database vector is normalized"
    );

    remove_fixture(directory);
}

// --------------------------------------------------
// Query loading
// --------------------------------------------------

void test_queries_loaded_raw() {
    const auto directory = fixture_directory();
    write_fixture(directory);

    BenchmarkDataset dataset(directory.string());
    bool correct = true;

    for (std::size_t i = 0; i < FIXTURE_NUM_QUERIES; ++i) {
        const auto query = dataset.get_query(i);
        for (std::size_t j = 0; j < FIXTURE_DIMENSIONS; ++j) {
            const float expected = FIXTURE_QUERIES[i * FIXTURE_DIMENSIONS + j];
            if (!approximately_equal(
                    *std::next(query, static_cast<std::ptrdiff_t>(j)),
                    expected)) {
                correct = false;
            }
        }
    }

    check(correct, "Every query is loaded with its raw values from queries.f32");
    remove_fixture(directory);
}

// --------------------------------------------------
// Ground truth loading
// --------------------------------------------------

void test_ground_truth_neighbors() {
    const auto directory = fixture_directory();
    write_fixture(directory);

    BenchmarkDataset dataset(directory.string());
    bool correct = true;

    for (std::size_t i = 0; i < FIXTURE_NUM_QUERIES; ++i) {
        auto neighbors = dataset.get_ground_truth(i);
        for (std::size_t j = 0; j < FIXTURE_K; ++j) {
            const std::int32_t expected = FIXTURE_NEIGHBORS[i * FIXTURE_K + j];
            if (*std::next(neighbors, static_cast<std::ptrdiff_t>(j)) != expected) {
                correct = false;
            }
        }
    }

    check(correct, "Ground truth neighbors match neighbors.i32 for every query");
    remove_fixture(directory);
}

void test_ground_truth_distances() {
    const auto directory = fixture_directory();
    write_fixture(directory);

    BenchmarkDataset dataset(directory.string());
    bool correct = true;

    for (std::size_t i = 0; i < FIXTURE_NUM_QUERIES; ++i) {
        const auto distances = dataset.get_distance_ground_truth(i);
        for (std::size_t j = 0; j < FIXTURE_K; ++j) {
            const float expected = FIXTURE_DISTANCES[i * FIXTURE_K + j];
            if (!approximately_equal(
                    *std::next(distances, static_cast<std::ptrdiff_t>(j)),
                    expected)) {
                correct = false;
            }
        }
    }

    check(correct, "Ground truth distances match distances.f32 for every query");
    remove_fixture(directory);
}

void test_invalid_ground_truth_neighbor_rejected() {
    const auto directory = fixture_directory();
    write_fixture(directory, false, FIXTURE_NUM_QUERIES, true);

    expect_exception(
        [&] { BenchmarkDataset dataset(directory.string()); },
        "Loader rejects a ground-truth neighbor index outside the database"
    );

    remove_fixture(directory);
}

// --------------------------------------------------
// Corrupt input
// --------------------------------------------------

void test_truncated_database_rejected() {
    const auto directory = fixture_directory();
    write_fixture(directory, true);

    expect_exception(
        [&] { BenchmarkDataset dataset(directory.string()); },
        "Truncated base.f32 is rejected"
    );

    remove_fixture(directory);
}

// --------------------------------------------------
// Loader agrees with CPU search
// --------------------------------------------------

void test_search_matches_ground_truth() {
    const auto directory = fixture_directory();
    write_fixture(directory);

    BenchmarkDataset dataset(directory.string());
    bool correct = true;

    for (std::size_t i = 0; i < FIXTURE_NUM_QUERIES; ++i) {
        const auto query_begin = dataset.get_query(i);
        const std::vector<float> query(
            query_begin,
            std::next(query_begin, static_cast<std::ptrdiff_t>(FIXTURE_DIMENSIONS))
        );

        const auto results = topK_search(query, dataset.get_database(), FIXTURE_K);
        if (results.size() != FIXTURE_K) {
            correct = false;
            break;
        }

        const auto neighbors = dataset.get_ground_truth(i);
        for (std::size_t j = 0; j < FIXTURE_K; ++j) {
            const std::int32_t expected =
                *std::next(neighbors, static_cast<std::ptrdiff_t>(j));
            if (results[j].index != static_cast<std::size_t>(expected)) {
                correct = false;
            }
        }
    }

    check(correct, "topK_search on the loaded fixture reproduces ground truth");
    remove_fixture(directory);
}

// --------------------------------------------------
// Real GloVe-200-angular smoke test (optional)
// --------------------------------------------------

void test_real_glove_dataset(const std::filesystem::path& dataset_path) {
    std::cout << "\n===== Real GloVe Dataset Smoke Test =====\n";
    std::cout << "Dataset: " << dataset_path << '\n';

    BenchmarkDataset dataset(dataset_path.string());
    const VecDB& database = dataset.get_database();

    check(dataset.get_dimensions() == 200, "GloVe dimensions are 200");
    check(dataset.get_num_queries() == 10000, "GloVe contains 10,000 queries");
    check(dataset.get_ground_truth_k() == 100, "GloVe ground-truth K is 100");
    check(database.size() == 1183514, "GloVe contains 1,183,514 database vectors");
    check(database.data().size() == 1183514ULL * 200ULL,
          "GloVe database contains the expected number of float values");

    // Check all one million ground-truth IDs and distances. This is a cheap
    // validation compared with exhaustive search over the full database.
    bool ground_truth_valid = true;
    bool distances_finite = true;

    for (std::size_t query_index = 0;
         query_index < dataset.get_num_queries();
         ++query_index) {
        auto neighbor = dataset.get_ground_truth(query_index);
        auto distance = dataset.get_distance_ground_truth(query_index);

        for (std::size_t rank = 0; rank < dataset.get_ground_truth_k(); ++rank) {
            const std::int32_t id = *neighbor++;
            const float value = *distance++;

            if (id < 0 || static_cast<std::size_t>(id) >= database.size()) {
                ground_truth_valid = false;
            }
            if (!std::isfinite(value)) {
                distances_finite = false;
            }
        }
    }

    check(ground_truth_valid, "All GloVe ground-truth neighbor IDs are in range");
    check(distances_finite, "All GloVe ground-truth distances are finite");

    // Run a small deterministic sample. This is deliberately not the full
    // 10,000-query benchmark; it verifies end-to-end search against the real
    // ground truth without making normal test runs excessively long.
    const std::vector<std::size_t> sampled_queries = {0, 5000, 9999};
    constexpr std::size_t EVALUATION_K = 10;
    constexpr double MIN_AVERAGE_RECALL_AT_10 = 0.90;

    double recall_sum = 0.0;
    std::size_t evaluated_queries = 0;

    for (const std::size_t query_index : sampled_queries) {
        const auto query_begin = dataset.get_query(query_index);
        const std::vector<float> query(
            query_begin,
            query_begin + static_cast<std::ptrdiff_t>(dataset.get_dimensions())
        );

        const auto results = topK_search(query, database, EVALUATION_K);
        if (results.size() != EVALUATION_K) {
            check(false, "GloVe query returns 10 search results");
            continue;
        }

        std::unordered_set<std::int32_t> expected_top10;
        auto gt = dataset.get_ground_truth(query_index);
        for (std::size_t i = 0; i < EVALUATION_K; ++i) {
            expected_top10.insert(*gt++);
        }

        std::size_t matches = 0;
        for (const SearchResult& result : results) {
            if (expected_top10.count(static_cast<std::int32_t>(result.index)) != 0) {
                ++matches;
            }
        }

        const double recall = expected_top10.empty()
            ? 0.0
            : static_cast<double>(matches) / expected_top10.size();

        std::cout << "Query " << query_index
                  << " Recall@10: " << std::fixed << std::setprecision(3)
                  << recall << '\n';

        recall_sum += recall;
        ++evaluated_queries;
    }

    const double average_recall = evaluated_queries == 0
        ? 0.0
        : recall_sum / static_cast<double>(evaluated_queries);

    std::cout << "Average sampled Recall@10: "
              << std::fixed << std::setprecision(3)
              << average_recall << '\n';

    check(
        evaluated_queries == sampled_queries.size(),
        "All sampled GloVe queries completed"
    );
    check(
        average_recall >= MIN_AVERAGE_RECALL_AT_10,
        "Sampled GloVe average Recall@10 is at least 0.90"
    );
}

// --------------------------------------------------
// Main
// --------------------------------------------------

int main(int argc, char* argv[]) {
    std::cout << "\n===== CudaVec: Benchmark Dataset Tests =====\n\n";

    if (argc != 1 && (argc != 3 || std::string(argv[1]) != "--glove")) {
        std::cerr << "Usage: " << argv[0]
                  << " [--glove <processed-dataset-directory>]\n";
        return 2;
    }

    try {
        test_constructor_rejects_missing_directory();
        test_constructor_rejects_file_path();
        test_constructor_rejects_missing_metadata();
        test_constructor_accepts_valid_fixture();
        test_inconsistent_query_count_rejected();

        test_scalar_getters();
        test_query_out_of_bounds_rejected();
        test_ground_truth_out_of_bounds_rejected();
        test_distance_ground_truth_out_of_bounds_rejected();

        test_database_size_and_dimensions();
        test_database_vectors_normalized();
        test_queries_loaded_raw();
        test_ground_truth_neighbors();
        test_ground_truth_distances();
        test_invalid_ground_truth_neighbor_rejected();
        test_truncated_database_rejected();
        test_search_matches_ground_truth();

        if (argc == 3) {
            test_real_glove_dataset(argv[2]);
        } else {
            std::cout << "\n[SKIP] Real GloVe smoke test. Pass --glove <dataset-directory> to run it.\n";
        }
    } catch (const std::exception& error) {
        std::cerr << "\n[FATAL] Unhandled test exception: " << error.what() << '\n';
        ++failures;
    }

    std::cout << "\n===== Test Summary =====\n";
    std::cout << "Failures: " << failures << '\n';

    if (failures == 0) {
        std::cout << "All tests passed!\n";
        return 0;
    }

    return 1;
}
