
#include "similarity.h"

#include <iostream>
#include <vector>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <algorithm>

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
    }
    catch (const std::exception&) {
        check(true, message);
    }
}

bool approximately_equal(
    double a,
    double b,
    double tolerance = 1e-6
) {
    return std::abs(a - b) <= tolerance;
}

// A small database with manually calculable similarities.
// Query direction: [1, 0]
VecDB create_test_database() {
    VecDB db(2);

    db.add_vector({0.0f, 1.0f});  // Index 0: similarity  0.0
    db.add_vector({3.0f, 4.0f});  // Index 1: similarity  0.6
    db.add_vector({-1.0f, 0.0f}); // Index 2: similarity -1.0
    db.add_vector({1.0f, 0.0f});  // Index 3: similarity  1.0
    db.add_vector({4.0f, 3.0f});  // Index 4: similarity  0.8

    return db;
}

// --------------------------------------------------
// Dot product
// --------------------------------------------------

void test_dot_product() {
    const std::vector<float> query = {1.0f, 2.0f, 3.0f};
    const std::vector<float> vector = {4.0f, 5.0f, 6.0f};

    double result = dot_product(
        query.cbegin(),
        vector.cbegin(),
        vector.cend()
    );

    check(
        approximately_equal(result, 32.0),
        "Dot product calculates the correct result"
    );
}

// --------------------------------------------------
// Complete search
// --------------------------------------------------

void test_complete_search_scores() {
    VecDB db = create_test_database();

    const std::vector<float> query = {2.0f, 0.0f};

    const std::vector<double> expected = {
        0.0, 0.6, -1.0, 1.0, 0.8
    };

    const auto results = complete_search(query, db);

    check(
        results.size() == expected.size(),
        "Complete search returns all database vectors"
    );

    if (results.size() == expected.size()) {
        bool correct = true;

        for (std::size_t i = 0; i < results.size(); ++i) {
            if (results[i].index != i ||
                !approximately_equal(
                    results[i].similarity,
                    expected[i]
                )) {
                correct = false;
                break;
            }
        }

        check(
            correct,
            "Complete search returns correct scores and indices"
        );
    }
}

void test_complete_search_empty_database() {
    VecDB db(2);

    const std::vector<float> query = {1.0f, 0.0f};

    const auto results = complete_search(query, db);

    check(
        results.empty(),
        "Complete search handles an empty database"
    );
}

void test_complete_search_invalid_dimensions() {
    VecDB db(2);
    db.add_vector({1.0f, 0.0f});

    const std::vector<float> query = {1.0f, 2.0f, 3.0f};

    expect_exception(
        [&] { complete_search(query, db); },
        "Complete search rejects incorrect query dimensions"
    );
}

void test_complete_search_zero_query() {
    VecDB db(2);
    db.add_vector({1.0f, 0.0f});

    const std::vector<float> query = {0.0f, 0.0f};

    expect_exception(
        [&] { complete_search(query, db); },
        "Complete search rejects a zero query"
    );
}

void test_complete_search_does_not_modify_inputs() {
    VecDB db = create_test_database();

    const std::vector<float> query = {3.0f, 4.0f};

    const auto original_query = query;
    const auto original_database = db.data();

    complete_search(query, db);

    check(
        query == original_query,
        "Complete search does not modify the original query"
    );

    check(
        db.data() == original_database,
        "Complete search does not modify the database"
    );
}

// --------------------------------------------------
// Top-K search
// --------------------------------------------------

void test_topk_scores_and_order() {
    VecDB db = create_test_database();

    const std::vector<float> query = {1.0f, 0.0f};

    const auto results = topK_search(query, db, 3);

    const std::vector<std::size_t> expected_indices = {
        3, 4, 1
    };

    const std::vector<double> expected_scores = {
        1.0, 0.8, 0.6
    };

    check(
        results.size() == 3,
        "Top-K returns exactly K results"
    );

    if (results.size() == 3) {
        bool correct = true;

        for (std::size_t i = 0; i < results.size(); ++i) {
            if (results[i].index != expected_indices[i] ||
                !approximately_equal(
                    results[i].similarity,
                    expected_scores[i]
                )) {
                correct = false;
                break;
            }
        }

        check(
            correct,
            "Top-K returns the highest similarities in descending order"
        );
    }
}

void test_topk_k_equals_one() {
    VecDB db = create_test_database();

    const std::vector<float> query = {1.0f, 0.0f};

    const auto results = topK_search(query, db, 1);

    check(
        results.size() == 1 && results[0].index == 3,
        "Top-K correctly handles K = 1"
    );
}

void test_topk_k_zero() {
    VecDB db = create_test_database();

    const std::vector<float> query = {1.0f, 0.0f};

    const auto results = topK_search(query, db, 0);

    check(
        results.empty(),
        "Top-K returns an empty result for K = 0"
    );
}

void test_topk_k_greater_than_database_size() {
    VecDB db = create_test_database();

    const std::vector<float> query = {1.0f, 0.0f};

    const auto results = topK_search(query, db, 100);

    check(
        results.size() == db.size(),
        "Top-K returns all vectors when K exceeds database size"
    );

    if (results.size() == db.size()) {
        check(
            results.front().index == 3 &&
            results.back().index == 2,
            "Top-K still sorts results when K exceeds database size"
        );
    }
}

void test_topk_tie_breaking() {
    VecDB db(2);

    db.add_vector({3.0f, 4.0f}); // Index 0: similarity 0.6
    db.add_vector({6.0f, 8.0f}); // Index 1: similarity 0.6
    db.add_vector({0.0f, 1.0f}); // Index 2: similarity 0.0

    const std::vector<float> query = {1.0f, 0.0f};

    const auto results = topK_search(query, db, 1);

    check(
        results.size() == 1 && results[0].index == 0,
        "Top-K prefers the smaller index when similarities tie"
    );

    const auto top_two = topK_search(query, db, 2);

    check(
        top_two.size() == 2 &&
        top_two[0].index == 0 &&
        top_two[1].index == 1,
        "Top-K orders tied results by ascending index"
    );
}

void test_topk_matches_exhaustive_search() {
    VecDB db = create_test_database();

    const std::vector<float> query = {1.0f, 0.0f};

    auto exhaustive_results = complete_search(query, db);

    auto ranking = [](const SearchResult& a, const SearchResult& b) {
        if (a.similarity != b.similarity) {
            return a.similarity > b.similarity;
        }
        return a.index < b.index;
    };

    std::sort(
        exhaustive_results.begin(),
        exhaustive_results.end(),
        ranking
    );

    const std::vector<std::size_t> k_values = {1, 2, 3, 5, 10};

    bool all_correct = true;

    for (std::size_t k : k_values) {
        const auto topk_results = topK_search(query, db, k);

        const std::size_t expected_size =
            std::min(k, exhaustive_results.size());

        if (topk_results.size() != expected_size) {
            all_correct = false;
            break;
        }

        for (std::size_t i = 0; i < expected_size; ++i) {
            if (topk_results[i].index != exhaustive_results[i].index ||
                !approximately_equal(
                    topk_results[i].similarity,
                    exhaustive_results[i].similarity
                )) {
                all_correct = false;
                break;
            }
        }

        if (!all_correct) {
            break;
        }
    }

    check(
        all_correct,
        "Top-K agrees with sorted exhaustive search for multiple K values"
    );
}

void test_topk_invalid_dimensions() {
    VecDB db(2);
    db.add_vector({1.0f, 0.0f});

    const std::vector<float> query = {1.0f, 2.0f, 3.0f};

    expect_exception(
        [&] { topK_search(query, db, 1); },
        "Top-K rejects incorrect query dimensions"
    );
}

void test_topk_zero_query() {
    VecDB db(2);
    db.add_vector({1.0f, 0.0f});

    const std::vector<float> query = {0.0f, 0.0f};

    expect_exception(
        [&] { topK_search(query, db, 1); },
        "Top-K rejects a zero query"
    );
}

void test_topk_does_not_modify_inputs() {
    VecDB db = create_test_database();

    const std::vector<float> query = {3.0f, 4.0f};

    const auto original_query = query;
    const auto original_database = db.data();

    topK_search(query, db, 3);

    check(
        query == original_query,
        "Top-K does not modify the original query"
    );

    check(
        db.data() == original_database,
        "Top-K does not modify the database"
    );
}

// --------------------------------------------------
// Main
// --------------------------------------------------

int main() {
    std::cout << "\n===== CudaVec: CPU Similarity Tests =====\n\n";

    test_dot_product();

    test_complete_search_scores();
    test_complete_search_empty_database();
    test_complete_search_invalid_dimensions();
    test_complete_search_zero_query();
    test_complete_search_does_not_modify_inputs();

    test_topk_scores_and_order();
    test_topk_k_equals_one();
    test_topk_k_zero();
    test_topk_k_greater_than_database_size();
    test_topk_tie_breaking();
    test_topk_matches_exhaustive_search();
    test_topk_invalid_dimensions();
    test_topk_zero_query();
    test_topk_does_not_modify_inputs();

    std::cout << "\n===== Test Summary =====\n";
    std::cout << "Failures: " << failures << '\n';

    if (failures == 0) {
        std::cout << "All tests passed!\n";
        return 0;
    }

    return 1;
}
