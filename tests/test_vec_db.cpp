
#include "vector_dataset.h"

#include <iostream>
#include <vector>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

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
    catch (const std::runtime_error&) {
        check(true, message);
    }
    catch (...) {
        check(false, message + " (unexpected exception type)");
    }
}

bool approximately_equal(float a, float b, float tolerance = 1e-5f) {
    return std::abs(a - b) <= tolerance;
}

void test_empty_constructor() {
    VecDB db(3);

    check(db.get_dimensions() == 3,
          "Empty constructor initializes dimensions");

    check(db.size() == 0,
          "New database is empty");
}

void test_capacity_constructor() {
    VecDB db(100, 3);

    check(db.get_dimensions() == 3,
          "Capacity constructor initializes dimensions");

    check(db.size() == 0,
          "Reserved capacity does not insert vectors");
}

void test_zero_dimension() {
    expect_exception(
        [] { VecDB db(0); },
        "Reject zero-dimensional database"
    );
}

void test_add_vector() {
    VecDB db(2);

    std::vector<float> vec = {3.0f, 4.0f};

    db.add_vector(vec);

    const auto& stored = db.data();

    check(db.size() == 1,
          "One vector inserted");

    check(approximately_equal(stored[0], 0.6f) &&
          approximately_equal(stored[1], 0.8f),
          "Vector is normalized correctly");
}

void test_multiple_vectors() {
    VecDB db(2);

    db.add_vector({3.0f, 4.0f});
    db.add_vector({0.0f, 5.0f});
    db.add_vector({-5.0f, 12.0f});

    check(db.size() == 3,
          "Multiple vectors inserted");

    check(db.data().size() == 6,
          "Flat storage contains the expected number of elements");
}

void test_input_unchanged() {
    VecDB db(2);

    std::vector<float> original = {3.0f, 4.0f};

    db.add_vector(original);

    check(original[0] == 3.0f && original[1] == 4.0f,
          "Original input remains unchanged");
}

void test_wrong_dimensions() {
    VecDB db(3);

    expect_exception(
        [&] { db.add_vector({1.0f, 2.0f}); },
        "Reject vector with incorrect dimensions"
    );

    check(db.size() == 0,
          "Invalid vector is not inserted");
}

void test_zero_vector() {
    VecDB db(3);

    expect_exception(
        [&] { db.add_vector({0.0f, 0.0f, 0.0f}); },
        "Reject zero vector"
    );

    check(db.size() == 0,
          "Zero vector is not inserted");
}

void test_nan_vector() {
    VecDB db(2);

    float nan = std::numeric_limits<float>::quiet_NaN();

    expect_exception(
        [&] { db.add_vector({nan, 1.0f}); },
        "Reject NaN values"
    );
}

void test_infinite_vector() {
    VecDB db(2);

    float inf = std::numeric_limits<float>::infinity();

    expect_exception(
        [&] { db.add_vector({inf, 1.0f}); },
        "Reject infinite values"
    );
}

void test_database_constructor() {
    std::vector<float> raw = {
        3.0f, 4.0f,
        0.0f, 5.0f,
        -5.0f, 12.0f
    };

    VecDB db(raw, 2);

    check(db.size() == 3,
          "Raw database constructor loads all vectors");

    check(db.data().size() == 6,
          "Raw database constructor maintains flat storage");

    check(approximately_equal(db.data()[0], 0.6f) &&
          approximately_equal(db.data()[1], 0.8f),
          "First loaded vector normalized");

    check(approximately_equal(db.data()[2], 0.0f) &&
          approximately_equal(db.data()[3], 1.0f),
          "Second loaded vector normalized");

    check(approximately_equal(db.data()[4], -5.0f / 13.0f) &&
          approximately_equal(db.data()[5], 12.0f / 13.0f),
          "Third loaded vector normalized");
}

void test_incomplete_database() {
    std::vector<float> raw = {
        1.0f, 2.0f, 3.0f,
        4.0f
    };

    expect_exception(
        [&] { VecDB db(raw, 3); },
        "Reject database with incomplete vector"
    );
}

void test_database_with_zero_vector() {
    std::vector<float> raw = {
        3.0f, 4.0f,
        0.0f, 0.0f
    };

    expect_exception(
        [&] { VecDB db(raw, 2); },
        "Reject database containing a zero vector"
    );
}

void test_all_vectors_normalized() {
    VecDB db(3);

    db.add_vector({3.0f, 4.0f, 0.0f});
    db.add_vector({1.0f, 2.0f, 2.0f});
    db.add_vector({-2.0f, 1.0f, 2.0f});

    const auto& data = db.data();
    const std::size_t dimensions = db.get_dimensions();

    bool all_normalized = true;

    for (std::size_t i = 0; i < db.size(); ++i) {
        double norm_squared = 0.0;

        for (std::size_t j = 0; j < dimensions; ++j) {
            double value = data[i * dimensions + j];
            norm_squared += value * value;
        }

        double norm = std::sqrt(norm_squared);

        if (std::abs(norm - 1.0) > 1e-5) {
            all_normalized = false;
            break;
        }
    }

    check(all_normalized,
          "Every stored vector has unit L2 norm");
}

int main() {
    std::cout << "\n===== CudaVec: VecDB Tests =====\n\n";

    test_empty_constructor();
    test_capacity_constructor();
    test_zero_dimension();

    test_add_vector();
    test_multiple_vectors();
    test_input_unchanged();

    test_wrong_dimensions();
    test_zero_vector();
    test_nan_vector();
    test_infinite_vector();

    test_database_constructor();
    test_incomplete_database();
    test_database_with_zero_vector();

    test_all_vectors_normalized();

    std::cout << "\n===== Test Summary =====\n";
    std::cout << "Failures: " << failures << '\n';

    if (failures == 0) {
        std::cout << "All tests passed!\n";
        return 0;
    }

    return 1;
}
