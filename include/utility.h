#pragma once


#include <vector>
#include <string>
#include <cstddef>
#include <filesystem>
#include <memory>
#include <cstdint>
#include <nlohmann/json.hpp>

#include "vector_dataset.h"



void normalize_vector(std::vector<float> &vec);

struct BenchmarkDataset{
    private:
        std::filesystem::path database_path;
        std::unique_ptr<VecDB> db;
        std::vector<float> query;
        std::vector<std::int32_t> ground_truth;
        std::vector<float> distance_ground_truth;
        nlohmann::json files_data;
        std::size_t dimensions=0;
        std::size_t num_queries=0;
        std::size_t ground_truth_k=0;
        std::vector<float> read_database();
        void load_metadata();
        void load_queries();
        void load_ground_truth();
        void build_database();
        

    public:
        BenchmarkDataset(std::filesystem::path const& db_path);

        VecDB const& get_database() const;
        std::vector<float>::const_iterator  get_query(std::size_t index) const;
        std::vector<std::int32_t>::const_iterator get_ground_truth(std::size_t index) const;
        std::vector<float>::const_iterator get_distance_ground_truth(std::size_t index) const;
        std::size_t get_num_queries() const;
        std::size_t get_dimensions() const;
        std::size_t get_ground_truth_k() const;

};
