#include "utility.h"
#include <vector>
#include <cmath>
#include <stdexcept>
#include <nlohmann/json.hpp>
#include <fstream>
#include <string>
#include <cstddef>
#include <filesystem>
#include <memory>
#include <cstdint>
#include <limits>
#include <type_traits>

#include "vector_dataset.h"

void normalize_vector(std::vector<float> &vec){
    double denominator = 0;
    for(float const &num : vec){ 
        if(!std::isfinite(num)) throw std::runtime_error("invalid value");
        denominator += (static_cast<double>(num)*num);
    }
    
    if(!denominator) throw std::runtime_error("you inserted a zero vector");
    if(!std::isfinite(denominator)) throw std::runtime_error("can't normalize vector");
    
    denominator = std::sqrt(denominator);
    for(float &num : vec)
    num = num/denominator;
}

template<typename T> 
std::vector<T> load_data(
            std::filesystem::path const& filename,
            std::size_t expected_elements){

    static_assert(
        std::is_trivially_copyable_v<T>,
        "T must be trivially copyable"
    );
    std::ifstream f(filename, std::ios::binary);

    if(!f.is_open()) throw std::runtime_error("can't open database file");
    if(!f.good()) throw std::runtime_error("database is not readable");
    
    if (expected_elements > std::numeric_limits<std::size_t>::max() / sizeof(T))
    throw std::runtime_error("Database size overflow");

    std::size_t expected_bytes = expected_elements * sizeof(T);
    if (expected_bytes > static_cast<std::size_t>(
        std::numeric_limits<std::streamsize>::max()))
        throw std::runtime_error("File too large for stream read");

    bool conf_check = std::filesystem::file_size(filename) != expected_bytes;
    if(conf_check)throw std::runtime_error("check conf.");

    std::vector<T> tmp_data(expected_elements);

    f.read(reinterpret_cast<char*>(tmp_data.data()),
       static_cast<std::streamsize>(tmp_data.size() * sizeof(T)));

    
    if(static_cast<std::streamsize>(tmp_data.size() * sizeof(T)) != f.gcount()) throw std::runtime_error("check conf.");

    return tmp_data;


}

BenchmarkDataset::BenchmarkDataset(std::filesystem::path const& db_path){
    if(!std::filesystem::is_directory(db_path))
        throw std::runtime_error("invalid directory path");

    this->database_path = db_path;
    this->load_metadata();
    this->build_database();
    this->load_queries();
    this->load_ground_truth();
}

void BenchmarkDataset::load_metadata(){
    std::ifstream f(this->database_path / "metadata.json");

    if(!f.is_open()) throw std::runtime_error("can't open metadata file");
    if(!f.good()) throw std::runtime_error("metadata is not readable");
    nlohmann::json data = nlohmann::json::parse(f);

    this->dimensions = data.at("dimensions").get<std::size_t>();
    this->ground_truth_k = data.at("ground_truth_k").get<std::size_t>();
    this->num_queries = data.at("query_count").get<std::size_t>();
    this->files_data = data.at("files");

    if(this->dimensions == 0 || this->ground_truth_k==0 || this->num_queries==0)
        throw std::runtime_error("invalid configuration check dimensions, ground_truth_k, and query_count");
}

std::vector<float> BenchmarkDataset::read_database(){

    std::size_t db_elements = this->files_data.at("train").at("elements").get<std::size_t>();
    // std::size_t size_bytes = this->files_data.at("train").at("size_bytes").get<std::size_t>();
    bool conf_check = db_elements % this->dimensions != 0;
    if(conf_check)throw std::runtime_error("check conf.");
    
    return load_data<float>(
        this->database_path / "base.f32",
        db_elements
    );
}

void BenchmarkDataset::load_queries(){
    std::size_t query_elements = this->files_data.at("test").at("elements").get<std::size_t>();
    bool conf_check = query_elements % this->dimensions != 0;

    if(conf_check)throw std::runtime_error("check conf.");

    std::vector<float> tmp(load_data<float>(
            this->database_path / "queries.f32",
            query_elements
        ));

    this->query.swap(
            tmp
    );
}

void BenchmarkDataset::load_ground_truth(){
    std::size_t neighbors_elements = this->files_data.at("neighbors").at("elements").get<std::size_t>();
    std::size_t distances_elements = this->files_data.at("distances").at("elements").get<std::size_t>();
    
    if (this->num_queries > std::numeric_limits<std::size_t>::max() / this->ground_truth_k)
        throw std::runtime_error("Ground truth size overflow");
    bool conf_check = (neighbors_elements!=distances_elements)
        ||  (this->num_queries *this->ground_truth_k != neighbors_elements);

    if(conf_check)throw std::runtime_error("check conf.");
    
    std::vector<std::int32_t> tmp_n(
        load_data<std::int32_t>(
            this->database_path / "neighbors.i32",
            neighbors_elements
        )
    );
    for (std::int32_t index : tmp_n) {
        if (index < 0 ||
            static_cast<std::size_t>(index) >= this->db->size()) {
            throw std::runtime_error(
                "Ground-truth neighbor index outside database bounds"
            );
        }
    }

    std::vector<float> tmp_dist(
        load_data<float>(
            this->database_path / "distances.f32",
            distances_elements
        )
    );

    this->ground_truth.swap(
            tmp_n
    );
    this->distance_ground_truth.swap(
            tmp_dist
    );
}

void BenchmarkDataset::build_database(){
    std::vector<float> tmp(
        this->read_database()
    );
    VecDB tmp_db(tmp, this->dimensions);
    this->db = std::make_unique<VecDB>(tmp_db);
    
}

VecDB const& BenchmarkDataset::get_database() const{
    return *(this->db);
}

std::vector<float>::const_iterator BenchmarkDataset::get_query(std::size_t index) const{
    if(index >= this->num_queries)
        throw std::runtime_error("Out of bound index");
    return this->query.begin() + (index * this->dimensions);
}

std::vector<std::int32_t>::const_iterator BenchmarkDataset::get_ground_truth(std::size_t index) const{
    if(index >= this->ground_truth.size() / this->ground_truth_k)
        throw std::runtime_error("Out of bound index");

    return this->ground_truth.begin() + (index * this->ground_truth_k);
}

std::vector<float>::const_iterator BenchmarkDataset::get_distance_ground_truth(std::size_t index) const{
    if(index >= this->distance_ground_truth.size() / this->ground_truth_k)
        throw std::runtime_error("Out of bound index");

    return this->distance_ground_truth.begin() + (index * this->ground_truth_k);
}


std::size_t BenchmarkDataset::get_num_queries() const{
    return this->num_queries;
}
std::size_t BenchmarkDataset::get_dimensions() const{
    return this->dimensions;
}
std::size_t BenchmarkDataset::get_ground_truth_k() const{
    return this->ground_truth_k;
}