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

BenchmarkDataset::BenchmarkDataset(std::string const &db_path){
    if(!std::filesystem::is_directory(db_path))
        throw std::runtime_error("invalid directory path");

    this->database_path = db_path;
    this->load_metadata();
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


