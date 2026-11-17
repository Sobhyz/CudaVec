

#include "vector_dataset.h"
#include <cmath>
#include <stdexcept>

void VecDB::normalize_vector(std::vector<float> &vec){
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

void VecDB::add_vector(std::vector<float> vec){
    
    if(vec.size() != this->dimensions){
        throw std::runtime_error("vector dimensions doesn't align with database");
    }
    
    this->normalize_vector(vec);
    this->db.insert(this->db.end(), vec.begin(), vec.end());
}

void VecDB::add_vector(std::vector<float>::const_iterator it){
    
    std::vector<float> vec;
    vec.insert(vec.begin(), it, it+(this->dimensions));
    
    this->normalize_vector(vec);
    this->db.insert(this->db.end(), vec.begin(), vec.end());
}


VecDB::VecDB(std::size_t dim){
    if(!dim) throw std::runtime_error("can't have zero dimensions vector");
    this->dimensions = dim;
}

VecDB::VecDB(std::size_t num_vectors, std::size_t dim){
    if(!dim) throw std::runtime_error("can't have zero dimensions vector");
    this->dimensions = dim;
    this->db.reserve(num_vectors * dim);
}

VecDB::VecDB(std::vector<float> const &db, std::size_t dim){
    if(!dim) throw std::runtime_error("can't have zero dimensions vector");
    if(db.size()%dim != 0)throw std::runtime_error("DB and vectors dimensions mismatch");

    std::size_t vec_count = db.size()/dim;
    this->dimensions = dim;

    for(std::size_t i=0;i<vec_count;i++){
        this->add_vector(db.begin()+(i*dim));
    }
}

std::size_t VecDB::get_dimensions() const noexcept {
    return dimensions;
}

std::size_t VecDB::size() const noexcept {
    return db.size() / dimensions;
}

const std::vector<float>& VecDB::data() const noexcept {
    return db;
}
