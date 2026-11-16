#include "include/vector_dataset.h"
#include <cmath>
#include <stdexcept>

void VecDB::normalize_vector(std::vector<float> &vec){
    double denominator = 0;
    for(float &num : vec){ 
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