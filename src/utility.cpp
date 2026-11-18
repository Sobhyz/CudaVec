#include "utility.h"
#include <vector>
#include <cmath>
#include <stdexcept>

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