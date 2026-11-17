#pragma once


#include <vector>
#include <cstddef>

struct VecDB{
    private:
        std::size_t dimensions;
        std::vector<float> db;
        
    public:
        void add_vector(std::vector<float> &vec);
        void add_vector(std::vector<float>::const_iterator it);
        void normalize_vector(std::vector<float> vec);
        VecDB(std::size_t dim);
        VecDB(std::size_t num_vectors, std::size_t dim);
        VecDB(std::vector<float> const &db, std::size_t dim);
};