#pragma once

#include "vector_dataset.h"
#include <vector>
#include <cstddef>


struct SearchResult{
    std::size_t index;
    double similarity;
};



double dot_product(
    std::vector<float>::const_iterator query_it, 
    std::vector<float>::const_iterator it_begin,
    std::vector<float>::const_iterator it_end
);

std::vector<SearchResult> complete_search(
    std::vector<float> const &query, 
    VecDB const &database
);

std::vector<SearchResult> topK_search(
    std::vector<float> const &query, 
    VecDB const &database,
    std::size_t k
);