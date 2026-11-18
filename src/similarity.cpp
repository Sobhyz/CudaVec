#include "similarity.h"
#include "utility.h"
#include <vector>
#include <cstddef>
#include <stdexcept>
#include <set>
#include <iterator>




double dot_product(
    std::vector<float>::const_iterator query_it, 
    std::vector<float>::const_iterator it_begin,
    std::vector<float>::const_iterator it_end){
        double result = 0;
        while(it_begin!=it_end){
            result += (static_cast<double>(*it_begin)*static_cast<double>(*query_it));
            it_begin+=1;
            query_it+=1;
        }

        return result;
}

std::vector<SearchResult> complete_search(
    std::vector<float> const &query, 
    VecDB const &database){
        std::size_t const dim = database.get_dimensions();
        if(query.size() != dim) throw std::runtime_error("Query's dimensions doesn't align with database");

        std::vector<float> normalized_query(query);

        std::vector<float> const &data = database.data();
        normalize_vector(normalized_query);
        std::size_t num_vectors=data.size()/dim;
        std::vector<SearchResult> results(num_vectors);

        for(std::size_t i=0 ; i<num_vectors ; i++){
            results[i] = SearchResult{
                i,
                dot_product(
                    normalized_query.begin(),
                    data.begin() + (i*dim),
                    data.begin() + ((i+1)*dim)
                )
            };
        }

        return results;

}

std::vector<SearchResult> topK_search(
    std::vector<float> const &query, 
    VecDB const &database,
    std::size_t k){
        auto custom_comparator = [](SearchResult const& a, SearchResult const& b){
            if(a.similarity>b.similarity) return true;
            else if(a.similarity<b.similarity) return false;
            return  a.index<b.index;
        };
        std::multiset<SearchResult, decltype(custom_comparator)> my_set(custom_comparator);
        std::vector<SearchResult> results;
        
        if(k==0) return results;

        std::size_t const dim = database.get_dimensions();
        if(query.size() != dim) throw std::runtime_error("Query's dimensions doesn't align with database");

        std::vector<float> normalized_query(query);

        std::vector<float> const &data = database.data();
        normalize_vector(normalized_query);
        std::size_t num_vectors=data.size()/dim;

        for(std::size_t i=0 ; i<num_vectors ; i++){
            SearchResult tmp = SearchResult{
                i,
                dot_product(
                    normalized_query.begin(),
                    data.begin() + (i*dim),
                    data.begin() + ((i+1)*dim)
                )
            };

            if(my_set.size()<k){
                my_set.insert(tmp);
                continue;
            }

            if(custom_comparator(tmp, *my_set.rbegin())){
                my_set.erase(std::prev(my_set.end()));
                my_set.insert(tmp);
            }
            
        }

        for(SearchResult const &res : my_set){
            results.push_back(res);
        }

        return results;

}