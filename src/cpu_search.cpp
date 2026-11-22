#include "similarity.h"
#include "utility.h"
#include <filesystem>
#include <stdexcept>
#include <charconv>
#include <unordered_set>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>
#include <cstdint>
#include <chrono>
#include <algorithm>
#include <cmath>



double recall(
    std::size_t index,
    std::size_t k,
    BenchmarkDataset const& bdb,
    std::vector<std::chrono::nanoseconds>& search_timing
)
{
    if(k>bdb.get_ground_truth_k() || k > bdb.get_database().size() || k==0)
        throw std::runtime_error("invalid k");

    std::vector<float>::const_iterator query_it = bdb.get_query(index);
    std::vector<std::int32_t>::const_iterator gt_it = bdb.get_ground_truth(index);

    std::vector<float> query(
        query_it,
        query_it + bdb.get_dimensions()
    );

    auto start = std::chrono::steady_clock::now();

    std::vector<SearchResult> results = topK_search(
        query,
        bdb.get_database(),
        k
    );

    auto end = std::chrono::steady_clock::now();

    search_timing.push_back(std::chrono::duration_cast<std::chrono::nanoseconds>(
        end - start
    ));

    std::unordered_set<std::size_t> gt_set;
    for (std::size_t i = 0; i < k; ++i, ++gt_it) {
        gt_set.insert(static_cast<std::size_t>(*gt_it));
    }
    std::size_t matches = 0;

    for (SearchResult const& res : results) {
        if (gt_set.find(res.index) != gt_set.end()) {
            ++matches;
        }
    }

    return static_cast<double>(matches) / k;
}


int main(int argc, char *argv[]){

    if( (argc!=3 && argc!=5) ||
        std::string(argv[1]) != "--path" ||
        (argc==5 && std::string(argv[3]) != "-k")){
        std::cerr << "Usage: " << argv[0]
          << " --path <processed-dataset-directory> [-k <K>]\n";
        return 2;
    }

    std::filesystem::path data_path = argv[2];
    BenchmarkDataset data = BenchmarkDataset(data_path);
    std::size_t const queries_cnt = data.get_num_queries();

    if (queries_cnt == 0) {
        std::cerr << "Dataset contains no queries\n";
        return 1;
    }

    std::size_t k =10;
    if (argc == 5) {
        std::string_view arg = argv[4];

        auto [ptr, ec] = std::from_chars(
            arg.data(),
            arg.data() + arg.size(),
            k
        );

        if (ec != std::errc{} || ptr != arg.data() + arg.size() || k == 0) {
            std::cerr << "Invalid K value\n";
            return 2;
        }
    }

    if (k > data.get_ground_truth_k() || k > data.get_database().size()) {
        std::cerr << "K exceeds the available neighbours\n";
        return 2;
    }

    // Warm-up phase (excluded from benchmark measurements)
    std::cout << "Warming up CPU search...\n";

    for (std::size_t i = 0; i < 3 && i < queries_cnt; ++i) {

        std::vector<float>::const_iterator query_it = data.get_query(i);

        std::vector<float> query(
            query_it,
            query_it + data.get_dimensions()
        );

        // Execute search, but don't record its timing.
        (void)topK_search(
            query,
            data.get_database(),
            k
        );
    }

    std::cout << "Warm-up completed.\n";

    double average_recall = 0;
    std::vector<std::chrono::nanoseconds> search_timing;
    search_timing.reserve(queries_cnt);

    
    for(std::size_t i = 0 ; i<queries_cnt ; i++){

        if(i!=0 && i%100 ==0){
            std::cout<< "CHECK POINT " << i <<'/'<< queries_cnt<<std::endl;
            std::cout<<"Your Average Recall@"<<k<<" is: " << average_recall/i<<std::endl;
        }
        average_recall += recall(
            i,
            k,
            data,
            search_timing
        );
        
    }

    std::sort(search_timing.begin(), search_timing.end());

    std::chrono::nanoseconds total_time{0};

    for (auto const& duration : search_timing) {
        total_time += duration;
    }

    double average_latency =
        std::chrono::duration<double, std::milli>(total_time).count()
        / search_timing.size();

    double median_latency;

    if (search_timing.size() % 2 == 0) {
        std::size_t mid = search_timing.size() / 2;

        median_latency =
            std::chrono::duration<double, std::milli>(
                search_timing[mid - 1] + search_timing[mid]
            ).count() / 2.0;
    } else {
        median_latency =
            std::chrono::duration<double, std::milli>(
                search_timing[search_timing.size() / 2]
            ).count();
    }

    std::size_t p95_index =
        static_cast<std::size_t>(
            std::ceil(0.95 * search_timing.size())
        ) - 1;

    double p95_latency =
        std::chrono::duration<double, std::milli>(
            search_timing[p95_index]
        ).count();

    double max_latency =
        std::chrono::duration<double, std::milli>(
            search_timing.back()
        ).count();

    average_recall /= queries_cnt;
    double search_qps = 1000.0 / average_latency;

    std::cout << "\n===== CPU BENCHMARK RESULTS =====\n";
    std::cout << "Queries: " << queries_cnt << '\n';
    std::cout << "Recall@" << k << ": " << average_recall << '\n';
    std::cout << "Average latency: " << average_latency << " ms\n";
    std::cout << "Median latency: " << median_latency << " ms\n";
    std::cout << "P95 latency: " << p95_latency << " ms\n";
    std::cout << "Maximum latency: " << max_latency << " ms\n";
    std::cout << "Search throughput: " << search_qps << " queries/second\n";

    return 0;
}



