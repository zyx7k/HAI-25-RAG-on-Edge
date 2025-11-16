#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <chrono>
#include <iomanip>
#include <sys/stat.h>
#include "QnnRunner.h"
#include "hnsw_search.h"

void load_fvecs(const std::string &filename, std::vector<std::vector<float>> &vectors, int &dim)
{
    std::ifstream input(filename, std::ios::binary);
    if (!input)
        throw std::runtime_error("Cannot open file: " + filename);

    int d;
    while (input.read(reinterpret_cast<char *>(&d), sizeof(int)))
    {
        if (vectors.empty())
            dim = d;
        if (d != dim)
            throw std::runtime_error("Dimension mismatch in file");

        std::vector<float> vec(dim);
        input.read(reinterpret_cast<char *>(vec.data()), dim * sizeof(float));
        if (!input)
            throw std::runtime_error("Error reading vector data");

        vectors.push_back(vec);
    }
}

int main(int argc, char *argv[])
{
    if (argc != 8)
    {
        std::cerr << "Usage: " << argv[0]
                  << " <hnsw_index.bin> <vectors.fvecs> <queries.fvecs> <results_dir> "
                  << "<backend.so> <top_k> <ef_search>" << std::endl;
        std::cerr << "\nExample:" << std::endl;
        std::cerr << "  " << argv[0]
                  << " siftsmall_hnsw_M16.bin siftsmall_base.fvecs siftsmall_query.fvecs "
                  << "results libQnnHtp.so 10 50" << std::endl;
        return 1;
    }

    std::string hnsw_index_path = argv[1];
    std::string vectors_file = argv[2];
    std::string query_file = argv[3];
    std::string results_dir = argv[4];
    std::string backend_path = argv[5];
    const int TOP_K = std::stoi(argv[6]);
    const int EF_SEARCH = std::stoi(argv[7]);

    try
    {
        mkdir(results_dir.c_str(), 0755);

        std::string results_txt = results_dir + "/results.txt";
        std::string metrics_txt = results_dir + "/metrics.txt";

        std::cout << "Loading queries..." << std::endl;
        std::vector<std::vector<float>> query_vecs;
        int query_dim = 0;
        load_fvecs(query_file, query_vecs, query_dim);
        std::cout << "Loaded " << query_vecs.size() << " queries (dim=" << query_dim << ")." << std::endl;

        // Initialize QNN (for potential NPU-accelerated distance computations)
        // Note: For HNSW, we primarily use CPU for graph traversal
        // NPU can be used for batch distance computations if beneficial
        auto qnn_runner = std::make_shared<QnnRunner>("dummy_context.bin", backend_path);

        // Initialize HNSW searcher
        std::cout << "Loading HNSW index..." << std::endl;
        HNSWSearcher searcher(hnsw_index_path, vectors_file, qnn_runner, 32);

        std::ofstream results_file(results_txt);
        if (!results_file)
            throw std::runtime_error("Cannot open output file: " + results_txt);

        auto total_start = std::chrono::high_resolution_clock::now();

        std::vector<double> search_times;
        search_times.reserve(query_vecs.size());

        std::cout << "Processing queries with HNSW (ef_search=" << EF_SEARCH << ")..." << std::endl;

        for (size_t i = 0; i < query_vecs.size(); ++i)
        {
            auto query_start = std::chrono::high_resolution_clock::now();

            // HNSW search
            auto results = searcher.search(query_vecs[i], TOP_K, EF_SEARCH);

            auto query_end = std::chrono::high_resolution_clock::now();
            double query_time = std::chrono::duration<double, std::milli>(query_end - query_start).count();
            search_times.push_back(query_time);

            // Write results
            results_file << "Query " << i << ":";
            for (const auto &[idx, dist] : results)
            {
                results_file << " (" << idx << ", " << std::fixed << std::setprecision(0) << dist << ")";
            }
            results_file << "\n";

            if ((i + 1) % 100 == 0)
                std::cout << "Processed " << (i + 1) << "/" << query_vecs.size() << " queries." << std::endl;
        }

        auto total_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> total_time = total_end - total_start;

        results_file.close();

        // Calculate metrics
        int num_queries = query_vecs.size();
        double avg_search_time = 0.0;
        for (double t : search_times)
            avg_search_time += t;
        avg_search_time /= num_queries;

        // Standard deviation
        double variance = 0.0;
        for (double t : search_times)
        {
            variance += (t - avg_search_time) * (t - avg_search_time);
        }
        double stddev = std::sqrt(variance / num_queries);

        // Min/Max
        auto minmax = std::minmax_element(search_times.begin(), search_times.end());

        // Percentiles
        auto sorted_times = search_times;
        std::sort(sorted_times.begin(), sorted_times.end());
        double p50 = sorted_times[sorted_times.size() / 2];
        double p95 = sorted_times[(sorted_times.size() * 95) / 100];
        double p99 = sorted_times[(sorted_times.size() * 99) / 100];

        // Throughput
        double throughput = num_queries / total_time.count();

        // Write metrics
        std::ofstream metrics_file(metrics_txt);
        if (!metrics_file)
            throw std::runtime_error("Cannot open metrics file: " + metrics_txt);

        metrics_file << std::fixed << std::setprecision(6);
        metrics_file << "=== HNSW Search Performance Metrics ===\n\n";

        metrics_file << "Dataset Information:\n";
        metrics_file << "  Number of queries: " << num_queries << "\n";
        metrics_file << "  Query dimension: " << query_dim << "\n";
        metrics_file << "  Top-K: " << TOP_K << "\n";
        metrics_file << "  ef_search: " << EF_SEARCH << "\n\n";

        metrics_file << "Overall Performance:\n";
        metrics_file << "  Total execution time: " << total_time.count() << " s\n";
        metrics_file << "  Throughput: " << throughput << " queries/sec\n\n";

        metrics_file << "Search Performance:\n";
        metrics_file << "  Average latency: " << avg_search_time << " ms/query\n";
        metrics_file << "  Std deviation: " << stddev << " ms\n";
        metrics_file << "  Min latency: " << *minmax.first << " ms\n";
        metrics_file << "  Max latency: " << *minmax.second << " ms\n";
        metrics_file << "  P50 latency: " << p50 << " ms\n";
        metrics_file << "  P95 latency: " << p95 << " ms\n";
        metrics_file << "  P99 latency: " << p99 << " ms\n";

        metrics_file.close();

        std::cout << "\n=== Search Complete ===" << std::endl;
        std::cout << "Results: " << results_txt << std::endl;
        std::cout << "Metrics: " << metrics_txt << std::endl;
        std::cout << "\nPerformance Summary:" << std::endl;
        std::cout << "  Throughput: " << throughput << " queries/sec" << std::endl;
        std::cout << "  Avg latency: " << avg_search_time << " ms" << std::endl;
        std::cout << "  P95 latency: " << p95 << " ms" << std::endl;
    }
    catch (const std::exception &e)
    {
        std::cerr << "ERROR: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}