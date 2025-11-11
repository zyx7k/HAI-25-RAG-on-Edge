#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <stdexcept>
#include <chrono>
#include <algorithm>
#include <iomanip>
#include <cmath>
#include <sys/stat.h>
#include "QnnRunner.h"

void load_fvecs(const std::string& filename, std::vector<std::vector<float>>& vectors, int& dim) {
    std::ifstream input(filename, std::ios::binary);
    if (!input) throw std::runtime_error("Cannot open file: " + filename);

    int d;
    while (input.read(reinterpret_cast<char*>(&d), sizeof(int))) {
        if (vectors.empty()) dim = d;
        if (d != dim) throw std::runtime_error("Dimension mismatch in file");

        std::vector<float> vec(dim);
        input.read(reinterpret_cast<char*>(vec.data()), dim * sizeof(float));
        if (!input) throw std::runtime_error("Error reading vector data");

        vectors.push_back(vec);
    }
}

struct SearchResult {
    int index;
    float distance;
    bool operator<(const SearchResult& other) const { return distance < other.distance; }
};

int main(int argc, char* argv[]) {
    if (argc != 7) {
        std::cerr << "Usage: " << argv[0]
                  << " <context_binary> <queries.fvecs> <results_dir> <backend.so> <documents.fvecs> <top_k>" << std::endl;
        return 1;
    }

    std::string model_path = argv[1];
    std::string query_file = argv[2];
    std::string results_dir = argv[3];
    std::string backend_path = argv[4];
    std::string doc_file = argv[5];
    const int TOP_K = std::stoi(argv[6]);

    try {
        // Create results directory
        mkdir(results_dir.c_str(), 0755);

        std::string results_txt = results_dir + "/results.txt";
        std::string metrics_txt = results_dir + "/metrics.txt";

        std::cout << "Loading queries..." << std::endl;
        std::vector<std::vector<float>> query_vecs;
        int query_dim = 0;
        load_fvecs(query_file, query_vecs, query_dim);
        std::cout << "Loaded " << query_vecs.size() << " queries." << std::endl;

        std::cout << "Loading documents..." << std::endl;
        std::vector<std::vector<float>> doc_vecs;
        int doc_dim = 0;
        load_fvecs(doc_file, doc_vecs, doc_dim);
        std::cout << "Loaded " << doc_vecs.size() << " documents." << std::endl;
        
        if (query_dim != doc_dim) {
            throw std::runtime_error("Query and document dimensions don't match!");
        }

        std::cout << "Pre-computing document norms..." << std::endl;
        std::vector<float> doc_norms(doc_vecs.size());
        for (size_t i = 0; i < doc_vecs.size(); ++i) {
            float norm = 0.0f;
            for (float val : doc_vecs[i]) {
                norm += val * val;
            }
            doc_norms[i] = norm;
        }

        QnnRunner runner(model_path, backend_path);
        int model_input_dim = runner.getInputDims().back();
        int model_output_dim = runner.getOutputDims().back();
        if (model_input_dim != query_dim)
            throw std::runtime_error("Model input dim and query dim mismatch!");
        if (model_output_dim != doc_vecs.size())
            throw std::runtime_error("Model output dim doesn't match number of documents!");

        std::ofstream results_file(results_txt);
        if (!results_file) throw std::runtime_error("Cannot open output file: " + results_txt);

        std::vector<float> dot_products(model_output_dim);
        auto total_start = std::chrono::high_resolution_clock::now();
        
        double npu_time_total = 0.0;
        double cpu_time_total = 0.0;
        
        std::vector<double> npu_times_per_query;
        std::vector<double> cpu_times_per_query;
        std::vector<double> total_times_per_query;
        
        npu_times_per_query.reserve(query_vecs.size());
        cpu_times_per_query.reserve(query_vecs.size());
        total_times_per_query.reserve(query_vecs.size());

        std::cout << "Processing queries..." << std::endl;
        
        for (int i = 0; i < query_vecs.size(); ++i) {
            auto query_start = std::chrono::high_resolution_clock::now();
            
            float query_norm = 0.0f;
            for (float val : query_vecs[i]) {
                query_norm += val * val;
            }

            auto npu_start = std::chrono::high_resolution_clock::now();
            runner.execute(query_vecs[i], dot_products);
            auto npu_end = std::chrono::high_resolution_clock::now();
            double npu_time = std::chrono::duration<double, std::milli>(npu_end - npu_start).count();
            npu_time_total += npu_time;
            npu_times_per_query.push_back(npu_time);

            auto cpu_start = std::chrono::high_resolution_clock::now();
            std::vector<float> distances(doc_vecs.size());
            for (size_t j = 0; j < doc_vecs.size(); ++j) {
                distances[j] = query_norm + doc_norms[j] - 2.0f * dot_products[j];
            }

            std::vector<SearchResult> top_k;
            top_k.reserve(TOP_K);
            for (int j = 0; j < distances.size(); ++j) {
                if (top_k.size() < TOP_K) {
                    top_k.push_back({j, distances[j]});
                    std::make_heap(top_k.begin(), top_k.end());
                } else if (distances[j] < top_k.front().distance) {
                    std::pop_heap(top_k.begin(), top_k.end());
                    top_k.back() = {j, distances[j]};
                    std::push_heap(top_k.begin(), top_k.end());
                }
            }
            std::sort(top_k.begin(), top_k.end(), [](const SearchResult& a, const SearchResult& b) {
                return a.distance < b.distance;
            });
            auto cpu_end = std::chrono::high_resolution_clock::now();
            double cpu_time = std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();
            cpu_time_total += cpu_time;
            cpu_times_per_query.push_back(cpu_time);
            
            auto query_end = std::chrono::high_resolution_clock::now();
            double query_total_time = std::chrono::duration<double, std::milli>(query_end - query_start).count();
            total_times_per_query.push_back(query_total_time);

            results_file << "Query " << i << ":";
            for (const auto& r : top_k) {
                results_file << " (" << r.index << ", " << std::fixed << std::setprecision(0) << r.distance << ")";
            }
            results_file << "\n";

            if ((i + 1) % 10 == 0)
                std::cout << "Processed " << (i + 1) << "/" << query_vecs.size() << " queries." << std::endl;
        }

        auto total_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> total_time = total_end - total_start;
        
        results_file.close();
        
        // Calculate additional metrics
        int num_queries = query_vecs.size();
        double avg_npu_time = npu_time_total / num_queries;
        double avg_cpu_time = cpu_time_total / num_queries;
        double avg_query_time = total_time.count() * 1000.0 / num_queries;
        
        // Calculate standard deviation for latency
        double npu_variance = 0.0, cpu_variance = 0.0, total_variance = 0.0;
        for (int i = 0; i < num_queries; ++i) {
            npu_variance += (npu_times_per_query[i] - avg_npu_time) * (npu_times_per_query[i] - avg_npu_time);
            cpu_variance += (cpu_times_per_query[i] - avg_cpu_time) * (cpu_times_per_query[i] - avg_cpu_time);
            total_variance += (total_times_per_query[i] - avg_query_time) * (total_times_per_query[i] - avg_query_time);
        }
        double npu_stddev = std::sqrt(npu_variance / num_queries);
        double cpu_stddev = std::sqrt(cpu_variance / num_queries);
        double total_stddev = std::sqrt(total_variance / num_queries);
        
        // Find min/max latencies
        auto npu_minmax = std::minmax_element(npu_times_per_query.begin(), npu_times_per_query.end());
        auto cpu_minmax = std::minmax_element(cpu_times_per_query.begin(), cpu_times_per_query.end());
        auto total_minmax = std::minmax_element(total_times_per_query.begin(), total_times_per_query.end());
        
        // Calculate throughput
        double throughput_qps = num_queries / total_time.count();
        
        // Calculate percentiles (p50, p95, p99)
        auto calculate_percentile = [](std::vector<double> times, double percentile) {
            std::sort(times.begin(), times.end());
            size_t idx = static_cast<size_t>(percentile * times.size());
            if (idx >= times.size()) idx = times.size() - 1;
            return times[idx];
        };
        
        double npu_p50 = calculate_percentile(npu_times_per_query, 0.50);
        double npu_p95 = calculate_percentile(npu_times_per_query, 0.95);
        double npu_p99 = calculate_percentile(npu_times_per_query, 0.99);
        
        double cpu_p50 = calculate_percentile(cpu_times_per_query, 0.50);
        double cpu_p95 = calculate_percentile(cpu_times_per_query, 0.95);
        double cpu_p99 = calculate_percentile(cpu_times_per_query, 0.99);
        
        double total_p50 = calculate_percentile(total_times_per_query, 0.50);
        double total_p95 = calculate_percentile(total_times_per_query, 0.95);
        double total_p99 = calculate_percentile(total_times_per_query, 0.99);
        
        // Write metrics to file
        std::ofstream metrics_file(metrics_txt);
        if (!metrics_file) throw std::runtime_error("Cannot open metrics file: " + metrics_txt);
        
        metrics_file << std::fixed << std::setprecision(6);
        metrics_file << "=== QNN RAG Demo Performance Metrics ===\n\n";
        
        metrics_file << "Dataset Information:\n";
        metrics_file << "  Number of queries: " << num_queries << "\n";
        metrics_file << "  Number of documents: " << doc_vecs.size() << "\n";
        metrics_file << "  Dimension: " << query_dim << "\n";
        metrics_file << "  Top-K: " << TOP_K << "\n\n";
        
        metrics_file << "Overall Performance:\n";
        metrics_file << "  Total execution time: " << total_time.count() << " s\n";
        metrics_file << "  Throughput: " << throughput_qps << " queries/sec\n\n";
        
        metrics_file << "NPU Performance:\n";
        metrics_file << "  Total NPU time: " << (npu_time_total/1000.0) << " s\n";
        metrics_file << "  Average latency: " << avg_npu_time << " ms/query\n";
        metrics_file << "  Std deviation: " << npu_stddev << " ms\n";
        metrics_file << "  Min latency: " << *npu_minmax.first << " ms\n";
        metrics_file << "  Max latency: " << *npu_minmax.second << " ms\n";
        metrics_file << "  P50 latency: " << npu_p50 << " ms\n";
        metrics_file << "  P95 latency: " << npu_p95 << " ms\n";
        metrics_file << "  P99 latency: " << npu_p99 << " ms\n\n";
        
        metrics_file << "CPU Post-processing:\n";
        metrics_file << "  Total CPU time: " << (cpu_time_total/1000.0) << " s\n";
        metrics_file << "  Average latency: " << avg_cpu_time << " ms/query\n";
        metrics_file << "  Std deviation: " << cpu_stddev << " ms\n";
        metrics_file << "  Min latency: " << *cpu_minmax.first << " ms\n";
        metrics_file << "  Max latency: " << *cpu_minmax.second << " ms\n";
        metrics_file << "  P50 latency: " << cpu_p50 << " ms\n";
        metrics_file << "  P95 latency: " << cpu_p95 << " ms\n";
        metrics_file << "  P99 latency: " << cpu_p99 << " ms\n\n";
        
        metrics_file << "End-to-End Per Query:\n";
        metrics_file << "  Average latency: " << avg_query_time << " ms/query\n";
        metrics_file << "  Std deviation: " << total_stddev << " ms\n";
        metrics_file << "  Min latency: " << *total_minmax.first << " ms\n";
        metrics_file << "  Max latency: " << *total_minmax.second << " ms\n";
        metrics_file << "  P50 latency: " << total_p50 << " ms\n";
        metrics_file << "  P95 latency: " << total_p95 << " ms\n";
        metrics_file << "  P99 latency: " << total_p99 << " ms\n\n";
        
        metrics_file << "Time Breakdown:\n";
        metrics_file << "  NPU percentage: " << (npu_time_total / (total_time.count() * 1000.0) * 100.0) << "%\n";
        metrics_file << "  CPU percentage: " << (cpu_time_total / (total_time.count() * 1000.0) * 100.0) << "%\n";
        
        metrics_file.close();
        
        std::cout << "\nAll queries processed successfully." << std::endl;
        std::cout << "Results saved to: " << results_txt << std::endl;
        std::cout << "Metrics saved to: " << metrics_txt << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "FATAL ERROR: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
