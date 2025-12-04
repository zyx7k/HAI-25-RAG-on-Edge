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

struct SearchResultInt8 {
    int index;
    uint8_t similarity;
    bool operator<(const SearchResultInt8& other) const { return similarity > other.similarity; }
};

void find_top_k_int8(const uint8_t* scores, size_t num_scores, int top_k, 
                     std::vector<SearchResultInt8>& results) {
    results.clear();
    results.reserve(top_k);
    
    for (size_t j = 0; j < num_scores; ++j) {
        uint8_t score = scores[j];
        if (results.size() < static_cast<size_t>(top_k)) {
            results.push_back({static_cast<int>(j), score});
            std::push_heap(results.begin(), results.end());
        } else if (score > results.front().similarity) {
            std::pop_heap(results.begin(), results.end());
            results.back() = {static_cast<int>(j), score};
            std::push_heap(results.begin(), results.end());
        }
    }
    
    std::sort(results.begin(), results.end(), 
        [](const SearchResultInt8& a, const SearchResultInt8& b) {
            return a.similarity > b.similarity;
        });
}

void find_top_k_batch_parallel(const uint8_t* raw_scores, size_t num_docs, size_t actual_batch_size,
                               int top_k, std::vector<std::vector<SearchResultInt8>>& batch_results) {
    batch_results.resize(actual_batch_size);
    
    std::vector<SearchResultInt8> local_heap;
    local_heap.reserve(top_k);
    
    for (size_t i = 0; i < actual_batch_size; ++i) {
        const uint8_t* query_scores = raw_scores + i * num_docs;
        find_top_k_int8(query_scores, num_docs, top_k, local_heap);
        batch_results[i] = local_heap;
    }
}

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

        QnnRunner runner(model_path, backend_path);
        
        // Get model dimensions
        size_t model_batch_size = runner.getBatchSize();
        size_t model_dim = runner.getDim();
        size_t num_docs = runner.getNumDocs();
        
        std::cout << "Model configuration:" << std::endl;
        std::cout << "  Batch size: " << model_batch_size << std::endl;
        std::cout << "  Dimension: " << model_dim << std::endl;
        std::cout << "  Number of documents: " << num_docs << std::endl;
        
        if (model_dim != static_cast<size_t>(query_dim)) {
            throw std::runtime_error("Model input dim and query dim mismatch!");
        }
        if (num_docs != doc_vecs.size()) {
            throw std::runtime_error("Model output dim doesn't match number of documents!");
        }

        std::ofstream results_file(results_txt);
        if (!results_file) throw std::runtime_error("Cannot open output file: " + results_txt);

        // Use raw query vectors - model was calibrated on raw SIFT vectors (range 0-141)
        const std::vector<std::vector<float>>& query_vectors = query_vecs;
        float output_scale = runner.getOutputScale();

        double npu_time_total = 0.0;
        double cpu_time_total = 0.0;
        double quantize_time_total = 0.0;
        double graph_exec_time_total = 0.0;
        
        std::vector<double> npu_times_per_batch;
        std::vector<double> cpu_times_per_batch;
        std::vector<double> graph_exec_times_per_batch;
        
        std::vector<SearchResultInt8> top_k_heap;
        top_k_heap.reserve(TOP_K);

        size_t num_queries = query_vectors.size();
        size_t num_batches = (num_queries + model_batch_size - 1) / model_batch_size;
        
        std::cout << "Processing " << num_queries << " queries in " << num_batches 
                  << " batches (batch_size=" << model_batch_size << ")..." << std::endl;

        auto total_start = std::chrono::high_resolution_clock::now();

        if (model_batch_size == 1) {
            npu_times_per_batch.reserve(num_queries);
            cpu_times_per_batch.reserve(num_queries);
            graph_exec_times_per_batch.reserve(num_queries);
            
            for (size_t i = 0; i < num_queries; ++i) {
                ExecutionTiming exec_timing;
                auto npu_start = std::chrono::high_resolution_clock::now();
                runner.executeRaw(query_vectors[i], exec_timing);
                auto npu_end = std::chrono::high_resolution_clock::now();
                
                double npu_time = std::chrono::duration<double, std::milli>(npu_end - npu_start).count();
                npu_time_total += npu_time;
                npu_times_per_batch.push_back(npu_time);
                
                quantize_time_total += exec_timing.quantize_ms;
                graph_exec_time_total += exec_timing.graph_execute_ms;
                graph_exec_times_per_batch.push_back(exec_timing.graph_execute_ms);

                auto cpu_start = std::chrono::high_resolution_clock::now();
                const uint8_t* raw_scores = runner.getRawOutputBuffer();
                find_top_k_int8(raw_scores, num_docs, TOP_K, top_k_heap);
                auto cpu_end = std::chrono::high_resolution_clock::now();
                
                double cpu_time = std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();
                cpu_time_total += cpu_time;
                cpu_times_per_batch.push_back(cpu_time);

                results_file << "Query " << i << ":";
                for (const auto& r : top_k_heap) {
                    float float_score = static_cast<float>(r.similarity) * output_scale;
                    results_file << " (" << r.index << ", " << std::fixed << std::setprecision(4) << float_score << ")";
                }
                results_file << "\n";

                if ((i + 1) % 100 == 0 || i + 1 == num_queries) {
                    std::cout << "Processed " << (i + 1) << "/" << num_queries << " queries." << std::endl;
                }
            }
        } else {
            npu_times_per_batch.reserve(num_batches);
            cpu_times_per_batch.reserve(num_batches);
            graph_exec_times_per_batch.reserve(num_batches);
            
            std::vector<float> batch_input(model_batch_size * query_dim);
            
            for (size_t batch_idx = 0; batch_idx < num_batches; ++batch_idx) {
                size_t batch_start = batch_idx * model_batch_size;
                size_t batch_end = std::min(batch_start + model_batch_size, num_queries);
                size_t actual_batch_size = batch_end - batch_start;
                
                std::fill(batch_input.begin(), batch_input.end(), 0.0f);
                for (size_t i = 0; i < actual_batch_size; ++i) {
                    std::copy(query_vectors[batch_start + i].begin(),
                              query_vectors[batch_start + i].end(),
                              batch_input.begin() + i * query_dim);
                }
                
                ExecutionTiming exec_timing;
                auto npu_start = std::chrono::high_resolution_clock::now();
                runner.executeBatchRaw(batch_input, exec_timing);
                auto npu_end = std::chrono::high_resolution_clock::now();
                
                double npu_time = std::chrono::duration<double, std::milli>(npu_end - npu_start).count();
                npu_time_total += npu_time;
                npu_times_per_batch.push_back(npu_time);
                
                quantize_time_total += exec_timing.quantize_ms;
                graph_exec_time_total += exec_timing.graph_execute_ms;
                graph_exec_times_per_batch.push_back(exec_timing.graph_execute_ms);

                auto cpu_start = std::chrono::high_resolution_clock::now();
                const uint8_t* raw_scores = runner.getRawOutputBuffer();
                
                std::vector<std::vector<SearchResultInt8>> batch_results;
                find_top_k_batch_parallel(raw_scores, num_docs, actual_batch_size, TOP_K, batch_results);
                
                for (size_t i = 0; i < actual_batch_size; ++i) {
                    results_file << "Query " << (batch_start + i) << ":";
                    for (const auto& r : batch_results[i]) {
                        float float_score = static_cast<float>(r.similarity) * output_scale;
                        results_file << " (" << r.index << ", " << std::fixed << std::setprecision(4) << float_score << ")";
                    }
                    results_file << "\n";
                }
                
                auto cpu_end = std::chrono::high_resolution_clock::now();
                double cpu_time = std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();
                cpu_time_total += cpu_time;
                cpu_times_per_batch.push_back(cpu_time);

                if ((batch_idx + 1) % 10 == 0 || batch_idx + 1 == num_batches) {
                    std::cout << "Processed batch " << (batch_idx + 1) << "/" << num_batches 
                              << " (" << batch_end << "/" << num_queries << " queries)" << std::endl;
                }
            }
        }

        auto total_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> total_time = total_end - total_start;
        
        results_file.close();
        
        // Calculate metrics
        double avg_npu_time_per_batch = npu_time_total / npu_times_per_batch.size();
        double avg_cpu_time_per_batch = cpu_time_total / cpu_times_per_batch.size();
        double avg_graph_exec_time_per_batch = graph_exec_time_total / graph_exec_times_per_batch.size();
        double avg_quantize_time_per_batch = quantize_time_total / npu_times_per_batch.size();
        
        // Per-query metrics (amortized over batch)
        double avg_npu_time_per_query = npu_time_total / num_queries;
        double avg_cpu_time_per_query = cpu_time_total / num_queries;
        double avg_graph_exec_time_per_query = graph_exec_time_total / num_queries;
        
        // Calculate variance for batch times
        double graph_exec_variance = 0.0;
        for (auto& t : graph_exec_times_per_batch) {
            graph_exec_variance += (t - avg_graph_exec_time_per_batch) * (t - avg_graph_exec_time_per_batch);
        }
        double graph_exec_stddev = std::sqrt(graph_exec_variance / graph_exec_times_per_batch.size());
        
        // Min/max
        auto graph_exec_minmax = std::minmax_element(graph_exec_times_per_batch.begin(), graph_exec_times_per_batch.end());
        auto npu_minmax = std::minmax_element(npu_times_per_batch.begin(), npu_times_per_batch.end());
        auto cpu_minmax = std::minmax_element(cpu_times_per_batch.begin(), cpu_times_per_batch.end());
        
        // Calculate FLOPs and GFLOPS
        // MatMul: [B, D] x [D, N] = [B, N]
        // FLOPs per batch = 2 * B * D * N
        double flops_per_batch = 2.0 * model_batch_size * query_dim * num_docs;
        double flops_per_query = 2.0 * query_dim * num_docs;
        
        // GFLOPS based on pure graph execution time
        double avg_pure_npu_gflops = (flops_per_batch / (avg_graph_exec_time_per_batch / 1000.0)) / 1e9;
        double max_pure_npu_gflops = (flops_per_batch / (*graph_exec_minmax.first / 1000.0)) / 1e9;
        double min_pure_npu_gflops = (flops_per_batch / (*graph_exec_minmax.second / 1000.0)) / 1e9;
        
        // GFLOPS per query (amortized)
        double avg_gflops_per_query = (flops_per_query / (avg_graph_exec_time_per_query / 1000.0)) / 1e9;
        
        // Throughput
        double throughput_qps = num_queries / total_time.count();
        
        // Operational Intensity calculation
        // Bytes moved: query[B*D] + docs[D*N] (reused!) + output[B*N]
        // For INT8: 1 byte each
        double bytes_query = model_batch_size * query_dim;
        double bytes_docs = query_dim * num_docs;  // Read once, reused for all queries in batch
        double bytes_output = model_batch_size * num_docs;
        double total_bytes = bytes_query + bytes_docs + bytes_output;
        double operational_intensity = flops_per_batch / total_bytes;
        
        // Percentiles
        auto calculate_percentile = [](std::vector<double> times, double percentile) {
            if (times.empty()) return 0.0;
            std::sort(times.begin(), times.end());
            size_t idx = static_cast<size_t>(percentile * times.size());
            if (idx >= times.size()) idx = times.size() - 1;
            return times[idx];
        };
        
        double graph_exec_p50 = calculate_percentile(graph_exec_times_per_batch, 0.50);
        double graph_exec_p95 = calculate_percentile(graph_exec_times_per_batch, 0.95);
        double graph_exec_p99 = calculate_percentile(graph_exec_times_per_batch, 0.99);
        
        // Write metrics
        std::ofstream metrics_file(metrics_txt);
        if (!metrics_file) throw std::runtime_error("Cannot open metrics file: " + metrics_txt);
        
        metrics_file << std::fixed << std::setprecision(6);
        metrics_file << "=== QNN RAG Demo Performance Metrics (Batched) ===\n\n";
        
        metrics_file << "Dataset Information:\n";
        metrics_file << "  Number of queries: " << num_queries << "\n";
        metrics_file << "  Number of documents: " << num_docs << "\n";
        metrics_file << "  Dimension: " << query_dim << "\n";
        metrics_file << "  Batch size: " << model_batch_size << "\n";
        metrics_file << "  Number of batches: " << npu_times_per_batch.size() << "\n";
        metrics_file << "  Top-K: " << TOP_K << "\n\n";
        
        metrics_file << "Operational Intensity Analysis:\n";
        metrics_file << "  FLOPs per batch: " << std::scientific << flops_per_batch << "\n";
        metrics_file << "  Bytes moved per batch: " << std::fixed << total_bytes << "\n";
        metrics_file << "    - Query input: " << bytes_query << " bytes\n";
        metrics_file << "    - Doc matrix (reused!): " << bytes_docs << " bytes\n";
        metrics_file << "    - Output scores: " << bytes_output << " bytes\n";
        metrics_file << "  Operational Intensity: " << operational_intensity << " FLOPs/byte\n";
        if (model_batch_size == 1) {
            metrics_file << "  (Tip: Increase batch size to improve OI via data reuse)\n";
        } else {
            double oi_single = (2.0 * query_dim * num_docs) / (query_dim + query_dim * num_docs + num_docs);
            metrics_file << "  OI improvement vs batch=1: " << (operational_intensity / oi_single) << "x\n";
        }
        metrics_file << "\n";
        
        metrics_file << "Overall Performance:\n";
        metrics_file << "  Total execution time: " << total_time.count() << " s\n";
        metrics_file << "  Throughput: " << throughput_qps << " queries/sec\n\n";
        
        metrics_file << "NPU Execution (per batch):\n";
        metrics_file << "  Avg quantization time: " << avg_quantize_time_per_batch << " ms\n";
        metrics_file << "  Avg graph execute time: " << avg_graph_exec_time_per_batch << " ms\n";
        metrics_file << "  Avg total NPU time: " << avg_npu_time_per_batch << " ms\n";
        metrics_file << "  Std deviation (graph exec): " << graph_exec_stddev << " ms\n";
        metrics_file << "  Min graph exec time: " << *graph_exec_minmax.first << " ms\n";
        metrics_file << "  Max graph exec time: " << *graph_exec_minmax.second << " ms\n";
        metrics_file << "  P50 graph exec time: " << graph_exec_p50 << " ms\n";
        metrics_file << "  P95 graph exec time: " << graph_exec_p95 << " ms\n";
        metrics_file << "  P99 graph exec time: " << graph_exec_p99 << " ms\n\n";
        
        metrics_file << "NPU Performance (per batch):\n";
        metrics_file << "  Avg GFLOPS: " << avg_pure_npu_gflops << "\n";
        metrics_file << "  Max GFLOPS: " << max_pure_npu_gflops << "\n";
        metrics_file << "  Min GFLOPS: " << min_pure_npu_gflops << "\n\n";
        
        metrics_file << "Per-Query Amortized Performance:\n";
        metrics_file << "  Avg NPU time per query: " << avg_npu_time_per_query << " ms\n";
        metrics_file << "  Avg graph exec per query: " << avg_graph_exec_time_per_query << " ms\n";
        metrics_file << "  Avg CPU (top-K) per query: " << avg_cpu_time_per_query << " ms\n";
        metrics_file << "  Avg total per query: " << (total_time.count() * 1000.0 / num_queries) << " ms\n";
        metrics_file << "  Effective GFLOPS per query: " << avg_gflops_per_query << "\n\n";
        
        metrics_file << "CPU Post-processing (Heapify + Top-K per batch):\n";
        metrics_file << "  Total CPU time: " << (cpu_time_total/1000.0) << " s\n";
        metrics_file << "  Avg per batch: " << avg_cpu_time_per_batch << " ms\n";
        metrics_file << "  Min per batch: " << *cpu_minmax.first << " ms\n";
        metrics_file << "  Max per batch: " << *cpu_minmax.second << " ms\n\n";
        
        metrics_file << "Time Breakdown (% of end-to-end):\n";
        double total_ms = total_time.count() * 1000.0;
        metrics_file << "  Pure NPU (graphExecute): " << (graph_exec_time_total / total_ms * 100.0) << "%\n";
        metrics_file << "  Quantization (CPU):      " << (quantize_time_total / total_ms * 100.0) << "%\n";
        metrics_file << "  Heap + Top-K (CPU):      " << (cpu_time_total / total_ms * 100.0) << "%\n";
        metrics_file << "  Other overhead:          " << (100.0 - (graph_exec_time_total + quantize_time_total + cpu_time_total) / total_ms * 100.0) << "%\n";
        
        metrics_file.close();
        
        std::cout << "\nAll queries processed successfully." << std::endl;
        std::cout << "Batch size: " << model_batch_size << std::endl;
        std::cout << "Throughput: " << throughput_qps << " queries/sec" << std::endl;
        std::cout << "Avg GFLOPS: " << avg_pure_npu_gflops << std::endl;
        std::cout << "Operational Intensity: " << operational_intensity << " FLOPs/byte" << std::endl;
        std::cout << "Results saved to: " << results_txt << std::endl;
        std::cout << "Metrics saved to: " << metrics_txt << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "FATAL ERROR: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
