/**
 * IVF Vector Search - Main Entry Point
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <stdexcept>
#include <chrono>
#include <algorithm>
#include <iomanip>
#include <cmath>
#include <set>
#include <sys/stat.h>
#include "IVFIndex.h"

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

void load_ivecs(const std::string& filename, std::vector<std::vector<int>>& vectors, int& dim) {
    std::ifstream input(filename, std::ios::binary);
    if (!input) throw std::runtime_error("Cannot open file: " + filename);

    int d;
    while (input.read(reinterpret_cast<char*>(&d), sizeof(int))) {
        if (vectors.empty()) dim = d;
        if (d != dim) throw std::runtime_error("Dimension mismatch in file");

        std::vector<int> vec(dim);
        input.read(reinterpret_cast<char*>(vec.data()), dim * sizeof(int));
        if (!input) throw std::runtime_error("Error reading vector data");

        vectors.push_back(vec);
    }
}

double compute_recall(const std::vector<int>& predicted, const std::vector<int>& ground_truth, int k) {
    std::set<int> gt_set(ground_truth.begin(), ground_truth.begin() + std::min(k, (int)ground_truth.size()));
    int hits = 0;
    for (int i = 0; i < std::min(k, (int)predicted.size()); ++i) {
        if (gt_set.count(predicted[i])) hits++;
    }
    return static_cast<double>(hits) / k;
}

int main(int argc, char* argv[]) {
    if (argc < 6) {
        std::cerr << "Usage: " << argv[0]
                  << " <index_dir> <queries.fvecs> <results_dir> <backend.so> <top_k> [nprobe] [groundtruth.ivecs]" 
                  << std::endl;
        std::cerr << "\nExample:" << std::endl;
        std::cerr << "  " << argv[0] << " ./ivf_sift queries.fvecs results ./libQnnHtp.so 10 32" << std::endl;
        return 1;
    }

    std::string index_dir = argv[1];
    std::string query_file = argv[2];
    std::string results_dir = argv[3];
    std::string backend_path = argv[4];
    const int TOP_K = std::stoi(argv[5]);
    const int NPROBE = (argc > 6) ? std::stoi(argv[6]) : 16;
    std::string gt_file = (argc > 7) ? argv[7] : "";
    const int BATCH_SIZE = (argc > 8) ? std::stoi(argv[8]) : 1;

    try {
        mkdir(results_dir.c_str(), 0755);

        std::string results_txt = results_dir + "/results.txt";
        std::string metrics_txt = results_dir + "/metrics.txt";

        // Load queries
        std::cout << "Loading queries..." << std::endl;
        std::vector<std::vector<float>> query_vecs;
        int query_dim = 0;
        load_fvecs(query_file, query_vecs, query_dim);
        std::cout << "Loaded " << query_vecs.size() << " queries." << std::endl;

        // Load ground truth if provided
        std::vector<std::vector<int>> ground_truth;
        int gt_k = 0;
        if (!gt_file.empty()) {
            std::cout << "Loading ground truth..." << std::endl;
            load_ivecs(gt_file, ground_truth, gt_k);
            std::cout << "Loaded " << ground_truth.size() << " ground truth entries." << std::endl;
        }

        // Load IVF index
        std::cout << "Loading IVF index..." << std::endl;
        IVFIndex ivf(index_dir, backend_path);
        
        if (query_dim != static_cast<int>(ivf.getDim())) {
            throw std::runtime_error("Query dim (" + std::to_string(query_dim) + 
                                     ") != index dim (" + std::to_string(ivf.getDim()) + ")");
        }

        std::cout << "\nIndex configuration:" << std::endl;
        std::cout << "  Vectors: " << ivf.getNumVectors() << std::endl;
        std::cout << "  Clusters: " << ivf.getNumClusters() << std::endl;
        std::cout << "  Dimension: " << ivf.getDim() << std::endl;
        std::cout << "  nprobe: " << NPROBE << std::endl;
        std::cout << "  top_k: " << TOP_K << std::endl;

        std::ofstream results_file(results_txt);
        if (!results_file) throw std::runtime_error("Cannot open output file: " + results_txt);

        // Timing accumulators
        double total_centroid_ms = 0.0;
        double total_gather_ms = 0.0;
        double total_fine_ms = 0.0;
        double total_search_ms = 0.0;
        size_t total_candidates = 0;
        double total_recall = 0.0;
        
        std::vector<double> latencies;
        latencies.reserve(query_vecs.size());

        size_t num_queries = query_vecs.size();
        
        std::cout << "\nProcessing " << num_queries << " queries with nprobe=" << NPROBE 
                  << ", batch_size=" << BATCH_SIZE << "..." << std::endl;

        auto total_start = std::chrono::high_resolution_clock::now();

        for (size_t i = 0; i < num_queries; i += BATCH_SIZE) {
            size_t current_batch_size = std::min((size_t)BATCH_SIZE, num_queries - i);
            
            // Prepare batch
            std::vector<float> batch_queries;
            batch_queries.reserve(BATCH_SIZE * query_dim); // Reserve full batch size
            for (size_t j = 0; j < current_batch_size; ++j) {
                batch_queries.insert(batch_queries.end(), query_vecs[i + j].begin(), query_vecs[i + j].end());
            }
            
            // Pad with zeros if necessary
            if (current_batch_size < (size_t)BATCH_SIZE) {
                size_t padding_vectors = (size_t)BATCH_SIZE - current_batch_size;
                batch_queries.insert(batch_queries.end(), padding_vectors * query_dim, 0.0f);
            }
            
            std::vector<std::vector<int>> batch_indices;
            std::vector<std::vector<float>> batch_scores;
            IVFIndex::SearchTiming timing;
            
            // Always pass BATCH_SIZE to searchBatch, as the model expects fixed size
            size_t batch_candidates = ivf.searchBatch(batch_queries, BATCH_SIZE, TOP_K, NPROBE, 
                                                      batch_indices, batch_scores, timing);
            
            total_centroid_ms += timing.centroid_search_ms;
            total_gather_ms += timing.gather_ms;
            total_fine_ms += timing.fine_search_ms;
            total_search_ms += timing.total_ms;
            total_candidates += batch_candidates;
            
            // Record latency for each query in the batch (batch latency)
            for (size_t j = 0; j < current_batch_size; ++j) {
                latencies.push_back(timing.total_ms);
                
                // Compute recall
                if (!ground_truth.empty() && (i + j) < ground_truth.size()) {
                    total_recall += compute_recall(batch_indices[j], ground_truth[i + j], TOP_K);
                }
                
                // Write results
                results_file << "Query " << (i + j) << ":";
                for (size_t k = 0; k < batch_indices[j].size(); ++k) {
                    results_file << " (" << batch_indices[j][k] << ", " << std::fixed << std::setprecision(4) << batch_scores[j][k] << ")";
                }
                results_file << "\n";
            }

            if ((i + current_batch_size) % 100 == 0 || (i + current_batch_size) >= num_queries) {
                std::cout << "Processed " << (i + current_batch_size) << "/" << num_queries << " queries." << std::endl;
            }
        }

        auto total_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> total_time = total_end - total_start;
        
        results_file.close();
        
        // Calculate statistics
        double avg_latency = total_search_ms / num_queries;
        double avg_candidates = static_cast<double>(total_candidates) / num_queries;
        double avg_recall = ground_truth.empty() ? 0.0 : total_recall / num_queries;
        double throughput = num_queries / total_time.count();
        
        // Latency percentiles
        std::sort(latencies.begin(), latencies.end());
        double p50 = latencies[latencies.size() / 2];
        double p95 = latencies[static_cast<size_t>(latencies.size() * 0.95)];
        double p99 = latencies[static_cast<size_t>(latencies.size() * 0.99)];
        
        // Speedup vs brute force estimate
        double brute_force_candidates = ivf.getNumVectors();
        double speedup_candidates = brute_force_candidates / avg_candidates;

        // Write metrics
        std::ofstream metrics_file(metrics_txt);
        if (!metrics_file) throw std::runtime_error("Cannot open metrics file: " + metrics_txt);
        
        metrics_file << std::fixed << std::setprecision(6);
        metrics_file << "=== IVF Search Performance Metrics ===\n\n";
        
        metrics_file << "Index Configuration:\n";
        metrics_file << "  Total vectors: " << ivf.getNumVectors() << "\n";
        metrics_file << "  Number of clusters: " << ivf.getNumClusters() << "\n";
        metrics_file << "  Dimension: " << ivf.getDim() << "\n";
        metrics_file << "  nprobe: " << NPROBE << "\n";
        metrics_file << "  top_k: " << TOP_K << "\n";
        metrics_file << "  batch_size: " << BATCH_SIZE << "\n\n";
        
        metrics_file << "Query Statistics:\n";
        metrics_file << "  Number of queries: " << num_queries << "\n";
        metrics_file << "  Avg candidates searched: " << avg_candidates << "\n";
        metrics_file << "  Candidate reduction: " << speedup_candidates << "x\n\n";
        
        if (!ground_truth.empty()) {
            metrics_file << "Accuracy:\n";
            metrics_file << "  Recall@" << TOP_K << ": " << (avg_recall * 100.0) << "%\n\n";
        }
        
        metrics_file << "Latency:\n";
        metrics_file << "  Avg per query (amortized): " << avg_latency << " ms\n";
        metrics_file << "  Avg centroid search (NPU): " << (total_centroid_ms / num_queries) << " ms\n";
        metrics_file << "  Avg gather: " << (total_gather_ms / num_queries) << " ms\n";
        metrics_file << "  Avg fine search (NEON): " << (total_fine_ms / num_queries) << " ms\n";
        metrics_file << "  Batch P50: " << p50 << " ms\n";
        metrics_file << "  Batch P95: " << p95 << " ms\n";
        metrics_file << "  Batch P99: " << p99 << " ms\n\n";
        
        metrics_file << "Throughput:\n";
        metrics_file << "  Total time: " << total_time.count() << " s\n";
        metrics_file << "  QPS: " << throughput << "\n\n";
        
        // GFLOPS calculation
        // Centroid search: query (1 x dim) * centroids (dim x n_clusters) = 2 * dim * n_clusters FLOPs
        // Fine search: query (1 x dim) * candidates (dim x avg_candidates) = 2 * dim * avg_candidates FLOPs
        double centroid_flops_per_query = 2.0 * ivf.getDim() * ivf.getNumClusters();
        double fine_flops_per_query = 2.0 * ivf.getDim() * avg_candidates;
        double total_flops_per_query = centroid_flops_per_query + fine_flops_per_query;
        double total_flops = total_flops_per_query * num_queries;
        double gflops = total_flops / (total_time.count() * 1e9);
        double avg_gflops_per_query = (total_flops_per_query / 1e9) / (avg_latency / 1000.0);
        
        metrics_file << "Compute:\n";
        metrics_file << "  FLOPs per query (centroid): " << std::scientific << centroid_flops_per_query << "\n";
        metrics_file << "  FLOPs per query (fine): " << fine_flops_per_query << "\n";
        metrics_file << "  FLOPs per query (total): " << total_flops_per_query << "\n";
        metrics_file << "  Avg GFLOPS: " << std::fixed << avg_gflops_per_query << "\n";
        metrics_file << "  Total GFLOPS: " << gflops << "\n\n";
        
        metrics_file << "Time Breakdown:\n";
        double total_ms = total_time.count() * 1000.0;
        metrics_file << "  Centroid search (NPU): " << (total_centroid_ms / total_ms * 100.0) << "%\n";
        metrics_file << "  Gather candidates: " << (total_gather_ms / total_ms * 100.0) << "%\n";
        metrics_file << "  Fine search (NEON): " << (total_fine_ms / total_ms * 100.0) << "%\n";
        
        metrics_file.close();
        
        // Print summary
        std::cout << "\n=== IVF Search Complete ===" << std::endl;
        std::cout << "Throughput: " << throughput << " QPS" << std::endl;
        std::cout << "Avg latency: " << avg_latency << " ms" << std::endl;
        std::cout << "Avg GFLOPS: " << avg_gflops_per_query << std::endl;
        std::cout << "Avg candidates: " << avg_candidates << " (" << speedup_candidates << "x reduction)" << std::endl;
        if (!ground_truth.empty()) {
            std::cout << "Recall@" << TOP_K << ": " << (avg_recall * 100.0) << "%" << std::endl;
        }
        std::cout << "\nResults saved to: " << results_txt << std::endl;
        std::cout << "Metrics saved to: " << metrics_txt << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "FATAL ERROR: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
