#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <stdexcept>
#include <chrono>
#include <algorithm>
#include <iomanip>
#include <cmath>
#include "QnnRunner.h"

// --- Load .fvecs file ---
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

// --- Compute L2 distance (squared, no sqrt) ---
float compute_l2_distance_squared(const std::vector<float>& a, const std::vector<float>& b) {
    float dist = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        float diff = a[i] - b[i];
        dist += diff * diff;
    }
    return dist;
}

// --- Top-K selection by L2 distance ---
struct SearchResult {
    int index;
    float distance;
    // For max-heap: we want largest distance at root
    bool operator<(const SearchResult& other) const { return distance < other.distance; }
};

std::vector<SearchResult> find_top_k_by_l2(const std::vector<float>& query, 
                                            const std::vector<std::vector<float>>& docs, 
                                            int k) {
    std::vector<SearchResult> top_k;
    top_k.reserve(k);

    // Compute L2 distance to all documents and keep top-K (smallest distances)
    for (int i = 0; i < docs.size(); ++i) {
        float dist = compute_l2_distance_squared(query, docs[i]);
        
        if (top_k.size() < k) {
            top_k.push_back({i, dist});
            std::make_heap(top_k.begin(), top_k.end()); // max-heap
        } else if (dist < top_k.front().distance) {
            // New distance is smaller than the largest in our top-K
            std::pop_heap(top_k.begin(), top_k.end());
            top_k.back() = {i, dist};
            std::push_heap(top_k.begin(), top_k.end());
        }
    }

    // Sort by distance (ascending)
    std::sort(top_k.begin(), top_k.end(), [](const SearchResult& a, const SearchResult& b) {
        return a.distance < b.distance;
    });
    return top_k;
}

// --- Main ---
int main(int argc, char* argv[]) {
    if (argc != 6) {
        std::cerr << "Usage: " << argv[0]
                  << " <model.bin> <queries.fvecs> <results.txt> <qnn_backend.so> <documents.fvecs>" << std::endl;
        return 1;
    }

    std::string model_path = argv[1];
    std::string query_file = argv[2];
    std::string out_file = argv[3];
    std::string backend_path = argv[4];
    std::string doc_file = argv[5];
    const int TOP_K = 5;

    try {
        // 1. Load queries
        std::cout << "Loading queries from " << query_file << "..." << std::endl;
        std::vector<std::vector<float>> query_vecs;
        int query_dim = 0;
        load_fvecs(query_file, query_vecs, query_dim);
        std::cout << "Loaded " << query_vecs.size() << " queries." << std::endl;

        // 2. Load document embeddings
        std::cout << "Loading document embeddings from " << doc_file << "..." << std::endl;
        std::vector<std::vector<float>> doc_vecs;
        int doc_dim = 0;
        load_fvecs(doc_file, doc_vecs, doc_dim);
        std::cout << "Loaded " << doc_vecs.size() << " document embeddings." << std::endl;
        std::cout << "First doc first 5 dims: ";
        for (int i = 0; i < 5; ++i) std::cout << doc_vecs[0][i] << " ";
        std::cout << std::endl;
        std::cout << "Last doc index: " << (doc_vecs.size()-1) << ", first 5 dims: ";
        for (int i = 0; i < 5; ++i) std::cout << doc_vecs[doc_vecs.size()-1][i] << " ";
        std::cout << std::endl;
        
        if (query_dim != doc_dim) {
            throw std::runtime_error("Query and document dimensions don't match!");
        }

        // 3. Pre-compute document norms on CPU
        std::cout << "Computing document norms on CPU..." << std::endl;
        std::vector<float> doc_norms(doc_vecs.size());
        for (size_t i = 0; i < doc_vecs.size(); ++i) {
            float norm = 0.0f;
            for (float val : doc_vecs[i]) {
                norm += val * val;
            }
            doc_norms[i] = norm;
        }

        // 4. Initialize QNN for dot product computation
        QnnRunner runner(model_path, backend_path);
        int model_input_dim = runner.getInputDims().back();
        int model_output_dim = runner.getOutputDims().back();
        if (model_input_dim != query_dim)
            throw std::runtime_error("Model input dim and query dim mismatch!");
        if (model_output_dim != doc_vecs.size())
            throw std::runtime_error("Model output dim doesn't match number of documents!");

        // 5. Inference: NPU computes dot products, CPU converts to L2 distances
        std::ofstream results_file(out_file);
        if (!results_file) throw std::runtime_error("Cannot open output file: " + out_file);

        std::vector<float> dot_products(model_output_dim);
        auto total_start = std::chrono::high_resolution_clock::now();
        
        double npu_time_total = 0.0;
        double cpu_time_total = 0.0;

        for (int i = 0; i < query_vecs.size(); ++i) {
            // Compute query norm
            float query_norm = 0.0f;
            for (float val : query_vecs[i]) {
                query_norm += val * val;
            }

            // NPU: Compute dot products (q·d for all documents)
            auto npu_start = std::chrono::high_resolution_clock::now();
            runner.execute(query_vecs[i], dot_products);
            auto npu_end = std::chrono::high_resolution_clock::now();
            npu_time_total += std::chrono::duration<double, std::milli>(npu_end - npu_start).count();

            // CPU: Convert dot products to L2 distances: dist² = ||q||² + ||d||² - 2*(q·d)
            auto cpu_start = std::chrono::high_resolution_clock::now();
            std::vector<float> distances(doc_vecs.size());
            for (size_t j = 0; j < doc_vecs.size(); ++j) {
                distances[j] = query_norm + doc_norms[j] - 2.0f * dot_products[j];
            }

            // CPU: Find top-K smallest distances
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
            cpu_time_total += std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();

            // Write results
            results_file << "Query " << i << ":";
            for (const auto& r : top_k) {
                results_file << " (" << r.index << ", " << std::fixed << std::setprecision(0) << r.distance << ")";
            }
            results_file << "\n";

            if ((i + 1) % 10 == 0)
                std::cout << "Processed " << (i + 1) << "/" << query_vecs.size() << " queries.\n";
        }

        auto total_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> total_time = total_end - total_start;

        std::cout << "All queries processed.\n"
                  << "Total time: " << total_time.count() << " s\n"
                  << "  NPU time: " << (npu_time_total/1000.0) << " s (" << (npu_time_total/query_vecs.size()) << " ms/query)\n"
                  << "  CPU time: " << (cpu_time_total/1000.0) << " s (" << (cpu_time_total/query_vecs.size()) << " ms/query)\n"
                  << "Avg/query: " << (total_time.count() * 1000.0 / query_vecs.size()) << " ms\n"
                  << "Results saved to " << out_file << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "FATAL ERROR: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
