#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <stdexcept>
#include <chrono>
#include <algorithm>
#include <iomanip>
#include "QnnRunner.h"

// --- Helper: .fvecs Loader ---
void load_fvecs(const std::string& filename, std::vector<std::vector<float>>& vectors, int& dim) {
    std::ifstream input(filename, std::ios::binary);
    if (!input) {
        throw std::runtime_error("Cannot open file: " + filename);
    }

    int d;
    while (input.read(reinterpret_cast<char*>(&d), sizeof(int))) {
        if (vectors.empty()) {
            dim = d; // First vector sets the dimension
        }
        if (d != dim) {
            throw std::runtime_error("Dimension mismatch in file");
        }
        
        std::vector<float> vec(dim);
        input.read(reinterpret_cast<char*>(vec.data()), dim * sizeof(float));
        
        if (!input) {
            throw std::runtime_error("Error reading vector data");
        }
        vectors.push_back(vec);
    }
}

// --- Helper: Top-K Finder ---
struct SearchResult {
    int index;
    float score;
    bool operator>(const SearchResult& other) const {
        return score < other.score; // Min-heap
    }
};

std::vector<SearchResult> find_top_k(const std::vector<float>& scores, int k) {
    std::vector<SearchResult> top_k;
    top_k.reserve(k);

    for (int i = 0; i < scores.size(); ++i) {
        if (top_k.size() < k) {
            top_k.push_back({i, scores[i]});
            std::push_heap(top_k.begin(), top_k.end(), std::greater<SearchResult>());
        } else if (scores[i] > top_k[0].score) {
            std::pop_heap(top_k.begin(), top_k.end(), std::greater<SearchResult>());
            top_k.back() = {i, scores[i]};
            std::push_heap(top_k.begin(), top_k.end(), std::greater<SearchResult>());
        }
    }

    std::sort(top_k.begin(), top_k.end(), [](const SearchResult& a, const SearchResult& b) {
        return a.score > b.score; // Sort descending
    });
    
    return top_k;
}

// --- Main ---
int main(int argc, char* argv[]) {
    if (argc != 5) {
        std::cerr << "Usage: " << argv[0] << " <model.bin> <queries.fvecs> <results.txt> <qnn_backend.so>" << std::endl;
        return 1;
    }

    std::string model_path = argv[1];
    std::string query_file = argv[2];
    std::string out_file = argv[3];
    std::string backend_path = argv[4];
    const int TOP_K = 5;

    try {
        // --- 1. Load Query Data ---
        std::cout << "Loading query vectors from " << query_file << "..." << std::endl;
        std::vector<std::vector<float>> query_vecs;
        int query_dim = 0;
        load_fvecs(query_file, query_vecs, query_dim);
        std::cout << "Loaded " << query_vecs.size() << " query vectors." << std::endl;

        // --- 2. Initialize QNN Runner ---
        QnnRunner runner(model_path, backend_path);

        // Verification
        int model_input_dim = runner.getInputDims().back();
        int model_output_dim = runner.getOutputDims().back();
        if (model_input_dim != query_dim) {
            throw std::runtime_error("Model input dim and query dim mismatch!");
        }

        // --- 3. Run Inference Loop ---
        std::cout << "Running inference on " << query_vecs.size() << " queries..." << std::endl;
        std::ofstream results_file(out_file);
        if (!results_file) {
            throw std::runtime_error("Failed to open output file: " + out_file);
        }

        std::vector<float> scores(model_output_dim); // Allocate output buffer once
        
        auto total_start = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < query_vecs.size(); ++i) {
            auto query_start = std::chrono::high_resolution_clock::now();
            
            // Execute on NPU
            runner.execute(query_vecs[i], scores);
            
            // Find Top-K on CPU
            auto top_k_results = find_top_k(scores, TOP_K);
            
            auto query_end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> query_ms = query_end - query_start;
            
            // Log results
            results_file << "Query " << i << " (Time: " << std::fixed << std::setprecision(2) << query_ms.count() << " ms): ";
            for (const auto& res : top_k_results) {
                results_file << "[" << res.index << "](" << std::setprecision(4) << res.score << ") ";
            }
            results_file << "\n";
            
            if ((i + 1) % 10 == 0) {
                 std::cout << "Processed " << (i + 1) << "/" << query_vecs.size() << " queries." << std::endl;
            }
        }

        auto total_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> total_sec = total_end - total_start;
        
        std::cout << "All queries processed." << std::endl;
        std::cout << "Total time: " << total_sec.count() << " seconds" << std::endl;
        std::cout << "Avg time per query: " << (total_sec.count() * 1000.0 / query_vecs.size()) << " ms" << std::endl;
        std::cout << "Results saved to " << out_file << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "FATAL ERROR: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}