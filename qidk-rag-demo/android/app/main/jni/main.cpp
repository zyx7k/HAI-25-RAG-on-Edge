#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <stdexcept>
#include <chrono>
#include <algorithm>
#include <iomanip>
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

// --- Top-K selection ---
struct SearchResult {
    int index;
    float score;
    bool operator>(const SearchResult& other) const { return score < other.score; } // Min-heap
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
        return a.score > b.score;
    });
    return top_k;
}

// --- Main ---
int main(int argc, char* argv[]) {
    if (argc != 5) {
        std::cerr << "Usage: " << argv[0]
                  << " <model.bin> <queries.fvecs> <results.txt> <qnn_backend.so>" << std::endl;
        return 1;
    }

    std::string model_path = argv[1];
    std::string query_file = argv[2];
    std::string out_file = argv[3];
    std::string backend_path = argv[4];
    const int TOP_K = 5;

    try {
        // 1. Load queries
        std::cout << "Loading queries from " << query_file << "..." << std::endl;
        std::vector<std::vector<float>> query_vecs;
        int query_dim = 0;
        load_fvecs(query_file, query_vecs, query_dim);
        std::cout << "Loaded " << query_vecs.size() << " queries." << std::endl;

        // 2. Initialize QNN
        QnnRunner runner(model_path, backend_path);
        int model_input_dim = runner.getInputDims().back();
        int model_output_dim = runner.getOutputDims().back();
        if (model_input_dim != query_dim)
            throw std::runtime_error("Model input dim and query dim mismatch!");

        // 3. Inference
        std::ofstream results_file(out_file);
        if (!results_file) throw std::runtime_error("Cannot open output file: " + out_file);

        std::vector<float> scores(model_output_dim);
        auto total_start = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < query_vecs.size(); ++i) {
            auto q_start = std::chrono::high_resolution_clock::now();

            runner.execute(query_vecs[i], scores);
            auto top_k = find_top_k(scores, TOP_K);

            auto q_end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> q_time = q_end - q_start;

            results_file << "Query " << i << " (" << std::fixed << std::setprecision(2)
                         << q_time.count() << " ms): ";
            for (const auto& r : top_k)
                results_file << "[" << r.index << "](" << std::setprecision(4) << r.score << ") ";
            results_file << "\n";

            if ((i + 1) % 10 == 0)
                std::cout << "Processed " << (i + 1) << "/" << query_vecs.size() << " queries.\n";
        }

        auto total_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> total_time = total_end - total_start;

        std::cout << "All queries processed.\n"
                  << "Total time: " << total_time.count() << " s\n"
                  << "Avg/query: " << (total_time.count() * 1000.0 / query_vecs.size()) << " ms\n"
                  << "Results saved to " << out_file << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "FATAL ERROR: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
