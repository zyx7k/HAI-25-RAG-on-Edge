
// hnsw_search.h
#ifndef HNSW_SEARCH_H
#define HNSW_SEARCH_H

#include <vector>
#include <queue>
#include <unordered_set>
#include <string>
#include <memory>
#include "QnnRunner.h"

struct HNSWIndex
{
    uint32_t num_vectors;
    uint32_t dim;
    uint32_t M;
    uint32_t M_max;
    uint32_t M_max0;
    uint32_t entry_point;
    uint32_t max_layer;

    // Graph structure: graph[point_id][layer] = vector of neighbors
    std::vector<std::vector<std::vector<uint32_t>>> graph;

    // Vector data for CPU fallback distance computations
    std::vector<std::vector<float>> vectors;
};

class HNSWSearcher
{
public:
    HNSWSearcher(const std::string &index_path,
                 const std::string &vectors_path,
                 std::shared_ptr<QnnRunner> qnn_runner,
                 size_t batch_size = 32);

    // Search for k nearest neighbors
    std::vector<std::pair<uint32_t, float>> search(
        const std::vector<float> &query,
        size_t k,
        size_t ef_search = 50);

private:
    HNSWIndex index_;
    std::shared_ptr<QnnRunner> qnn_runner_;
    size_t batch_size_;

    // Batch buffer for NPU computation
    std::vector<float> batch_queries_;
    std::vector<float> batch_scores_;
    // Precomputed squared norms for document vectors (||d||^2)
    std::vector<float> doc_norms_;

    void load_index(const std::string &index_path);
    void load_vectors(const std::string &vectors_path);

    // Compute distances for a batch of candidates using NPU
    void compute_distances_batch(
        const std::vector<float> &query,
        const std::vector<uint32_t> &candidates,
        std::vector<float> &distances);

    // Core HNSW search at a specific layer
    std::vector<std::pair<float, uint32_t>> search_layer(
        const std::vector<float> &query,
        const std::unordered_set<uint32_t> &entry_points,
        size_t num_closest,
        uint32_t layer);
};

#endif // HNSW_SEARCH_H