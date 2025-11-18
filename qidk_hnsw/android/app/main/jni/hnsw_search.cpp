#include "hnsw_search.h"
#include <fstream>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <cstring>

HNSWSearcher::HNSWSearcher(const std::string &index_path,
                           const std::string &vectors_path,
                           std::shared_ptr<QnnRunner> qnn_runner,
                           size_t batch_size)
    : qnn_runner_(std::move(qnn_runner)), batch_size_(batch_size)
{
    if (!qnn_runner_)
    {
        throw std::runtime_error("QnnRunner must be provided for NPU-only execution");
    }

    load_index(index_path);
    load_vectors(vectors_path);

    // Pre-allocate batch buffers
    batch_queries_.resize(batch_size_ * index_.dim);
    batch_scores_.resize(batch_size_ * index_.num_vectors);

    std::cout << "HNSW Searcher initialized:" << std::endl;
    std::cout << "  Vectors: " << index_.num_vectors << std::endl;
    std::cout << "  Dimension: " << index_.dim << std::endl;
    std::cout << "  Entry point: " << index_.entry_point << std::endl;
    std::cout << "  Max layer: " << index_.max_layer << std::endl;
    std::cout << "  NPU batch size: " << batch_size_ << std::endl;
    std::cout << "  NPU acceleration: enforced" << std::endl;
}

void HNSWSearcher::load_index(const std::string &index_path)
{
    std::ifstream f(index_path, std::ios::binary);
    if (!f)
        throw std::runtime_error("Cannot open HNSW index: " + index_path);

    // Read header
    f.read(reinterpret_cast<char *>(&index_.num_vectors), sizeof(uint32_t));
    f.read(reinterpret_cast<char *>(&index_.dim), sizeof(uint32_t));
    f.read(reinterpret_cast<char *>(&index_.M), sizeof(uint32_t));
    f.read(reinterpret_cast<char *>(&index_.M_max), sizeof(uint32_t));
    f.read(reinterpret_cast<char *>(&index_.M_max0), sizeof(uint32_t));
    f.read(reinterpret_cast<char *>(&index_.entry_point), sizeof(uint32_t));
    f.read(reinterpret_cast<char *>(&index_.max_layer), sizeof(uint32_t));

    // Read graph
    index_.graph.resize(index_.num_vectors);
    for (uint32_t point_id = 0; point_id < index_.num_vectors; ++point_id)
    {
        uint32_t num_layers;
        f.read(reinterpret_cast<char *>(&num_layers), sizeof(uint32_t));

        index_.graph[point_id].resize(num_layers);
        for (uint32_t layer = 0; layer < num_layers; ++layer)
        {
            uint32_t num_neighbors;
            f.read(reinterpret_cast<char *>(&num_neighbors), sizeof(uint32_t));

            index_.graph[point_id][layer].resize(num_neighbors);
            f.read(reinterpret_cast<char *>(index_.graph[point_id][layer].data()),
                   num_neighbors * sizeof(uint32_t));
        }
    }
}

void HNSWSearcher::load_vectors(const std::string &vectors_path)
{
    std::ifstream f(vectors_path, std::ios::binary);
    if (!f)
        throw std::runtime_error("Cannot open vectors: " + vectors_path);

    index_.vectors.reserve(index_.num_vectors);

    for (uint32_t i = 0; i < index_.num_vectors; ++i)
    {
        int32_t dim;
        f.read(reinterpret_cast<char *>(&dim), sizeof(int32_t));
        if (dim != index_.dim)
            throw std::runtime_error("Dimension mismatch in vectors");

        std::vector<float> vec(dim);
        f.read(reinterpret_cast<char *>(vec.data()), dim * sizeof(float));
        index_.vectors.push_back(std::move(vec));
        // Precompute squared norm for this document vector
        float norm_sq = 0.0f;
        for (size_t d = 0; d < index_.dim; ++d) norm_sq += index_.vectors.back()[d] * index_.vectors.back()[d];
        doc_norms_.push_back(norm_sq);
    }
}

void HNSWSearcher::compute_distances_batch(
    const std::vector<float> &query,
    const std::vector<uint32_t> &candidates,
    std::vector<float> &distances)
{

    distances.resize(candidates.size());

    // Process in batches for NPU efficiency
    size_t num_batches = (candidates.size() + batch_size_ - 1) / batch_size_;

    for (size_t batch_idx = 0; batch_idx < num_batches; ++batch_idx)
    {
        size_t start_idx = batch_idx * batch_size_;
        size_t end_idx = std::min(start_idx + batch_size_, candidates.size());
        size_t current_batch_size = end_idx - start_idx;

        // For this deployment we require NPU execution for distance computations.
        // If the QNN runner is not available or NPU execution fails, throw an error
        // so the caller can abort — we do NOT fall back to CPU here.
        if (!qnn_runner_) {
            throw std::runtime_error("NPU runner not available: QNN must be initialized for NPU-only execution");
        }

        // Create batch matrix: [batch_size, dim] (replicate the query for each doc)
        for (size_t i = 0; i < current_batch_size; ++i)
        {
            std::memcpy(&batch_queries_[i * index_.dim],
                        query.data(),
                        index_.dim * sizeof(float));
        }

        // Call NPU with a batch of identical queries to get scores for all docs
        std::vector<float> npu_batch_scores;
        size_t batch_input_elems = current_batch_size * index_.dim;
        std::vector<float> batch_input_view(batch_queries_.begin(), batch_queries_.begin() + batch_input_elems);

        try {
            qnn_runner_->executeBatch(batch_input_view, current_batch_size, npu_batch_scores);
        } catch (const std::exception &e) {
            throw std::runtime_error(std::string("NPU execution failed: ") + e.what());
        }

        // Determine output size (per-sample)
        const auto outDims = qnn_runner_->getOutputDims();
        size_t out_elems_per_sample = 1;
        if (!outDims.empty()) {
            // assume batch is first dim
            for (size_t d = 1; d < outDims.size(); ++d) out_elems_per_sample *= outDims[d];
            if (outDims.size() == 1) out_elems_per_sample = outDims[0];
        } else {
            // If graph info is unavailable, we cannot safely interpret outputs
            throw std::runtime_error("QNN runner returned empty output dimensions; cannot map outputs to document scores");
        }

        // Convert NPU dot-product scores to L2 squared distances:
        // dist(q,d) = ||q||^2 + ||d||^2 - 2 * (q . d)
        // Precompute query norm
        float q_norm_sq = 0.0f;
        for (size_t t = 0; t < index_.dim; ++t) q_norm_sq += query[t] * query[t];

        // Sanity: out_elems_per_sample should match number of documents
        if (out_elems_per_sample != index_.num_vectors) {
            throw std::runtime_error("Unexpected QNN output shape: per-sample output size does not match number of indexed vectors");
        }

        for (size_t i = 0; i < current_batch_size; ++i)
        {
            uint32_t doc_id = candidates[start_idx + i];
            size_t offset = i * out_elems_per_sample;
            // dot product from NPU for this (query_i, all_docs)
            float dot = npu_batch_scores[offset + doc_id];
            // Use precomputed doc norm
            float doc_norm_sq = doc_norms_[doc_id];
            float dist = q_norm_sq + doc_norm_sq - 2.0f * dot;
            distances[start_idx + i] = dist;
        }
    }
}

std::vector<std::pair<float, uint32_t>> HNSWSearcher::search_layer(
    const std::vector<float> &query,
    const std::unordered_set<uint32_t> &entry_points,
    size_t num_closest,
    uint32_t layer)
{

    std::unordered_set<uint32_t> visited;

    // Max heap for candidates (to get minimum)
    std::priority_queue<std::pair<float, uint32_t>> candidates;
    // Min heap for results
    std::priority_queue<std::pair<float, uint32_t>,
                        std::vector<std::pair<float, uint32_t>>,
                        std::greater<std::pair<float, uint32_t>>>
        w;

    // Initialize with entry points
    std::vector<uint32_t> entry_vec(entry_points.begin(), entry_points.end());
    std::vector<float> entry_dists;
    compute_distances_batch(query, entry_vec, entry_dists);

    for (size_t i = 0; i < entry_vec.size(); ++i)
    {
        uint32_t point = entry_vec[i];
        float dist = entry_dists[i];
        candidates.push({-dist, point});
        w.push({dist, point});
        visited.insert(point);
    }

    while (!candidates.empty())
    {
        auto [neg_current_dist, current] = candidates.top();
        candidates.pop();
        float current_dist = -neg_current_dist;

        if (current_dist > w.top().first)
        {
            break;
        }

        // Get neighbors at this layer
        const auto &neighbors = (layer < index_.graph[current].size())
                                    ? index_.graph[current][layer]
                                    : std::vector<uint32_t>();

        // Batch compute distances for unvisited neighbors
        std::vector<uint32_t> unvisited_neighbors;
        for (uint32_t neighbor : neighbors)
        {
            if (visited.find(neighbor) == visited.end())
            {
                unvisited_neighbors.push_back(neighbor);
                visited.insert(neighbor);
            }
        }

        if (!unvisited_neighbors.empty())
        {
            std::vector<float> neighbor_dists;
            compute_distances_batch(query, unvisited_neighbors, neighbor_dists);

            for (size_t i = 0; i < unvisited_neighbors.size(); ++i)
            {
                float dist = neighbor_dists[i];
                uint32_t neighbor = unvisited_neighbors[i];

                if (dist < w.top().first || w.size() < num_closest)
                {
                    candidates.push({-dist, neighbor});
                    w.push({dist, neighbor});

                    if (w.size() > num_closest)
                    {
                        w.pop();
                    }
                }
            }
        }
    }

    // Convert to vector
    std::vector<std::pair<float, uint32_t>> result;
    while (!w.empty())
    {
        result.push_back(w.top());
        w.pop();
    }
    std::reverse(result.begin(), result.end());

    return result;
}

std::vector<std::pair<uint32_t, float>> HNSWSearcher::search(
    const std::vector<float> &query,
    size_t k,
    size_t ef_search)
{

    if (query.size() != index_.dim)
    {
        throw std::runtime_error("Query dimension mismatch");
    }

    std::unordered_set<uint32_t> current_nearest = {index_.entry_point};

    // Traverse from top layer to layer 1
    for (int layer = index_.max_layer; layer >= 1; --layer)
    {
        auto results = search_layer(query, current_nearest, 1, layer);
        current_nearest.clear();
        for (const auto &[dist, point] : results)
        {
            current_nearest.insert(point);
        }
    }

    // Search layer 0 with ef_search
    auto results = search_layer(query, current_nearest, ef_search, 0);

    // Return top-k
    std::vector<std::pair<uint32_t, float>> top_k;
    for (size_t i = 0; i < std::min(k, results.size()); ++i)
    {
        top_k.push_back({results[i].second, results[i].first});
    }

    return top_k;
}