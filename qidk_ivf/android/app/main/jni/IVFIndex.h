#ifndef IVFINDEX_H
#define IVFINDEX_H

#include <string>
#include <vector>
#include <cstdint>
#include "QnnRunner.h"

/**
 * IVF (Inverted File Index) for approximate nearest neighbor search.
 * Centroid search runs on NPU via QnnRunner.
 * Fine search uses NEON-optimized dot products on CPU.
 */
class IVFIndex {
public:
    /**
     * Load IVF index from directory.
     */
    IVFIndex(const std::string& indexDir, const std::string& backendPath = "./libQnnHtp.so");
    ~IVFIndex();
    
    /**
     * Search for k nearest neighbors.
     */
    size_t search(const std::vector<float>& query, int k, int nprobe,
                  std::vector<int>& indices, std::vector<float>& scores);
    
    /**
     * Search with timing information.
     */
    struct SearchTiming {
        double centroid_search_ms = 0.0;  // NPU centroid search time
        double gather_ms = 0.0;           // Candidate gathering time
        double fine_search_ms = 0.0;      // Fine search time (CPU/NEON)
        double total_ms = 0.0;
    };
    
    size_t search(const std::vector<float>& query, int k, int nprobe,
                  std::vector<int>& indices, std::vector<float>& scores,
                  SearchTiming& timing);

    /**
     * Batched search.
     */
    size_t searchBatch(const std::vector<float>& queries, int batchSize, int k, int nprobe,
                       std::vector<std::vector<int>>& allIndices, 
                       std::vector<std::vector<float>>& allScores,
                       SearchTiming& timing);
    
    // Getters
    size_t getNumVectors() const { return m_numVectors; }
    size_t getNumClusters() const { return m_numClusters; }
    size_t getDim() const { return m_dim; }
    float getAvgClusterSize() const { return m_avgClusterSize; }
    
private:
    void loadConfig(const std::string& configPath);
    void loadClusterData(const std::string& indexDir);
    void loadVectors(const std::string& indexDir);
    
    // NEON-optimized dot product for fine search (scattered indices)
    void computeDotProductsNeon(const float* query, const int* candidateIds, 
                                size_t numCandidates, float* scores);
    
    // NEON-optimized dot product for CONTIGUOUS range (cluster-reordered mode)
    void computeDotProductsContiguous(const float* query, size_t startIdx,
                                      size_t count, float* scores);
    
    // Top-K selection
    void findTopK(const float* scores, const int* candidateIds, size_t numCandidates,
                  int k, std::vector<int>& indices, std::vector<float>& topScores);
    
    // Top-K selection for reordered mode (processes cluster ranges)
    void findTopKReordered(const std::vector<std::pair<int, int>>& clusterRanges,
                           const float* query, int k,
                           std::vector<int>& indices, std::vector<float>& topScores);
    
    // Index data
    size_t m_numVectors;
    size_t m_numClusters;
    size_t m_dim;
    float m_avgClusterSize;
    bool m_isReordered;  // True if vectors are reordered by cluster
    
    // Cluster data
    std::vector<int32_t> m_clusterOffsets;
    std::vector<int32_t> m_clusterIndices;
    
    // Reordered mode mapping
    std::vector<int32_t> m_reorderToOriginal;
    
    // All vectors for fine search
    std::vector<float> m_vectors;
    
    // QNN runner for centroid search
    std::unique_ptr<QnnRunner> m_centroidRunner;
};

#endif // IVFINDEX_H
