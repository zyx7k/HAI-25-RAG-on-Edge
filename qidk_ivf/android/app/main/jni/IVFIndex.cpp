#include "IVFIndex.h"
#include <fstream>
#include <iostream>
#include <cstring>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <arm_neon.h>
#include <omp.h>

#define IVF_LOG(level, msg) std::cout << "[IVF " << level << "] " << msg << std::endl

static bool parseJsonValue(const std::string& json, const std::string& key, size_t& value) {
    size_t pos = json.find("\"" + key + "\"");
    if (pos == std::string::npos) return false;
    pos = json.find(":", pos);
    if (pos == std::string::npos) return false;
    pos++;
    while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\t')) pos++;
    value = std::stoul(json.substr(pos));
    return true;
}

static bool parseJsonFloat(const std::string& json, const std::string& key, float& value) {
    size_t pos = json.find("\"" + key + "\"");
    if (pos == std::string::npos) return false;
    pos = json.find(":", pos);
    if (pos == std::string::npos) return false;
    pos++;
    while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\t')) pos++;
    value = std::stof(json.substr(pos));
    return true;
}

static bool parseJsonBool(const std::string& json, const std::string& key, bool& value) {
    size_t pos = json.find("\"" + key + "\"");
    if (pos == std::string::npos) return false;
    pos = json.find(":", pos);
    if (pos == std::string::npos) return false;
    pos++;
    while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\t')) pos++;
    if (json.substr(pos, 4) == "true") {
        value = true;
        return true;
    } else if (json.substr(pos, 5) == "false") {
        value = false;
        return true;
    }
    return true;
}

static bool loadNpyFloat(const std::string& path, std::vector<float>& data, std::vector<size_t>& shape) {
    std::ifstream file(path, std::ios::binary);
    if (!file) return false;
    
    // Read magic number and version
    char magic[6];
    file.read(magic, 6);
    if (std::string(magic, 6) != "\x93NUMPY") return false;
    
    uint8_t major, minor;
    file.read(reinterpret_cast<char*>(&major), 1);
    file.read(reinterpret_cast<char*>(&minor), 1);
    
    // Read header length
    uint16_t headerLen;
    if (major == 1) {
        file.read(reinterpret_cast<char*>(&headerLen), 2);
    } else {
        uint32_t headerLen32;
        file.read(reinterpret_cast<char*>(&headerLen32), 4);
        headerLen = static_cast<uint16_t>(headerLen32);
    }
    
    // Read and parse header
    std::string header(headerLen, '\0');
    file.read(&header[0], headerLen);
    
    // Parse shape from header (simple parsing)
    size_t shapeStart = header.find("'shape': (");
    if (shapeStart == std::string::npos) return false;
    shapeStart += 10;
    size_t shapeEnd = header.find(")", shapeStart);
    std::string shapeStr = header.substr(shapeStart, shapeEnd - shapeStart);
    
    // Parse dimensions
    shape.clear();
    size_t totalSize = 1;
    size_t pos = 0;
    while (pos < shapeStr.size()) {
        while (pos < shapeStr.size() && !isdigit(shapeStr[pos])) pos++;
        if (pos >= shapeStr.size()) break;
        size_t dim = std::stoul(shapeStr.substr(pos));
        shape.push_back(dim);
        totalSize *= dim;
        while (pos < shapeStr.size() && isdigit(shapeStr[pos])) pos++;
    }
    
    // Read data
    data.resize(totalSize);
    file.read(reinterpret_cast<char*>(data.data()), totalSize * sizeof(float));
    
    return file.good();
}

static bool loadNpyInt32(const std::string& path, std::vector<int32_t>& data, std::vector<size_t>& shape) {
    std::ifstream file(path, std::ios::binary);
    if (!file) return false;
    
    char magic[6];
    file.read(magic, 6);
    if (std::string(magic, 6) != "\x93NUMPY") return false;
    
    uint8_t major, minor;
    file.read(reinterpret_cast<char*>(&major), 1);
    file.read(reinterpret_cast<char*>(&minor), 1);
    
    uint16_t headerLen;
    if (major == 1) {
        file.read(reinterpret_cast<char*>(&headerLen), 2);
    } else {
        uint32_t headerLen32;
        file.read(reinterpret_cast<char*>(&headerLen32), 4);
        headerLen = static_cast<uint16_t>(headerLen32);
    }
    
    std::string header(headerLen, '\0');
    file.read(&header[0], headerLen);
    
    size_t shapeStart = header.find("'shape': (");
    if (shapeStart == std::string::npos) return false;
    shapeStart += 10;
    size_t shapeEnd = header.find(")", shapeStart);
    std::string shapeStr = header.substr(shapeStart, shapeEnd - shapeStart);
    
    shape.clear();
    size_t totalSize = 1;
    size_t pos = 0;
    while (pos < shapeStr.size()) {
        while (pos < shapeStr.size() && !isdigit(shapeStr[pos])) pos++;
        if (pos >= shapeStr.size()) break;
        size_t dim = std::stoul(shapeStr.substr(pos));
        shape.push_back(dim);
        totalSize *= dim;
        while (pos < shapeStr.size() && isdigit(shapeStr[pos])) pos++;
    }
    
    data.resize(totalSize);
    file.read(reinterpret_cast<char*>(data.data()), totalSize * sizeof(int32_t));
    
    return file.good();
}

IVFIndex::IVFIndex(const std::string& indexDir, const std::string& backendPath) {
    IVF_LOG("INFO", "Loading IVF index from: " << indexDir);
    
    // Load configuration
    loadConfig(indexDir + "/ivf_config.json");
    
    // Load cluster data (inverted lists)
    loadClusterData(indexDir);
    
    // Load vectors for fine search
    loadVectors(indexDir);
    
    // Load centroid search model (QNN)
    std::string centroidBinPath = indexDir + "/centroids.bin";
    IVF_LOG("INFO", "Loading centroid model: " << centroidBinPath);
    m_centroidRunner = std::make_unique<QnnRunner>(centroidBinPath, backendPath);
    
    IVF_LOG("INFO", "IVF index loaded successfully!");
    IVF_LOG("INFO", "  Vectors: " << m_numVectors);
    IVF_LOG("INFO", "  Clusters: " << m_numClusters);
    IVF_LOG("INFO", "  Dimension: " << m_dim);
    IVF_LOG("INFO", "  Avg cluster size: " << m_avgClusterSize);
    IVF_LOG("INFO", "  Reordered mode: " << (m_isReordered ? "YES" : "NO"));
}

IVFIndex::~IVFIndex() = default;

void IVFIndex::loadConfig(const std::string& configPath) {
    std::ifstream file(configPath);
    if (!file) {
        throw std::runtime_error("Cannot open config file: " + configPath);
    }
    
    std::string json((std::istreambuf_iterator<char>(file)),
                     std::istreambuf_iterator<char>());
    
    if (!parseJsonValue(json, "n_vectors", m_numVectors)) {
        throw std::runtime_error("Missing n_vectors in config");
    }
    if (!parseJsonValue(json, "n_clusters", m_numClusters)) {
        throw std::runtime_error("Missing n_clusters in config");
    }
    if (!parseJsonValue(json, "dim", m_dim)) {
        throw std::runtime_error("Missing dim in config");
    }
    parseJsonFloat(json, "avg_cluster_size", m_avgClusterSize);
    
    // Check if vectors are reordered by cluster
    m_isReordered = false;
    parseJsonBool(json, "reordered", m_isReordered);
}

void IVFIndex::loadClusterData(const std::string& indexDir) {
    std::vector<size_t> shape;
    
    // Load cluster offsets
    std::string offsetsPath = indexDir + "/cluster_offsets.npy";
    if (!loadNpyInt32(offsetsPath, m_clusterOffsets, shape)) {
        throw std::runtime_error("Failed to load cluster offsets: " + offsetsPath);
    }
    IVF_LOG("INFO", "Loaded cluster offsets: " << m_clusterOffsets.size() << " entries");
    
    // Load cluster indices (only for non-reordered mode)
    if (!m_isReordered) {
        std::string indicesPath = indexDir + "/cluster_indices.npy";
        if (!loadNpyInt32(indicesPath, m_clusterIndices, shape)) {
            throw std::runtime_error("Failed to load cluster indices: " + indicesPath);
        }
        IVF_LOG("INFO", "Loaded cluster indices: " << m_clusterIndices.size() << " entries");
    } else {
        // Reordered mode: load the mapping from reordered index to original
        std::string mapPath = indexDir + "/reorder_to_original.npy";
        if (!loadNpyInt32(mapPath, m_reorderToOriginal, shape)) {
            throw std::runtime_error("Failed to load reorder map: " + mapPath);
        }
        IVF_LOG("INFO", "Loaded reorder-to-original map: " << m_reorderToOriginal.size() << " entries");
        IVF_LOG("INFO", "Using REORDERED mode (contiguous cluster vectors)");
    }
}

void IVFIndex::loadVectors(const std::string& indexDir) {
    std::vector<size_t> shape;
    
    // For reordered mode, load reordered vectors
    if (m_isReordered) {
        std::string reorderedPath = indexDir + "/vectors_reordered.npy";
        if (loadNpyFloat(reorderedPath, m_vectors, shape)) {
            IVF_LOG("INFO", "Loaded reordered vectors from NPY: " << m_vectors.size() << " floats");
            return;
        }
        throw std::runtime_error("Cannot open reordered vectors file: " + reorderedPath);
    }
    
    // Try loading NPY file first
    std::string npyPath = indexDir + "/vectors.npy";
    if (loadNpyFloat(npyPath, m_vectors, shape)) {
        IVF_LOG("INFO", "Loaded vectors from NPY: " << m_vectors.size() << " floats");
        return;
    }
    
    // Fall back to binary file
    std::string binPath = indexDir + "/vectors.bin";
    std::ifstream file(binPath, std::ios::binary | std::ios::ate);
    if (!file) {
        throw std::runtime_error("Cannot open vectors file: " + npyPath + " or " + binPath);
    }
    
    size_t fileSize = file.tellg();
    file.seekg(0);
    
    m_vectors.resize(fileSize / sizeof(float));
    file.read(reinterpret_cast<char*>(m_vectors.data()), fileSize);
    IVF_LOG("INFO", "Loaded vectors from BIN: " << m_vectors.size() << " floats");
}

// Compute dot products for contiguous vector range (cluster-reordered mode)
void IVFIndex::computeDotProductsContiguous(const float* query, size_t startIdx,
                                            size_t count, float* scores) {
    const size_t dim = m_dim;
    const float* vectors = m_vectors.data() + startIdx * dim;
    
    size_t i = 0;
    
    // Process 8 vectors at a time for maximum throughput with contiguous access
    for (; i + 7 < count; i += 8) {
        // Prefetch next cache lines (64 bytes = 16 floats)
        __builtin_prefetch(vectors + (i + 8) * dim, 0, 0);
        __builtin_prefetch(vectors + (i + 8) * dim + 64, 0, 0);
        
        float32x4_t sum0 = vdupq_n_f32(0.0f);
        float32x4_t sum1 = vdupq_n_f32(0.0f);
        float32x4_t sum2 = vdupq_n_f32(0.0f);
        float32x4_t sum3 = vdupq_n_f32(0.0f);
        float32x4_t sum4 = vdupq_n_f32(0.0f);
        float32x4_t sum5 = vdupq_n_f32(0.0f);
        float32x4_t sum6 = vdupq_n_f32(0.0f);
        float32x4_t sum7 = vdupq_n_f32(0.0f);
        
        const float* v0 = vectors + i * dim;
        const float* v1 = vectors + (i + 1) * dim;
        const float* v2 = vectors + (i + 2) * dim;
        const float* v3 = vectors + (i + 3) * dim;
        const float* v4 = vectors + (i + 4) * dim;
        const float* v5 = vectors + (i + 5) * dim;
        const float* v6 = vectors + (i + 6) * dim;
        const float* v7 = vectors + (i + 7) * dim;
        
        for (size_t j = 0; j < dim; j += 4) {
            float32x4_t q = vld1q_f32(query + j);
            sum0 = vmlaq_f32(sum0, q, vld1q_f32(v0 + j));
            sum1 = vmlaq_f32(sum1, q, vld1q_f32(v1 + j));
            sum2 = vmlaq_f32(sum2, q, vld1q_f32(v2 + j));
            sum3 = vmlaq_f32(sum3, q, vld1q_f32(v3 + j));
            sum4 = vmlaq_f32(sum4, q, vld1q_f32(v4 + j));
            sum5 = vmlaq_f32(sum5, q, vld1q_f32(v5 + j));
            sum6 = vmlaq_f32(sum6, q, vld1q_f32(v6 + j));
            sum7 = vmlaq_f32(sum7, q, vld1q_f32(v7 + j));
        }
        
        scores[i]   = vaddvq_f32(sum0);
        scores[i+1] = vaddvq_f32(sum1);
        scores[i+2] = vaddvq_f32(sum2);
        scores[i+3] = vaddvq_f32(sum3);
        scores[i+4] = vaddvq_f32(sum4);
        scores[i+5] = vaddvq_f32(sum5);
        scores[i+6] = vaddvq_f32(sum6);
        scores[i+7] = vaddvq_f32(sum7);
    }
    
    // Handle remaining vectors 4 at a time
    for (; i + 3 < count; i += 4) {
        float32x4_t sum0 = vdupq_n_f32(0.0f);
        float32x4_t sum1 = vdupq_n_f32(0.0f);
        float32x4_t sum2 = vdupq_n_f32(0.0f);
        float32x4_t sum3 = vdupq_n_f32(0.0f);
        
        const float* v0 = vectors + i * dim;
        const float* v1 = vectors + (i + 1) * dim;
        const float* v2 = vectors + (i + 2) * dim;
        const float* v3 = vectors + (i + 3) * dim;
        
        for (size_t j = 0; j < dim; j += 4) {
            float32x4_t q = vld1q_f32(query + j);
            sum0 = vmlaq_f32(sum0, q, vld1q_f32(v0 + j));
            sum1 = vmlaq_f32(sum1, q, vld1q_f32(v1 + j));
            sum2 = vmlaq_f32(sum2, q, vld1q_f32(v2 + j));
            sum3 = vmlaq_f32(sum3, q, vld1q_f32(v3 + j));
        }
        
        scores[i]   = vaddvq_f32(sum0);
        scores[i+1] = vaddvq_f32(sum1);
        scores[i+2] = vaddvq_f32(sum2);
        scores[i+3] = vaddvq_f32(sum3);
    }
    
    // Handle remaining 1-3 vectors
    for (; i < count; ++i) {
        float32x4_t sum = vdupq_n_f32(0.0f);
        const float* v = vectors + i * dim;
        for (size_t j = 0; j < dim; j += 4) {
            sum = vmlaq_f32(sum, vld1q_f32(query + j), vld1q_f32(v + j));
        }
        scores[i] = vaddvq_f32(sum);
    }
}

// Compute dot products for scattered indices (non-reordered mode)
void IVFIndex::computeDotProductsNeon(const float* query, const int* candidateIds,
                                       size_t numCandidates, float* scores) {
    const size_t dim = m_dim;
    const float* vectors = m_vectors.data();
    
    // Prefetch query into cache (it's reused for all candidates)
    for (size_t j = 0; j < dim; j += 64/sizeof(float)) {
        __builtin_prefetch(query + j, 0, 3);
    }
    
    size_t i = 0;
    
    // Process 4 candidates at a time for better ILP
    for (; i + 3 < numCandidates; i += 4) {
        const float* vec0 = vectors + candidateIds[i] * dim;
        const float* vec1 = vectors + candidateIds[i+1] * dim;
        const float* vec2 = vectors + candidateIds[i+2] * dim;
        const float* vec3 = vectors + candidateIds[i+3] * dim;
        
        // Prefetch next batch of vectors
        if (i + 7 < numCandidates) {
            __builtin_prefetch(vectors + candidateIds[i+4] * dim, 0, 0);
            __builtin_prefetch(vectors + candidateIds[i+5] * dim, 0, 0);
            __builtin_prefetch(vectors + candidateIds[i+6] * dim, 0, 0);
            __builtin_prefetch(vectors + candidateIds[i+7] * dim, 0, 0);
        }
        
        float32x4_t sum0 = vdupq_n_f32(0.0f);
        float32x4_t sum1 = vdupq_n_f32(0.0f);
        float32x4_t sum2 = vdupq_n_f32(0.0f);
        float32x4_t sum3 = vdupq_n_f32(0.0f);
        
        // Process all dimensions (dim=128 is perfectly aligned)
        for (size_t j = 0; j < dim; j += 16) {
            float32x4_t q0 = vld1q_f32(query + j);
            float32x4_t q1 = vld1q_f32(query + j + 4);
            float32x4_t q2 = vld1q_f32(query + j + 8);
            float32x4_t q3 = vld1q_f32(query + j + 12);
            
            // Vector 0
            sum0 = vmlaq_f32(sum0, q0, vld1q_f32(vec0 + j));
            sum0 = vmlaq_f32(sum0, q1, vld1q_f32(vec0 + j + 4));
            sum0 = vmlaq_f32(sum0, q2, vld1q_f32(vec0 + j + 8));
            sum0 = vmlaq_f32(sum0, q3, vld1q_f32(vec0 + j + 12));
            
            // Vector 1
            sum1 = vmlaq_f32(sum1, q0, vld1q_f32(vec1 + j));
            sum1 = vmlaq_f32(sum1, q1, vld1q_f32(vec1 + j + 4));
            sum1 = vmlaq_f32(sum1, q2, vld1q_f32(vec1 + j + 8));
            sum1 = vmlaq_f32(sum1, q3, vld1q_f32(vec1 + j + 12));
            
            // Vector 2
            sum2 = vmlaq_f32(sum2, q0, vld1q_f32(vec2 + j));
            sum2 = vmlaq_f32(sum2, q1, vld1q_f32(vec2 + j + 4));
            sum2 = vmlaq_f32(sum2, q2, vld1q_f32(vec2 + j + 8));
            sum2 = vmlaq_f32(sum2, q3, vld1q_f32(vec2 + j + 12));
            
            // Vector 3
            sum3 = vmlaq_f32(sum3, q0, vld1q_f32(vec3 + j));
            sum3 = vmlaq_f32(sum3, q1, vld1q_f32(vec3 + j + 4));
            sum3 = vmlaq_f32(sum3, q2, vld1q_f32(vec3 + j + 8));
            sum3 = vmlaq_f32(sum3, q3, vld1q_f32(vec3 + j + 12));
        }
        
        // Horizontal reduction
        scores[i]   = vaddvq_f32(sum0);
        scores[i+1] = vaddvq_f32(sum1);
        scores[i+2] = vaddvq_f32(sum2);
        scores[i+3] = vaddvq_f32(sum3);
    }
    
    // Handle remaining candidates (less than 4)
    for (; i < numCandidates; ++i) {
        const float* vec = vectors + candidateIds[i] * dim;
        
        float32x4_t sum0 = vdupq_n_f32(0.0f);
        float32x4_t sum1 = vdupq_n_f32(0.0f);
        
        for (size_t j = 0; j < dim; j += 8) {
            sum0 = vmlaq_f32(sum0, vld1q_f32(query + j), vld1q_f32(vec + j));
            sum1 = vmlaq_f32(sum1, vld1q_f32(query + j + 4), vld1q_f32(vec + j + 4));
        }
        
        scores[i] = vaddvq_f32(vaddq_f32(sum0, sum1));
    }
}

// Top-K selection using min-heap
void IVFIndex::findTopK(const float* scores, const int* candidateIds, size_t numCandidates,
                        int k, std::vector<int>& indices, std::vector<float>& topScores) {
    if (numCandidates == 0 || k <= 0) {
        indices.clear();
        topScores.clear();
        return;
    }
    
    size_t kActual = std::min(static_cast<size_t>(k), numCandidates);
    
    // Use a min-heap of size K - keeps track of K largest elements
    // Heap element: (score, index)
    std::vector<std::pair<float, int>> heap;
    heap.reserve(kActual);
    
    // Fill heap with first K elements
    for (size_t i = 0; i < kActual; ++i) {
        heap.push_back({scores[i], candidateIds[i]});
    }
    // Build min-heap (smallest at top)
    std::make_heap(heap.begin(), heap.end(), 
        [](const auto& a, const auto& b) { return a.first > b.first; });
    
    // Process remaining elements
    for (size_t i = kActual; i < numCandidates; ++i) {
        float score = scores[i];
        // If current score is larger than smallest in heap, replace it
        if (score > heap.front().first) {
            std::pop_heap(heap.begin(), heap.end(),
                [](const auto& a, const auto& b) { return a.first > b.first; });
            heap.back() = {score, candidateIds[i]};
            std::push_heap(heap.begin(), heap.end(),
                [](const auto& a, const auto& b) { return a.first > b.first; });
        }
    }
    
    // Sort results by score (descending)
    std::sort(heap.begin(), heap.end(),
        [](const auto& a, const auto& b) { return a.first > b.first; });
    
    // Extract results
    indices.resize(kActual);
    topScores.resize(kActual);
    for (size_t i = 0; i < kActual; ++i) {
        topScores[i] = heap[i].first;
        indices[i] = heap[i].second;
    }
}

// Top-K selection for reordered mode with cluster ranges\nvoid IVFIndex::findTopKReordered(const std::vector<std::pair<int, int>>& clusterRanges,
                                  const float* query, int k,
                                  std::vector<int>& indices, std::vector<float>& topScores) {
    // Count total candidates
    size_t totalCandidates = 0;
    for (const auto& range : clusterRanges) {
        totalCandidates += range.second;  // (start, count)
    }
    
    if (totalCandidates == 0 || k <= 0) {
        indices.clear();
        topScores.clear();
        return;
    }
    
    size_t kActual = std::min(static_cast<size_t>(k), totalCandidates);
    
    // Min-heap for top-K (score, reordered_index)
    std::vector<std::pair<float, int>> heap;
    heap.reserve(kActual);
    
    const int32_t* reorderMap = m_reorderToOriginal.data();
    
    // Process all cluster ranges
    for (const auto& range : clusterRanges) {
        int start = range.first;
        int count = range.second;
        
        // Allocate scores buffer for this cluster
        std::vector<float> scores(count);
        
        // Compute dot products for this contiguous range
        computeDotProductsContiguous(query, start, count, scores.data());
        
        // Update heap
        for (int i = 0; i < count; ++i) {
            float score = scores[i];
            int reorderedIdx = start + i;
            
            if (heap.size() < kActual) {
                heap.push_back({score, reorderedIdx});
                if (heap.size() == kActual) {
                    std::make_heap(heap.begin(), heap.end(),
                        [](const auto& a, const auto& b) { return a.first > b.first; });
                }
            } else if (score > heap.front().first) {
                std::pop_heap(heap.begin(), heap.end(),
                    [](const auto& a, const auto& b) { return a.first > b.first; });
                heap.back() = {score, reorderedIdx};
                std::push_heap(heap.begin(), heap.end(),
                    [](const auto& a, const auto& b) { return a.first > b.first; });
            }
        }
    }
    
    // Build heap if we didn't fill it completely
    if (heap.size() < kActual) {
        std::make_heap(heap.begin(), heap.end(),
            [](const auto& a, const auto& b) { return a.first > b.first; });
    }
    
    // Sort by score descending
    std::sort(heap.begin(), heap.end(),
        [](const auto& a, const auto& b) { return a.first > b.first; });
    
    // Extract results, converting back to original indices
    indices.resize(heap.size());
    topScores.resize(heap.size());
    for (size_t i = 0; i < heap.size(); ++i) {
        topScores[i] = heap[i].first;
        indices[i] = reorderMap[heap[i].second];  // Map back to original index
    }
}

size_t IVFIndex::search(const std::vector<float>& query, int k, int nprobe,
                        std::vector<int>& indices, std::vector<float>& scores) {
    SearchTiming timing;
    return search(query, k, nprobe, indices, scores, timing);
}

size_t IVFIndex::search(const std::vector<float>& query, int k, int nprobe,
                        std::vector<int>& indices, std::vector<float>& scores,
                        SearchTiming& timing) {
    auto totalStart = std::chrono::high_resolution_clock::now();
    
    // Clamp nprobe
    nprobe = std::min(nprobe, static_cast<int>(m_numClusters));
    
    // Step 1: Centroid search (NPU)
    auto centroidStart = std::chrono::high_resolution_clock::now();
    
    std::vector<float> centroidScores;
    ExecutionTiming execTiming;
    m_centroidRunner->execute(query, centroidScores, execTiming);
    
    // Find top nprobe clusters
    std::vector<std::pair<float, int>> clusterScores(m_numClusters);
    for (size_t i = 0; i < m_numClusters; ++i) {
        clusterScores[i] = {centroidScores[i], static_cast<int>(i)};
    }
    std::partial_sort(clusterScores.begin(), clusterScores.begin() + nprobe, clusterScores.end(),
                      [](const auto& a, const auto& b) { return a.first > b.first; });
    
    auto centroidEnd = std::chrono::high_resolution_clock::now();
    timing.centroid_search_ms = std::chrono::duration<double, std::milli>(centroidEnd - centroidStart).count();
    
    // Step 2: Gather candidates from selected clusters
    auto gatherStart = std::chrono::high_resolution_clock::now();
    
    std::vector<int32_t> candidateIds;
    candidateIds.reserve(static_cast<size_t>(m_avgClusterSize * nprobe * 1.5));
    
    for (int i = 0; i < nprobe; ++i) {
        int clusterId = clusterScores[i].second;
        int32_t start = m_clusterOffsets[clusterId];
        int32_t end = m_clusterOffsets[clusterId + 1];
        candidateIds.insert(candidateIds.end(),
                            m_clusterIndices.begin() + start,
                            m_clusterIndices.begin() + end);
    }
    
    auto gatherEnd = std::chrono::high_resolution_clock::now();
    timing.gather_ms = std::chrono::duration<double, std::milli>(gatherEnd - gatherStart).count();
    
    // Step 3: Fine search on candidates (NEON-optimized)
    auto fineStart = std::chrono::high_resolution_clock::now();
    
    std::vector<float> candidateScores(candidateIds.size());
    computeDotProductsNeon(query.data(), candidateIds.data(), candidateIds.size(), candidateScores.data());
    
    // Find top-K
    findTopK(candidateScores.data(), candidateIds.data(), candidateIds.size(), k, indices, scores);
    
    auto fineEnd = std::chrono::high_resolution_clock::now();
    timing.fine_search_ms = std::chrono::duration<double, std::milli>(fineEnd - fineStart).count();
    
    auto totalEnd = std::chrono::high_resolution_clock::now();
    timing.total_ms = std::chrono::duration<double, std::milli>(totalEnd - totalStart).count();
    
    return candidateIds.size();
}

size_t IVFIndex::searchBatch(const std::vector<float>& queries, int batchSize, int k, int nprobe,
                             std::vector<std::vector<int>>& allIndices, 
                             std::vector<std::vector<float>>& allScores,
                             SearchTiming& timing) {
    auto totalStart = std::chrono::high_resolution_clock::now();
    
    // Clamp nprobe
    nprobe = std::min(nprobe, static_cast<int>(m_numClusters));
    
    // Resize outputs
    allIndices.resize(batchSize);
    allScores.resize(batchSize);
    
    // Step 1: Batched Centroid Search (NPU)
    auto centroidStart = std::chrono::high_resolution_clock::now();
    
    ExecutionTiming execTiming;
    m_centroidRunner->executeBatchRaw(queries, execTiming);
    
    // Access raw output
    const uint8_t* rawOutput = m_centroidRunner->getRawOutputBuffer();
    bool isFloat = m_centroidRunner->isFloatModel();
    float scale = m_centroidRunner->getOutputScale();
    size_t numClusters = m_numClusters;
    
    auto centroidEnd = std::chrono::high_resolution_clock::now();
    timing.centroid_search_ms = std::chrono::duration<double, std::milli>(centroidEnd - centroidStart).count();
    
    // Per-thread timing accumulators
    double gather_ms_total = 0.0;
    double fine_ms_total = 0.0;
    size_t totalCandidates = 0;
    
    auto fineStart = std::chrono::high_resolution_clock::now();
    
    if (m_isReordered) {
        // REORDERED MODE: Use contiguous cluster ranges for fast sequential access
        #pragma omp parallel reduction(+:totalCandidates, gather_ms_total, fine_ms_total)
        {
            // Per-thread buffers
            std::vector<std::pair<float, int>> clusterScores(numClusters);
            std::vector<std::pair<int, int>> clusterRanges;  // (start, count) pairs
            clusterRanges.reserve(nprobe);
            
            // Buffer for scores within a cluster
            size_t maxClusterSize = static_cast<size_t>(m_avgClusterSize * 3);  // Some margin
            std::vector<float> clusterCandidateScores(maxClusterSize);
            
            // Min-heap for top-K
            std::vector<std::pair<float, int>> heap;
            heap.reserve(k);
            
            #pragma omp for schedule(dynamic, 4)
            for (int b = 0; b < batchSize; ++b) {
                auto gatherStart = std::chrono::high_resolution_clock::now();
                
                // Extract scores for this query
                if (isFloat) {
                    const float* floatOutput = reinterpret_cast<const float*>(rawOutput);
                    const float* queryScores = floatOutput + b * numClusters;
                    for (size_t i = 0; i < numClusters; ++i) {
                        clusterScores[i] = {queryScores[i], static_cast<int>(i)};
                    }
                } else {
                    const uint8_t* queryScores = rawOutput + b * numClusters;
                    for (size_t i = 0; i < numClusters; ++i) {
                        clusterScores[i] = {static_cast<float>(queryScores[i]) * scale, static_cast<int>(i)};
                    }
                }
                
                // Use nth_element for faster top-nprobe selection
                std::nth_element(clusterScores.begin(), clusterScores.begin() + nprobe, clusterScores.end(),
                                [](const auto& a, const auto& b) { return a.first > b.first; });
                
                // Build cluster ranges (no need to gather indices!)
                clusterRanges.clear();
                size_t candidatesThisQuery = 0;
                for (int i = 0; i < nprobe; ++i) {
                    int clusterId = clusterScores[i].second;
                    int32_t start = m_clusterOffsets[clusterId];
                    int32_t count = m_clusterOffsets[clusterId + 1] - start;
                    clusterRanges.push_back({start, count});
                    candidatesThisQuery += count;
                }
                
                auto gatherEnd = std::chrono::high_resolution_clock::now();
                gather_ms_total += std::chrono::duration<double, std::milli>(gatherEnd - gatherStart).count();
                
                totalCandidates += candidatesThisQuery;
                
                // Fine search using contiguous access
                const float* queryVec = queries.data() + b * m_dim;
                
                // Reset heap
                heap.clear();
                size_t kActual = std::min(static_cast<size_t>(k), candidatesThisQuery);
                
                // Process each cluster range with contiguous access
                for (const auto& range : clusterRanges) {
                    int start = range.first;
                    int count = range.second;
                    
                    if (count > static_cast<int>(clusterCandidateScores.size())) {
                        clusterCandidateScores.resize(count);
                    }
                    
                    // Fast contiguous dot products
                    computeDotProductsContiguous(queryVec, start, count, clusterCandidateScores.data());
                    
                    // Update heap with results from this cluster
                    for (int i = 0; i < count; ++i) {
                        float score = clusterCandidateScores[i];
                        int reorderedIdx = start + i;
                        
                        if (heap.size() < kActual) {
                            heap.push_back({score, reorderedIdx});
                            if (heap.size() == kActual) {
                                std::make_heap(heap.begin(), heap.end(),
                                    [](const auto& a, const auto& b) { return a.first > b.first; });
                            }
                        } else if (score > heap.front().first) {
                            std::pop_heap(heap.begin(), heap.end(),
                                [](const auto& a, const auto& b) { return a.first > b.first; });
                            heap.back() = {score, reorderedIdx};
                            std::push_heap(heap.begin(), heap.end(),
                                [](const auto& a, const auto& b) { return a.first > b.first; });
                        }
                    }
                }
                
                // Sort and extract results
                std::sort(heap.begin(), heap.end(),
                    [](const auto& a, const auto& b) { return a.first > b.first; });
                
                allIndices[b].resize(heap.size());
                allScores[b].resize(heap.size());
                for (size_t i = 0; i < heap.size(); ++i) {
                    allScores[b][i] = heap[i].first;
                    allIndices[b][i] = m_reorderToOriginal[heap[i].second];  // Map back to original
                }
                
                auto fineEnd = std::chrono::high_resolution_clock::now();
                fine_ms_total += std::chrono::duration<double, std::milli>(fineEnd - gatherEnd).count();
            }
        }
    } else {
        // SCATTERED MODE: Original implementation with random vector access
        #pragma omp parallel reduction(+:totalCandidates, gather_ms_total, fine_ms_total)
        {
            // Per-thread buffers
            size_t maxCandidates = static_cast<size_t>(m_avgClusterSize * nprobe * 2);
            std::vector<int32_t> candidateIds;
            candidateIds.reserve(maxCandidates);
            std::vector<float> candidateScores;
            candidateScores.reserve(maxCandidates);
            std::vector<std::pair<float, int>> clusterScores(numClusters);
            
            #pragma omp for schedule(dynamic, 4)
            for (int b = 0; b < batchSize; ++b) {
                auto gatherStart = std::chrono::high_resolution_clock::now();
                
                // Extract scores for this query
                if (isFloat) {
                    const float* floatOutput = reinterpret_cast<const float*>(rawOutput);
                    const float* queryScores = floatOutput + b * numClusters;
                    for (size_t i = 0; i < numClusters; ++i) {
                        clusterScores[i] = {queryScores[i], static_cast<int>(i)};
                    }
                } else {
                    const uint8_t* queryScores = rawOutput + b * numClusters;
                    for (size_t i = 0; i < numClusters; ++i) {
                        clusterScores[i] = {static_cast<float>(queryScores[i]) * scale, static_cast<int>(i)};
                    }
                }
                
                // Use nth_element for faster top-nprobe selection
                std::nth_element(clusterScores.begin(), clusterScores.begin() + nprobe, clusterScores.end(),
                                [](const auto& a, const auto& b) { return a.first > b.first; });
                
                // Gather candidates
                candidateIds.clear();
                for (int i = 0; i < nprobe; ++i) {
                    int clusterId = clusterScores[i].second;
                    int32_t start = m_clusterOffsets[clusterId];
                    int32_t end = m_clusterOffsets[clusterId + 1];
                    candidateIds.insert(candidateIds.end(),
                                        m_clusterIndices.begin() + start,
                                        m_clusterIndices.begin() + end);
                }
                
                auto gatherEnd = std::chrono::high_resolution_clock::now();
                gather_ms_total += std::chrono::duration<double, std::milli>(gatherEnd - gatherStart).count();
                
                totalCandidates += candidateIds.size();
                
                // Fine search
                candidateScores.resize(candidateIds.size());
                const float* queryVec = queries.data() + b * m_dim;
                computeDotProductsNeon(queryVec, candidateIds.data(), candidateIds.size(), candidateScores.data());
                
                // Find top-K
                findTopK(candidateScores.data(), candidateIds.data(), candidateIds.size(), k, allIndices[b], allScores[b]);
                
                auto fineEnd = std::chrono::high_resolution_clock::now();
                fine_ms_total += std::chrono::duration<double, std::milli>(fineEnd - gatherEnd).count();
            }
        }
    }
    
    auto fineEnd = std::chrono::high_resolution_clock::now();
    
    // Store timing (these are wall-clock times now, not accumulated)
    timing.gather_ms = gather_ms_total / omp_get_max_threads();  // Average per-thread time
    timing.fine_search_ms = std::chrono::duration<double, std::milli>(fineEnd - fineStart).count();
    
    auto totalEnd = std::chrono::high_resolution_clock::now();
    timing.total_ms = std::chrono::duration<double, std::milli>(totalEnd - totalStart).count();
    
    return totalCandidates;
}
