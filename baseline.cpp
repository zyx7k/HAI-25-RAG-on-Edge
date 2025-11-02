#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <cstring>
#include <string>
#include <omp.h>
#include <cblas.h>
#include <immintrin.h> // Directly include AVX2 header

// ============================================================================
// File I/O (No changes)
// ============================================================================

bool read_fvecs(const std::string& filename, std::vector<float>& data, int& rows, int& dim) {
    std::ifstream input(filename, std::ios::binary);
    if (!input) {
        std::cerr << "Error: Cannot open file " << filename << std::endl;
        return false;
    }
    rows = 0;
    int d;
    while (input.read(reinterpret_cast<char*>(&d), sizeof(int))) {
        if (rows == 0) {
            dim = d;
        } else if (d != dim) {
            std::cerr << "Error: Inconsistent dimension." << std::endl;
            return false;
        }

        data.resize(data.size() + dim);
        input.read(reinterpret_cast<char*>(&data[rows * dim]), dim * sizeof(float));
        rows++;
    }
    if (input.gcount() > 0) {
        std::cerr << "Error: File seems truncated." << std::endl;
        return false;
    }
    return true;
}


// ============================================================================
// SIMD-Optimized Norm Computation (AVX2-Only)
// ============================================================================

inline float compute_norm_avx2(const float* vec, int dim) {
    __m256 sum_vec = _mm256_setzero_ps();
    int i = 0;
    
    // Process 8 floats at a time
    for (; i + 7 < dim; i += 8) {
        __m256 v = _mm256_loadu_ps(&vec[i]);
        sum_vec = _mm256_fmadd_ps(v, v, sum_vec);
    }
    
    // Horizontal sum
    float result[8];
    _mm256_storeu_ps(result, sum_vec);
    float sum = result[0] + result[1] + result[2] + result[3] + 
                result[4] + result[5] + result[6] + result[7];
    
    // Handle remaining elements
    for (; i < dim; ++i) {
        sum += vec[i] * vec[i];
    }
    
    return sum;
}

void compute_norms(const std::vector<float>& data, std::vector<float>& norms, 
                   int rows, int dim) {
    norms.resize(rows);
    
    #pragma omp parallel for
    for (int i = 0; i < rows; ++i) {
        const float* vec = &data[i * dim];
        // No more conditional logic, just call the AVX2 version
        norms[i] = compute_norm_avx2(vec, dim);
    }
}

// ============================================================================
// Matrix Multiplication (BLAS) (No changes)
// ============================================================================

void compute_dot_products(const std::vector<float>& Q_data, 
                          const std::vector<float>& B_data,
                          std::vector<float>& C, int M, int K, int N) {
    C.resize(M * N);
    cblas_sgemm(
        CblasRowMajor, CblasNoTrans, CblasTrans,
        M, N, K,
        1.0f,
        Q_data.data(), K,
        B_data.data(), K,
        0.0f,
        C.data(), N
    );
}

// ============================================================================
// Distance Conversion (AVX2-Only)
// ============================================================================

inline void convert_to_distances_avx2(float* C_row, const float q_norm, 
                                      const float* B_norms, int N) {
    __m256 q_norm_vec = _mm256_set1_ps(q_norm);
    int j = 0;
    
    for (; j + 7 < N; j += 8) {
        __m256 dot = _mm256_loadu_ps(&C_row[j]);
        __m256 b_norm = _mm256_loadu_ps(&B_norms[j]);
        
        // distance = q_norm + b_norm - 2*dot
        __m256 dist = _mm256_add_ps(q_norm_vec, b_norm);
        dist = _mm256_fnmadd_ps(_mm256_set1_ps(2.0f), dot, dist);
        
        _mm256_storeu_ps(&C_row[j], dist);
    }
    
    // Handle remainder
    for (; j < N; ++j) {
        C_row[j] = q_norm + B_norms[j] - 2.0f * C_row[j];
    }
}

void convert_to_distances(std::vector<float>& C, 
                          const std::vector<float>& Q_norms,
                          const std::vector<float>& B_norms,
                          int M, int N) {
    #pragma omp parallel for
    for (int i = 0; i < M; ++i) {
        float* C_row = &C[i * N];
        float q_norm = Q_norms[i];
        
        // No more conditional logic, just call the AVX2 version
        convert_to_distances_avx2(C_row, q_norm, B_norms.data(), N);
    }
}

// ============================================================================
// Top-K Selection (No changes)
// ============================================================================

struct Result {
    float dist;
    int idx;
    bool operator<(const Result& other) const {
        return dist < other.dist;
    }
};

void select_topk(const float* distances, int N, int k, std::vector<Result>& topk) {
    topk.resize(k);
    for (int i = 0; i < k; ++i) {
        topk[i] = {distances[i], i};
    }
    
    int max_idx = 0;
    for (int i = 1; i < k; ++i) {
        if (topk[i].dist > topk[max_idx].dist) {
            max_idx = i;
        }
    }
    
    for (int j = k; j < N; ++j) {
        if (distances[j] < topk[max_idx].dist) {
            topk[max_idx] = {distances[j], j};
            max_idx = 0;
            for (int i = 1; i < k; ++i) {
                if (topk[i].dist > topk[max_idx].dist) {
                    max_idx = i;
                }
            }
        }
    }
    std::sort(topk.begin(), topk.end());
}

void find_knn(const std::vector<float>& distances, int M, int N, int k,
              std::vector<std::vector<Result>>& results) {
    results.resize(M);
    #pragma omp parallel for
    for (int i = 0; i < M; ++i) {
        const float* dist_row = &distances[i * N];
        select_topk(dist_row, N, k, results[i]);
    }
}

// ============================================================================
// Output (No changes)
// ============================================================================

bool write_results(const std::string& filename, 
                   const std::vector<std::vector<Result>>& results) {
    std::ofstream output(filename);
    if (!output) {
        std::cerr << "Error: Cannot open output file " << filename << std::endl;
        return false;
    }

    const size_t buffer_size = 1024 * 1024; // 1MB buffer
    std::vector<char> buffer(buffer_size);
    output.rdbuf()->pubsetbuf(buffer.data(), buffer_size);

    for (size_t i = 0; i < results.size(); ++i) {
        output << "Query " << i << ":";
        for (size_t t = 0; t < results[i].size(); ++t) {
            output << " (" << results[i][t].idx << ", " << results[i][t].dist << ")";
        }
        output << "\n";
    }
    return output.good();
}

// ============================================================================
// Main Pipeline (Simplified info printout)
// ============================================================================

int main(int argc, char* argv[]) {
    // --- 1. Parse Command-Line Arguments ---
    if (argc != 5) {
        std::cerr << "Usage: " << argv[0] << " <base_file.fvecs> <query_file.fvecs> <k> <output_file.txt>" << std::endl;
        return 1;
    }

    std::string base_file   = argv[1];
    std::string query_file  = argv[2];
    int k                   = std::stoi(argv[3]);
    std::string output_file = argv[4];

    // --- 2. Print System Info ---
    std::cout << "--- System Configuration ---" << std::endl;
    std::cout << "Using " << omp_get_max_threads() << " OpenMP threads." << std::endl;
    std::cout << "SIMD: AVX2 support is required and enabled." << std::endl;
    std::cout << "----------------------------" << std::endl;

    // --- 3. Load Data ---
    std::vector<float> Q_data, B_data;
    int Q_rows = 0, Q_dim = 0;
    int B_rows = 0, B_dim = 0;

    std::cout << "Loading base file: " << base_file << std::endl;
    if (!read_fvecs(base_file, B_data, B_rows, B_dim)) {
        return 1;
    }
    std::cout << "Loading query file: " << query_file << std::endl;
    if (!read_fvecs(query_file, Q_data, Q_rows, Q_dim)) {
        return 1;
    }

    std::cout << "Loaded " << Q_rows << " query vectors of dimension " << Q_dim << std::endl;
    std::cout << "Loaded " << B_rows << " base vectors of dimension " << B_dim << std::endl;

    if (Q_dim != B_dim) {
        std::cerr << "Error: Query and Base dimensions must be equal." << std::endl;
        return 1;
    }
    
    // --- 4. Main Pipeline ---
    std::cout << "Computing norms..." << std::endl;
    std::vector<float> Q_norms, B_norms;
    compute_norms(Q_data, Q_norms, Q_rows, Q_dim);
    compute_norms(B_data, B_norms, B_rows, B_dim);

    std::cout << "Computing dot products (sgemm)..." << std::endl;
    std::vector<float> distances;
    compute_dot_products(Q_data, B_data, distances, Q_rows, Q_dim, B_rows);
    
    std::cout << "Converting to distances..." << std::endl;
    convert_to_distances(distances, Q_norms, B_norms, Q_rows, B_rows);

    std::cout << "Finding top " << k << " nearest neighbors..." << std::endl;
    std::vector<std::vector<Result>> results;
    find_knn(distances, Q_rows, B_rows, k, results);

    // --- 5. Write Results ---
    std::cout << "Writing results to " << output_file << "..." << std::endl;
    if (!write_results(output_file, results)) {
        return 1;
    }

    std::cout << "Done." << std::endl;
    return 0;
}