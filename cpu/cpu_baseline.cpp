#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <cstring>
#include <string>
#include <cmath>
#include <chrono>
#include <omp.h>
#include <cblas.h>
#include <immintrin.h>

struct Result {
    float dist;
    int idx;
    bool operator<(const Result& other) const {
        return dist < other.dist;
    }
};

struct TimingStats {
    double mean;
    double std_dev;
    double min_val;
    double max_val;
    double p50;
    double p95;
    double p99;
};

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

TimingStats compute_statistics(std::vector<double>& times) {
    TimingStats stats;
    
    if (times.empty()) {
        stats.mean = stats.std_dev = stats.min_val = stats.max_val = 0.0;
        stats.p50 = stats.p95 = stats.p99 = 0.0;
        return stats;
    }
    
    std::sort(times.begin(), times.end());
    
    stats.min_val = times.front();
    stats.max_val = times.back();
    
    double sum = 0.0;
    for (double t : times) {
        sum += t;
    }
    stats.mean = sum / times.size();
    
    double sq_sum = 0.0;
    for (double t : times) {
        double diff = t - stats.mean;
        sq_sum += diff * diff;
    }
    stats.std_dev = std::sqrt(sq_sum / times.size());
    
    size_t n = times.size();
    stats.p50 = times[n * 50 / 100];
    stats.p95 = times[n * 95 / 100];
    stats.p99 = times[n * 99 / 100];
    
    return stats;
}

inline float compute_norm_avx2(const float* vec, int dim) {
    __m256 sum_vec = _mm256_setzero_ps();
    int i = 0;
    
    for (; i + 7 < dim; i += 8) {
        __m256 v = _mm256_loadu_ps(&vec[i]);
        sum_vec = _mm256_fmadd_ps(v, v, sum_vec);
    }
    
    float result[8];
    _mm256_storeu_ps(result, sum_vec);
    float sum = result[0] + result[1] + result[2] + result[3] + 
                result[4] + result[5] + result[6] + result[7];
    
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
        norms[i] = compute_norm_avx2(vec, dim);
    }
}

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

bool write_results(const std::string& filename, 
                   const std::vector<std::vector<Result>>& results) {
    std::ofstream output(filename);
    if (!output) {
        std::cerr << "Error: Cannot open output file " << filename << std::endl;
        return false;
    }

    const size_t buffer_size = 1024 * 1024;
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

void run_benchmark(const std::string& dataset_name,
                   const std::string& base_file,
                   const std::string& query_file,
                   int k,
                   const std::string& output_file) {
    
    using namespace std::chrono;
    
    std::cout << "\n========================================" << std::endl;
    std::cout << "Processing dataset: " << dataset_name << std::endl;
    std::cout << "========================================\n" << std::endl;
    
    std::vector<float> Q_data, B_data;
    int Q_rows = 0, Q_dim = 0;
    int B_rows = 0, B_dim = 0;

    std::cout << "Loading base file: " << base_file << std::endl;
    if (!read_fvecs(base_file, B_data, B_rows, B_dim)) {
        std::cerr << "Failed to load base file!" << std::endl;
        return;
    }
    std::cout << "Loading query file: " << query_file << std::endl;
    if (!read_fvecs(query_file, Q_data, Q_rows, Q_dim)) {
        std::cerr << "Failed to load query file!" << std::endl;
        return;
    }

    if (Q_dim != B_dim) {
        std::cerr << "Error: Query and Base dimensions must be equal." << std::endl;
        return;
    }
    
    std::cout << "Pre-computing norms..." << std::endl;
    std::vector<float> Q_norms, B_norms;
    compute_norms(Q_data, Q_norms, Q_rows, Q_dim);
    compute_norms(B_data, B_norms, B_rows, B_dim);

    std::vector<double> distance_times(Q_rows);
    std::vector<double> topk_times(Q_rows);
    std::vector<double> e2e_times(Q_rows);
    
    std::vector<std::vector<Result>> results(Q_rows);
    
    auto total_start = high_resolution_clock::now();
    
    for (int i = 0; i < Q_rows; ++i) {
        auto query_start = high_resolution_clock::now();
        
        auto dist_start = high_resolution_clock::now();
        
        std::vector<float> distances(B_rows);
        
        cblas_sgemm(
            CblasRowMajor, CblasNoTrans, CblasTrans,
            1, B_rows, Q_dim,
            1.0f,
            &Q_data[i * Q_dim], Q_dim,
            B_data.data(), Q_dim,
            0.0f,
            distances.data(), B_rows
        );
        
        float q_norm = Q_norms[i];
        for (int j = 0; j < B_rows; ++j) {
            distances[j] = q_norm + B_norms[j] - 2.0f * distances[j];
        }
        
        auto dist_end = high_resolution_clock::now();
        distance_times[i] = duration_cast<duration<double>>(dist_end - dist_start).count();
        
        auto topk_start = high_resolution_clock::now();
        select_topk(distances.data(), B_rows, k, results[i]);
        auto topk_end = high_resolution_clock::now();
        topk_times[i] = duration_cast<duration<double>>(topk_end - topk_start).count();
        
        auto query_end = high_resolution_clock::now();
        e2e_times[i] = duration_cast<duration<double>>(query_end - query_start).count();
    }
    
    auto total_end = high_resolution_clock::now();
    double total_time = duration_cast<duration<double>>(total_end - total_start).count();
    
    TimingStats dist_stats = compute_statistics(distance_times);
    TimingStats topk_stats = compute_statistics(topk_times);
    TimingStats e2e_stats = compute_statistics(e2e_times);
    
    double total_dist_time = 0.0;
    double total_topk_time = 0.0;
    for (int i = 0; i < Q_rows; ++i) {
        total_dist_time += distance_times[i];
        total_topk_time += topk_times[i];
    }
    
    std::cout << "\n=== CPU RAG Performance Metrics ===" << std::endl;
    std::cout << "\nDataset Information:" << std::endl;
    std::cout << "  Number of queries: " << Q_rows << std::endl;
    std::cout << "  Number of documents: " << B_rows << std::endl;
    std::cout << "  Dimension: " << Q_dim << std::endl;
    std::cout << "  Top-K: " << k << std::endl;
    
    std::cout << "\nOverall Performance:" << std::endl;
    std::cout << "  Total execution time: " << total_time << " s" << std::endl;
    std::cout << "  Throughput: " << (Q_rows / total_time) << " queries/sec" << std::endl;
    
    std::cout << "\nDistance Computation:" << std::endl;
    std::cout << "  Total time: " << total_dist_time << " s" << std::endl;
    std::cout << "  Average latency: " << (dist_stats.mean * 1000.0) << " ms/query" << std::endl;
    std::cout << "  Std deviation: " << (dist_stats.std_dev * 1000.0) << " ms" << std::endl;
    std::cout << "  Min latency: " << (dist_stats.min_val * 1000.0) << " ms" << std::endl;
    std::cout << "  Max latency: " << (dist_stats.max_val * 1000.0) << " ms" << std::endl;
    std::cout << "  P50 latency: " << (dist_stats.p50 * 1000.0) << " ms" << std::endl;
    std::cout << "  P95 latency: " << (dist_stats.p95 * 1000.0) << " ms" << std::endl;
    std::cout << "  P99 latency: " << (dist_stats.p99 * 1000.0) << " ms" << std::endl;
    
    std::cout << "\nTop-K Selection:" << std::endl;
    std::cout << "  Total time: " << total_topk_time << " s" << std::endl;
    std::cout << "  Average latency: " << (topk_stats.mean * 1000.0) << " ms/query" << std::endl;
    std::cout << "  Std deviation: " << (topk_stats.std_dev * 1000.0) << " ms" << std::endl;
    std::cout << "  Min latency: " << (topk_stats.min_val * 1000.0) << " ms" << std::endl;
    std::cout << "  Max latency: " << (topk_stats.max_val * 1000.0) << " ms" << std::endl;
    std::cout << "  P50 latency: " << (topk_stats.p50 * 1000.0) << " ms" << std::endl;
    std::cout << "  P95 latency: " << (topk_stats.p95 * 1000.0) << " ms" << std::endl;
    std::cout << "  P99 latency: " << (topk_stats.p99 * 1000.0) << " ms" << std::endl;
    
    std::cout << "\nEnd-to-End Per Query:" << std::endl;
    std::cout << "  Average latency: " << (e2e_stats.mean * 1000.0) << " ms/query" << std::endl;
    std::cout << "  Std deviation: " << (e2e_stats.std_dev * 1000.0) << " ms" << std::endl;
    std::cout << "  Min latency: " << (e2e_stats.min_val * 1000.0) << " ms" << std::endl;
    std::cout << "  Max latency: " << (e2e_stats.max_val * 1000.0) << " ms" << std::endl;
    std::cout << "  P50 latency: " << (e2e_stats.p50 * 1000.0) << " ms" << std::endl;
    std::cout << "  P95 latency: " << (e2e_stats.p95 * 1000.0) << " ms" << std::endl;
    std::cout << "  P99 latency: " << (e2e_stats.p99 * 1000.0) << " ms" << std::endl;
    
    std::cout << "\nTime Breakdown:" << std::endl;
    std::cout << "  Distance computation: " << (total_dist_time / total_time * 100.0) << "%" << std::endl;
    std::cout << "  Top-K selection: " << (total_topk_time / total_time * 100.0) << "%" << std::endl;
    
    std::cout << "\nWriting results to " << output_file << "..." << std::endl;
    if (!write_results(output_file, results)) {
        std::cerr << "Failed to write results!" << std::endl;
        return;
    }
    
    std::cout << "\nDone processing " << dataset_name << "!\n" << std::endl;
}

int main(int argc, char* argv[]) {
    std::cout << "=== CPU Baseline for k-NN Search ===" << std::endl;
    std::cout << "Using " << omp_get_max_threads() << " OpenMP threads" << std::endl;
    std::cout << "SIMD: AVX2 + FMA3 enabled" << std::endl;
    std::cout << "====================================\n" << std::endl;
    
    int k = 5;
    
    run_benchmark(
        "SIFT-small",
        "siftsmall/siftsmall_base.fvecs",
        "siftsmall/siftsmall_query.fvecs",
        k,
        "siftsmall_results.txt"
    );
    
    run_benchmark(
        "SIFT",
        "sift/sift_base.fvecs",
        "sift/sift_query.fvecs",
        k,
        "sift_results.txt"
    );
    
    std::cout << "\n========================================" << std::endl;
    std::cout << "All benchmarks completed!" << std::endl;
    std::cout << "========================================" << std::endl;
    
    return 0;
}
