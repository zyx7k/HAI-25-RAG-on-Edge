"""IVF Benchmark Script.

Compares IVF search performance against brute force.
"""

import os
import sys
import json
import time
import numpy as np
import onnxruntime as ort


def read_fvecs(filename, count=-1):
    """Reads an .fvecs file."""
    vectors = []
    dim = None
    with open(filename, 'rb') as f:
        while True:
            dim_data = f.read(4)
            if not dim_data:
                break
            current_dim = np.frombuffer(dim_data, dtype='int32')[0]
            if dim is None:
                dim = current_dim
            vec = np.frombuffer(f.read(dim * 4), dtype='float32')
            vectors.append(vec)
            if 0 < count <= len(vectors):
                break
    return np.array(vectors, dtype='float32'), int(dim)


def read_ivecs(filename, count=-1):
    """Reads an .ivecs file (for ground truth)."""
    vectors = []
    dim = None
    with open(filename, 'rb') as f:
        while True:
            dim_data = f.read(4)
            if not dim_data:
                break
            current_dim = np.frombuffer(dim_data, dtype='int32')[0]
            if dim is None:
                dim = current_dim
            vec = np.frombuffer(f.read(dim * 4), dtype='int32')
            vectors.append(vec)
            if 0 < count <= len(vectors):
                break
    return np.array(vectors, dtype='int32')


class IVFSearcher:
    """IVF search implementation using ONNX Runtime."""
    
    def __init__(self, index_dir):
        self.index_dir = index_dir
        
        # Load config
        with open(os.path.join(index_dir, "ivf_config.json")) as f:
            self.config = json.load(f)
        
        self.n_vectors = self.config["n_vectors"]
        self.n_clusters = self.config["n_clusters"]
        self.dim = self.config["dim"]
        
        # Load ONNX model for centroid search
        centroid_model_path = os.path.join(index_dir, "centroids.onnx")
        self.centroid_session = ort.InferenceSession(
            centroid_model_path,
            providers=['CPUExecutionProvider']
        )
        
        # Load inverted lists
        self.cluster_offsets = np.load(os.path.join(index_dir, "cluster_offsets.npy"))
        self.cluster_indices = np.load(os.path.join(index_dir, "cluster_indices.npy"))
        
        # Load all vectors for fine search (contiguous for fast access)
        self.vectors = np.ascontiguousarray(
            np.load(os.path.join(index_dir, "vectors.npy")),
            dtype=np.float32
        )
        
        # Pre-compute cluster data for faster gathering
        self.cluster_vectors = []
        self.cluster_original_ids = []
        for i in range(self.n_clusters):
            start = self.cluster_offsets[i]
            end = self.cluster_offsets[i + 1]
            ids = self.cluster_indices[start:end]
            self.cluster_original_ids.append(ids)
            self.cluster_vectors.append(np.ascontiguousarray(self.vectors[ids]))
        
        print(f"Loaded IVF index: {self.n_vectors:,} vectors, {self.n_clusters} clusters")
    
    def search(self, query, k=10, nprobe=16):
        """
        Search for k nearest neighbors (optimized).
        """
        if query.ndim == 1:
            query = query.reshape(1, -1)
        
        query = np.ascontiguousarray(query, dtype=np.float32)
        
        # Step 1: Find nearest centroids (fast ONNX inference)
        centroid_scores = self.centroid_session.run(
            None, {"query": query}
        )[0][0]
        
        # Get top nprobe clusters using argpartition (faster than argsort for large arrays)
        if nprobe < self.n_clusters:
            top_clusters = np.argpartition(centroid_scores, -nprobe)[-nprobe:]
        else:
            top_clusters = np.arange(self.n_clusters)
        
        # Step 2 & 3: Gather candidates and compute scores in one pass
        # Pre-allocate arrays for efficiency
        total_candidates = sum(len(self.cluster_original_ids[c]) for c in top_clusters)
        all_scores = np.empty(total_candidates, dtype=np.float32)
        all_ids = np.empty(total_candidates, dtype=np.int32)
        
        query_flat = query.flatten()
        offset = 0
        for cluster_id in top_clusters:
            cluster_vecs = self.cluster_vectors[cluster_id]
            cluster_ids = self.cluster_original_ids[cluster_id]
            n = len(cluster_ids)
            
            # Compute dot products
            all_scores[offset:offset+n] = cluster_vecs @ query_flat
            all_ids[offset:offset+n] = cluster_ids
            offset += n
        
        # Get top k
        if total_candidates <= k:
            top_k_local = np.argsort(all_scores)[::-1]
        else:
            top_k_local = np.argpartition(all_scores, -k)[-k:]
            top_k_local = top_k_local[np.argsort(all_scores[top_k_local])[::-1]]
        
        return all_ids[top_k_local], all_scores[top_k_local], total_candidates


class BruteForceSearcher:
    """Brute force search for comparison."""
    
    def __init__(self, vectors):
        self.vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        self.n_vectors = vectors.shape[0]
        print(f"Loaded brute force searcher: {self.n_vectors:,} vectors")
    
    def search(self, query, k=10):
        if query.ndim == 1:
            query = query.reshape(1, -1)
        
        query = np.ascontiguousarray(query, dtype=np.float32)
        scores = self.vectors @ query.T
        scores = scores.flatten()
        
        if self.n_vectors <= k:
            top_k = np.argsort(scores)[::-1]
        else:
            top_k = np.argpartition(scores, -k)[-k:]
            top_k = top_k[np.argsort(scores[top_k])[::-1]]
        
        return top_k, scores[top_k], self.n_vectors


def compute_recall(predicted, ground_truth, k=10):
    """Compute recall@k."""
    gt_set = set(ground_truth[:k])
    pred_set = set(predicted[:k])
    return len(gt_set & pred_set) / k


def benchmark(dataset_name, nprobe_values=[1, 2, 4, 8, 16, 32, 64], k=10, n_queries=100):
    """Run benchmark comparing IVF vs brute force."""
    
    print("=" * 70)
    print(f"IVF Benchmark: {dataset_name}")
    print("=" * 70)
    
    # Load data
    index_dir = f"models/ivf_{dataset_name}"
    query_file = f"data/{dataset_name}/{dataset_name}_query.fvecs"
    gt_file = f"data/{dataset_name}/{dataset_name}_groundtruth.ivecs"
    
    queries, dim = read_fvecs(query_file, count=n_queries)
    ground_truth = read_ivecs(gt_file, count=n_queries)
    
    print(f"\nLoaded {len(queries)} queries")
    
    # Initialize searchers
    ivf = IVFSearcher(index_dir)
    brute = BruteForceSearcher(ivf.vectors)
    
    # Warmup
    print("\nWarming up...")
    for _ in range(10):
        ivf.search(queries[0], k=k, nprobe=16)
        brute.search(queries[0], k=k)
    
    # Benchmark brute force
    print("\n" + "-" * 70)
    print("Brute Force Baseline")
    print("-" * 70)
    
    start = time.perf_counter()
    for query in queries:
        brute.search(query, k=k)
    bf_time = (time.perf_counter() - start) * 1000 / len(queries)
    
    print(f"  Latency: {bf_time:.3f} ms/query")
    print(f"  Throughput: {1000/bf_time:.1f} QPS")
    print(f"  Vectors searched: {brute.n_vectors:,}")
    
    # Benchmark IVF with different nprobe values
    print("\n" + "-" * 70)
    print("IVF Results")
    print("-" * 70)
    print(f"{'nprobe':>8} {'Latency(ms)':>12} {'QPS':>10} {'Recall@'+str(k):>10} {'Speedup':>10} {'Candidates':>12}")
    print("-" * 70)
    
    results = []
    for nprobe in nprobe_values:
        if nprobe > ivf.n_clusters:
            continue
            
        # Measure latency and recall
        latencies = []
        recalls = []
        total_candidates = 0
        
        for i, query in enumerate(queries):
            start = time.perf_counter()
            pred_indices, pred_scores, n_candidates = ivf.search(query, k=k, nprobe=nprobe)
            latencies.append((time.perf_counter() - start) * 1000)
            
            recall = compute_recall(pred_indices, ground_truth[i], k=k)
            recalls.append(recall)
            total_candidates += n_candidates
        
        avg_latency = np.mean(latencies)
        avg_recall = np.mean(recalls)
        avg_candidates = total_candidates / len(queries)
        speedup = bf_time / avg_latency
        qps = 1000 / avg_latency
        
        print(f"{nprobe:>8} {avg_latency:>12.3f} {qps:>10.1f} {avg_recall:>10.2%} {speedup:>10.1f}x {avg_candidates:>12.0f}")
        
        results.append({
            "nprobe": nprobe,
            "latency_ms": avg_latency,
            "qps": qps,
            "recall": avg_recall,
            "speedup": speedup,
            "candidates": avg_candidates
        })
    
    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    
    # Find best nprobe for 95% recall
    best_for_95 = None
    for r in results:
        if r["recall"] >= 0.95:
            if best_for_95 is None or r["latency_ms"] < best_for_95["latency_ms"]:
                best_for_95 = r
    
    if best_for_95:
        print(f"\nBest config for ≥95% recall:")
        print(f"  nprobe={best_for_95['nprobe']}: {best_for_95['recall']:.1%} recall, {best_for_95['speedup']:.1f}x speedup")
    
    # Find best speedup with >90% recall
    best_speedup = None
    for r in results:
        if r["recall"] >= 0.90:
            if best_speedup is None or r["speedup"] > best_speedup["speedup"]:
                best_speedup = r
    
    if best_speedup:
        print(f"\nBest speedup with ≥90% recall:")
        print(f"  nprobe={best_speedup['nprobe']}: {best_speedup['recall']:.1%} recall, {best_speedup['speedup']:.1f}x speedup")
    
    # Find best speedup with >80% recall
    best_80 = None
    for r in results:
        if r["recall"] >= 0.80:
            if best_80 is None or r["speedup"] > best_80["speedup"]:
                best_80 = r
    
    if best_80:
        print(f"\nBest speedup with ≥80% recall:")
        print(f"  nprobe={best_80['nprobe']}: {best_80['recall']:.1%} recall, {best_80['speedup']:.1f}x speedup")
    
    return results


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python benchmark_ivf.py <dataset_name> [n_queries]")
        print("  dataset_name: 'siftsmall' or 'sift'")
        print("  n_queries: number of queries to run (default: 100)")
        sys.exit(1)
    
    dataset_name = sys.argv[1]
    n_queries = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    
    benchmark(dataset_name, n_queries=n_queries)
