"""IVF Index Builder with Cluster-Reordered Vectors.

Reorders vectors by cluster for contiguous memory access during fine search.
"""

import os
import sys
import json
import onnx
import numpy as np
from sklearn.cluster import KMeans
from onnx import helper, TensorProto


def read_fvecs(filename, count=-1):
    """Read .fvecs file and return (vectors, dim) tuple."""
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
            elif current_dim != dim:
                raise IOError(f"Invalid dim {current_dim}, expected {dim}")
            
            vec = np.frombuffer(f.read(dim * 4), dtype='float32')
            if len(vec) != dim:
                raise IOError("Incomplete vector data")
            vectors.append(vec)
            if 0 < count <= len(vectors):
                break
    
    return np.array(vectors, dtype='float32'), int(dim)


def create_matmul_model(embeddings, output_path, input_name="query", output_name="scores", batch_size=1):
    """Create MatMul ONNX model for similarity search."""
    num_vecs, dim = embeddings.shape
    embeddings_T = np.ascontiguousarray(embeddings.T, dtype=np.float32)

    query_input = helper.make_tensor_value_info(input_name, TensorProto.FLOAT, [batch_size, dim])
    embeddings_tensor = helper.make_tensor(
        name="embeddings_T",
        data_type=TensorProto.FLOAT,
        dims=[dim, num_vecs],
        vals=embeddings_T.flatten().tolist(),
    )
    scores_output = helper.make_tensor_value_info(output_name, TensorProto.FLOAT, [batch_size, num_vecs])
    
    matmul_node = helper.make_node(
        "MatMul",
        inputs=[input_name, "embeddings_T"],
        outputs=[output_name],
        name="SimilarityMatMul",
    )

    graph = helper.make_graph(
        nodes=[matmul_node],
        name="SimilarityGraph",
        inputs=[query_input],
        outputs=[scores_output],
        initializer=[embeddings_tensor],
    )
    
    opset = helper.make_opsetid("", 11)
    model = helper.make_model(graph, opset_imports=[opset])
    model.ir_version = 6

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    onnx.checker.check_model(model)
    onnx.save(model, output_path)
    
    return num_vecs, dim


def build_ivf_index_reordered(base_fvecs, output_dir, n_clusters=1024, batch_size=32):
    """Build IVF index with cluster-reordered vectors."""
    print(f"Building IVF Index (Cluster-Reordered)")
    print(f"=" * 60)
    
    print(f"Loading vectors from {base_fvecs}...")
    vectors, dim = read_fvecs(base_fvecs)
    n_vectors = vectors.shape[0]
    print(f"Loaded {n_vectors:,} vectors with dimension {dim}")
    
    if n_clusters > n_vectors // 10:
        n_clusters = max(16, n_vectors // 100)
        print(f"Adjusted n_clusters to {n_clusters}")
    
    print(f"Training KMeans with {n_clusters} clusters...")
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=1,
        max_iter=100,
        verbose=1
    )
    cluster_ids = kmeans.fit_predict(vectors)
    centroids = kmeans.cluster_centers_.astype(np.float32)
    print(f"KMeans converged in {kmeans.n_iter_} iterations")
    
    print(f"Reordering vectors by cluster...")
    
    # Sort vectors by cluster
    sorted_indices = np.argsort(cluster_ids)
    reordered_vectors = vectors[sorted_indices]
    reordered_cluster_ids = cluster_ids[sorted_indices]
    
    # Build mapping: original_index -> reordered_index
    original_to_reordered = np.zeros(n_vectors, dtype=np.int32)
    original_to_reordered[sorted_indices] = np.arange(n_vectors, dtype=np.int32)
    
    # Compute cluster offsets in reordered array
    cluster_sizes = []
    cluster_offsets = [0]
    for i in range(n_clusters):
        size = np.sum(reordered_cluster_ids == i)
        cluster_sizes.append(size)
        cluster_offsets.append(cluster_offsets[-1] + size)
    
    cluster_offsets = np.array(cluster_offsets, dtype=np.int32)
    cluster_sizes = np.array(cluster_sizes, dtype=np.int32)
    
    avg_cluster_size = np.mean(cluster_sizes)
    min_cluster_size = np.min(cluster_sizes)
    max_cluster_size = np.max(cluster_sizes)
    print(f"Cluster sizes: min={min_cluster_size}, avg={avg_cluster_size:.1f}, max={max_cluster_size}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Creating centroid ONNX model...")
    centroid_model_path = os.path.join(output_dir, "centroids.onnx")
    create_matmul_model(centroids, centroid_model_path, batch_size=batch_size)
    
    print(f"Saving index files...")
    vectors_path = os.path.join(output_dir, "vectors_reordered.npy")
    np.save(vectors_path, reordered_vectors)
    
    reorder_map_path = os.path.join(output_dir, "reorder_to_original.npy")
    np.save(reorder_map_path, sorted_indices.astype(np.int32)))
    
    config = {
        "n_vectors": n_vectors,
        "n_clusters": n_clusters,
        "dim": dim,
        "batch_size": batch_size,
        "avg_cluster_size": float(avg_cluster_size),
        "min_cluster_size": int(min_cluster_size),
        "max_cluster_size": int(max_cluster_size),
        "reordered": True,
    }
    config_path = os.path.join(output_dir, "ivf_config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    
    offsets_path = os.path.join(output_dir, "cluster_offsets.npy")
    np.save(offsets_path, cluster_offsets)
    
    sizes_path = os.path.join(output_dir, "cluster_sizes.npy")
    np.save(sizes_path, cluster_sizes)
    
    centroids_path = os.path.join(output_dir, "centroids.npy")
    np.save(centroids_path, centroids)
    
    print(f"\nIVF Index Built (Cluster-Reordered)")
    print(f"  Vectors: {n_vectors:,}")
    print(f"  Clusters: {n_clusters}")
    print(f"  Dimension: {dim}")
    print(f"  Output: {output_dir}/")
    
    return config


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python create_ivf_model_reordered.py <dataset_name> [n_clusters] [batch_size]")
        sys.exit(1)
    
    dataset_name = sys.argv[1]
    
    if dataset_name == "siftsmall":
        default_clusters = 100
    elif dataset_name == "sift":
        default_clusters = 1024
    else:
        default_clusters = 256
    
    n_clusters = int(sys.argv[2]) if len(sys.argv) > 2 else default_clusters
    batch_size = int(sys.argv[3]) if len(sys.argv) > 3 else 32
    
    base_fvecs = f"data/{dataset_name}/{dataset_name}_base.fvecs"
    output_dir = f"models/ivf_{dataset_name}_reordered"
    
    if not os.path.exists(base_fvecs):
        print(f"ERROR: Base vectors file not found: {base_fvecs}")
        sys.exit(1)
    
    build_ivf_index_reordered(base_fvecs, output_dir, n_clusters, batch_size)
