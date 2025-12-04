"""
Create ONNX model for IVF Fine Search on NPU.

Strategy: Process one cluster at a time with a fixed-size model.

For SIFT 1M with 1024 clusters:
- Average cluster size: ~1000 vectors
- Max cluster size: ~2000-3000 vectors

We create a model that handles max_cluster_size vectors.
For smaller clusters, we pad with zeros (scores will be very negative).

Model: query [1, dim] × cluster_vectors_T [dim, max_cluster_size] = scores [1, max_cluster_size]
"""

import os
import sys
import onnx
import numpy as np
from onnx import helper, TensorProto


def create_cluster_search_model(dim, max_cluster_size, output_path):
    """
    Create ONNX model for single-cluster fine search.
    
    This is a TWO-INPUT model:
    - query: [1, dim] - the query vector
    - cluster_vectors_T: [dim, max_cluster_size] - transposed cluster vectors
    
    Output:
    - scores: [1, max_cluster_size] - dot product scores
    """
    print(f"Creating Cluster Search Model:")
    print(f"  Dimension: {dim}")
    print(f"  Max cluster size: {max_cluster_size}")
    
    # Input: query [1, dim]
    query_input = helper.make_tensor_value_info(
        "query", TensorProto.FLOAT, [1, dim]
    )
    
    # Input: cluster vectors transposed [dim, max_cluster_size]
    cluster_input = helper.make_tensor_value_info(
        "cluster_vectors_T", TensorProto.FLOAT, [dim, max_cluster_size]
    )
    
    # Output: scores [1, max_cluster_size]
    scores_output = helper.make_tensor_value_info(
        "scores", TensorProto.FLOAT, [1, max_cluster_size]
    )
    
    # MatMul: [1, dim] × [dim, max_cluster_size] = [1, max_cluster_size]
    matmul_node = helper.make_node(
        "MatMul",
        inputs=["query", "cluster_vectors_T"],
        outputs=["scores"],
        name="ClusterSearchMatMul",
    )
    
    # Build graph
    graph = helper.make_graph(
        nodes=[matmul_node],
        name="ClusterSearchGraph",
        inputs=[query_input, cluster_input],
        outputs=[scores_output],
    )
    
    # Create model
    opset = helper.make_opsetid("", 13)
    model = helper.make_model(graph, opset_imports=[opset])
    model.ir_version = 7
    
    # Validate and save
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    onnx.checker.check_model(model)
    onnx.save(model, output_path)
    
    # Stats
    flops = 2 * dim * max_cluster_size
    print(f"\nModel saved: {output_path}")
    print(f"  Input 'query': [1, {dim}]")
    print(f"  Input 'cluster_vectors_T': [{dim}, {max_cluster_size}]")
    print(f"  Output 'scores': [1, {max_cluster_size}]")
    print(f"  FLOPs per query-cluster: {flops:,}")
    
    return output_path


def create_batched_fine_search_model(dim, max_candidates, batch_size, output_path):
    """
    Create ONNX model for batched fine search.
    
    This processes multiple queries against the SAME candidate set.
    Good when you have many queries to process and candidates don't vary much.
    
    Inputs:
    - queries: [batch_size, dim]
    - candidates_T: [dim, max_candidates]
    
    Output:
    - scores: [batch_size, max_candidates]
    """
    print(f"Creating Batched Fine Search Model:")
    print(f"  Dimension: {dim}")
    print(f"  Max candidates: {max_candidates}")
    print(f"  Batch size: {batch_size}")
    
    # Input: queries [batch_size, dim]
    queries_input = helper.make_tensor_value_info(
        "queries", TensorProto.FLOAT, [batch_size, dim]
    )
    
    # Input: candidate vectors transposed [dim, max_candidates]
    candidates_input = helper.make_tensor_value_info(
        "candidates_T", TensorProto.FLOAT, [dim, max_candidates]
    )
    
    # Output: scores [batch_size, max_candidates]
    scores_output = helper.make_tensor_value_info(
        "scores", TensorProto.FLOAT, [batch_size, max_candidates]
    )
    
    # MatMul: [batch, dim] × [dim, candidates] = [batch, candidates]
    matmul_node = helper.make_node(
        "MatMul",
        inputs=["queries", "candidates_T"],
        outputs=["scores"],
        name="BatchedFineSearchMatMul",
    )
    
    # Build graph
    graph = helper.make_graph(
        nodes=[matmul_node],
        name="BatchedFineSearchGraph",
        inputs=[queries_input, candidates_input],
        outputs=[scores_output],
    )
    
    # Create model
    opset = helper.make_opsetid("", 13)
    model = helper.make_model(graph, opset_imports=[opset])
    model.ir_version = 7
    
    # Validate and save
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    onnx.checker.check_model(model)
    onnx.save(model, output_path)
    
    # Stats
    flops = 2 * batch_size * dim * max_candidates
    print(f"\nModel saved: {output_path}")
    print(f"  Input 'queries': [{batch_size}, {dim}]")
    print(f"  Input 'candidates_T': [{dim}, {max_candidates}]")
    print(f"  Output 'scores': [{batch_size}, {max_candidates}]")
    print(f"  FLOPs per batch: {flops:,}")
    print(f"  GFLOPs per batch: {flops / 1e9:.3f}")
    
    return output_path


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python create_fine_search_model.py cluster <output_dir> [dim] [max_cluster_size]")
        print("  python create_fine_search_model.py batched <output_dir> [dim] [max_candidates] [batch_size]")
        print("\nExamples:")
        print("  python create_fine_search_model.py cluster models/ivf_sift 128 2048")
        print("  python create_fine_search_model.py batched models/ivf_sift 128 32768 32")
        sys.exit(1)
    
    mode = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "models"
    
    if mode == "cluster":
        dim = int(sys.argv[3]) if len(sys.argv) > 3 else 128
        max_cluster_size = int(sys.argv[4]) if len(sys.argv) > 4 else 2048
        output_path = os.path.join(output_dir, f"fine_search_cluster_{max_cluster_size}.onnx")
        create_cluster_search_model(dim, max_cluster_size, output_path)
        
    elif mode == "batched":
        dim = int(sys.argv[3]) if len(sys.argv) > 3 else 128
        max_candidates = int(sys.argv[4]) if len(sys.argv) > 4 else 32768
        batch_size = int(sys.argv[5]) if len(sys.argv) > 5 else 32
        output_path = os.path.join(output_dir, f"fine_search_{max_candidates//1024}k_b{batch_size}.onnx")
        create_batched_fine_search_model(dim, max_candidates, batch_size, output_path)
    
    else:
        print(f"Unknown mode: {mode}")
        sys.exit(1)
