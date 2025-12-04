import os
import sys
import onnx
import numpy as np
from onnx import helper, TensorProto


# --- Minimal .fvecs Reader ---
def read_fvecs(filename, count=-1, offset=0):
    """Reads an .fvecs file and returns a (num_vectors, dim) float32 numpy array."""
    vectors = []
    dim = None
    
    with open(filename, 'rb') as f:
        if offset > 0 and dim is not None:
            f.seek(offset * (4 + dim * 4))  # Skip vectors

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


# --- ONNX Model Creation ---
def create_matmul_model(base_fvecs, output_model_path, num_docs=-1, batch_size=1):
    """
    Create an ONNX MatMul model for vector search with optional batching.
    
    Args:
        base_fvecs: Path to the base document vectors file
        output_model_path: Path to save the ONNX model
        num_docs: Number of documents to use (-1 for all)
        batch_size: Number of queries to process in a single batch (default 1)
    """
    print(f"Loading document vectors from {base_fvecs}...")
    doc_embeddings, dim = read_fvecs(base_fvecs, count=num_docs)
    num_docs_loaded = doc_embeddings.shape[0]
    
    print(f"Loaded {num_docs_loaded} document vectors with dimension {dim}.")
    print(f"Batch size: {batch_size}")

    # Transpose to [DIM, NUM_DOCS]
    doc_embeddings_T = np.ascontiguousarray(doc_embeddings.T, dtype=np.float32)

    # Define input with batch dimension: [BATCH_SIZE, DIM]
    query_input = helper.make_tensor_value_info("query", TensorProto.FLOAT, [batch_size, dim])
    doc_matrix_T = helper.make_tensor(
        name="doc_embeddings_T",
        data_type=TensorProto.FLOAT,
        dims=[dim, num_docs_loaded],
        vals=doc_embeddings_T.flatten().tolist(),
    )
    # Output: [BATCH_SIZE, NUM_DOCS]
    scores_output = helper.make_tensor_value_info("scores", TensorProto.FLOAT, [batch_size, num_docs_loaded])

    # MatMul operation: [B, D] x [D, N] = [B, N]
    matmul_node = helper.make_node(
        "MatMul",
        inputs=["query", "doc_embeddings_T"],
        outputs=["scores"],
        name="VecSearchMatMul",
    )

    # Build graph & model
    graph = helper.make_graph(
        nodes=[matmul_node],
        name="VectorSearchGraph",
        inputs=[query_input],
        outputs=[scores_output],
        initializer=[doc_matrix_T],
    )
    model = helper.make_model(graph)
    model.opset_import[0].version = 13

    os.makedirs(os.path.dirname(output_model_path), exist_ok=True)
    onnx.checker.check_model(model)
    onnx.save(model, output_model_path)

    print(f"Model saved: {output_model_path}")
    print(
        f"Input: 'query' [{batch_size}, {dim}] | Constant: 'doc_embeddings_T' [{dim}, {num_docs_loaded}] | Output: 'scores' [{batch_size}, {num_docs_loaded}]"
    )
    
    # Calculate and display operational intensity improvement
    flops = 2 * batch_size * dim * num_docs_loaded
    bytes_moved = batch_size * dim + dim * num_docs_loaded + batch_size * num_docs_loaded  # Approximate
    oi = flops / bytes_moved
    print(f"\nOperational Intensity Analysis:")
    print(f"  FLOPs per batch: {flops:,}")
    print(f"  Bytes moved (approx): {bytes_moved:,}")
    print(f"  Operational Intensity: {oi:.2f} FLOPs/byte")
    if batch_size == 1:
        print(f"  (Tip: Use batch_size > 1 to improve OI via data reuse)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python create_model.py <dataset_name> [num_docs] [batch_size]")
        print("  dataset_name: 'siftsmall' or 'sift'")
        print("  num_docs: optional, number of documents to use (-1 for all)")
        print("  batch_size: optional, number of queries per batch (default 1)")
        print("\nExamples:")
        print("  python create_model.py siftsmall")
        print("  python create_model.py sift")
        print("  python create_model.py sift -1 32    # Batch size 32")
        print("  python create_model.py siftsmall -1 16  # Batch size 16")
        sys.exit(1)
    
    dataset_name = sys.argv[1]
    num_docs = int(sys.argv[2]) if len(sys.argv) > 2 else -1
    batch_size = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    
    # Determine paths based on dataset name
    base_fvecs = f"data/{dataset_name}/{dataset_name}_base.fvecs"
    
    if not os.path.exists(base_fvecs):
        print(f"ERROR: Base vectors file not found: {base_fvecs}")
        sys.exit(1)
    
    # Read to determine actual size
    temp_data, dim = read_fvecs(base_fvecs, count=1)
    
    # Count total vectors
    with open(base_fvecs, 'rb') as f:
        total_count = 0
        while True:
            dim_data = f.read(4)
            if not dim_data:
                break
            d = np.frombuffer(dim_data, dtype='int32')[0]
            f.seek(d * 4, 1)  # Skip the vector data
            total_count += 1
    
    if num_docs == -1:
        num_docs = total_count
    else:
        num_docs = min(num_docs, total_count)
    
    # Create output model path with proper naming (include batch size if > 1)
    if num_docs >= 1_000_000:
        size_suffix = f"{num_docs // 1_000_000}M"
    elif num_docs >= 1_000:
        size_suffix = f"{num_docs // 1_000}k"
    else:
        size_suffix = str(num_docs)
    
    if batch_size > 1:
        output_model_path = f"models/vector_search_{size_suffix}_b{batch_size}.onnx"
    else:
        output_model_path = f"models/vector_search_{size_suffix}.onnx"
    
    print(f"Creating model for {dataset_name} dataset:")
    print(f"  Base vectors: {base_fvecs}")
    print(f"  Total available: {total_count} vectors")
    print(f"  Using: {num_docs} vectors")
    print(f"  Dimension: {dim}")
    print(f"  Batch size: {batch_size}")
    print(f"  Output model: {output_model_path}")
    print()
    
    create_matmul_model(base_fvecs, output_model_path, num_docs, batch_size)
