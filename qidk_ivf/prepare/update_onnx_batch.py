import os
import sys
import numpy as np
import onnx
from onnx import helper, TensorProto

def create_matmul_model(centroids, output_path, input_name="query", output_name="scores", batch_size=1):
    """Create a simple MatMul ONNX model for similarity search."""
    n_clusters, dim = centroids.shape
    
    # Transpose to [DIM, NUM_CLUSTERS]
    centroids_T = np.ascontiguousarray(centroids.T, dtype=np.float32)

    query_input = helper.make_tensor_value_info(input_name, TensorProto.FLOAT, [batch_size, dim])
    embeddings_tensor = helper.make_tensor(
        name="embeddings_T",
        data_type=TensorProto.FLOAT,
        dims=[dim, n_clusters],
        vals=centroids_T.flatten().tolist(),
    )
    scores_output = helper.make_tensor_value_info(output_name, TensorProto.FLOAT, [batch_size, n_clusters])
    
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
    
    # Create model with proper opset version for ONNX Runtime compatibility
    opset = helper.make_opsetid("", 11)
    model = helper.make_model(graph, opset_imports=[opset])
    model.ir_version = 6

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    onnx.checker.check_model(model)
    onnx.save(model, output_path)
    print(f"Saved model to {output_path} with batch_size={batch_size}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python update_onnx_batch.py <dataset_name> <batch_size>")
        sys.exit(1)

    dataset_name = sys.argv[1]
    batch_size = int(sys.argv[2])
    
    model_dir = f"models/ivf_{dataset_name}"
    centroids_path = os.path.join(model_dir, "centroids.npy")
    output_path = os.path.join(model_dir, "centroids.onnx")
    
    if not os.path.exists(centroids_path):
        print(f"Error: {centroids_path} not found.")
        sys.exit(1)
        
    centroids = np.load(centroids_path)
    create_matmul_model(centroids, output_path, batch_size=batch_size)
