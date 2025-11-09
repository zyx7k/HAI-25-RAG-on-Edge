import onnx
from onnx import helper, TensorProto
import numpy as np
import os
from download_data import read_fvecs

# --- Configuration ---
# Set to True to use the full 1M dataset (slower, ~500MB ONNX file)
# Set to False to use the 10k "dev" dataset (faster, ~5MB ONNX file)
USE_FULL_DATASET = False

if USE_FULL_DATASET:
    BASE_FVECS = "data/siftsmall_base.fvecs"
    NUM_DOCS = 1_000_000
    ONNX_MODEL_PATH = "models/vector_search_1M.onnx"
else:
    BASE_FVECS = "data/siftsmall_base.fvecs"
    NUM_DOCS = 10_000
    ONNX_MODEL_PATH = "models/vector_search_10k.onnx"

DIM = 128
# --- End Configuration ---

def create_matmul_model():
    print(f"Loading document vectors from {BASE_FVECS}...")
    # Load the document database
    # Shape: (NUM_DOCS, DIM) e.g., (10000, 128)
    doc_embeddings = read_fvecs(BASE_FVECS, count=NUM_DOCS)
    
    if doc_embeddings.shape[0] != NUM_DOCS:
        raise ValueError(f"Loaded {doc_embeddings.shape[0]} vectors, expected {NUM_DOCS}")
    
    print(f"Loaded {doc_embeddings.shape[0]} document vectors.")

    # We need to compute: C = A * B^T
    # A = query (Input) -> Shape: [1, 128]
    # B = docs (Constant) -> Shape: [10000, 128]
    # We must transpose B to B^T -> Shape: [128, 10000]
    # MatMul(A, B^T) -> [1, 128] * [128, 10000] = [1, 10000] (Scores)
    
    # Transpose and ensure data is C-contiguous
    doc_embeddings_T = np.ascontiguousarray(doc_embeddings.T, dtype=np.float32)
    
    # 1. Define the (dynamic) query input
    query_input = helper.make_tensor_value_info(
        "query", 
        TensorProto.FLOAT, 
        [1, DIM] # Batch size 1, dimension 128
    )

    # 2. Define the (constant) document matrix (transposed)
    # This is an "Initializer", which means it's a constant
    # weight embedded in the model file.
    doc_matrix_T = helper.make_tensor(
        name="doc_embeddings_T",
        data_type=TensorProto.FLOAT,
        dims=[DIM, NUM_DOCS], # e.g., [128, 10000]
        vals=doc_embeddings_T.flatten().tolist()
    )

    # 3. Define the output
    scores_output = helper.make_tensor_value_info(
        "scores",
        TensorProto.FLOAT,
        [1, NUM_DOCS] # e.g., [1, 10000]
    )

    # 4. Define the MatMul node
    matmul_node = helper.make_node(
        "MatMul",
        inputs=["query", "doc_embeddings_T"], # Input name, Constant name
        outputs=["scores"],
        name="VecSearchMatMul"
    )

    # 5. Create the graph
    graph = helper.make_graph(
        nodes=[matmul_node],
        name="VectorSearchGraph",
        inputs=[query_input],    # Only dynamic inputs go here
        outputs=[scores_output],
        initializer=[doc_matrix_T] # Constants/weights go here
    )

    # 6. Create the model
    model = helper.make_model(graph)
    model.opset_import[0].version = 13 # Use a common opset
    
    # 7. Check and save the model
    os.makedirs("models", exist_ok=True)
    onnx.checker.check_model(model)
    onnx.save(model, ONNX_MODEL_PATH)

    print(f"Model saved to {ONNX_MODEL_PATH}")
    print(f"  Input: 'query' (float32, [1, {DIM}])")
    print(f"  Constant: 'doc_embeddings_T' (float32, [{DIM}, {NUM_DOCS}])")
    print(f"  Output: 'scores' (float32, [1, {NUM_DOCS}])")

if __name__ == "__main__":
    create_matmul_model()