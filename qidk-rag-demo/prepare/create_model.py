import os
import onnx
import numpy as np
from onnx import helper, TensorProto

# --- Configuration ---
BASE_FVECS = "data/siftsmall_base.fvecs"
NUM_DOCS = 10_000
DIM = 128
ONNX_MODEL_PATH = "models/vector_search_10k.onnx"
# --- End Configuration ---


# --- Minimal .fvecs Reader ---
def read_fvecs(filename, count=-1, offset=0):
    """Reads an .fvecs file and returns a (num_vectors, dim) float32 numpy array."""
    with open(filename, 'rb') as f:
        if offset > 0:
            f.seek(offset * (4 + DIM * 4))  # Skip vectors

        vectors = []
        while True:
            dim_data = f.read(4)
            if not dim_data:
                break
            dim = np.frombuffer(dim_data, dtype='int32')[0]
            if dim != DIM:
                raise IOError(f"Invalid dim {dim}, expected {DIM}")
            vec = np.frombuffer(f.read(dim * 4), dtype='float32')
            if len(vec) != dim:
                raise IOError("Incomplete vector data")
            vectors.append(vec)
            if 0 < count <= len(vectors):
                break
    return np.array(vectors, dtype='float32')


# --- ONNX Model Creation ---
def create_matmul_model():
    print(f"Loading document vectors from {BASE_FVECS}...")
    doc_embeddings = read_fvecs(BASE_FVECS, count=NUM_DOCS)
    if doc_embeddings.shape[0] != NUM_DOCS:
        raise ValueError(f"Loaded {doc_embeddings.shape[0]} vectors, expected {NUM_DOCS}")
    print(f"Loaded {doc_embeddings.shape[0]} document vectors.")

    # Transpose to [DIM, NUM_DOCS]
    doc_embeddings_T = np.ascontiguousarray(doc_embeddings.T, dtype=np.float32)

    # Define input, constant, output
    query_input = helper.make_tensor_value_info("query", TensorProto.FLOAT, [1, DIM])
    doc_matrix_T = helper.make_tensor(
        name="doc_embeddings_T",
        data_type=TensorProto.FLOAT,
        dims=[DIM, NUM_DOCS],
        vals=doc_embeddings_T.flatten().tolist(),
    )
    scores_output = helper.make_tensor_value_info("scores", TensorProto.FLOAT, [1, NUM_DOCS])

    # MatMul operation
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

    os.makedirs("models", exist_ok=True)
    onnx.checker.check_model(model)
    onnx.save(model, ONNX_MODEL_PATH)

    print(f"Model saved: {ONNX_MODEL_PATH}")
    print(
        f"Input: 'query' [1, {DIM}] | Constant: 'doc_embeddings_T' [{DIM}, {NUM_DOCS}] | Output: 'scores' [1, {NUM_DOCS}]"
    )


if __name__ == "__main__":
    create_matmul_model()
