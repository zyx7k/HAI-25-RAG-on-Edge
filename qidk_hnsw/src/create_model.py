#!/usr/bin/env python3
import os
import sys
import numpy as np
from onnx import helper, TensorProto


def read_fvecs(filename, count=-1):
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


def create_matmul_model(base_fvecs, output_model_path, num_docs=-1):
    print(f"Loading document vectors from {base_fvecs}...")
    doc_embeddings, dim = read_fvecs(base_fvecs, count=num_docs)
    num_docs_loaded = doc_embeddings.shape[0]
    print(f"Loaded {num_docs_loaded} document vectors with dimension {dim}.")

    # Transpose to [DIM, NUM_DOCS]
    doc_embeddings_T = np.ascontiguousarray(doc_embeddings.T, dtype=np.float32)

    # Define input and output
    query_input = helper.make_tensor_value_info("query", TensorProto.FLOAT, [1, dim])
    doc_matrix_T = helper.make_tensor(
        name="doc_embeddings_T",
        data_type=TensorProto.FLOAT,
        dims=[dim, num_docs_loaded],
        vals=doc_embeddings_T.flatten().tolist(),
    )
    scores_output = helper.make_tensor_value_info("scores", TensorProto.FLOAT, [1, num_docs_loaded])

    matmul_node = helper.make_node(
        "MatMul",
        inputs=["query", "doc_embeddings_T"],
        outputs=["scores"],
        name="VecSearchMatMul",
    )

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
    import onnx
    onnx.checker.check_model(model)
    onnx.save(model, output_model_path)
    print(f"Model saved: {output_model_path}")
    print(f"Input: 'query' [1, {dim}] | Constant: 'doc_embeddings_T' [{dim}, {num_docs_loaded}] | Output: 'scores' [1, {num_docs_loaded}]")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python create_model.py <dataset_name>")
        sys.exit(1)
    dataset = sys.argv[1]
    base_file = f"data/{dataset}/{dataset}_base.fvecs"
    if not os.path.isfile(base_file):
        print(f"Base fvecs file not found: {base_file}")
        sys.exit(1)
    # output to android jni directory for conversion
    out_path = f"android/app/main/jni/vector_search_{dataset}.onnx"
    create_matmul_model(base_file, out_path)
