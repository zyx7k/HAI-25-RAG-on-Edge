#!/usr/bin/env python3
"""
Create a minimal ONNX model to test HTP backend
"""
import numpy as np
import onnx
from onnx import helper, TensorProto

# Create a very simple MatMul: [1, 4] @ [4, 8] = [1, 8]
def create_simple_matmul():
    print("Creating simple 4x8 MatMul model...")
    
    # Input
    input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 4])
    
    # Weight matrix
    weights = np.random.randn(4, 8).astype(np.float32)
    weight_tensor = helper.make_tensor(
        name="weights",
        data_type=TensorProto.FLOAT,
        dims=[4, 8],
        vals=weights.flatten().tolist()
    )
    
    # Output
    output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 8])
    
    # MatMul node
    matmul_node = helper.make_node(
        "MatMul",
        inputs=["input", "weights"],
        outputs=["output"],
        name="TestMatMul"
    )
    
    # Build graph
    graph = helper.make_graph(
        nodes=[matmul_node],
        name="SimpleMatMul",
        inputs=[input_tensor],
        outputs=[output_tensor],
        initializer=[weight_tensor]
    )
    
    model = helper.make_model(graph)
    model.opset_import[0].version = 13
    
    onnx.checker.check_model(model)
    onnx.save(model, "models/test_matmul.onnx")
    print("✓ Saved: models/test_matmul.onnx")
    
    # Create test input
    test_input = np.random.randn(1, 4).astype(np.float32)
    test_input.tofile("data/test_input.raw")
    print("✓ Saved: data/test_input.raw")
    
    # Compute expected output
    expected_output = test_input @ weights
    expected_output.tofile("data/test_expected.raw")
    print(f"✓ Expected output shape: {expected_output.shape}")

if __name__ == "__main__":
    create_simple_matmul()
