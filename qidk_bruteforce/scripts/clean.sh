#!/bin/bash
echo "Cleaning project build and temporary files..."

rm -rf .venv \
       __pycache__ \
       prepare/__pycache__ \
       models/*.onnx \
       android/app/src/main/assets/*.bin \
       android/app/main/libs \
       android/app/main/obj \
       android/output \
       qnn/qnn_artifacts \
       qnn/raw_inputs \
       qnn/input_list.txt \
       output \
       results \
       results.txt

echo "Clean complete!"
