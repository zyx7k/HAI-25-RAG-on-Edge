#!/bin/bash
echo "Cleaning project build and temp files..."

rm -rf .venv venv __pycache__ \
       models/*.onnx \
       android/app/src/main/assets/*.bin \
       android/output \
       qnn/qnn_artifacts qnn/raw_inputs qnn/input_list.txt \
       output results.txt

echo "Clean complete!"
