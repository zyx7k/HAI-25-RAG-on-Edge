#!/usr/bin/env python3
"""
Generate input_list.txt for QNN quantization calibration.
Creates a small set of representative query vectors in the format expected by qnn-onnx-converter.
"""
import struct
import os
import sys
import numpy as np

def read_fvecs(filename, max_vectors=100):
    """Read up to max_vectors from a .fvecs file."""
    vectors = []
    with open(filename, 'rb') as f:
        while len(vectors) < max_vectors:
            dim_bytes = f.read(4)
            if not dim_bytes:
                break
            dim = struct.unpack('i', dim_bytes)[0]
            vec = struct.unpack(f'{dim}f', f.read(dim * 4))
            vectors.append(vec)
    return np.array(vectors, dtype=np.float32)

def main():
    if len(sys.argv) < 2:
        print("Usage: generate_input_list.py <dataset_name>")
        sys.exit(1)
    
    dataset = sys.argv[1]
    query_file = f"data/{dataset}/{dataset}_query.fvecs"
    
    if not os.path.exists(query_file):
        print(f"ERROR: Query file not found: {query_file}")
        sys.exit(1)
    
    # Read a subset of queries for calibration (100 is enough)
    print(f"Reading calibration data from {query_file}...")
    queries = read_fvecs(query_file, max_vectors=100)
    print(f"Loaded {len(queries)} calibration vectors (dim={queries.shape[1]})")
    
    # Create output directory
    calib_dir = f"qnn/calibration/{dataset}"
    os.makedirs(calib_dir, exist_ok=True)
    
    # Write input_list.txt (one line per calibration sample)
    input_list_path = f"{calib_dir}/input_list.txt"
    with open(input_list_path, 'w') as f:
        for i in range(len(queries)):
            raw_file = f"{calib_dir}/input_{i:04d}.raw"
            # Write raw binary file
            queries[i].tofile(raw_file)
            # Add to input list (format: input_name:=path)
            f.write(f"query:={raw_file}\n")
    
    print(f"Generated {len(queries)} calibration inputs")
    print(f"Input list: {input_list_path}")
    print(f"Calibration complete!")

if __name__ == "__main__":
    main()
