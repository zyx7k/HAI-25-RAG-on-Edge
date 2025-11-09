import numpy as np
import os

#
# --- MODIFICATION ---
# All download and extraction logic has been removed.
# This file now *only* provides the helper function `read_fvecs`
# which is imported by other scripts.
#

# Helper to read .fvecs files
def read_fvecs(filename, count=-1, offset=0):
    """
    Reads an .fvecs file.
    
    Args:
        filename (str): Path to the .fvecs file.
        count (int): Max number of vectors to read (-1 for all).
        offset (int): Vector offset to start reading from.

    Returns:
        np.ndarray: A 2D numpy array of shape (num_vectors, dim).
    """
    with open(filename, 'rb') as f:
        # Seek to the offset
        # Each vector record is (4 bytes for dim) + (dim * 4 bytes for data)
        # Assuming dim=128, (4 + 512) = 516 bytes per vector
        if offset > 0:
            f.seek(offset * (4 + 128 * 4)) 
            
        vectors = []
        while True:
            dim_data = f.read(4)
            if not dim_data:
                break # End of file
            
            dim = np.frombuffer(dim_data, dtype='int32')[0]
            if dim != 128:
                raise IOError(f"Invalid dimension {dim} at vector {len(vectors)} in {filename}. Expected 128.")
            
            vec_data = f.read(dim * 4) # 4 bytes per float
            if len(vec_data) != dim * 4:
                raise IOError(f"Incomplete vector data at {len(vectors)} in {filename}.")
                
            vectors.append(np.frombuffer(vec_data, dtype='float32'))
            
            if count > 0 and len(vectors) >= count:
                break # Reached read limit
                
    return np.array(vectors, dtype='float32')

#
# --- MODIFICATION ---
# The `if __name__ == "__main__":` block has been removed
# so running this file directly does nothing.
#