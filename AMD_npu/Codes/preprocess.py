import numpy as np

# ---- read fvecs ----
def read_fvecs(fname):
    data = np.fromfile(fname, dtype=np.int32)
    dim = data[0]
    return np.fromfile(fname, dtype=np.float32).reshape(-1, dim+1)[:, 1:]

def pad_to_multiple(x, mult, axis=0):
    size = x.shape[axis]
    pad = (mult - size % mult) % mult
    if pad == 0:
        return x
    pad_shape = list(x.shape)
    pad_shape[axis] = pad
    padding = np.zeros(pad_shape, dtype=x.dtype)
    return np.concatenate([x, padding], axis=axis)

# ---- load fvecs ----
base = read_fvecs("siftsmall/siftsmall_base.fvecs")     # 10k × 128
query = read_fvecs("siftsmall/siftsmall_query.fvecs")   # 100 × 128

# ---- L2 normalize rows (critical!) ----
def l2_norm(x):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)

base = l2_norm(base)
query = l2_norm(query)

# ---- scale before int16 ----
SCALE = 1000
base = (base * SCALE)
query = (query * SCALE)

# ---- pad A (queries) to multiple of 32 ----
A = pad_to_multiple(query, 32, axis=0).astype(np.int16)
A.tofile("A.bin")
print("A.bin shape:", A.shape)

# ---- pad B to multiple of 256 ----
B = pad_to_multiple(base, 256, axis=0).astype(np.int16)
B.tofile("B.bin")

print("Original B rows:", base.shape[0])
print("Padded   B rows:", B.shape[0])
print("B.bin shape:", B.shape)
print("Use N =", B.shape[0])
