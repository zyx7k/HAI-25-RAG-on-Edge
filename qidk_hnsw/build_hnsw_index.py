#!/usr/bin/env python3
"""
HNSW Index Builder for QNN Deployment
Builds HNSW graph structure offline for on-device search
"""

import numpy as np
import struct
import os
from typing import List, Tuple, Set
import heapq


class HNSWIndexBuilder:
    """Build HNSW index with serialization for device deployment"""

    def __init__(self, M: int = 16, M_max: int = 16, M_max0: int = 32,
                 ef_construction: int = 200, ml: float = 1.0/np.log(2.0)):
        """
        Args:
            M: Number of bi-directional links per element (except layer 0)
            M_max: Maximum number of connections per element per layer
            M_max0: Maximum connections for layer 0
            ef_construction: Size of dynamic candidate list during construction
            ml: Normalization factor for level assignment
        """
        self.M = M
        self.M_max = M_max
        self.M_max0 = M_max0
        self.ef_construction = ef_construction
        self.ml = ml

        self.dim = None
        self.vectors = None
        self.graph = []  # graph[point_id][layer] = list of neighbors
        self.entry_point = None
        self.max_layer = 0

    def add_vectors(self, vectors: np.ndarray):
        """Add vectors to the index"""
        if vectors.ndim != 2:
            raise ValueError("Vectors must be 2D array")

        self.dim = vectors.shape[1]
        self.vectors = vectors.astype(np.float32)
        n = len(vectors)

        print(f"Building HNSW index for {n} vectors (dim={self.dim})")
        print(
            f"Parameters: M={self.M}, M_max={self.M_max}, ef_construction={self.ef_construction}")

        # Initialize graph structure
        self.graph = [[] for _ in range(n)]

        for idx in range(n):
            if idx % 1000 == 0:
                print(f"  Processing vector {idx}/{n}...")
            self._insert(idx)

    def _get_random_level(self) -> int:
        """Select level for new point using exponential decay"""
        return int(-np.log(np.random.uniform(0, 1)) * self.ml)

    def _distance(self, idx1: int, idx2: int) -> float:
        """Compute L2 distance between two vectors"""
        diff = self.vectors[idx1] - self.vectors[idx2]
        return np.dot(diff, diff)

    def _search_layer(self, query_idx: int, entry_points: Set[int],
                      num_closest: int, layer: int) -> List[Tuple[float, int]]:
        """Search for num_closest nearest neighbors at given layer"""
        visited = set()
        candidates = []
        w = []

        for point in entry_points:
            dist = self._distance(query_idx, point)
            heapq.heappush(candidates, (-dist, point))
            heapq.heappush(w, (dist, point))
            visited.add(point)

        while candidates:
            current_dist, current = heapq.heappop(candidates)
            current_dist = -current_dist

            if current_dist > w[0][0]:
                break

            # Get neighbors at this layer
            neighbors = self.graph[current][layer] if layer < len(
                self.graph[current]) else []

            for neighbor in neighbors:
                if neighbor not in visited:
                    visited.add(neighbor)
                    dist = self._distance(query_idx, neighbor)

                    if dist < w[0][0] or len(w) < num_closest:
                        heapq.heappush(candidates, (-dist, neighbor))
                        heapq.heappush(w, (dist, neighbor))

                        if len(w) > num_closest:
                            heapq.heappop(w)

        return w

    def _get_neighbors(self, candidates: List[Tuple[float, int]], M: int) -> List[int]:
        """Select M neighbors using heuristic"""
        # Simple heuristic: return M closest points
        candidates.sort()
        return [idx for _, idx in candidates[:M]]

    def _insert(self, idx: int):
        """Insert a point into the HNSW structure"""
        level = self._get_random_level()

        # Initialize connections for this point
        self.graph[idx] = [[] for _ in range(level + 1)]

        if self.entry_point is None:
            # First point
            self.entry_point = idx
            self.max_layer = level
            return

        # Search for nearest neighbors
        nearest = {self.entry_point}

        # Traverse from top layer to target layer
        for lc in range(self.max_layer, level, -1):
            nearest = set(point for _, point in
                          self._search_layer(idx, nearest, 1, lc))

        # Insert at all layers from level down to 0
        for lc in range(level, -1, -1):
            candidates = self._search_layer(
                idx, nearest, self.ef_construction, lc)
            M = self.M if lc > 0 else self.M_max0
            neighbors = self._get_neighbors(candidates, M)

            # Add bidirectional links
            self.graph[idx][lc] = neighbors
            for neighbor in neighbors:
                if lc >= len(self.graph[neighbor]):
                    self.graph[neighbor].extend(
                        [[] for _ in range(lc + 1 - len(self.graph[neighbor]))])
                self.graph[neighbor][lc].append(idx)

                # Prune neighbors if needed
                max_conn = self.M_max if lc > 0 else self.M_max0
                if len(self.graph[neighbor][lc]) > max_conn:
                    # Prune to best connections
                    neighbor_dists = [(self._distance(neighbor, n), n)
                                      for n in self.graph[neighbor][lc]]
                    neighbor_dists.sort()
                    self.graph[neighbor][lc] = [
                        n for _, n in neighbor_dists[:max_conn]]

            nearest = set(neighbors)

        if level > self.max_layer:
            self.max_layer = level
            self.entry_point = idx

    def save(self, output_path: str):
        """Save HNSW index in binary format for device deployment"""
        print(f"\nSaving HNSW index to {output_path}")

        with open(output_path, 'wb') as f:
            # Header
            f.write(struct.pack('I', len(self.vectors)))  # num_vectors
            f.write(struct.pack('I', self.dim))  # dimension
            f.write(struct.pack('I', self.M))
            f.write(struct.pack('I', self.M_max))
            f.write(struct.pack('I', self.M_max0))
            f.write(struct.pack('I', self.entry_point))
            f.write(struct.pack('I', self.max_layer))

            # Graph structure: for each point, write layers and neighbors
            for point_id, layers in enumerate(self.graph):
                # num_layers for this point
                f.write(struct.pack('I', len(layers)))
                for layer_neighbors in layers:
                    # num_neighbors
                    f.write(struct.pack('I', len(layer_neighbors)))
                    for neighbor in layer_neighbors:
                        f.write(struct.pack('I', neighbor))  # neighbor_id

        print(
            f"  Index saved: {os.path.getsize(output_path) / (1024**2):.2f} MB")
        print(f"  Entry point: {self.entry_point}")
        print(f"  Max layer: {self.max_layer}")

        # Statistics
        total_connections = sum(sum(len(layer)
                                for layer in layers) for layers in self.graph)
        avg_connections = total_connections / len(self.graph)
        print(f"  Average connections per node: {avg_connections:.1f}")


def read_fvecs(filename: str, count: int = -1) -> Tuple[np.ndarray, int]:
    """Read vectors from .fvecs file"""
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


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print(
            "Usage: python build_hnsw_index.py <dataset_name> [M] [ef_construction]")
        print("  dataset_name: 'siftsmall' or 'sift'")
        print("  M: Number of connections (default: 16)")
        print("  ef_construction: Construction parameter (default: 200)")
        sys.exit(1)

    dataset_name = sys.argv[1]
    M = int(sys.argv[2]) if len(sys.argv) > 2 else 16
    ef_construction = int(sys.argv[3]) if len(sys.argv) > 3 else 200

    base_file = f"data/{dataset_name}/{dataset_name}_base.fvecs"
    output_file = f"data/{dataset_name}/{dataset_name}_hnsw_M{M}.bin"

    print(f"Loading vectors from {base_file}...")
    vectors, dim = read_fvecs(base_file)
    print(f"Loaded {len(vectors)} vectors (dim={dim})")

    # Build index
    builder = HNSWIndexBuilder(
        M=M, M_max=M, M_max0=M*2, ef_construction=ef_construction)
    builder.add_vectors(vectors)
    builder.save(output_file)

    print("\nDone! Index ready for deployment.")
#!/usr/bin/env python3
"""
HNSW Index Builder for QNN Deployment
Builds HNSW graph structure offline for on-device search
"""


class HNSWIndexBuilder:
    """Build HNSW index with serialization for device deployment"""

    def __init__(self, M: int = 16, M_max: int = 16, M_max0: int = 32,
                 ef_construction: int = 200, ml: float = 1.0/np.log(2.0)):
        """
        Args:
            M: Number of bi-directional links per element (except layer 0)
            M_max: Maximum number of connections per element per layer
            M_max0: Maximum connections for layer 0
            ef_construction: Size of dynamic candidate list during construction
            ml: Normalization factor for level assignment
        """
        self.M = M
        self.M_max = M_max
        self.M_max0 = M_max0
        self.ef_construction = ef_construction
        self.ml = ml

        self.dim = None
        self.vectors = None
        self.graph = []  # graph[point_id][layer] = list of neighbors
        self.entry_point = None
        self.max_layer = 0

    def add_vectors(self, vectors: np.ndarray):
        """Add vectors to the index"""
        if vectors.ndim != 2:
            raise ValueError("Vectors must be 2D array")

        self.dim = vectors.shape[1]
        self.vectors = vectors.astype(np.float32)
        n = len(vectors)

        print(f"Building HNSW index for {n} vectors (dim={self.dim})")
        print(
            f"Parameters: M={self.M}, M_max={self.M_max}, ef_construction={self.ef_construction}")

        # Initialize graph structure
        self.graph = [[] for _ in range(n)]

        for idx in range(n):
            if idx % 1000 == 0:
                print(f"  Processing vector {idx}/{n}...")
            self._insert(idx)

    def _get_random_level(self) -> int:
        """Select level for new point using exponential decay"""
        return int(-np.log(np.random.uniform(0, 1)) * self.ml)

    def _distance(self, idx1: int, idx2: int) -> float:
        """Compute L2 distance between two vectors"""
        diff = self.vectors[idx1] - self.vectors[idx2]
        return np.dot(diff, diff)

    def _search_layer(self, query_idx: int, entry_points: Set[int],
                      num_closest: int, layer: int) -> List[Tuple[float, int]]:
        """Search for num_closest nearest neighbors at given layer"""
        visited = set()
        candidates = []
        w = []

        for point in entry_points:
            dist = self._distance(query_idx, point)
            heapq.heappush(candidates, (-dist, point))
            heapq.heappush(w, (dist, point))
            visited.add(point)

        while candidates:
            current_dist, current = heapq.heappop(candidates)
            current_dist = -current_dist

            if current_dist > w[0][0]:
                break

            # Get neighbors at this layer
            neighbors = self.graph[current][layer] if layer < len(
                self.graph[current]) else []

            for neighbor in neighbors:
                if neighbor not in visited:
                    visited.add(neighbor)
                    dist = self._distance(query_idx, neighbor)

                    if dist < w[0][0] or len(w) < num_closest:
                        heapq.heappush(candidates, (-dist, neighbor))
                        heapq.heappush(w, (dist, neighbor))

                        if len(w) > num_closest:
                            heapq.heappop(w)

        return w

    def _get_neighbors(self, candidates: List[Tuple[float, int]], M: int) -> List[int]:
        """Select M neighbors using heuristic"""
        # Simple heuristic: return M closest points
        candidates.sort()
        return [idx for _, idx in candidates[:M]]

    def _insert(self, idx: int):
        """Insert a point into the HNSW structure"""
        level = self._get_random_level()

        # Initialize connections for this point
        self.graph[idx] = [[] for _ in range(level + 1)]

        if self.entry_point is None:
            # First point
            self.entry_point = idx
            self.max_layer = level
            return

        # Search for nearest neighbors
        nearest = {self.entry_point}

        # Traverse from top layer to target layer
        for lc in range(self.max_layer, level, -1):
            nearest = set(point for _, point in
                          self._search_layer(idx, nearest, 1, lc))

        # Insert at all layers from level down to 0
        for lc in range(level, -1, -1):
            candidates = self._search_layer(
                idx, nearest, self.ef_construction, lc)
            M = self.M if lc > 0 else self.M_max0
            neighbors = self._get_neighbors(candidates, M)

            # Add bidirectional links
            self.graph[idx][lc] = neighbors
            for neighbor in neighbors:
                if lc >= len(self.graph[neighbor]):
                    self.graph[neighbor].extend(
                        [[] for _ in range(lc + 1 - len(self.graph[neighbor]))])
                self.graph[neighbor][lc].append(idx)

                # Prune neighbors if needed
                max_conn = self.M_max if lc > 0 else self.M_max0
                if len(self.graph[neighbor][lc]) > max_conn:
                    # Prune to best connections
                    neighbor_dists = [(self._distance(neighbor, n), n)
                                      for n in self.graph[neighbor][lc]]
                    neighbor_dists.sort()
                    self.graph[neighbor][lc] = [
                        n for _, n in neighbor_dists[:max_conn]]

            nearest = set(neighbors)

        if level > self.max_layer:
            self.max_layer = level
            self.entry_point = idx

    def save(self, output_path: str):
        """Save HNSW index in binary format for device deployment"""
        print(f"\nSaving HNSW index to {output_path}")

        with open(output_path, 'wb') as f:
            # Header
            f.write(struct.pack('I', len(self.vectors)))  # num_vectors
            f.write(struct.pack('I', self.dim))  # dimension
            f.write(struct.pack('I', self.M))
            f.write(struct.pack('I', self.M_max))
            f.write(struct.pack('I', self.M_max0))
            f.write(struct.pack('I', self.entry_point))
            f.write(struct.pack('I', self.max_layer))

            # Graph structure: for each point, write layers and neighbors
            for point_id, layers in enumerate(self.graph):
                # num_layers for this point
                f.write(struct.pack('I', len(layers)))
                for layer_neighbors in layers:
                    # num_neighbors
                    f.write(struct.pack('I', len(layer_neighbors)))
                    for neighbor in layer_neighbors:
                        f.write(struct.pack('I', neighbor))  # neighbor_id

        print(
            f"  Index saved: {os.path.getsize(output_path) / (1024**2):.2f} MB")
        print(f"  Entry point: {self.entry_point}")
        print(f"  Max layer: {self.max_layer}")

        # Statistics
        total_connections = sum(sum(len(layer)
                                for layer in layers) for layers in self.graph)
        avg_connections = total_connections / len(self.graph)
        print(f"  Average connections per node: {avg_connections:.1f}")


def read_fvecs(filename: str, count: int = -1) -> Tuple[np.ndarray, int]:
    """Read vectors from .fvecs file"""
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


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print(
            "Usage: python build_hnsw_index.py <dataset_name> [M] [ef_construction]")
        print("  dataset_name: 'siftsmall' or 'sift'")
        print("  M: Number of connections (default: 16)")
        print("  ef_construction: Construction parameter (default: 200)")
        sys.exit(1)

    dataset_name = sys.argv[1]
    M = int(sys.argv[2]) if len(sys.argv) > 2 else 16
    ef_construction = int(sys.argv[3]) if len(sys.argv) > 3 else 200

    base_file = f"data/{dataset_name}/{dataset_name}_base.fvecs"
    output_file = f"data/{dataset_name}/{dataset_name}_hnsw_M{M}.bin"

    print(f"Loading vectors from {base_file}...")
    vectors, dim = read_fvecs(base_file)
    print(f"Loaded {len(vectors)} vectors (dim={dim})")

    # Build index
    builder = HNSWIndexBuilder(
        M=M, M_max=M, M_max0=M*2, ef_construction=ef_construction)
    builder.add_vectors(vectors)
    builder.save(output_file)

    print("\nDone! Index ready for deployment.")
