"""
Accuracy Comparison Tool for k-NN Search Results

This script compares the accuracy of different k-NN implementations
(e.g., CPU vs NPU, CPU vs QIDK) by computing Top-K accuracy metrics.
"""

import re
import sys
from pathlib import Path


def parse_results(filename):
    """Parse k-NN results from a text file.
    
    Args:
        filename: Path to the results file
        
    Returns:
        Dictionary mapping query ID to list of document indices
    """
    results = {}
    try:
        with open(filename, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                
                match = re.search(r"Query (\d+):", line)
                if not match:
                    continue
                    
                query_id = int(match.group(1))
                pairs = re.findall(r"\((\d+),\s*[\d\.]+\)", line)
                results[query_id] = [int(idx) for idx in pairs]
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found")
        sys.exit(1)
    except Exception as e:
        print(f"Error parsing file '{filename}': {e}")
        sys.exit(1)
        
    return results


def compute_topk_accuracy(reference_results, test_results, k=5):
    """Compute Top-K accuracy between two result sets.
    
    Args:
        reference_results: Ground truth or reference results
        test_results: Results to evaluate
        k: Number of top results to consider
        
    Returns:
        Top-K accuracy as a fraction (0.0 to 1.0)
    """
    if not test_results:
        return 0.0
        
    total_queries = len(test_results)
    total_accuracy = 0.0
    
    for query_id in test_results:
        if query_id not in reference_results:
            continue
            
        ref_topk = set(reference_results[query_id][:k])
        test_topk = set(test_results[query_id][:k])
        overlap = len(ref_topk & test_topk)
        total_accuracy += overlap / k
    
    return total_accuracy / total_queries


def print_comparison(dataset_name, reference_file, test_file):
    """Print accuracy comparison for a dataset.
    
    Args:
        dataset_name: Name of the dataset being compared
        reference_file: Path to reference results
        test_file: Path to test results
    """
    print("=" * 70)
    print(f"{dataset_name.upper()} DATASET COMPARISON")
    print("=" * 70)
    
    ref_results = parse_results(reference_file)
    test_results = parse_results(test_file)
    
    print(f"\nReference file: {reference_file}")
    print(f"Test file: {test_file}")
    print(f"Number of queries: {len(test_results)}")
    print()
    
    for k in [1, 3, 5]:
        accuracy = compute_topk_accuracy(ref_results, test_results, k)
        print(f"Top-{k} Accuracy: {accuracy*100:.2f}%")
    print()


def main():
    """Main function to compare CPU and QIDK results."""
    results_dir = Path(__file__).parent / "results"
    
    print("\n" + "=" * 70)
    print("k-NN ACCURACY COMPARISON: CPU vs QIDK")
    print("=" * 70)
    print()
    
    datasets = [
        ("SIFT-small", "siftsmall_results.txt", "siftsmall_results_qidk.txt"),
        ("SIFT", "sift_results.txt", "sift_results_qidk.txt")
    ]
    
    summary = {}
    
    for dataset_name, cpu_file, qidk_file in datasets:
        cpu_path = results_dir / cpu_file
        qidk_path = results_dir / qidk_file
        
        if not cpu_path.exists() or not qidk_path.exists():
            print(f"Skipping {dataset_name}: Missing result files")
            continue
        
        print_comparison(dataset_name, cpu_path, qidk_path)
        
        ref_results = parse_results(cpu_path)
        test_results = parse_results(qidk_path)
        
        summary[dataset_name] = {
            k: compute_topk_accuracy(ref_results, test_results, k)
            for k in [1, 3, 5]
        }
    
    if summary:
        print("=" * 70)
        print("SUMMARY")
        print("=" * 70)
        print()
        
        for dataset_name, accuracies in summary.items():
            print(f"{dataset_name}:")
            for k, acc in accuracies.items():
                print(f"  Top-{k}: {acc*100:.2f}%")
            print()


if __name__ == "__main__":
    main()
