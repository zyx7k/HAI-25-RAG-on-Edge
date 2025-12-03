"""
Accuracy comparison script for SIFT benchmark results.
Compares results from different batch sizes against ground truth.
"""

import os
import re
from pathlib import Path

def parse_results_file(filepath):
    """Parse a results file and extract query results as dictionary."""
    results = {}
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or not line.startswith('Query'):
                continue
            
            # Parse query number
            match = re.match(r'Query (\d+):', line)
            if not match:
                continue
            
            query_num = int(match.group(1))
            
            # Extract all (id, distance) pairs
            pairs = re.findall(r'\((\d+),\s*[\d.]+\)', line)
            ids = [int(p) for p in pairs]
            results[query_num] = ids
    
    return results

def calculate_recall_at_k(predicted, ground_truth, k=5):
    """
    Calculate Recall@K for a set of queries.
    Recall@K = (number of relevant items in top-K predictions) / K
    """
    total_recall = 0
    num_queries = 0
    
    for query_num in ground_truth:
        if query_num not in predicted:
            continue
        
        gt_ids = set(ground_truth[query_num][:k])
        pred_ids = set(predicted[query_num][:k])
        
        # Count how many ground truth items appear in predictions
        matches = len(gt_ids.intersection(pred_ids))
        recall = matches / k
        total_recall += recall
        num_queries += 1
    
    if num_queries == 0:
        return 0.0
    
    return total_recall / num_queries

def calculate_precision_at_k(predicted, ground_truth, k=5):
    """
    Calculate Precision@K - proportion of top-K predictions that are correct.
    """
    return calculate_recall_at_k(predicted, ground_truth, k)  # Same as recall when K is same for both

def calculate_hit_rate_at_k(predicted, ground_truth, k=5):
    """
    Calculate Hit Rate@K - proportion of queries where at least one ground truth item is in top-K.
    """
    hits = 0
    num_queries = 0
    
    for query_num in ground_truth:
        if query_num not in predicted:
            continue
        
        gt_ids = set(ground_truth[query_num][:k])
        pred_ids = set(predicted[query_num][:k])
        
        if len(gt_ids.intersection(pred_ids)) > 0:
            hits += 1
        num_queries += 1
    
    if num_queries == 0:
        return 0.0
    
    return hits / num_queries

def calculate_mrr(predicted, ground_truth, k=5):
    """
    Calculate Mean Reciprocal Rank - average of 1/rank of first correct result.
    """
    total_rr = 0
    num_queries = 0
    
    for query_num in ground_truth:
        if query_num not in predicted:
            continue
        
        gt_ids = set(ground_truth[query_num][:k])
        pred_ids = predicted[query_num][:k]
        
        # Find the rank of the first correct prediction
        rr = 0
        for i, pred_id in enumerate(pred_ids):
            if pred_id in gt_ids:
                rr = 1 / (i + 1)
                break
        
        total_rr += rr
        num_queries += 1
    
    if num_queries == 0:
        return 0.0
    
    return total_rr / num_queries

def main():
    base_path = Path(__file__).parent
    
    # Ground truth files
    sift_1m_gt = base_path / "sift_results.txt"
    siftsmall_gt = base_path / "siftsmall_results.txt"
    
    # Result folders for SIFT 1M
    sift_1m_folders = [
        "sift_1M",
        "sift_1M_b8",
        "sift_1M_b16",
        "sift_1M_b32",
        "sift_1M_b64",
        "sift_1M_b128"
    ]
    
    # Result folders for SIFTSmall 10K
    siftsmall_folders = [
        "siftsmall_10k",
        "siftsmall_10k_b8",
        "siftsmall_10k_b16",
        "siftsmall_10k_b32",
        "siftsmall_10k_b64"
    ]
    
    print("=" * 80)
    print("SIFT Benchmark Accuracy Comparison")
    print("=" * 80)
    
    # Parse ground truth files
    print("\nLoading ground truth files...")
    sift_1m_truth = parse_results_file(sift_1m_gt)
    siftsmall_truth = parse_results_file(siftsmall_gt)
    print(f"  SIFT 1M ground truth: {len(sift_1m_truth)} queries")
    print(f"  SIFTSmall ground truth: {len(siftsmall_truth)} queries")
    
    # Analyze SIFT 1M results
    print("\n" + "-" * 80)
    print("SIFT 1M Dataset Results")
    print("-" * 80)
    print(f"{'Folder':<20} {'Queries':<10} {'Recall@1':<12} {'Recall@5':<12} {'Hit@5':<12} {'MRR@5':<12}")
    print("-" * 80)
    
    for folder in sift_1m_folders:
        results_file = base_path / folder / "results.txt"
        if not results_file.exists():
            print(f"{folder:<20} File not found")
            continue
        
        predicted = parse_results_file(results_file)
        
        recall_1 = calculate_recall_at_k(predicted, sift_1m_truth, k=1)
        recall_5 = calculate_recall_at_k(predicted, sift_1m_truth, k=5)
        hit_5 = calculate_hit_rate_at_k(predicted, sift_1m_truth, k=5)
        mrr_5 = calculate_mrr(predicted, sift_1m_truth, k=5)
        
        print(f"{folder:<20} {len(predicted):<10} {recall_1:<12.4f} {recall_5:<12.4f} {hit_5:<12.4f} {mrr_5:<12.4f}")
    
    # Analyze SIFTSmall results
    print("\n" + "-" * 80)
    print("SIFTSmall 10K Dataset Results")
    print("-" * 80)
    print(f"{'Folder':<20} {'Queries':<10} {'Recall@1':<12} {'Recall@5':<12} {'Hit@5':<12} {'MRR@5':<12}")
    print("-" * 80)
    
    for folder in siftsmall_folders:
        results_file = base_path / folder / "results.txt"
        if not results_file.exists():
            print(f"{folder:<20} File not found")
            continue
        
        predicted = parse_results_file(results_file)
        
        recall_1 = calculate_recall_at_k(predicted, siftsmall_truth, k=1)
        recall_5 = calculate_recall_at_k(predicted, siftsmall_truth, k=5)
        hit_5 = calculate_hit_rate_at_k(predicted, siftsmall_truth, k=5)
        mrr_5 = calculate_mrr(predicted, siftsmall_truth, k=5)
        
        print(f"{folder:<20} {len(predicted):<10} {recall_1:<12.4f} {recall_5:<12.4f} {hit_5:<12.4f} {mrr_5:<12.4f}")
    
    print("\n" + "=" * 80)
    print("Metrics Explanation:")
    print("  Recall@K: Fraction of ground truth top-K items found in predicted top-K")
    print("  Hit@K: Fraction of queries with at least one correct item in top-K")
    print("  MRR@K: Mean Reciprocal Rank - average of 1/rank of first correct result")
    print("=" * 80)

if __name__ == "__main__":
    main()
