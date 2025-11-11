#!/bin/bash
set -e

# This script runs the full pipeline for both siftsmall and sift datasets
# and saves results to separate directories

PROJECT_ROOT=$(realpath "$(dirname "$0")/..")
cd "$PROJECT_ROOT"

echo "=========================================="
echo "  QNN RAG Demo - Multi-Dataset Pipeline"
echo "=========================================="
echo ""
echo "This script will:"
echo "  1. Build and run the pipeline for siftsmall dataset (10k docs, 100 queries)"
echo "  2. Build and run the pipeline for sift dataset (1M docs, 10k queries)"
echo ""
echo "Results will be saved to:"
echo "  - results/siftsmall_10k/"
echo "  - results/sift_1M/"
echo ""
read -p "Continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 1
fi

echo ""
echo "=========================================="
echo "  Dataset 1: siftsmall (10k documents)"
echo "=========================================="
echo ""

echo "Step 1/3: Building for siftsmall..."
bash ./scripts/build.sh siftsmall 10k

echo ""
echo "Step 2/3: Deploying and running siftsmall..."
bash ./scripts/deploy.sh siftsmall 10k 5

echo ""
echo "=========================================="
echo "  Dataset 2: sift (1M documents)"
echo "=========================================="
echo ""

echo "Step 1/3: Building for sift..."
bash ./scripts/build.sh sift 1M

echo ""
echo "Step 2/3: Deploying and running sift..."
bash ./scripts/deploy.sh sift 1M 5

echo ""
echo "=========================================="
echo "  All Datasets Complete!"
echo "=========================================="
echo ""
echo "Results Summary:"
echo ""
echo "--- siftsmall (10k docs, 100 queries) ---"
if [ -f "results/siftsmall_10k/metrics.txt" ]; then
    echo "Location: results/siftsmall_10k/"
    grep -E "Number of queries:|Number of documents:|Total execution time|Throughput|Average latency" results/siftsmall_10k/metrics.txt | head -8
else
    echo "ERROR: Results not found"
fi

echo ""
echo "--- sift (1M docs, 10k queries) ---"
if [ -f "results/sift_1M/metrics.txt" ]; then
    echo "Location: results/sift_1M/"
    grep -E "Number of queries:|Number of documents:|Total execution time|Throughput|Average latency" results/sift_1M/metrics.txt | head -8
else
    echo "ERROR: Results not found"
fi

echo ""
echo "=========================================="
echo "Compare results:"
echo "  cat results/siftsmall_10k/metrics.txt"
echo "  cat results/sift_1M/metrics.txt"
echo "=========================================="
