#!/bin/bash
set -e

PROJECT_ROOT=$(realpath "$(dirname "$0")/..")
cd "$PROJECT_ROOT"

print_usage() {
    echo "Usage: $0 [dataset] [batch_sizes]"
    echo ""
    echo "Arguments:"
    echo "  dataset      Dataset to use: siftsmall, sift, or all (default: all)"
    echo "  batch_sizes  Comma-separated list of batch sizes (default: 1,8,16,32,64)"
    echo ""
    echo "Examples:"
    echo "  $0                          # Run both datasets with default batch sizes"
    echo "  $0 siftsmall                # Run siftsmall with default batch sizes"
    echo "  $0 sift                     # Run sift with default batch sizes"
    echo "  $0 all 1,8,32               # Run both datasets with specific batch sizes"
}

DATASET="${1:-all}"
BATCH_SIZES="${2:-1,8,16,32,64}"

if [[ "$DATASET" == "-h" ]] || [[ "$DATASET" == "--help" ]]; then
    print_usage
    exit 0
fi

if [[ "$DATASET" != "siftsmall" ]] && [[ "$DATASET" != "sift" ]] && [[ "$DATASET" != "all" ]]; then
    echo "Error: Invalid dataset '$DATASET'"
    echo ""
    print_usage
    exit 1
fi

# Determine which datasets to run
if [[ "$DATASET" == "all" ]]; then
    DATASETS=("siftsmall" "sift")
else
    DATASETS=("$DATASET")
fi

IFS=',' read -ra SIZES <<< "$BATCH_SIZES"

echo "=========================================="
echo "  QNN Vector Search Benchmark"
echo "=========================================="
echo "Datasets:     ${DATASETS[*]}"
echo "Batch sizes:  ${SIZES[*]}"
echo "=========================================="
echo ""

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_FILE="results/benchmark_${TIMESTAMP}.csv"
mkdir -p results

echo "dataset,batch_size,throughput_qps,gflops,avg_latency_ms,p95_latency_ms,p99_latency_ms" > "$RESULTS_FILE"

for CURRENT_DATASET in "${DATASETS[@]}"; do
    if [[ "$CURRENT_DATASET" == "siftsmall" ]]; then
        MODEL_SIZE="10k"
    else
        MODEL_SIZE="1M"
    fi
    
    echo ""
    echo "=========================================="
    echo "  Dataset: $CURRENT_DATASET ($MODEL_SIZE)"
    echo "=========================================="

    for BATCH in "${SIZES[@]}"; do
        echo ""
        echo ">>> Running $CURRENT_DATASET batch size: $BATCH"
        echo "=========================================="
        
        if [[ "$BATCH" == "1" ]]; then
            SUFFIX=""
        else
            SUFFIX="_b${BATCH}"
        fi

        bash scripts/build.sh "$CURRENT_DATASET" "$MODEL_SIZE" "$BATCH"
        bash scripts/deploy.sh "$CURRENT_DATASET" "${MODEL_SIZE}${SUFFIX}" 5

        RESULTS_DIR="results/${CURRENT_DATASET}_${MODEL_SIZE}${SUFFIX}"
        if [[ -f "$RESULTS_DIR/metrics.txt" ]]; then
            THROUGHPUT=$(grep "Throughput:" "$RESULTS_DIR/metrics.txt" | awk '{print $2}' | head -1)
            GFLOPS=$(grep "Avg GFLOPS:" "$RESULTS_DIR/metrics.txt" | awk '{print $3}' | head -1)
            AVG_LATENCY=$(grep "Avg graph execute time:" "$RESULTS_DIR/metrics.txt" | awk '{print $5}' | head -1)
            P95_LATENCY=$(grep "P95 graph exec time:" "$RESULTS_DIR/metrics.txt" | awk '{print $5}' | head -1)
            P99_LATENCY=$(grep "P99 graph exec time:" "$RESULTS_DIR/metrics.txt" | awk '{print $5}' | head -1)
            
            echo "$CURRENT_DATASET,$BATCH,$THROUGHPUT,$GFLOPS,$AVG_LATENCY,$P95_LATENCY,$P99_LATENCY" >> "$RESULTS_FILE"
        fi
    done
done

echo ""
echo "=========================================="
echo "  Benchmark Complete"
echo "=========================================="
echo ""
echo "Results saved to: $RESULTS_FILE"
echo ""

echo "Summary:"
echo "--------"
column -t -s',' "$RESULTS_FILE"
