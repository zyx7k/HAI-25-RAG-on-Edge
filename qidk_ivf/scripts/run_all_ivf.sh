#!/bin/bash
# Run IVF benchmark for all datasets with multiple nprobe values

set -e

PROJECT_ROOT=$(realpath "$(dirname "$0")/..")
cd "$PROJECT_ROOT"

: "${QNN_SDK_ROOT:=$HOME/qualcomm/qnn}"

print_usage() {
    echo "Usage: $0 [dataset] [nprobe_values]"
    echo ""
    echo "Arguments:"
    echo "  dataset       Dataset to use: siftsmall, sift, or all (default: all)"
    echo "  nprobe_values Comma-separated list of nprobe values (default: 8,16,32,64)"
    echo ""
    echo "Examples:"
    echo "  $0                          # Run both datasets with default nprobe values"
    echo "  $0 siftsmall                # Run siftsmall only"
    echo "  $0 sift                     # Run sift only"
    echo "  $0 all 8,16,32              # Run both datasets with specific nprobe values"
}

DATASET="${1:-all}"
NPROBE_VALUES="${2:-8,16,32,64}"
TOP_K=10

if [[ "$DATASET" == "-h" ]] || [[ "$DATASET" == "--help" ]]; then
    print_usage
    exit 0
fi

if [[ "$DATASET" != "siftsmall" ]] && [[ "$DATASET" != "sift" ]] && [[ "$DATASET" != "all" ]]; then
    echo "Error: Invalid dataset '$DATASET'"
    print_usage
    exit 1
fi

# Determine which datasets to run
if [[ "$DATASET" == "all" ]]; then
    DATASETS=("siftsmall" "sift")
else
    DATASETS=("$DATASET")
fi

IFS=',' read -ra NPROBES <<< "$NPROBE_VALUES"

echo "=========================================="
echo "  IVF Vector Search Benchmark"
echo "=========================================="
echo "Datasets:     ${DATASETS[*]}"
echo "nprobe values: ${NPROBES[*]}"
echo "top_k:        $TOP_K"
echo "=========================================="
echo ""

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_FILE="results/ivf_benchmark_${TIMESTAMP}.csv"
mkdir -p results

echo "dataset,nprobe,top_k,recall,qps,avg_latency_ms,p50_latency_ms,p95_latency_ms,p99_latency_ms,avg_candidates,candidate_reduction" > "$RESULTS_FILE"

for CURRENT_DATASET in "${DATASETS[@]}"; do
    INDEX_DIR="models/ivf_${CURRENT_DATASET}"
    
    echo ""
    echo "=========================================="
    echo "  Dataset: $CURRENT_DATASET"
    echo "=========================================="
    
    # Check if IVF index exists
    if [ ! -f "${INDEX_DIR}/ivf_config.json" ]; then
        echo "IVF index not found. Creating..."
        # Create virtual environment if needed
        if [ ! -d ".venv" ]; then
            python3 -m venv .venv
        fi
        source .venv/bin/activate
        pip install -q numpy scikit-learn onnx==1.13.1 protobuf==3.20.3 2>&1 | grep -v "WARNING" || true
        
        if [[ "$CURRENT_DATASET" == "siftsmall" ]]; then
            python3 prepare/create_ivf_model.py siftsmall
        else
            python3 prepare/create_ivf_model.py sift 1024
        fi
        deactivate 2>/dev/null || true
    fi
    
    # Check if QNN context binary exists
    if [ ! -f "${INDEX_DIR}/centroids.bin" ]; then
        echo "QNN context binary not found. Converting..."
        bash qnn/convert_centroids.sh "$CURRENT_DATASET"
    fi
    
    for NPROBE in "${NPROBES[@]}"; do
        echo ""
        echo ">>> Running $CURRENT_DATASET with nprobe=$NPROBE"
        echo "=========================================="
        
        # Deploy and run
        bash scripts/deploy_ivf.sh "$CURRENT_DATASET" "$NPROBE"
        
        # Run on device
        DEVICE_DIR="/data/local/tmp/ivf_search"
        REMOTE_RESULTS="${DEVICE_DIR}/results_${CURRENT_DATASET}_np${NPROBE}"
        
        echo "Running IVF search on device..."
        adb shell "cd ${DEVICE_DIR} && export LD_LIBRARY_PATH=. && \
            ./qidk_ivf ./ivf_${CURRENT_DATASET} ${CURRENT_DATASET}_query.fvecs \
            ${REMOTE_RESULTS} ./libQnnHtp.so ${TOP_K} ${NPROBE} \
            ${CURRENT_DATASET}_groundtruth.ivecs"
        
        # Pull results
        LOCAL_RESULTS="results/ivf_${CURRENT_DATASET}_np${NPROBE}"
        mkdir -p "$LOCAL_RESULTS"
        adb pull "${REMOTE_RESULTS}/metrics.txt" "${LOCAL_RESULTS}/"
        adb pull "${REMOTE_RESULTS}/results.txt" "${LOCAL_RESULTS}/" 2>/dev/null || true
        
        # Parse metrics and add to CSV
        if [[ -f "${LOCAL_RESULTS}/metrics.txt" ]]; then
            RECALL=$(grep "Recall@" "${LOCAL_RESULTS}/metrics.txt" | awk '{print $2}' | sed 's/%//')
            QPS=$(grep "QPS:" "${LOCAL_RESULTS}/metrics.txt" | awk '{print $2}')
            AVG_LATENCY=$(grep "Avg total:" "${LOCAL_RESULTS}/metrics.txt" | awk '{print $3}')
            P50=$(grep "P50:" "${LOCAL_RESULTS}/metrics.txt" | awk '{print $2}')
            P95=$(grep "P95:" "${LOCAL_RESULTS}/metrics.txt" | awk '{print $2}')
            P99=$(grep "P99:" "${LOCAL_RESULTS}/metrics.txt" | awk '{print $2}')
            AVG_CANDIDATES=$(grep "Avg candidates" "${LOCAL_RESULTS}/metrics.txt" | awk '{print $4}')
            REDUCTION=$(grep "Candidate reduction:" "${LOCAL_RESULTS}/metrics.txt" | awk '{print $3}' | sed 's/x//')
            
            echo "${CURRENT_DATASET},${NPROBE},${TOP_K},${RECALL},${QPS},${AVG_LATENCY},${P50},${P95},${P99},${AVG_CANDIDATES},${REDUCTION}" >> "$RESULTS_FILE"
            
            echo "  Recall@${TOP_K}: ${RECALL}%"
            echo "  QPS: ${QPS}"
            echo "  Avg latency: ${AVG_LATENCY} ms"
            echo "  Candidate reduction: ${REDUCTION}x"
        fi
    done
done

echo ""
echo "=========================================="
echo "  IVF Benchmark Complete"
echo "=========================================="
echo ""
echo "Results saved to: $RESULTS_FILE"
echo ""

echo "Summary:"
echo "--------"
column -t -s',' "$RESULTS_FILE"
