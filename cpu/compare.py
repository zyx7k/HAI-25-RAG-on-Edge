import re

def parse_results(filename):
    results = {}
    with open(filename, "r") as f:
        for line in f:
            if not line.strip():
                continue
            # Example: Query 0: (3752, 76269) (2176, 77121) ...
            query_id = int(re.search(r"Query (\d+):", line).group(1))
            pairs = re.findall(r"\((\d+),\s*([\d\.]+)\)", line)
            results[query_id] = [int(x[0]) for x in pairs]
    return results

def compute_topk_accuracy(npu_results, cpu_results, k=5):
    total = len(cpu_results)
    correct = 0
    for qid in cpu_results:
        npu_topk = npu_results.get(qid, [])[:k]
        cpu_topk = cpu_results[qid][:k]
        overlap = len(set(npu_topk) & set(cpu_topk))
        correct += overlap / k
    return correct / total

# --- Example Usage ---
# Save your two files as npu.txt and cpu.txt in same folder before running

npu_results = parse_results("npu.txt")
cpu_results = parse_results("cpu.txt")

for k in [1, 3, 5]:
    acc = compute_topk_accuracy(npu_results, cpu_results, k)
    print(f"Top-{k} Accuracy: {acc*100:.2f}%")
