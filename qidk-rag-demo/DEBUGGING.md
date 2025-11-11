# QNN RAG Demo Debugging Status

## Current Status (BLOCKED at Execute)

### ✅ Working
- Context binary generation with `libQnnHtpPrepare.so`
- Backend loading (`libQnnHtp.so`)
- Device creation (with warning but continues)
- Context loading from binary
- Graph retrieval from context
- Tensor setup for UFIXED_POINT_8 quantization

### ❌ Failing
- `QnnGraph_execute()` returns error (no detailed message)

## Environment
- Device: QIDK with Snapdragon 8 Gen 2 (Hexagon V73)
- QNN SDK: v2.29.0.241129103708_105762
- Backend: HTP (Hexagon Tensor Processor)
- Model: 128-dim input → 10K-dim output (MatMul/FullyConnected)
- Quantization: UFIXED_POINT_8 (input scale=0.6627451, output scale=1013.4312)

## Execution Flow
```
1. loadBackend("libQnnHtp.so") ✅
2. createDevice() ✅ (warning: "Device handle not required for context binary")
3. createContext("model_context.bin") ✅
4. setupGraph("model") ✅
5. setupTensors() ✅
   - Input tensor: [1, 128] UFIXED_POINT_8, 128 bytes
   - Output tensor: [1, 10000] UFIXED_POINT_8, 10000 bytes
6. execute(query_vector) ❌
   - Quantizes float32 → uint8
   - Calls QnnGraph_execute → FAILS HERE
```

## Possible Root Causes

### 1. Tensor Descriptor Mismatch
**Symptom**: Execute fails silently  
**Theory**: Custom tensor descriptors created in `setupTensors()` don't match context binary expectations

**Test**: Try using graph's internal tensors instead of custom ones
```cpp
// In execute(), try passing graph tensors directly:
Qnn_Tensor_t* graphInputTensors = nullptr;
Qnn_Tensor_t* graphOutputTensors = nullptr;
uint32_t numInputs, numOutputs;

// Get tensors from graph
m_qnnInterface.graphGetInfo(m_graphHandle, QNN_GRAPH_INFO_INPUT_TENSORS, 
                             &graphInputTensors, &numInputs);
m_qnnInterface.graphGetInfo(m_graphHandle, QNN_GRAPH_INFO_OUTPUT_TENSORS,
                             &graphOutputTensors, &numOutputs);

// Copy data to graph tensor's clientBuf
memcpy(graphInputTensors[0].clientBuf.data, quantized_input, 128);

// Execute with graph tensors
m_qnnInterface.graphExecute(m_graphHandle, graphInputTensors, numInputs,
                             graphOutputTensors, numOutputs, nullptr, nullptr);
```

### 2. Quantization Parameter Mismatch
**Symptom**: Scale/offset hardcoded from model.cpp  
**Theory**: Context binary may store different quantization params

**Test**: Extract actual quantization from loaded tensors
```cpp
// After graphGetInfo for input tensors:
Qnn_QuantizeParams_t inputQuant = graphInputTensors[0].quantizeParams;
float actual_scale = inputQuant.scaleOffsetEncoding.scale;
int32_t actual_offset = inputQuant.scaleOffsetEncoding.offset;
std::cout << "Actual input scale: " << actual_scale << ", offset: " << actual_offset << std::endl;
```

### 3. Missing Graph Finalization
**Symptom**: Execute fails on first call  
**Theory**: Graph needs explicit finalization after context load

**Test**: Call graphFinalize after graphRetrieve
```cpp
// In setupGraph(), after successful retrieve:
if (m_qnnInterface.graphFinalize(m_graphHandle, nullptr, nullptr) != QNN_SUCCESS) {
    std::cerr << "Warning: Graph finalization failed" << std::endl;
}
```

### 4. Buffer Alignment Issues
**Symptom**: Silent failure  
**Theory**: HTP backend requires specific memory alignment

**Test**: Use aligned allocation
```cpp
// In setupTensors():
m_inputBuffer = aligned_alloc(128, 128);  // 128-byte alignment
m_outputBuffer = aligned_alloc(128, 10000);
```

## Debug Commands

### Enable QNN verbose logging
```bash
adb shell "cd /data/local/tmp/qnn-rag-demo && \
    export LD_LIBRARY_PATH=.:\$LD_LIBRARY_PATH && \
    export ADSP_LIBRARY_PATH=. && \
    export QNN_LOG_LEVEL=verbose && \
    export QNN_ENABLE_DEBUG=1 && \
    ./qidk_rag_demo model_context.bin query.fvecs results.txt libQnnHtp.so 2>&1"
```

### Check tensor info in context
```bash
adb shell "cd /data/local/tmp/qnn-rag-demo && \
    export LD_LIBRARY_PATH=.:\$LD_LIBRARY_PATH && \
    ./qnn-net-run --backend libQnnHtp.so --retrieve_context model_context.bin --dump_tensors"
```

### Test with minimal model
```bash
# Use the 4x8 test model instead
adb shell "cd /data/local/tmp/qnn-rag-demo && \
    ./qnn-context-binary-generator --model libtest_model.so --backend libQnnHtp.so \
        --binary_file test_ctx --output_dir . && \
    mv output/test_ctx.bin test_context.bin"

# Update QnnRunner to use [1,4] input, [1,8] output
# Run with smaller test data
```

## Next Steps

1. **Immediate**: Check QNN SampleApp for proper tensor passing after contextCreateFromBinary
   - Location: `/root/qualcomm/qnn/examples/QNN/SampleApp/src/QnnSampleApp.cpp`
   - Look for: How they handle graphExecute with context binaries

2. **Alternative 1**: Use `qnn-net-run` tool to verify context binary is valid
   ```bash
   qnn-net-run --backend libQnnHtp.so --retrieve_context model_context.bin \
               --input_list input.txt --output_dir output/
   ```

3. **Alternative 2**: Generate context with debug symbols
   ```bash
   qnn-context-binary-generator --model libmodel.so --backend libQnnHtp.so \
                                 --binary_file model_ctx --log_level verbose
   ```

4. **Contact Support**: If all tests fail, contact Qualcomm with:
   - SDK version: v2.29.0.241129103708_105762
   - Device: Snapdragon 8 Gen 2 (Hexagon V73)
   - Error: "QnnGraph_execute fails after successful contextCreateFromBinary"
   - Logs: Attach full output with QNN_LOG_LEVEL=verbose

## Files Modified
- `android/app/main/jni/QnnRunner.cpp`: Quantization implementation
- `android/app/main/jni/QnnRunner.h`: Context binary path constructor
- `scripts/deploy.sh`: Added libQnnHtpPrepare.so, auto context generation
- `qnn/convert_to_qnn.sh`: ARM64 target with quantization

## Key Discovery
**libQnnHtpPrepare.so (46MB) is REQUIRED for HTP backend** - without it, context binary generation fails with "Graph Compose failure".
