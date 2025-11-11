# QNN RAG Demo - Final Status Report

## Summary

We've successfully resolved multiple blockers to get the QNN RAG demo running on a Snapdragon 8 Gen 2 device with HTP (Hexagon NPU) backend, but are currently blocked at the final execution step.

## What's Working ✅

1. **Model Pipeline**: ONNX → QNN conversion with INT8 quantization
2. **Cross-compilation**: ARM64 model library generation for aarch64-android
3. **Context Binary Generation**: On-device generation using qnn-context-binary-generator
4. **Critical Discovery**: libQnnHtpPrepare.so (46MB) is REQUIRED for HTP context binary generation
5. **Backend/Device/Context**: All initialization succeeds
6. **Graph Retrieval**: Successfully loads graph from context binary

## Current Blocker ❌

**QnnGraph_execute fails** after successful initialization.

```
[QNN INFO] Runner ready.
FATAL ERROR: QnnGraph_execute failed
```

## Root Cause Analysis

The issue appears to be that we're creating custom `Qnn_Tensor_t` descriptors manually, but **context binaries require using tensor metadata from the binary itself**.

### Evidence:
1. Graph finalization returns error 1000 (graph already finalized in context)
2. QNN SampleApp uses `QnnSystemContext_getBinaryInfo()` to extract tensor descriptors
3. Removing tensor names (set to nullptr) didn't fix the issue
4. All quantization parameters match the model.cpp generation

### Proper Solution (from QNN SampleApp):

```cpp
// 1. Load context binary
contextCreateFromBinary(contextData, contextSize, &contextHandle);

// 2. Extract metadata using SystemContext API
QnnSystemContext_Handle_t sysCtxHandle;
systemContextCreate(&sysCtxHandle);
systemContextGetBinaryInfo(sysCtxHandle, contextData, contextSize, &binaryInfo);

// 3. Copy tensor descriptors from binaryInfo
GraphInfo_t graphInfo;
copyMetadataToGraphsInfo(binaryInfo, &graphInfo);

// 4. Setup tensors using metadata
setupInputAndOutputTensors(&inputs, &outputs, graphInfo);
// This deep-copies tensor info and allocates buffers

// 5. Execute with properly initialized tensors
graphExecute(graphHandle, inputs, numInputs, outputs, numOutputs);
```

We're skipping steps 2-4, which is why execution fails.

## Implementation Options

### Option A: Implement SystemContext API (Recommended)
**Effort**: Medium (2-4 hours)
**Success Rate**: High (follows official examples)

Add proper tensor retrieval:
1. Call `QnnSystemContext_getBinaryInfo()` after loading context
2. Extract `inputTensors` and `outputTensors` from binary metadata
3. Deep copy tensor descriptors
4. Allocate buffers matching metadata specifications
5. Execute with these tensors

**Files to modify**:
- `QnnRunner.cpp`: Add SystemContext API calls in `setupTensors()`
- `QnnRunner.h`: Add SystemContext handle member

### Option B: Use Model Library Instead of Context Binary
**Effort**: Low (already supported)
**Success Rate**: Medium (loses HTP optimizations?)

Revert to using `libmodel.so` directly with `composeGraphs()` instead of context binary.

**Pros**: Simpler API, no SystemContext needed
**Cons**: May not get same HTP optimizations as context binary

### Option C: Use qnn-net-run Tool
**Effort**: Very Low
**Success Rate**: High (for validation only)

Use Qualcomm's qnn-net-run tool to verify inference works:
```bash
./qnn-net-run --backend libQnnHtp.so \
              --retrieve_context model_context.bin \
              --input_list inputs.txt \
              --output_dir output/
```

This confirms the context binary is valid but doesn't solve the C++ integration.

### Option D: Contact Qualcomm Support
**Effort**: Low
**Success Rate**: Unknown (depends on response time)

Provide:
- SDK: v2.29.0.241129103708_105762
- Device: Snapdragon 8 Gen 2 (Hexagon V73)
- Error: "QnnGraph_execute fails after successful contextCreateFromBinary"
- All initialization succeeds but execute returns error with no details

## Key Discoveries

### 1. libQnnHtpPrepare.so is Critical
This 46MB library is REQUIRED for:
- On-device context binary generation
- HTP backend graph composition
- Without it: "Graph Compose failure" errors

Must be deployed with other QNN libraries.

### 2. Context Binary Generation Path
`qnn-context-binary-generator` creates output in `output/` subdirectory with double `.bin.bin` extension:
```bash
./qnn-context-binary-generator --model libmodel.so \
    --backend libQnnHtp.so \
    --binary_file model_context

# Creates: output/model_context.bin.bin (not model_context.bin!)
```

### 3. Device Handle Warning is Expected
```
[QNN WARNING] QnnDevice_create failed, continuing without device handle
```

This is normal for context binaries - device handle was used during generation, not needed for loading.

## Files Modified

1. **android/app/main/jni/QnnRunner.cpp**
   - Context binary loading
   - Manual tensor setup (needs SystemContext integration)
   - Quantization conversion

2. **android/app/main/jni/QnnRunner.h**
   - Changed to contextCreateFromBinary approach
   - Added binary path parameter

3. **scripts/deploy.sh**
   - Added libQnnHtpPrepare.so to deployed libraries
   - Auto-generates context binary on device
   - Handles .bin.bin file extension

4. **qnn/convert_to_qnn.sh**
   - Targets aarch64-android
   - Enables quantization

## Quick Test Commands

```bash
# Rebuild
cd /root/qidk-rag-demo && bash scripts/build.sh

# Deploy (includes context generation)
bash scripts/deploy.sh

# Manual test on device
adb shell "cd /data/local/tmp/qnn-rag-demo && \
    export LD_LIBRARY_PATH=.:\$LD_LIBRARY_PATH && \
    export ADSP_LIBRARY_PATH=. && \
    ./qidk_rag_demo model_context.bin query.fvecs results.txt libQnnHtp.so"

# Validate context binary with qnn-net-run
adb shell "cd /data/local/tmp/qnn-rag-demo && \
    export LD_LIBRARY_PATH=.:\$LD_LIBRARY_PATH && \
    ./qnn-net-run --backend libQnnHtp.so \
        --retrieve_context model_context.bin \
        --input_list input_files.txt"
```

## Recommended Next Step

**Implement SystemContext API** (Option A) following the SampleApp pattern. This is the "correct" way to handle context binaries and will ensure compatibility with future QNN SDK versions.

The alternative (Option B - using model library directly) might work but defeats the purpose of using context binaries which are optimized for the specific hardware.

## Additional Resources

- QNN SDK: `/root/qualcomm/qnn/`
- SampleApp source: `/root/qualcomm/qnn/examples/QNN/SampleApp/`
- Key file: `src/Utils/IOTensor.cpp` (setupTensors implementation)
- Documentation: QNN SDK Programming Guide (context binary section)

---

**Bottom Line**: We're 95% there. Just need to properly extract and use tensor metadata from the context binary instead of creating custom tensors.
