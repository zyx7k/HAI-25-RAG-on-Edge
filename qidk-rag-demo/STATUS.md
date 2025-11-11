# QNN RAG Demo - Current Status

## ✅ WORKING
1. **Model Conversion**: ONNX → QNN with quantization (UFIXED_POINT_8)
2. **Cross-compilation**: ARM64 model library generation
3. **Context Binary Generation**: On-device using qnn-context-binary-generator + libQnnHtpPrepare.so
4. **Backend Loading**: libQnnHtp.so loads successfully
5. **Device Creation**: QnnDevice_create succeeds (with expected warning for context binaries)
6. **Context Loading**: QnnContext_createFromBinary succeeds with 1.3MB context binary
7. **Graph Retrieval**: QnnGraph_retrieve("model") succeeds
8. **Tensor Setup**: Input/output tensors configured with correct dimensions and quantization params

## ❌ BLOCKED: Graph Execution Fails

```
[QNN INFO] Runner ready.
FATAL ERROR: QnnGraph_execute failed
```

**Symptoms:**
- All initialization steps succeed
- Context binary loads without errors
- Graph handle retrieved successfully
- Tensors configured properly
- `QnnGraph_execute()` returns error (no detailed message even with QNN_ENABLE_DEBUG=1)

**Context:**
- Device: Snapdragon 8 Gen 2 (Hexagon V73)
- Model: 128-dim input → 10K-dim output (MatMul/FullyConnected)
- Quantization: UFIXED_POINT_8, input scale=0.6627451, output scale=1013.4312
- Context binary: 1.3MB, generated on-device with HTP backend

## 🔍 Investigation Needed

### Hypothesis 1: Tensor Name Mismatch
**Theory**: Tensor names in code ("query", "scores") don't match context binary's internal names

**Test**:
```cpp
// Try without names
m_inputTensor.v1.name = nullptr;
m_outputTensor.v1.name = nullptr;
```

### Hypothesis 2: Tensor IDs Required
**Theory**: Context binaries may need correct tensor IDs (currently using id=0)

**Solution**: Query graph for actual tensor metadata using SystemContext API

### Hypothesis 3: Missing Graph Finalization
**Theory**: Graph needs explicit finalization after retrieve from context

**Test**:
```cpp
// After graphRetrieve:
if (m_qnnInterface->QNN_INTERFACE_VER_NAME.graphFinalize) {
    m_qnnInterface->QNN_INTERFACE_VER_NAME.graphFinalize(
        m_graphHandle, nullptr, nullptr);
}
```

### Hypothesis 4: Wrong Tensor Descriptors
**Theory**: When using context binaries, QNN expects to use the graph's internal tensors, not custom-created ones

**Solution Approach** (from QNN SampleApp):
1. Load context binary
2. Use `QnnSystemContext_getBinaryInfo` to get metadata  
3. Call `copyMetadataToGraphsInfo` to populate tensor descriptors from binary
4. Deep copy tensor info and allocate buffers matching binary's expectations
5. Pass these properly initialized tensors to graphExecute

**Key Insight from SampleApp**:
```cpp
// SampleApp does this:
iotensor::StatusCode iotensor::IOTensor::setupInputAndOutputTensors(
    Qnn_Tensor_t** inputs, Qnn_Tensor_t** outputs, 
    qnn_wrapper_api::GraphInfo_t graphInfo) {
  
  // graphInfo.inputTensors comes from context binary metadata
  setupTensors(inputs, graphInfo.numInputTensors, graphInfo.inputTensors);
  setupTensors(outputs, graphInfo.numOutputTensors, graphInfo.outputTensors);
  
  // For each tensor:
  //   - Deep copy tensor info from graphInfo
  //   - Allocate buffer matching size/type
  //   - Set clientBuf to point to allocated buffer
}
```

### Hypothesis 5: Quantization Parameters Wrong
**Theory**: Hardcoded scale/offset values don't match what context binary expects

**Test**: Extract actual quantization from loaded graph (requires SystemContext API)

### Hypothesis 6: Buffer Alignment
**Theory**: HTP backend requires specific memory alignment (128-byte?)

**Test**:
```cpp
m_inputBuffer = aligned_alloc(128, 128);   // 128-byte aligned
m_outputBuffer = aligned_alloc(128, 10000);
```

## 📋 Next Steps (Priority Order)

### Option A: Follow SampleApp Pattern (RECOMMENDED)
This is the "official" way to handle context binaries:

1. **Add SystemContext API calls** to get binary metadata:
   ```cpp
   // After loading context binary:
   QnnSystemContext_Handle_t sysCtxHandle;
   m_sysInterface->systemContextCreate(&sysCtxHandle);
   
   const QnnSystemContext_BinaryInfo_t* binaryInfo;
   m_sysInterface->systemContextGetBinaryInfo(sysCtxHandle, 
       contextBinaryData, contextBinarySize, &binaryInfo, &binaryInfoSize);
   
   // Extract tensor info from binaryInfo
   // graphInfo->inputTensors = binaryInfo->graphsInfo[0].inputTensors;
   // graphInfo->outputTensors = binaryInfo->graphsInfo[0].outputTensors;
   ```

2. **Deep copy tensor descriptors** from binary metadata

3. **Allocate buffers** matching binary's specifications

4. **Execute** with properly initialized tensors

**Pros**: Most reliable, matches Qualcomm's examples
**Cons**: More complex, requires understanding SystemContext API

### Option B: Test Simple Fixes First
Try quick tests before major refactor:

1. ✅ Set tensor names to nullptr
2. ✅ Try aligned_alloc for buffers  
3. ✅ Add graphFinalize call
4. ✅ Try passing nullptr for profile handle explicitly

### Option C: Use qnn-net-run Tool
Verify context binary is valid:
```bash
cd /data/local/tmp/qnn-rag-demo
export LD_LIBRARY_PATH=.:$LD_LIBRARY_PATH
./qnn-net-run --backend libQnnHtp.so \
              --retrieve_context model_context.bin \
              --input_list input_files.txt \
              --output_dir output/
```

If qnn-net-run works, context is valid and issue is in our tensor setup.

### Option D: Contact Qualcomm Support
Provide:
- SDK version: v2.29.0.241129103708_105762
- Device: Snapdragon 8 Gen 2 (Hexagon V73)  
- Error: "QnnGraph_execute fails after successful contextCreateFromBinary"
- Observation: Context loads, graph retrieves, but execute fails with no error details

## 🔧 Quick Fixes to Try Now

Let me try the simplest fixes first. Update `QnnRunner.cpp` setupTensors():

```cpp
// Try removing names (context binary may not match)
m_inputTensor.v1.name = nullptr;
m_outputTensor.v1.name = nullptr;

// Try removing IDs
// m_inputTensor.v1.id = 0;  // Comment this out
// m_outputTensor.v1.id = 0;  // Comment this out
```

Update `setupGraph()` to add finalization:

```cpp
void QnnRunner::setupGraph() {
    QNN_LOG("INFO", "Retrieving graph handle...");
    
    if (m_qnnInterface->QNN_INTERFACE_VER_NAME.graphRetrieve(
            m_contextHandle, "model", &m_graphHandle) != QNN_SUCCESS) {
        throw std::runtime_error("Failed to retrieve graph");
    }
    
    QNN_LOG("INFO", "Graph retrieved, finalizing...");
    
    // Try finalizing graph after retrieve
    if (m_qnnInterface->QNN_INTERFACE_VER_NAME.graphFinalize) {
        auto status = m_qnnInterface->QNN_INTERFACE_VER_NAME.graphFinalize(
            m_graphHandle, nullptr, nullptr);
        if (status != QNN_SUCCESS) {
            QNN_LOG("WARNING", "Graph finalization returned: " + std::to_string(status));
        } else {
            QNN_LOG("INFO", "Graph finalized successfully.");
        }
    }
}
```

## 📁 Key Files
- **QnnRunner.cpp**: Main inference wrapper (lines 98-231)
- **deploy.sh**: Generates context binary on device (line 107-113)
- **model.cpp**: Generated by qnn-model-lib-generator (has tensor metadata)
- **Context binary**: /data/local/tmp/qnn-rag-demo/model_context.bin (1.3MB)

## 🔑 Critical Discovery
**libQnnHtpPrepare.so (46MB) is REQUIRED** for context binary generation. Without it:
- qnn-context-binary-generator fails with "Graph Compose failure"
- Context binaries cannot be created on-device
- Must be deployed alongside other QNN libraries

This was the blocker that took longest to identify.
