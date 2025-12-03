#ifndef QNNRUNNER_H
#define QNNRUNNER_H

#include <string>
#include <vector>
#include <QNN/QnnInterface.h>
#include <QNN/QnnTypes.h>
#include <QNN/System/QnnSystemInterface.h>
#include <QNN/System/QnnSystemContext.h>

// Struct to hold detailed execution timing
struct ExecutionTiming {
    double quantize_ms;      // Input quantization time (CPU)
    double graph_execute_ms; // Actual NPU graph execution time
    double dequantize_ms;    // Output dequantization time (CPU)
    double total_ms;         // Total execute() time
};

class QnnRunner {
public:
    QnnRunner(const std::string& modelBinaryPath, const std::string& backendPath = "./libQnnHtp.so");
    ~QnnRunner();

    // Original methods (with dequantization)
    void execute(const std::vector<float>& query, std::vector<float>& scores);
    void execute(const std::vector<float>& query, std::vector<float>& scores, ExecutionTiming& timing);
    
    // Optimized method: returns raw INT8 output, skips dequantization
    // For top-K, INT8 ordering is preserved (higher INT8 = higher similarity)
    void executeRaw(const std::vector<float>& query, ExecutionTiming& timing);
    
    // Batched execution: process multiple queries at once
    // Input: [batch_size * dim] flattened queries (normalized)
    // Output: raw INT8 buffer of size [batch_size * num_docs]
    void executeBatchRaw(const std::vector<float>& batch_queries, ExecutionTiming& timing);
    
    // Get raw output buffer pointer (valid after executeRaw or executeBatchRaw)
    const uint8_t* getRawOutputBuffer() const { return static_cast<const uint8_t*>(m_outputBuffer); }
    size_t getOutputSize() const { return m_outputBufferSize; }
    
    // Get quantization scale for converting INT8 back to float if needed
    float getOutputScale() const { return m_outputScale; }
    
    const std::vector<uint32_t>& getInputDims() const { return m_inputDims; }
    const std::vector<uint32_t>& getOutputDims() const { return m_outputDims; }
    
    // Get batch size from model (first dimension of input)
    size_t getBatchSize() const { return m_inputDims.size() > 0 ? m_inputDims[0] : 1; }
    // Get dimension (second dimension of input, or first if batch=1)
    size_t getDim() const { return m_inputDims.size() > 1 ? m_inputDims[1] : (m_inputDims.size() > 0 ? m_inputDims[0] : 0); }
    // Get number of documents (last dimension of output)
    size_t getNumDocs() const { return m_outputDims.size() > 0 ? m_outputDims.back() : 0; }

private:
    void loadBackend();
    void initializeBackend();
    void createDevice();
    void createContext();
    void setupGraph();
    void setupTensors();
    void setupTensorsManual();
    bool setupTensorsFromGraphInfo(const QnnSystemContext_GraphInfoV1_t& graphInfo);
    bool setupTensorsFromGraphInfoV3(const QnnSystemContext_GraphInfoV3_t& graphInfo);
    bool deepCopyTensorInfo(Qnn_Tensor_t* dst, const Qnn_Tensor_t* src);
    size_t calculateTensorSize(const Qnn_Tensor_t* tensor);
    void cleanup();

    void* m_backendLibHandle;
    void* m_systemLibHandle;
    Qnn_BackendHandle_t m_backendHandle;
    Qnn_DeviceHandle_t m_deviceHandle;
    Qnn_ContextHandle_t m_contextHandle;
    Qnn_GraphHandle_t m_graphHandle;
    QnnSystemInterface_t* m_sysInterface;
    QnnInterface_t* m_qnnInterface;

    void* m_inputBuffer;
    void* m_outputBuffer;
    size_t m_inputBufferSize;
    size_t m_outputBufferSize;
    
    // Cached quantization parameters
    float m_inputScale;
    float m_outputScale;
    
    std::vector<uint32_t> m_inputDims;
    std::vector<uint32_t> m_outputDims;
    Qnn_Tensor_t m_inputTensor;
    Qnn_Tensor_t m_outputTensor;
    
    // Store context binary data for SystemContext queries
    std::vector<uint8_t> m_contextBinaryData;

    std::string m_backendPath;
    std::string m_modelBinaryPath;
};

#endif // QNNRUNNER_H