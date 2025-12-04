#ifndef QNNRUNNER_H
#define QNNRUNNER_H

#include <string>
#include <vector>
#include <QNN/QnnInterface.h>
#include <QNN/QnnTypes.h>
#include <QNN/System/QnnSystemInterface.h>
#include <QNN/System/QnnSystemContext.h>

struct ExecutionTiming {
    double quantize_ms;
    double graph_execute_ms;
    double dequantize_ms;
    double total_ms;
};

class QnnRunner {
public:
    QnnRunner(const std::string& modelBinaryPath, const std::string& backendPath = "./libQnnHtp.so");
    ~QnnRunner();

    void execute(const std::vector<float>& query, std::vector<float>& scores);
    void execute(const std::vector<float>& query, std::vector<float>& scores, ExecutionTiming& timing);
    
    void executeRaw(const std::vector<float>& query, ExecutionTiming& timing);
    
    void executeBatchRaw(const std::vector<float>& batch_queries, ExecutionTiming& timing);
    
    const uint8_t* getRawOutputBuffer() const { return static_cast<const uint8_t*>(m_outputBuffer); }
    size_t getOutputSize() const { return m_outputBufferSize; }
    float getOutputScale() const { return m_outputScale; }
    
    const std::vector<uint32_t>& getInputDims() const { return m_inputDims; }
    const std::vector<uint32_t>& getOutputDims() const { return m_outputDims; }
    
    size_t getBatchSize() const { return m_inputDims.size() > 0 ? m_inputDims[0] : 1; }
    size_t getDim() const { return m_inputDims.size() > 1 ? m_inputDims[1] : (m_inputDims.size() > 0 ? m_inputDims[0] : 0); }
    size_t getNumDocs() const { return m_outputDims.size() > 0 ? m_outputDims.back() : 0; }

    bool isFloatModel() const { return m_isFloatModel; }

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
    
    bool m_isFloatModel;
};

#endif // QNNRUNNER_H