#ifndef QNNRUNNER_H
#define QNNRUNNER_H

#include <string>
#include <vector>
#include <QNN/QnnInterface.h>
#include <QNN/QnnTypes.h>
#include <QNN/System/QnnSystemInterface.h>
#include <QNN/System/QnnSystemContext.h>

class QnnRunner {
public:
    QnnRunner(const std::string& modelBinaryPath, const std::string& backendPath = "./libQnnHtp.so");
    ~QnnRunner();

    void execute(const std::vector<float>& query, std::vector<float>& scores);
    const std::vector<uint32_t>& getInputDims() const { return m_inputDims; }
    const std::vector<uint32_t>& getOutputDims() const { return m_outputDims; }

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