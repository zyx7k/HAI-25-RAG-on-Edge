#ifndef QNN_RUNNER_H
#define QNN_RUNNER_H

#include <string>
#include <vector>
#include <stdexcept>
#include <cstdint>

// --- QNN SDK Includes ---
// Component headers first
#include "QNN/QnnTypes.h"
#include "QNN/QnnCommon.h"
#include "QNN/QnnBackend.h"
#include "QNN/QnnContext.h"
#include "QNN/QnnGraph.h"
#include "QNN/QnnTensor.h"
#include "QNN/System/QnnSystemInterface.h"

// Main interface header last
#include "QNN/QnnInterface.h"

class QnnRunner {
public:
    QnnRunner(const std::string& contextBinaryPath, const std::string& backendPath);
    ~QnnRunner();

    void execute(const std::vector<float>& query, std::vector<float>& scores);
    const std::vector<uint32_t>& getInputDims() const { return m_inputDims; }
    const std::vector<uint32_t>& getOutputDims() const { return m_outputDims; }

private:
    void loadBackend();
    void initializeBackend();
    void createContext();
    void setupGraph();
    void setupTensors();
    void cleanup();
    size_t getDataTypeSize(Qnn_DataType_t type);

    // Handles
    void* m_backendLibHandle = nullptr;
    void* m_systemLibHandle = nullptr;
    Qnn_BackendHandle_t m_backendHandle = nullptr;
    Qnn_ContextHandle_t m_contextHandle = nullptr;
    Qnn_GraphHandle_t m_graphHandle = nullptr;
    QnnSystemInterface_t* m_sysInterface = nullptr;
    QnnInterface_t* m_qnnInterface = nullptr;

    // Tensors
    std::vector<uint32_t> m_inputDims;
    std::vector<uint32_t> m_outputDims;
    Qnn_Tensor_t m_inputTensor{};
    Qnn_Tensor_t m_outputTensor{};
    void* m_inputBuffer = nullptr;
    void* m_outputBuffer = nullptr;
    size_t m_inputBufferSize = 0;
    size_t m_outputBufferSize = 0;

    // Paths
    std::string m_backendPath;
    std::string m_contextBinaryPath;
};

#endif // QNN_RUNNER_H