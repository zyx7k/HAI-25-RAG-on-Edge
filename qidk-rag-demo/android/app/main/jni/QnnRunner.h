#ifndef QNN_RUNNER_H
#define QNN_RUNNER_H

#include <string>
#include <vector>
#include <stdexcept>
#include <cstdint>

// --- QNN SDK includes (FIXED: Correct Order + "QNN/" Prefix) ---

// 1. Include the component definitions *FIRST*.
//    These paths are relative to the .../include path.
#include "QNN/QnnTypes.h"
#include "QNN/QnnCommon.h"
#include "QNN/QnnBackend.h"
#include "QNN/QnnContext.h"
#include "QNN/QnnGraph.h"
#include "QNN/QnnTensor.h"
#include "QNN/System/QnnSystemInterface.h"

// 2. Include the main interface header *LAST*.
//    Now the compiler knows the full definitions for all members.
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
    void* m_backendLibHandle;
    void* m_systemLibHandle;
    Qnn_BackendHandle_t m_backendHandle;
    Qnn_ContextHandle_t m_contextHandle;
    Qnn_GraphHandle_t m_graphHandle;
    QnnSystemInterface_t* m_sysInterface;
    QnnInterface_t* m_qnnInterface;

    // Tensors
    std::vector<uint32_t> m_inputDims;
    std::vector<uint32_t> m_outputDims;
    Qnn_Tensor_t m_inputTensor;
    Qnn_Tensor_t m_outputTensor;
    void* m_inputBuffer;
    void* m_outputBuffer;
    size_t m_inputBufferSize;
    size_t m_outputBufferSize;

    // Paths
    std::string m_backendPath;
    std::string m_contextBinaryPath;
};

#endif