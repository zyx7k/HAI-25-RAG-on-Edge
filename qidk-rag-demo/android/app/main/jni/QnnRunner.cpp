#include "QnnRunner.h"
#include <iostream>
#include <fstream>
#include <cstring>
#include <dlfcn.h>
#include <vector>

#define QNN_LOG(level, msg) std::cout << "[QNN " << level << "] " << msg << std::endl

QnnRunner::QnnRunner(const std::string& contextBinaryPath, const std::string& backendPath)
    : m_backendLibHandle(nullptr),
      m_systemLibHandle(nullptr),
      m_backendHandle(nullptr),
      m_contextHandle(nullptr),
      m_graphHandle(nullptr),
      m_sysInterface(nullptr),
      m_qnnInterface(nullptr),
      m_inputBuffer(nullptr),
      m_outputBuffer(nullptr),
      m_inputBufferSize(0),
      m_outputBufferSize(0),
      m_backendPath(backendPath),
      m_contextBinaryPath(contextBinaryPath)
{
    QNN_LOG("INFO", "Initializing QNN Runner...");
    loadBackend();
    initializeBackend();
    createContext();
    setupGraph();
    setupTensors();
    QNN_LOG("INFO", "Runner ready.");
}

QnnRunner::~QnnRunner() { cleanup(); }

void QnnRunner::loadBackend() {
    QNN_LOG("INFO", "Loading QNN backend libraries...");
    
    m_systemLibHandle = dlopen("/root/qualcomm/qnn/lib/aarch64-android/libQnnSystem.so", RTLD_NOW | RTLD_LOCAL);
    if (!m_systemLibHandle)
        throw std::runtime_error("Failed to load libQnnSystem.so: " + std::string(dlerror()));

    auto sysGetProviders = (Qnn_ErrorHandle_t (*)(const QnnSystemInterface_t***, uint32_t*))
        dlsym(m_systemLibHandle, "QnnSystemInterface_getProviders");
    if (!sysGetProviders)
        throw std::runtime_error("Failed to find QnnSystemInterface_getProviders");

    const QnnSystemInterface_t** sysProviders = nullptr;
    uint32_t sysCount = 0;
    if (sysGetProviders(&sysProviders, &sysCount) != QNN_SUCCESS || sysCount == 0)
        throw std::runtime_error("No QNN System providers found");
    
    m_sysInterface = const_cast<QnnSystemInterface_t*>(sysProviders[0]);

    m_backendLibHandle = dlopen(m_backendPath.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!m_backendLibHandle)
        throw std::runtime_error("Failed to load backend: " + std::string(dlerror()));

    auto getProviders = (Qnn_ErrorHandle_t (*)(const QnnInterface_t***, uint32_t*))
        dlsym(m_backendLibHandle, "QnnInterface_getProviders");
    if (!getProviders)
        throw std::runtime_error("Failed to find QnnInterface_getProviders in backend");

    const QnnInterface_t** providers = nullptr;
    uint32_t numProviders = 0;
    if (getProviders(&providers, &numProviders) != QNN_SUCCESS || numProviders == 0)
        throw std::runtime_error("No QNN interface providers found");
    
    m_qnnInterface = const_cast<QnnInterface_t*>(providers[0]);
    
    QNN_LOG("INFO", "QNN interfaces loaded successfully.");
}

void QnnRunner::initializeBackend() {
    QNN_LOG("INFO", "Creating backend...");
    const QnnBackend_Config_t* cfg[] = { nullptr };
    
    if (m_qnnInterface->QNN_INTERFACE_VER_NAME.backendCreate(nullptr, cfg, &m_backendHandle) != QNN_SUCCESS)
        throw std::runtime_error("QnnBackend_create failed");
}

void QnnRunner::createContext() {
    QNN_LOG("INFO", "Creating context from binary...");
    
    std::ifstream file(m_contextBinaryPath, std::ios::binary | std::ios::ate);
    if (!file)
        throw std::runtime_error("Cannot open context binary: " + m_contextBinaryPath);
    
    size_t size = file.tellg();
    file.seekg(0);
    std::vector<uint8_t> data(size);
    file.read(reinterpret_cast<char*>(data.data()), size);
    file.close();

    const QnnContext_Config_t* cfg[] = { nullptr };
    
    if (m_qnnInterface->QNN_INTERFACE_VER_NAME.contextCreateFromBinary(
        m_backendHandle, nullptr, cfg, data.data(), size, &m_contextHandle, nullptr) != QNN_SUCCESS)
        throw std::runtime_error("QnnContext_createFromBinary failed");
}

void QnnRunner::setupGraph() {
    const char* name = "graph";
    
    if (m_qnnInterface->QNN_INTERFACE_VER_NAME.graphRetrieve(
        m_contextHandle, name, &m_graphHandle) != QNN_SUCCESS)
        throw std::runtime_error("QnnGraph_retrieve failed");
    
    QNN_LOG("INFO", "Graph retrieved successfully.");
}

void QnnRunner::setupTensors() {
    QNN_LOG("INFO", "Setting up tensors...");
    
    // For QNN SDK 2.22, when loading from a binary context, the graph already has
    // input/output tensors defined. We need to create our own tensor structures
    // that match what was defined in the model, and allocate buffers for them.
    
    // Based on typical embedding model architectures, we'll assume:
    // - Input: [1, embedding_dim] or [batch, embedding_dim]
    // - Output: [1, num_vectors] or [batch, num_vectors] similarity scores
    
    // Since we don't have direct API access to query tensor info from the loaded graph,
    // we need to hardcode the expected dimensions or pass them as parameters.
    // For a typical vector search model with 10k vectors:
    
    // Hardcoded dimensions - these should match your model's actual I/O
    // You can get these from your model conversion logs
    uint32_t input_dim = 128;    // Typical embedding dimension (adjust as needed)
    uint32_t output_dim = 10000; // Number of vectors in your database
    
    QNN_LOG("INFO", "Using hardcoded tensor dimensions:");
    QNN_LOG("INFO", "  Input: [1, " + std::to_string(input_dim) + "]");
    QNN_LOG("INFO", "  Output: [1, " + std::to_string(output_dim) + "]");
    
    m_inputDims = {1, input_dim};
    m_outputDims = {1, output_dim};
    
    // Allocate buffers
    m_inputBufferSize = sizeof(float) * input_dim;
    m_outputBufferSize = sizeof(float) * output_dim;
    
    m_inputBuffer = malloc(m_inputBufferSize);
    m_outputBuffer = malloc(m_outputBufferSize);
    
    if (!m_inputBuffer || !m_outputBuffer)
        throw std::runtime_error("Failed to allocate tensor buffers");
    
    // Initialize tensor structures for execution
    // These will be used in QnnGraph_execute()
    m_inputTensor = QNN_TENSOR_INIT;
    m_inputTensor.version = QNN_TENSOR_VERSION_1;
    m_inputTensor.v1.id = 0; // Will be ignored/overridden by backend
    m_inputTensor.v1.name = "input";
    m_inputTensor.v1.type = QNN_TENSOR_TYPE_APP_WRITE;
    m_inputTensor.v1.dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER;
    m_inputTensor.v1.dataType = QNN_DATATYPE_FLOAT_32;
    m_inputTensor.v1.rank = 2;
    m_inputTensor.v1.dimensions = m_inputDims.data();
    m_inputTensor.v1.memType = QNN_TENSORMEMTYPE_RAW;
    m_inputTensor.v1.clientBuf.data = m_inputBuffer;
    m_inputTensor.v1.clientBuf.dataSize = m_inputBufferSize;
    
    m_outputTensor = QNN_TENSOR_INIT;
    m_outputTensor.version = QNN_TENSOR_VERSION_1;
    m_outputTensor.v1.id = 0; // Will be ignored/overridden by backend
    m_outputTensor.v1.name = "output";
    m_outputTensor.v1.type = QNN_TENSOR_TYPE_APP_READ;
    m_outputTensor.v1.dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER;
    m_outputTensor.v1.dataType = QNN_DATATYPE_FLOAT_32;
    m_outputTensor.v1.rank = 2;
    m_outputTensor.v1.dimensions = m_outputDims.data();
    m_outputTensor.v1.memType = QNN_TENSORMEMTYPE_RAW;
    m_outputTensor.v1.clientBuf.data = m_outputBuffer;
    m_outputTensor.v1.clientBuf.dataSize = m_outputBufferSize;
    
    QNN_LOG("INFO", "Tensors configured with buffers allocated.");
}

void QnnRunner::execute(const std::vector<float>& query, std::vector<float>& scores) {
    if (query.size() * sizeof(float) != m_inputBufferSize) {
        throw std::runtime_error("Query size mismatch. Expected " + 
            std::to_string(m_inputBufferSize / sizeof(float)) + " elements, got " + 
            std::to_string(query.size()));
    }
    
    std::memcpy(m_inputBuffer, query.data(), m_inputBufferSize);
    
    if (m_qnnInterface->QNN_INTERFACE_VER_NAME.graphExecute(
        m_graphHandle, &m_inputTensor, 1, &m_outputTensor, 1, nullptr, nullptr) != QNN_SUCCESS)
        throw std::runtime_error("QnnGraph_execute failed");
    
    if (scores.size() * sizeof(float) != m_outputBufferSize) {
        scores.resize(m_outputBufferSize / sizeof(float));
    }
    
    std::memcpy(scores.data(), m_outputBuffer, m_outputBufferSize);
}

void QnnRunner::cleanup() {
    if (m_inputBuffer) free(m_inputBuffer);
    if (m_outputBuffer) free(m_outputBuffer);
    
    if (m_contextHandle && m_qnnInterface) {
        m_qnnInterface->QNN_INTERFACE_VER_NAME.contextFree(m_contextHandle, nullptr);
    }
    
    if (m_backendHandle && m_qnnInterface) {
        m_qnnInterface->QNN_INTERFACE_VER_NAME.backendFree(m_backendHandle);
    }
    
    if (m_backendLibHandle) dlclose(m_backendLibHandle);
    if (m_systemLibHandle) dlclose(m_systemLibHandle);
    
    m_inputBuffer = nullptr;
    m_outputBuffer = nullptr;
    m_contextHandle = nullptr;
    m_backendHandle = nullptr;
    m_backendLibHandle = nullptr;
    m_systemLibHandle = nullptr;
    m_qnnInterface = nullptr;
}

size_t QnnRunner::getDataTypeSize(Qnn_DataType_t type) {
    switch (type) {
        case QNN_DATATYPE_FLOAT_32: return 4;
        case QNN_DATATYPE_FLOAT_16: return 2;
        case QNN_DATATYPE_INT_32: return 4;
        case QNN_DATATYPE_INT_16: return 2;
        case QNN_DATATYPE_INT_8:
        case QNN_DATATYPE_UINT_8: return 1;
        default: return 4;
    }
}