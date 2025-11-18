#include "QnnRunner.h"
#include <iostream>
#include <fstream>
#include <cstring>
#include <cmath>
#include <dlfcn.h>
#include <vector>
#include <utility>

#define QNN_LOG(level, msg) std::cout << "[QNN " << level << "] " << msg << std::endl

QnnRunner::QnnRunner(const std::string& modelBinaryPath, const std::string& backendPath)
        : m_backendLibHandle(nullptr),
            m_systemLibHandle(nullptr),
            m_backendHandle(nullptr),
            m_deviceHandle(nullptr),
            m_contextHandle(nullptr),
            m_graphHandle(nullptr),
            m_sysInterface(nullptr),
            m_qnnInterface(nullptr),
            m_inputBuffer(nullptr),
            m_outputBuffer(nullptr),
            m_inputBufferSize(0),
            m_outputBufferSize(0),
            m_backendPath(backendPath),
            m_modelBinaryPath(modelBinaryPath),
            m_graphName("model") {
    QNN_LOG("INFO", "Initializing QNN Runner...");
    loadBackend();
    initializeBackend();
    createDevice();
    createContext();
    setupTensors();
        setupGraph();
    QNN_LOG("INFO", "Runner ready.");
}

QnnRunner::~QnnRunner() { cleanup(); }

void QnnRunner::loadBackend() {
    QNN_LOG("INFO", "Loading QNN backend libraries...");

    // Load System library
    m_systemLibHandle = dlopen("libQnnSystem.so", RTLD_NOW | RTLD_LOCAL);
    if (!m_systemLibHandle) {
        m_systemLibHandle = dlopen("./libQnnSystem.so", RTLD_NOW | RTLD_LOCAL);
    }
    if (!m_systemLibHandle)
        throw std::runtime_error("Failed to load libQnnSystem.so: " + std::string(dlerror()));

    auto sysGetProviders = (Qnn_ErrorHandle_t (*)(const QnnSystemInterface_t***, uint32_t*))
        dlsym(m_systemLibHandle, "QnnSystemInterface_getProviders");
    if (!sysGetProviders)
        throw std::runtime_error("Missing QnnSystemInterface_getProviders");

    const QnnSystemInterface_t** sysProviders = nullptr;
    uint32_t sysCount = 0;
    if (sysGetProviders(&sysProviders, &sysCount) != QNN_SUCCESS || sysCount == 0)
        throw std::runtime_error("No QNN System providers found");

    m_sysInterface = const_cast<QnnSystemInterface_t*>(sysProviders[0]);

    // Load backend library
    m_backendLibHandle = dlopen(m_backendPath.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!m_backendLibHandle)
        throw std::runtime_error("Failed to load backend: " + std::string(dlerror()));

    auto getProviders = (Qnn_ErrorHandle_t (*)(const QnnInterface_t***, uint32_t*))
        dlsym(m_backendLibHandle, "QnnInterface_getProviders");
    if (!getProviders)
        throw std::runtime_error("Missing QnnInterface_getProviders in backend");

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

void QnnRunner::createDevice() {
    QNN_LOG("INFO", "Creating device...");
    const QnnDevice_Config_t* deviceCfg[] = { nullptr };
    if (m_qnnInterface->QNN_INTERFACE_VER_NAME.deviceCreate(nullptr, deviceCfg, &m_deviceHandle) != QNN_SUCCESS) {
        QNN_LOG("WARNING", "QnnDevice_create failed, continuing without device handle");
        m_deviceHandle = nullptr;
    } else {
        QNN_LOG("INFO", "Device created successfully.");
    }
}

void QnnRunner::createContext() {
    QNN_LOG("INFO", "Creating context from binary...");

    std::ifstream file(m_modelBinaryPath, std::ios::binary | std::ios::ate);
    if (!file) throw std::runtime_error("Cannot open model binary: " + m_modelBinaryPath);

    size_t size = file.tellg();
    file.seekg(0);
    m_contextBinaryData.resize(size);
    file.read(reinterpret_cast<char*>(m_contextBinaryData.data()), size);
    file.close();

    const QnnContext_Config_t* cfg[] = { nullptr };
    if (m_qnnInterface->QNN_INTERFACE_VER_NAME.contextCreateFromBinary(
            m_backendHandle, m_deviceHandle, cfg, m_contextBinaryData.data(), size, &m_contextHandle, nullptr) != QNN_SUCCESS)
        throw std::runtime_error("QnnContext_createFromBinary failed");
    
    QNN_LOG("INFO", "Context created successfully from binary.");
}

void QnnRunner::setupGraph() {
    QNN_LOG("INFO", "Retrieving graph handle...");
    
    if (m_qnnInterface->QNN_INTERFACE_VER_NAME.graphRetrieve(m_contextHandle, m_graphName.c_str(), &m_graphHandle) != QNN_SUCCESS) {
        throw std::runtime_error("Failed to retrieve graph '" + m_graphName + "' from context");
    }
    
    QNN_LOG("INFO", "Graph handle obtained, attempting finalization...");
    
    // Try finalizing the graph (may be needed for context binaries)
    if (m_qnnInterface->QNN_INTERFACE_VER_NAME.graphFinalize) {
        auto status = m_qnnInterface->QNN_INTERFACE_VER_NAME.graphFinalize(m_graphHandle, nullptr, nullptr);
        if (status != QNN_SUCCESS) {
            QNN_LOG("WARNING", "Graph finalization returned error: " + std::to_string(status));
        } else {
            QNN_LOG("INFO", "Graph finalized successfully.");
        }
    } else {
        QNN_LOG("INFO", "No graphFinalize function available (already finalized in context).");
    }
}

void QnnRunner::setupTensors() {
    QNN_LOG("INFO", "Setting up tensors using SystemContext API...");

    // Check if SystemContext functions are available
    if (!m_sysInterface->QNN_SYSTEM_INTERFACE_VER_NAME.systemContextCreate || 
        !m_sysInterface->QNN_SYSTEM_INTERFACE_VER_NAME.systemContextGetBinaryInfo || 
        !m_sysInterface->QNN_SYSTEM_INTERFACE_VER_NAME.systemContextFree) {
        QNN_LOG("WARNING", "SystemContext API not available, using manual tensor setup");
        setupTensorsManual();
        return;
    }

    // Create SystemContext handle
    QnnSystemContext_Handle_t sysCtxHandle = nullptr;
    if (m_sysInterface->QNN_SYSTEM_INTERFACE_VER_NAME.systemContextCreate(&sysCtxHandle) != QNN_SUCCESS) {
        QNN_LOG("WARNING", "Failed to create SystemContext, using manual setup");
        setupTensorsManual();
        return;
    }

    // Get binary info
    const QnnSystemContext_BinaryInfo_t* binaryInfo = nullptr;
    Qnn_ContextBinarySize_t binaryInfoSize = 0;
    
    if (m_sysInterface->QNN_SYSTEM_INTERFACE_VER_NAME.systemContextGetBinaryInfo(
            sysCtxHandle,
            m_contextBinaryData.data(),
            m_contextBinaryData.size(),
            &binaryInfo,
            &binaryInfoSize) != QNN_SUCCESS) {
        QNN_LOG("WARNING", "Failed to get binary info, using manual setup");
        m_sysInterface->QNN_SYSTEM_INTERFACE_VER_NAME.systemContextFree(sysCtxHandle);
        setupTensorsManual();
        return;
    }

    // Extract tensor info from binaryInfo
    bool success = false;
    if (binaryInfo->version == QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_1) {
        if (binaryInfo->contextBinaryInfoV1.graphs && binaryInfo->contextBinaryInfoV1.numGraphs > 0) {
            auto& graphInfo = binaryInfo->contextBinaryInfoV1.graphs[0];
            if (graphInfo.version == QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_1) {
                success = setupTensorsFromGraphInfo(graphInfo.graphInfoV1);
            }
        }
    } else if (binaryInfo->version == QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_2) {
        if (binaryInfo->contextBinaryInfoV2.graphs && binaryInfo->contextBinaryInfoV2.numGraphs > 0) {
            auto& graphInfo = binaryInfo->contextBinaryInfoV2.graphs[0];
            if (graphInfo.version == QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_1) {
                success = setupTensorsFromGraphInfo(graphInfo.graphInfoV1);
            }
        }
    } else if (binaryInfo->version == QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_3) {
        if (binaryInfo->contextBinaryInfoV3.graphs && binaryInfo->contextBinaryInfoV3.numGraphs > 0) {
            auto& graphInfo = binaryInfo->contextBinaryInfoV3.graphs[0];
            if (graphInfo.version == QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_3) {
                success = setupTensorsFromGraphInfoV3(graphInfo.graphInfoV3);
            } else if (graphInfo.version == QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_1) {
                success = setupTensorsFromGraphInfo(graphInfo.graphInfoV1);
            }
        }
    }

    m_sysInterface->QNN_SYSTEM_INTERFACE_VER_NAME.systemContextFree(sysCtxHandle);

    if (!success) {
        QNN_LOG("WARNING", "Failed to extract tensor info from binary, using manual setup");
        setupTensorsManual();
    } else {
        QNN_LOG("INFO", "Tensors configured from binary metadata.");
    }
}

void QnnRunner::updateGraphName(const char* graphName) {
    if (graphName && graphName[0] != '\0') {
        std::string newName(graphName);
        if (newName != m_graphName) {
            QNN_LOG("INFO", "Detected graph in context: " + newName);
        }
        m_graphName = std::move(newName);
    }
}

bool QnnRunner::setupTensorsFromGraphInfo(const QnnSystemContext_GraphInfoV1_t& graphInfo) {
    if (!graphInfo.graphInputs || graphInfo.numGraphInputs == 0 ||
        !graphInfo.graphOutputs || graphInfo.numGraphOutputs == 0) {
        return false;
    }

    updateGraphName(graphInfo.graphName);

    // Use first input and output tensors
    const Qnn_Tensor_t& srcInput = graphInfo.graphInputs[0];
    const Qnn_Tensor_t& srcOutput = graphInfo.graphOutputs[0];

    // Deep copy input tensor info
    m_inputTensor = QNN_TENSOR_INIT;
    if (!deepCopyTensorInfo(&m_inputTensor, &srcInput)) {
        return false;
    }

    // Deep copy output tensor info
    m_outputTensor = QNN_TENSOR_INIT;
    if (!deepCopyTensorInfo(&m_outputTensor, &srcOutput)) {
        return false;
    }

    // Calculate buffer sizes and allocate
    m_inputBufferSize = calculateTensorSize(&m_inputTensor);
    m_outputBufferSize = calculateTensorSize(&m_outputTensor);

    m_inputBuffer = malloc(m_inputBufferSize);
    m_outputBuffer = malloc(m_outputBufferSize);
    
    if (!m_inputBuffer || !m_outputBuffer) {
        return false;
    }

    // Update tensors to use our buffers
    m_inputTensor.v1.clientBuf.data = m_inputBuffer;
    m_inputTensor.v1.clientBuf.dataSize = m_inputBufferSize;
    m_outputTensor.v1.clientBuf.data = m_outputBuffer;
    m_outputTensor.v1.clientBuf.dataSize = m_outputBufferSize;

    // Populate m_inputDims and m_outputDims for getInputDims/getOutputDims accessors
    if (m_inputTensor.version == QNN_TENSOR_VERSION_1 && m_inputTensor.v1.dimensions) {
        m_inputDims.assign(m_inputTensor.v1.dimensions, 
                          m_inputTensor.v1.dimensions + m_inputTensor.v1.rank);
    }
    if (m_outputTensor.version == QNN_TENSOR_VERSION_1 && m_outputTensor.v1.dimensions) {
        m_outputDims.assign(m_outputTensor.v1.dimensions,
                           m_outputTensor.v1.dimensions + m_outputTensor.v1.rank);
    }

    QNN_LOG("INFO", "Input: " + std::to_string(m_inputBufferSize) + " bytes, Output: " + 
            std::to_string(m_outputBufferSize) + " bytes");

    return true;
}

bool QnnRunner::setupTensorsFromGraphInfoV3(const QnnSystemContext_GraphInfoV3_t& graphInfo) {
    // V3 has same structure for our purposes
    if (!graphInfo.graphInputs || graphInfo.numGraphInputs == 0 ||
        !graphInfo.graphOutputs || graphInfo.numGraphOutputs == 0) {
        return false;
    }

    updateGraphName(graphInfo.graphName);

    const Qnn_Tensor_t& srcInput = graphInfo.graphInputs[0];
    const Qnn_Tensor_t& srcOutput = graphInfo.graphOutputs[0];

    m_inputTensor = QNN_TENSOR_INIT;
    if (!deepCopyTensorInfo(&m_inputTensor, &srcInput)) {
        return false;
    }

    m_outputTensor = QNN_TENSOR_INIT;
    if (!deepCopyTensorInfo(&m_outputTensor, &srcOutput)) {
        return false;
    }

    m_inputBufferSize = calculateTensorSize(&m_inputTensor);
    m_outputBufferSize = calculateTensorSize(&m_outputTensor);

    m_inputBuffer = malloc(m_inputBufferSize);
    m_outputBuffer = malloc(m_outputBufferSize);
    
    if (!m_inputBuffer || !m_outputBuffer) {
        return false;
    }

    m_inputTensor.v1.clientBuf.data = m_inputBuffer;
    m_inputTensor.v1.clientBuf.dataSize = m_inputBufferSize;
    m_outputTensor.v1.clientBuf.data = m_outputBuffer;
    m_outputTensor.v1.clientBuf.dataSize = m_outputBufferSize;

    // Populate m_inputDims and m_outputDims for getInputDims/getOutputDims accessors
    if (m_inputTensor.version == QNN_TENSOR_VERSION_1 && m_inputTensor.v1.dimensions) {
        m_inputDims.assign(m_inputTensor.v1.dimensions, 
                          m_inputTensor.v1.dimensions + m_inputTensor.v1.rank);
    }
    if (m_outputTensor.version == QNN_TENSOR_VERSION_1 && m_outputTensor.v1.dimensions) {
        m_outputDims.assign(m_outputTensor.v1.dimensions,
                           m_outputTensor.v1.dimensions + m_outputTensor.v1.rank);
    }

    QNN_LOG("INFO", "Input: " + std::to_string(m_inputBufferSize) + " bytes, Output: " + 
            std::to_string(m_outputBufferSize) + " bytes");

    return true;
}

bool QnnRunner::deepCopyTensorInfo(Qnn_Tensor_t* dst, const Qnn_Tensor_t* src) {
    if (!dst || !src) return false;

    dst->version = src->version;
    
    // Copy name
    const char* name = (src->version == QNN_TENSOR_VERSION_1) ? src->v1.name : 
                       (src->version == QNN_TENSOR_VERSION_2) ? src->v2.name : nullptr;
    if (name) {
        char* nameCopy = (char*)malloc(strlen(name) + 1);
        if (nameCopy) {
            strcpy(nameCopy, name);
            if (dst->version == QNN_TENSOR_VERSION_1) dst->v1.name = nameCopy;
            else if (dst->version == QNN_TENSOR_VERSION_2) dst->v2.name = nameCopy;
        }
    }

    // Copy fields based on version
    if (dst->version == QNN_TENSOR_VERSION_1) {
        dst->v1.id = src->v1.id;
        dst->v1.type = src->v1.type;
        dst->v1.dataFormat = src->v1.dataFormat;
        dst->v1.dataType = src->v1.dataType;
        dst->v1.quantizeParams = src->v1.quantizeParams;
        dst->v1.rank = src->v1.rank;
        
        // Copy dimensions
        if (src->v1.rank > 0 && src->v1.dimensions) {
            dst->v1.dimensions = (uint32_t*)malloc(src->v1.rank * sizeof(uint32_t));
            if (dst->v1.dimensions) {
                memcpy(dst->v1.dimensions, src->v1.dimensions, src->v1.rank * sizeof(uint32_t));
            }
        }
        
        dst->v1.memType = QNN_TENSORMEMTYPE_RAW;
        dst->v1.clientBuf = QNN_CLIENT_BUFFER_INIT;
    } else if (dst->version == QNN_TENSOR_VERSION_2) {
        dst->v2.id = src->v2.id;
        dst->v2.type = src->v2.type;
        dst->v2.dataFormat = src->v2.dataFormat;
        dst->v2.dataType = src->v2.dataType;
        dst->v2.quantizeParams = src->v2.quantizeParams;
        dst->v2.rank = src->v2.rank;
        
        if (src->v2.rank > 0 && src->v2.dimensions) {
            dst->v2.dimensions = (uint32_t*)malloc(src->v2.rank * sizeof(uint32_t));
            if (dst->v2.dimensions) {
                memcpy(dst->v2.dimensions, src->v2.dimensions, src->v2.rank * sizeof(uint32_t));
            }
        }
        
        dst->v2.memType = QNN_TENSORMEMTYPE_RAW;
        dst->v2.clientBuf = QNN_CLIENT_BUFFER_INIT;
    }

    return true;
}

size_t QnnRunner::calculateTensorSize(const Qnn_Tensor_t* tensor) {
    if (!tensor) return 0;

    size_t size = 1;
    uint32_t rank = 0;
    uint32_t* dimensions = nullptr;

    if (tensor->version == QNN_TENSOR_VERSION_1) {
        rank = tensor->v1.rank;
        dimensions = tensor->v1.dimensions;
    } else if (tensor->version == QNN_TENSOR_VERSION_2) {
        rank = tensor->v2.rank;
        dimensions = tensor->v2.dimensions;
    }

    if (rank > 0 && dimensions) {
        for (uint32_t i = 0; i < rank; i++) {
            size *= dimensions[i];
        }
    }

    return size;
}

void QnnRunner::setupTensorsManual() {
    QNN_LOG("INFO", "Using manual tensor setup (fallback)...");

    // For quantized INT8 model: input [1,128], output [1,10000]
    m_inputDims = {1, 128};
    m_outputDims = {1, 10000};
    
    m_inputBufferSize = 128;  // 1 byte per element for UFIXED_POINT_8
    m_outputBufferSize = 10000;
    
    m_inputBuffer = malloc(m_inputBufferSize);
    m_outputBuffer = malloc(m_outputBufferSize);
    if (!m_inputBuffer || !m_outputBuffer)
        throw std::runtime_error("Failed to allocate tensor buffers");

    // Setup input tensor with proper quantization params
    m_inputTensor = QNN_TENSOR_INIT;
    m_inputTensor.version = QNN_TENSOR_VERSION_1;
    m_inputTensor.v1.id = 0;
    m_inputTensor.v1.name = nullptr;  // Let QNN use internal name from context
    m_inputTensor.v1.type = QNN_TENSOR_TYPE_APP_WRITE;
    m_inputTensor.v1.dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER;
    m_inputTensor.v1.dataType = QNN_DATATYPE_UFIXED_POINT_8;
    m_inputTensor.v1.quantizeParams.quantizationEncoding = QNN_QUANTIZATION_ENCODING_SCALE_OFFSET;
    m_inputTensor.v1.quantizeParams.scaleOffsetEncoding.scale = 0.6627451181411743f;
    m_inputTensor.v1.quantizeParams.scaleOffsetEncoding.offset = 0;
    m_inputTensor.v1.rank = 2;
    m_inputTensor.v1.dimensions = m_inputDims.data();
    m_inputTensor.v1.memType = QNN_TENSORMEMTYPE_RAW;
    m_inputTensor.v1.clientBuf.data = m_inputBuffer;
    m_inputTensor.v1.clientBuf.dataSize = m_inputBufferSize;

    // Setup output tensor with proper quantization params
    m_outputTensor = QNN_TENSOR_INIT;
    m_outputTensor.version = QNN_TENSOR_VERSION_1;
    m_outputTensor.v1.id = 0;
    m_outputTensor.v1.name = nullptr;  // Let QNN use internal name from context
    m_outputTensor.v1.type = QNN_TENSOR_TYPE_APP_READ;
    m_outputTensor.v1.dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER;
    m_outputTensor.v1.dataType = QNN_DATATYPE_UFIXED_POINT_8;
    m_outputTensor.v1.quantizeParams.quantizationEncoding = QNN_QUANTIZATION_ENCODING_SCALE_OFFSET;
    m_outputTensor.v1.quantizeParams.scaleOffsetEncoding.scale = 1013.4312133789062500f;
    m_outputTensor.v1.quantizeParams.scaleOffsetEncoding.offset = 0;
    m_outputTensor.v1.rank = 2;
    m_outputTensor.v1.dimensions = m_outputDims.data();
    m_outputTensor.v1.memType = QNN_TENSORMEMTYPE_RAW;
    m_outputTensor.v1.clientBuf.data = m_outputBuffer;
    m_outputTensor.v1.clientBuf.dataSize = m_outputBufferSize;

    QNN_LOG("INFO", "Input: [1, 128] UFIXED_POINT_8 (128 bytes)");
    QNN_LOG("INFO", "Output: [1, 10000] UFIXED_POINT_8 (10000 bytes)");
    QNN_LOG("INFO", "Manual tensors configured.");
}

void QnnRunner::execute(const std::vector<float>& query, std::vector<float>& scores) {
    // Get quantization parameters and expected size from tensor
    float input_scale = 0.6627451f;
    float output_scale = 1013.4312f;
    size_t expected_input_size = 128;
    size_t expected_output_size = 10000;
    
    if (m_inputTensor.version == QNN_TENSOR_VERSION_1) {
        if (m_inputTensor.v1.quantizeParams.quantizationEncoding == QNN_QUANTIZATION_ENCODING_SCALE_OFFSET) {
            input_scale = m_inputTensor.v1.quantizeParams.scaleOffsetEncoding.scale;
        }
        expected_input_size = calculateTensorSize(&m_inputTensor);
    }
    
    if (m_outputTensor.version == QNN_TENSOR_VERSION_1) {
        if (m_outputTensor.v1.quantizeParams.quantizationEncoding == QNN_QUANTIZATION_ENCODING_SCALE_OFFSET) {
            output_scale = m_outputTensor.v1.quantizeParams.scaleOffsetEncoding.scale;
        }
        expected_output_size = calculateTensorSize(&m_outputTensor);
    }
    
    if (query.size() != expected_input_size) {
        throw std::runtime_error("Query size mismatch: expected " + std::to_string(expected_input_size) + 
                                 ", got " + std::to_string(query.size()));
    }

    // Quantize: UFIXED_POINT_8
    uint8_t* input_bytes = static_cast<uint8_t*>(m_inputBuffer);
    for (size_t i = 0; i < query.size(); i++) {
        int32_t quantized = static_cast<int32_t>(std::round(query[i] / input_scale));
        quantized = std::max(0, std::min(255, quantized));
        input_bytes[i] = static_cast<uint8_t>(quantized);
    }

    // Execute
    if (m_qnnInterface->QNN_INTERFACE_VER_NAME.graphExecute(
            m_graphHandle, &m_inputTensor, 1, &m_outputTensor, 1, nullptr, nullptr) != QNN_SUCCESS)
        throw std::runtime_error("QnnGraph_execute failed");

    // Dequantize
    uint8_t* output_bytes = static_cast<uint8_t*>(m_outputBuffer);
    scores.resize(expected_output_size);
    for (size_t i = 0; i < expected_output_size; i++) {
        scores[i] = static_cast<float>(output_bytes[i]) * output_scale;
    }
}

void QnnRunner::executeBatch(const std::vector<float>& batchData, size_t batchSize, std::vector<float>& scores) {
    if (batchSize == 0) return;

    // Currently only handling V1 tensors (as in existing code)
    if (m_inputTensor.version != QNN_TENSOR_VERSION_1 || m_outputTensor.version != QNN_TENSOR_VERSION_1) {
        throw std::runtime_error("executeBatch: only tensor version 1 supported in this implementation");
    }

    // Determine per-sample input size (elements) assuming batch is first dimension
    uint32_t in_rank = m_inputTensor.v1.rank;
    uint32_t* in_dims = m_inputTensor.v1.dimensions;
    size_t per_sample_input = 0;
    uint32_t original_in_batch = 0;
    if (in_rank > 0 && in_dims) {
        original_in_batch = in_dims[0];
        // product of remaining dims (if rank == 1, treat per_sample as 1)
        per_sample_input = 1;
        for (uint32_t i = 1; i < in_rank; ++i) per_sample_input *= in_dims[i];
        // If rank == 1, then per_sample_input should be the single dim
        if (in_rank == 1) per_sample_input = in_dims[0];
    } else {
        per_sample_input = calculateTensorSize(&m_inputTensor);
    }

    if (batchData.size() != batchSize * per_sample_input) {
        throw std::runtime_error("executeBatch: batchData size mismatch with batchSize * per-sample-input");
    }

    // Ensure input buffer can hold whole batch
    size_t required_input_size = batchData.size();
    if (m_inputBufferSize < required_input_size) {
        if (m_inputBuffer) free(m_inputBuffer);
        m_inputBuffer = malloc(required_input_size);
        if (!m_inputBuffer) throw std::runtime_error("executeBatch: failed to allocate input buffer");
        m_inputBufferSize = required_input_size;
        m_inputTensor.v1.clientBuf.data = m_inputBuffer;
        m_inputTensor.v1.clientBuf.dataSize = m_inputBufferSize;
        // Update m_inputDims if present
        if (!m_inputDims.empty()) m_inputDims[0] = static_cast<uint32_t>(batchSize);
        if (in_dims) in_dims[0] = static_cast<uint32_t>(batchSize);
        m_inputTensor.v1.dimensions = in_dims;
    } else {
        // Update dims even if buffer already large enough
        if (!m_inputDims.empty()) m_inputDims[0] = static_cast<uint32_t>(batchSize);
        if (in_dims) in_dims[0] = static_cast<uint32_t>(batchSize);
    }

    // Quantize entire batch (UFIXED_POINT_8 assumed as in execute)
    float input_scale = 0.6627451f;
    if (m_inputTensor.v1.quantizeParams.quantizationEncoding == QNN_QUANTIZATION_ENCODING_SCALE_OFFSET) {
        input_scale = m_inputTensor.v1.quantizeParams.scaleOffsetEncoding.scale;
    }

    uint8_t* input_bytes = static_cast<uint8_t*>(m_inputBuffer);
    for (size_t i = 0; i < required_input_size; ++i) {
        int32_t quantized = static_cast<int32_t>(std::round(batchData[i] / input_scale));
        quantized = std::max(0, std::min(255, quantized));
        input_bytes[i] = static_cast<uint8_t>(quantized);
    }

    // Prepare output buffer for batch
    uint32_t out_rank = m_outputTensor.v1.rank;
    uint32_t* out_dims = m_outputTensor.v1.dimensions;
    size_t per_sample_output = 0;
    uint32_t original_out_batch = 0;
    if (out_rank > 0 && out_dims) {
        original_out_batch = out_dims[0];
        per_sample_output = 1;
        for (uint32_t i = 1; i < out_rank; ++i) per_sample_output *= out_dims[i];
        if (out_rank == 1) per_sample_output = out_dims[0];
    } else {
        per_sample_output = calculateTensorSize(&m_outputTensor);
    }

    size_t required_output_size = batchSize * per_sample_output;
    if (m_outputBufferSize < required_output_size) {
        if (m_outputBuffer) free(m_outputBuffer);
        m_outputBuffer = malloc(required_output_size);
        if (!m_outputBuffer) throw std::runtime_error("executeBatch: failed to allocate output buffer");
        m_outputBufferSize = required_output_size;
        m_outputTensor.v1.clientBuf.data = m_outputBuffer;
        m_outputTensor.v1.clientBuf.dataSize = m_outputBufferSize;
        if (!m_outputDims.empty()) m_outputDims[0] = static_cast<uint32_t>(batchSize);
        if (out_dims) out_dims[0] = static_cast<uint32_t>(batchSize);
        m_outputTensor.v1.dimensions = out_dims;
    } else {
        if (!m_outputDims.empty()) m_outputDims[0] = static_cast<uint32_t>(batchSize);
        if (out_dims) out_dims[0] = static_cast<uint32_t>(batchSize);
    }

    // Execute
    if (m_qnnInterface->QNN_INTERFACE_VER_NAME.graphExecute(
            m_graphHandle, &m_inputTensor, 1, &m_outputTensor, 1, nullptr, nullptr) != QNN_SUCCESS)
        throw std::runtime_error("QnnGraph_execute failed (batch)");

    // Dequantize output
    float output_scale = 1013.4312f;
    if (m_outputTensor.v1.quantizeParams.quantizationEncoding == QNN_QUANTIZATION_ENCODING_SCALE_OFFSET) {
        output_scale = m_outputTensor.v1.quantizeParams.scaleOffsetEncoding.scale;
    }

    uint8_t* output_bytes = static_cast<uint8_t*>(m_outputBuffer);
    scores.resize(required_output_size);
    for (size_t i = 0; i < required_output_size; ++i) {
        scores[i] = static_cast<float>(output_bytes[i]) * output_scale;
    }

    // Restore original batch dimensions so single-sample execute remains consistent
    if (in_dims && original_in_batch != 0) {
        in_dims[0] = original_in_batch;
    }
    if (!m_inputDims.empty() && original_in_batch != 0) {
        m_inputDims[0] = original_in_batch;
    }
    if (out_dims && original_out_batch != 0) {
        out_dims[0] = original_out_batch;
    }
    if (!m_outputDims.empty() && original_out_batch != 0) {
        m_outputDims[0] = original_out_batch;
    }
}

void QnnRunner::cleanup() {
    if (m_inputBuffer) free(m_inputBuffer);
    if (m_outputBuffer) free(m_outputBuffer);

    if (m_contextHandle && m_qnnInterface)
        m_qnnInterface->QNN_INTERFACE_VER_NAME.contextFree(m_contextHandle, nullptr);

    if (m_deviceHandle && m_qnnInterface)
        m_qnnInterface->QNN_INTERFACE_VER_NAME.deviceFree(m_deviceHandle);

    if (m_backendHandle && m_qnnInterface)
        m_qnnInterface->QNN_INTERFACE_VER_NAME.backendFree(m_backendHandle);

    if (m_backendLibHandle) dlclose(m_backendLibHandle);
    if (m_systemLibHandle) dlclose(m_systemLibHandle);

    m_inputBuffer = nullptr;
    m_outputBuffer = nullptr;
    m_contextHandle = nullptr;
    m_deviceHandle = nullptr;
    m_backendHandle = nullptr;
    m_backendLibHandle = nullptr;
    m_systemLibHandle = nullptr;
    m_qnnInterface = nullptr;
}