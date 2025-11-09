#ifndef QNN_RUNNER_H
#define QNN_RUNNER_H

#include <string>
#include <vector>
#include <memory>
#include <stdexcept>

// QNN API Headers (correct headers from SDK)
#include "QnnInterface.h"
#include "QnnTypes.h"
#include "QnnCommon.h"

// Simple stub implementation for now
class QnnRunner {
public:
    QnnRunner(const std::string& contextBinaryPath, const std::string& backendPath);
    ~QnnRunner();

    // Execute the model (stub for now)
    void execute(const std::vector<float>& query, std::vector<float>& scores);

    // Get input/output shapes (hardcoded for now)
    const std::vector<uint32_t>& getInputDims() const { return m_inputDims; }
    const std::vector<uint32_t>& getOutputDims() const { return m_outputDims; }

private:
    std::vector<uint32_t> m_inputDims;
    std::vector<uint32_t> m_outputDims;
    std::string m_modelPath;
};

#endif // QNN_RUNNER_H