#include "QnnRunner.h"
#include <iostream>
#include <cstring>
#include <cstdlib>
#include <ctime>

QnnRunner::QnnRunner(const std::string& contextBinaryPath, const std::string& backendPath) 
    : m_modelPath(contextBinaryPath) {
    std::cout << "=== QNN Runner Stub ===" << std::endl;
    std::cout << "Model path: " << contextBinaryPath << std::endl;
    std::cout << "Backend path: " << backendPath << std::endl;
    
    // Hardcode dimensions for vector_search_10k model
    // Input: query [1, 128]
    // Output: scores [1, 10000]
    m_inputDims = {1, 128};
    m_outputDims = {1, 10000};
    
    std::cout << "Input dims: [1, 128]" << std::endl;
    std::cout << "Output dims: [1, 10000]" << std::endl;
    std::cout << "QNN Runner initialized (STUB MODE - NOT RUNNING ON NPU YET)" << std::endl;
    std::cout << "===================" << std::endl;
    
    // Initialize random seed for stub execution
    srand(time(NULL));
}

QnnRunner::~QnnRunner() {
    std::cout << "QNN Runner destroyed" << std::endl;
}

void QnnRunner::execute(const std::vector<float>& query, std::vector<float>& scores) {
    // Stub implementation: just generate random scores for now
    // This allows the build to complete and the program to run
    
    if (query.size() != 128) {
        throw std::runtime_error("Query size must be 128");
    }
    
    if (scores.size() != 10000) {
        throw std::runtime_error("Scores size must be 10000");
    }
    
    // Generate random scores between 0 and 1
    for (size_t i = 0; i < scores.size(); ++i) {
        scores[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    }
    
    // Make a few scores higher so top-k returns something interesting
    scores[42] = 0.95f;
    scores[100] = 0.92f;
    scores[500] = 0.88f;
    scores[1000] = 0.85f;
    scores[5000] = 0.82f;
}