#ifndef NN_SERIALIZATION_HPP
#define NN_SERIALIZATION_HPP

/**
 * @file serialization.hpp
 * @brief Model serialization (save/load) utilities
 */

#include "nn/tensor.hpp"
#include <fstream>
#include <string>
#include <map>
#include <stdexcept>

namespace nn {

/**
 * @brief Collection of named tensors for serialization
 */
class StateDict {
public:
    using TensorMap = std::map<std::string, Tensor>;
    
    void add(const std::string& name, const Tensor& tensor) {
        tensors_[name] = tensor;
    }
    
    const Tensor& get(const std::string& name) const {
        auto it = tensors_.find(name);
        if (it == tensors_.end()) {
            throw std::runtime_error("Tensor not found: " + name);
        }
        return it->second;
    }
    
    bool contains(const std::string& name) const {
        return tensors_.find(name) != tensors_.end();
    }
    
    std::vector<std::string> keys() const {
        std::vector<std::string> names;
        for (const auto& [name, _] : tensors_) {
            names.push_back(name);
        }
        return names;
    }
    
    TensorMap& tensors() { return tensors_; }
    const TensorMap& tensors() const { return tensors_; }
    size_t size() const { return tensors_.size(); }

private:
    TensorMap tensors_;
};

inline void save_state_dict(const StateDict& state_dict, const std::string& filepath) {
    std::ofstream file(filepath, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Cannot open file for writing: " + filepath);
    }
    
    const char magic[] = "NNET";
    file.write(magic, 4);
    
    uint32_t version = 1;
    file.write(reinterpret_cast<const char*>(&version), sizeof(version));
    
    uint32_t num_tensors = static_cast<uint32_t>(state_dict.size());
    file.write(reinterpret_cast<const char*>(&num_tensors), sizeof(num_tensors));
    
    for (const auto& [name, tensor] : state_dict.tensors()) {
        uint32_t name_len = static_cast<uint32_t>(name.length());
        file.write(reinterpret_cast<const char*>(&name_len), sizeof(name_len));
        file.write(name.data(), name_len);
        
        uint32_t ndim = static_cast<uint32_t>(tensor.ndim());
        file.write(reinterpret_cast<const char*>(&ndim), sizeof(ndim));
        
        for (size_t i = 0; i < tensor.ndim(); ++i) {
            uint32_t dim = static_cast<uint32_t>(tensor.dim(i));
            file.write(reinterpret_cast<const char*>(&dim), sizeof(dim));
        }
        
        file.write(reinterpret_cast<const char*>(tensor.data()), 
                   tensor.size() * sizeof(float));
    }
}

inline StateDict load_state_dict(const std::string& filepath) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Cannot open file for reading: " + filepath);
    }
    
    char magic[5] = {0};
    file.read(magic, 4);
    if (std::string(magic) != "NNET") {
        throw std::runtime_error("Invalid file format");
    }
    
    uint32_t version;
    file.read(reinterpret_cast<char*>(&version), sizeof(version));
    
    uint32_t num_tensors;
    file.read(reinterpret_cast<char*>(&num_tensors), sizeof(num_tensors));
    
    StateDict state_dict;
    
    for (uint32_t t = 0; t < num_tensors; ++t) {
        uint32_t name_len;
        file.read(reinterpret_cast<char*>(&name_len), sizeof(name_len));
        std::string name(name_len, '\0');
        file.read(name.data(), name_len);
        
        uint32_t ndim;
        file.read(reinterpret_cast<char*>(&ndim), sizeof(ndim));
        
        Tensor::Shape shape(ndim);
        for (uint32_t i = 0; i < ndim; ++i) {
            uint32_t dim;
            file.read(reinterpret_cast<char*>(&dim), sizeof(dim));
            shape[i] = dim;
        }
        
        Tensor tensor(shape);
        file.read(reinterpret_cast<char*>(tensor.data()), 
                  tensor.size() * sizeof(float));
        
        state_dict.add(name, tensor);
    }
    
    return state_dict;
}

} // namespace nn

#endif // NN_SERIALIZATION_HPP
