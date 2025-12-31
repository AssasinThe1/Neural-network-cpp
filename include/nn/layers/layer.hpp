#ifndef NN_LAYER_HPP
#define NN_LAYER_HPP

/**
 * @file layer.hpp
 * @brief Abstract base class for all neural network layers
 * 
 * This file defines the Layer interface that all layer types must implement.
 * Using an abstract base class allows us to treat different layer types
 * uniformly when building networks.
 * 
 * Key C++ concepts demonstrated:
 * - Abstract classes (classes with pure virtual functions)
 * - Virtual functions and polymorphism
 * - Smart pointers for memory management
 */

#include "nn/tensor.hpp"
#include <memory>
#include <string>
#include <vector>

namespace nn {

/**
 * @brief Abstract base class for neural network layers
 * 
 * All layer types (Dense, Conv2D, activations, etc.) inherit from this class
 * and implement its pure virtual functions.
 * 
 * Design Pattern: This uses the Template Method pattern where the base class
 * defines the interface and derived classes provide the implementation.
 * 
 * The forward pass computes: output = f(input, weights, biases)
 * The backward pass computes gradients for backpropagation
 */
class Layer {
public:
    /**
     * @brief Virtual destructor
     * 
     * Important C++ concept: When using polymorphism (virtual functions),
     * always declare the destructor as virtual. This ensures that when
     * deleting through a base class pointer, the derived class destructor
     * is called first.
     */
    virtual ~Layer() = default;
    
    // ========================================================================
    // Core Operations (Pure Virtual - Must Be Implemented by Derived Classes)
    // ========================================================================
    
    /**
     * @brief Perform forward pass
     * 
     * @param input Input tensor from previous layer
     * @return Output tensor to pass to next layer
     * 
     * Pure virtual function (= 0) means derived classes MUST implement this.
     */
    virtual Tensor forward(const Tensor& input) = 0;
    
    /**
     * @brief Perform backward pass (for training)
     * 
     * @param grad_output Gradient from the next layer (∂L/∂output)
     * @return Gradient to pass to previous layer (∂L/∂input)
     * 
     * Uses the chain rule: ∂L/∂input = (∂output/∂input)ᵀ × ∂L/∂output
     */
    virtual Tensor backward(const Tensor& grad_output) = 0;
    
    // ========================================================================
    // Parameter Access
    // ========================================================================
    
    /**
     * @brief Get all trainable parameters (weights, biases)
     * @return Vector of pointers to parameter tensors
     */
    virtual std::vector<Tensor*> parameters() { return {}; }
    
    /**
     * @brief Get gradients for all trainable parameters
     * @return Vector of pointers to gradient tensors
     */
    virtual std::vector<Tensor*> gradients() { return {}; }
    
    /**
     * @brief Get the number of trainable parameters
     */
    virtual size_t num_parameters() const { return 0; }
    
    // ========================================================================
    // Layer Information
    // ========================================================================
    
    /**
     * @brief Get the layer type name (for debugging/serialization)
     */
    virtual std::string name() const = 0;
    
    /**
     * @brief Get output shape given input shape
     * 
     * This is useful for building networks and validating layer compatibility.
     */
    virtual Tensor::Shape output_shape(const Tensor::Shape& input_shape) const = 0;
    
    /**
     * @brief Get a string description of the layer
     */
    virtual std::string summary() const {
        return name() + " (params: " + std::to_string(num_parameters()) + ")";
    }
    
    // ========================================================================
    // Training Mode
    // ========================================================================
    
    /**
     * @brief Set training mode
     * 
     * Some layers behave differently during training vs inference
     * (e.g., Dropout, BatchNorm)
     */
    virtual void set_training(bool training) { training_ = training; }
    
    /**
     * @brief Check if in training mode
     */
    bool is_training() const { return training_; }
    
    // ========================================================================
    // Initialization
    // ========================================================================
    
    /**
     * @brief Reset/reinitialize parameters
     */
    virtual void reset_parameters() {}

protected:
    /**
     * @brief Cache for storing values needed during backward pass
     * 
     * During forward pass, we often need to save intermediate values
     * (like the input) that are used in the backward pass.
     */
    Tensor input_cache_;
    
    /**
     * @brief Training mode flag
     */
    bool training_ = true;
};

/**
 * @brief Unique pointer type for layers (for memory management)
 * 
 * std::unique_ptr is a smart pointer that:
 * - Automatically deletes the object when it goes out of scope
 * - Ensures only one owner of the object at a time
 * - Can be moved but not copied
 */
using LayerPtr = std::unique_ptr<Layer>;

} // namespace nn

#endif // NN_LAYER_HPP
