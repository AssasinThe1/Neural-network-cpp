#ifndef NN_ACTIVATIONS_HPP
#define NN_ACTIVATIONS_HPP

/**
 * @file activations.hpp
 * @brief Activation function layers
 * 
 * Activation functions introduce non-linearity into neural networks.
 * Without them, a neural network would just be a linear function,
 * no matter how many layers it has!
 * 
 * Common activation functions:
 * - ReLU: max(0, x) - most popular for hidden layers
 * - Sigmoid: 1/(1+e^(-x)) - good for binary classification output
 * - Softmax: e^x_i / Σe^x_j - good for multi-class classification output
 * - Tanh: (e^x - e^(-x))/(e^x + e^(-x)) - similar to sigmoid but centered at 0
 */

#include "nn/layers/layer.hpp"
#include <cmath>
#include <algorithm>

namespace nn {

// ============================================================================
// ReLU (Rectified Linear Unit)
// ============================================================================

/**
 * @brief ReLU activation: f(x) = max(0, x)
 * 
 * Properties:
 * - Simple and fast to compute
 * - Helps avoid vanishing gradient problem
 * - Derivative: 1 if x > 0, else 0
 * 
 * Potential issue: "Dead ReLU" - neurons can become permanently inactive
 * if they output negative values during training.
 */
class ReLU : public Layer {
public:
    Tensor forward(const Tensor& input) override {
        // Cache input for backward pass
        input_cache_ = input;
        
        // Apply ReLU: max(0, x)
        Tensor output(input.shape());
        for (size_t i = 0; i < input.size(); ++i) {
            output[i] = std::max(0.0f, input[i]);
        }
        return output;
    }
    
    Tensor backward(const Tensor& grad_output) override {
        // Derivative of ReLU:
        // d/dx max(0, x) = 1 if x > 0, else 0
        
        Tensor grad_input(input_cache_.shape());
        for (size_t i = 0; i < input_cache_.size(); ++i) {
            // Gradient flows through only where input was positive
            grad_input[i] = input_cache_[i] > 0 ? grad_output[i] : 0.0f;
        }
        return grad_input;
    }
    
    std::string name() const override { return "ReLU"; }
    
    Tensor::Shape output_shape(const Tensor::Shape& input_shape) const override {
        return input_shape; // Shape doesn't change
    }
};

// ============================================================================
// Leaky ReLU
// ============================================================================

/**
 * @brief Leaky ReLU: f(x) = x if x > 0, else alpha*x
 * 
 * Addresses the "dying ReLU" problem by allowing small gradients
 * when the input is negative.
 */
class LeakyReLU : public Layer {
public:
    /**
     * @param alpha Slope for negative inputs (default: 0.01)
     */
    explicit LeakyReLU(float alpha = 0.01f) : alpha_(alpha) {}
    
    Tensor forward(const Tensor& input) override {
        input_cache_ = input;
        
        Tensor output(input.shape());
        for (size_t i = 0; i < input.size(); ++i) {
            output[i] = input[i] > 0 ? input[i] : alpha_ * input[i];
        }
        return output;
    }
    
    Tensor backward(const Tensor& grad_output) override {
        Tensor grad_input(input_cache_.shape());
        for (size_t i = 0; i < input_cache_.size(); ++i) {
            grad_input[i] = input_cache_[i] > 0 ? grad_output[i] : alpha_ * grad_output[i];
        }
        return grad_input;
    }
    
    std::string name() const override { return "LeakyReLU"; }
    
    Tensor::Shape output_shape(const Tensor::Shape& input_shape) const override {
        return input_shape;
    }

private:
    float alpha_;
};

// ============================================================================
// Sigmoid
// ============================================================================

/**
 * @brief Sigmoid activation: f(x) = 1 / (1 + e^(-x))
 * 
 * Properties:
 * - Output range: (0, 1)
 * - Useful for binary classification
 * - Derivative: f(x) * (1 - f(x))
 * 
 * Issues:
 * - Vanishing gradients for very large/small inputs
 * - Not zero-centered
 */
class Sigmoid : public Layer {
public:
    Tensor forward(const Tensor& input) override {
        // Compute sigmoid and cache for backward
        output_cache_ = Tensor(input.shape());
        
        for (size_t i = 0; i < input.size(); ++i) {
            // Numerically stable sigmoid:
            // For x >= 0: 1 / (1 + exp(-x))
            // For x < 0: exp(x) / (1 + exp(x))
            float x = input[i];
            if (x >= 0) {
                output_cache_[i] = 1.0f / (1.0f + std::exp(-x));
            } else {
                float exp_x = std::exp(x);
                output_cache_[i] = exp_x / (1.0f + exp_x);
            }
        }
        
        return output_cache_;
    }
    
    Tensor backward(const Tensor& grad_output) override {
        // d/dx sigmoid(x) = sigmoid(x) * (1 - sigmoid(x))
        Tensor grad_input(output_cache_.shape());
        
        for (size_t i = 0; i < output_cache_.size(); ++i) {
            float s = output_cache_[i];
            grad_input[i] = grad_output[i] * s * (1.0f - s);
        }
        
        return grad_input;
    }
    
    std::string name() const override { return "Sigmoid"; }
    
    Tensor::Shape output_shape(const Tensor::Shape& input_shape) const override {
        return input_shape;
    }

private:
    Tensor output_cache_;
};

// ============================================================================
// Tanh
// ============================================================================

/**
 * @brief Tanh activation: f(x) = (e^x - e^(-x)) / (e^x + e^(-x))
 * 
 * Properties:
 * - Output range: (-1, 1)
 * - Zero-centered (unlike sigmoid)
 * - Derivative: 1 - tanh²(x)
 */
class Tanh : public Layer {
public:
    Tensor forward(const Tensor& input) override {
        output_cache_ = Tensor(input.shape());
        
        for (size_t i = 0; i < input.size(); ++i) {
            output_cache_[i] = std::tanh(input[i]);
        }
        
        return output_cache_;
    }
    
    Tensor backward(const Tensor& grad_output) override {
        // d/dx tanh(x) = 1 - tanh²(x)
        Tensor grad_input(output_cache_.shape());
        
        for (size_t i = 0; i < output_cache_.size(); ++i) {
            float t = output_cache_[i];
            grad_input[i] = grad_output[i] * (1.0f - t * t);
        }
        
        return grad_input;
    }
    
    std::string name() const override { return "Tanh"; }
    
    Tensor::Shape output_shape(const Tensor::Shape& input_shape) const override {
        return input_shape;
    }

private:
    Tensor output_cache_;
};

// ============================================================================
// Softmax
// ============================================================================

/**
 * @brief Softmax activation: f(x_i) = e^(x_i) / Σ_j e^(x_j)
 * 
 * Properties:
 * - Output is a probability distribution (sums to 1)
 * - Used for multi-class classification output layer
 * - Applied along the last axis (features dimension)
 * 
 * Note: For numerical stability, we compute softmax as:
 * softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))
 */
class Softmax : public Layer {
public:
    Tensor forward(const Tensor& input) override {
        // Assume input shape is (batch_size, num_classes) for 2D
        // or (num_classes,) for 1D
        
        output_cache_ = Tensor(input.shape());
        
        if (input.ndim() == 1) {
            // Single sample
            apply_softmax_1d(input.data(), output_cache_.data(), input.size());
        } else if (input.ndim() == 2) {
            // Batch of samples
            size_t batch_size = input.dim(0);
            size_t num_classes = input.dim(1);
            
            for (size_t b = 0; b < batch_size; ++b) {
                apply_softmax_1d(
                    input.data() + b * num_classes,
                    output_cache_.data() + b * num_classes,
                    num_classes
                );
            }
        } else {
            throw std::invalid_argument("Softmax expects 1D or 2D input");
        }
        
        return output_cache_;
    }
    
    Tensor backward(const Tensor& grad_output) override {
        // The Jacobian of softmax is:
        // ∂softmax(x)_i / ∂x_j = softmax(x)_i * (δ_ij - softmax(x)_j)
        //
        // For cross-entropy loss with softmax, the combined gradient simplifies to:
        // ∂L/∂x = softmax(x) - y (one-hot target)
        //
        // Here we compute the general Jacobian-vector product
        
        Tensor grad_input(output_cache_.shape());
        
        if (output_cache_.ndim() == 1) {
            compute_softmax_grad_1d(
                output_cache_.data(),
                grad_output.data(),
                grad_input.data(),
                output_cache_.size()
            );
        } else if (output_cache_.ndim() == 2) {
            size_t batch_size = output_cache_.dim(0);
            size_t num_classes = output_cache_.dim(1);
            
            for (size_t b = 0; b < batch_size; ++b) {
                compute_softmax_grad_1d(
                    output_cache_.data() + b * num_classes,
                    grad_output.data() + b * num_classes,
                    grad_input.data() + b * num_classes,
                    num_classes
                );
            }
        }
        
        return grad_input;
    }
    
    std::string name() const override { return "Softmax"; }
    
    Tensor::Shape output_shape(const Tensor::Shape& input_shape) const override {
        return input_shape;
    }

private:
    Tensor output_cache_;
    
    /**
     * @brief Apply softmax to a single 1D array (numerically stable)
     */
    static void apply_softmax_1d(const float* input, float* output, size_t size) {
        // Find max for numerical stability
        float max_val = *std::max_element(input, input + size);
        
        // Compute exp(x - max) and sum
        float sum = 0.0f;
        for (size_t i = 0; i < size; ++i) {
            output[i] = std::exp(input[i] - max_val);
            sum += output[i];
        }
        
        // Normalize
        for (size_t i = 0; i < size; ++i) {
            output[i] /= sum;
        }
    }
    
    /**
     * @brief Compute softmax gradient (Jacobian-vector product)
     */
    static void compute_softmax_grad_1d(
        const float* softmax_out,
        const float* grad_output,
        float* grad_input,
        size_t size
    ) {
        // grad_input[i] = Σ_j softmax[j] * (δ_ij - softmax[i]) * grad_output[j]
        //               = softmax[i] * grad_output[i] 
        //                 - softmax[i] * Σ_j (softmax[j] * grad_output[j])
        
        // First compute the dot product: Σ_j (softmax[j] * grad_output[j])
        float dot = 0.0f;
        for (size_t i = 0; i < size; ++i) {
            dot += softmax_out[i] * grad_output[i];
        }
        
        // Now compute gradient for each element
        for (size_t i = 0; i < size; ++i) {
            grad_input[i] = softmax_out[i] * (grad_output[i] - dot);
        }
    }
};

// ============================================================================
// Factory function for creating activations by name
// ============================================================================

/**
 * @brief Create an activation layer by name
 * @param name Activation name: "relu", "sigmoid", "tanh", "softmax", "leaky_relu"
 * @return Unique pointer to the activation layer
 */
inline LayerPtr create_activation(const std::string& activation_name) {
    if (activation_name == "relu") {
        return std::make_unique<ReLU>();
    } else if (activation_name == "sigmoid") {
        return std::make_unique<Sigmoid>();
    } else if (activation_name == "tanh") {
        return std::make_unique<Tanh>();
    } else if (activation_name == "softmax") {
        return std::make_unique<Softmax>();
    } else if (activation_name == "leaky_relu") {
        return std::make_unique<LeakyReLU>();
    } else {
        throw std::invalid_argument("Unknown activation: " + activation_name);
    }
}

} // namespace nn

#endif // NN_ACTIVATIONS_HPP
