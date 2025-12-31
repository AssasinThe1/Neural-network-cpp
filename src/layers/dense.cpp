/**
 * @file dense.cpp
 * @brief Implementation of the Dense (Fully Connected) layer
 */

#include "nn/layers/dense.hpp"
#include <sstream>

namespace nn {

// ============================================================================
// Constructor
// ============================================================================

Dense::Dense(size_t in_features, 
             size_t out_features, 
             bool use_bias,
             const std::string& weight_init)
    : in_features_(in_features)
    , out_features_(out_features)
    , use_bias_(use_bias)
    , weight_init_(weight_init)
    , weights_(Tensor::Shape{in_features, out_features})
    , bias_(use_bias ? Tensor(Tensor::Shape{out_features}) : Tensor())
    , grad_weights_(Tensor::Shape{in_features, out_features})
    , grad_bias_(use_bias ? Tensor(Tensor::Shape{out_features}) : Tensor())
{
    initialize_weights();
}

// ============================================================================
// Forward Pass
// ============================================================================

Tensor Dense::forward(const Tensor& input) {
    // Handle both single sample and batch input
    bool single_sample = (input.ndim() == 1);
    
    // Reshape single sample to batch of 1
    Tensor batch_input = single_sample 
        ? input.reshape({1, in_features_}) 
        : input;
    
    // Validate input shape
    if (batch_input.ndim() != 2 || batch_input.dim(1) != in_features_) {
        throw std::invalid_argument(
            "Dense: Expected input of shape (batch, " + std::to_string(in_features_) + 
            "), got " + batch_input.shape_string()
        );
    }
    
    // Cache input for backward pass
    // Important: We store the batch version
    input_cache_ = batch_input;
    
    // Compute output = input × weights
    // (batch_size, in_features) × (in_features, out_features) = (batch_size, out_features)
    Tensor output = batch_input.matmul(weights_);
    
    // Add bias if present
    if (use_bias_) {
        // Broadcast bias across batch dimension
        // bias is (out_features,), we need to add it to each row
        for (size_t b = 0; b < output.dim(0); ++b) {
            for (size_t j = 0; j < out_features_; ++j) {
                output.at({b, j}) += bias_[j];
            }
        }
    }
    
    // Return in original format
    return single_sample ? output.reshape({out_features_}) : output;
}

// ============================================================================
// Backward Pass
// ============================================================================

Tensor Dense::backward(const Tensor& grad_output) {
    // Ensure batch format
    bool single_sample = (grad_output.ndim() == 1);
    Tensor batch_grad = single_sample 
        ? grad_output.reshape({1, out_features_}) 
        : grad_output;
    
    size_t batch_size = batch_grad.dim(0);
    
    // ========================================================================
    // Compute gradient of loss w.r.t weights
    // ∂L/∂W = Xᵀ × ∂L/∂Y
    // 
    // Shape: (in_features, batch_size) × (batch_size, out_features) 
    //      = (in_features, out_features)
    // ========================================================================
    Tensor input_t = input_cache_.transpose();
    grad_weights_ = input_t.matmul(batch_grad);
    
    // ========================================================================
    // Compute gradient of loss w.r.t bias
    // ∂L/∂b = sum(∂L/∂Y, axis=0)
    //
    // The bias gradient is just the sum of output gradients across the batch
    // ========================================================================
    if (use_bias_) {
        grad_bias_.fill(0.0f);
        for (size_t b = 0; b < batch_size; ++b) {
            for (size_t j = 0; j < out_features_; ++j) {
                grad_bias_[j] += batch_grad.at({b, j});
            }
        }
    }
    
    // ========================================================================
    // Compute gradient to pass to previous layer
    // ∂L/∂X = ∂L/∂Y × Wᵀ
    //
    // Shape: (batch_size, out_features) × (out_features, in_features) 
    //      = (batch_size, in_features)
    // ========================================================================
    Tensor weights_t = weights_.transpose();
    Tensor grad_input = batch_grad.matmul(weights_t);
    
    // Return in original format
    return single_sample ? grad_input.reshape({in_features_}) : grad_input;
}

// ============================================================================
// Parameter Access
// ============================================================================

std::vector<Tensor*> Dense::parameters() {
    if (use_bias_) {
        return {&weights_, &bias_};
    }
    return {&weights_};
}

std::vector<Tensor*> Dense::gradients() {
    if (use_bias_) {
        return {&grad_weights_, &grad_bias_};
    }
    return {&grad_weights_};
}

size_t Dense::num_parameters() const {
    size_t count = in_features_ * out_features_;  // weights
    if (use_bias_) {
        count += out_features_;  // bias
    }
    return count;
}

// ============================================================================
// Layer Info
// ============================================================================

Tensor::Shape Dense::output_shape(const Tensor::Shape& input_shape) const {
    if (input_shape.size() == 1) {
        return {out_features_};
    } else if (input_shape.size() == 2) {
        return {input_shape[0], out_features_};
    }
    throw std::invalid_argument("Dense layer expects 1D or 2D input");
}

std::string Dense::summary() const {
    std::ostringstream oss;
    oss << "Dense(" << in_features_ << " -> " << out_features_;
    if (use_bias_) oss << ", bias=true";
    oss << ") [" << num_parameters() << " params]";
    return oss.str();
}

// ============================================================================
// Initialization
// ============================================================================

void Dense::reset_parameters() {
    initialize_weights();
}

void Dense::initialize_weights() {
    /*
     * Weight initialization is crucial for training deep networks!
     * 
     * Bad initialization can cause:
     * - Vanishing gradients (weights too small)
     * - Exploding gradients (weights too large)
     * - Symmetric weights (all neurons learn the same thing)
     * 
     * Common strategies:
     * - Xavier/Glorot: Good for sigmoid/tanh activations
     * - He/Kaiming: Good for ReLU activations
     */
    
    if (weight_init_ == "xavier" || weight_init_ == "glorot") {
        // Xavier uniform: U(-sqrt(6/(fan_in+fan_out)), sqrt(6/(fan_in+fan_out)))
        float limit = std::sqrt(6.0f / static_cast<float>(in_features_ + out_features_));
        weights_ = Tensor::random_uniform({in_features_, out_features_}, -limit, limit);
    } 
    else if (weight_init_ == "he" || weight_init_ == "kaiming") {
        // He uniform: U(-sqrt(6/fan_in), sqrt(6/fan_in))
        float limit = std::sqrt(6.0f / static_cast<float>(in_features_));
        weights_ = Tensor::random_uniform({in_features_, out_features_}, -limit, limit);
    }
    else if (weight_init_ == "normal") {
        // Small random values from normal distribution
        float std = std::sqrt(2.0f / static_cast<float>(in_features_ + out_features_));
        weights_ = Tensor::random_normal({in_features_, out_features_}, 0.0f, std);
    }
    else if (weight_init_ == "zeros") {
        // Not recommended for training, but useful for testing
        weights_ = Tensor::zeros({in_features_, out_features_});
    }
    else {
        throw std::invalid_argument("Unknown weight initialization: " + weight_init_);
    }
    
    // Initialize biases to zero (common practice)
    if (use_bias_) {
        bias_.fill(0.0f);
    }
    
    // Clear gradients
    grad_weights_.fill(0.0f);
    if (use_bias_) {
        grad_bias_.fill(0.0f);
    }
}

} // namespace nn
