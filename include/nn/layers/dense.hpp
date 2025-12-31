#ifndef NN_DENSE_HPP
#define NN_DENSE_HPP

/**
 * @file dense.hpp
 * @brief Dense (Fully Connected) Layer
 * 
 * The Dense layer, also called a Fully Connected layer, connects every
 * input neuron to every output neuron. It's the most fundamental layer
 * type in neural networks.
 * 
 * Mathematical operation:
 *   output = input × weights + bias
 * 
 * Where:
 *   - input: (batch_size, in_features)
 *   - weights: (in_features, out_features)
 *   - bias: (out_features,)
 *   - output: (batch_size, out_features)
 * 
 * Parameters to learn:
 *   - weights: in_features × out_features values
 *   - bias: out_features values (optional)
 */

#include "nn/layers/layer.hpp"

namespace nn {

/**
 * @brief Fully connected (dense) layer
 * 
 * Each output is a weighted sum of all inputs plus a bias:
 *   y_j = Σ_i (x_i * W_ij) + b_j
 */
class Dense : public Layer {
public:
    /**
     * @brief Construct a Dense layer
     * 
     * @param in_features Number of input features
     * @param out_features Number of output features (neurons)
     * @param use_bias Whether to include a bias term (default: true)
     * @param weight_init Weight initialization method: "xavier", "he", "normal", "zeros"
     */
    Dense(size_t in_features, 
          size_t out_features, 
          bool use_bias = true,
          const std::string& weight_init = "xavier");
    
    // ========================================================================
    // Core Operations
    // ========================================================================
    
    /**
     * @brief Forward pass: output = input × weights + bias
     * 
     * @param input Tensor of shape (batch_size, in_features) or (in_features,)
     * @return Tensor of shape (batch_size, out_features) or (out_features,)
     */
    Tensor forward(const Tensor& input) override;
    
    /**
     * @brief Backward pass: compute gradients
     * 
     * Given ∂L/∂output (grad_output), compute:
     *   - ∂L/∂input = grad_output × weightsᵀ
     *   - ∂L/∂weights = inputᵀ × grad_output
     *   - ∂L/∂bias = sum(grad_output, axis=0)
     * 
     * @param grad_output Gradient from next layer (batch_size, out_features)
     * @return Gradient to pass to previous layer (batch_size, in_features)
     */
    Tensor backward(const Tensor& grad_output) override;
    
    // ========================================================================
    // Parameter Access
    // ========================================================================
    
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    size_t num_parameters() const override;
    
    /**
     * @brief Get the weight tensor (for inspection/debugging)
     */
    const Tensor& weights() const { return weights_; }
    Tensor& weights() { return weights_; }
    
    /**
     * @brief Get the bias tensor
     */
    const Tensor& bias() const { return bias_; }
    Tensor& bias() { return bias_; }
    
    // ========================================================================
    // Layer Info
    // ========================================================================
    
    std::string name() const override { return "Dense"; }
    
    Tensor::Shape output_shape(const Tensor::Shape& input_shape) const override;
    
    std::string summary() const override;
    
    /**
     * @brief Reset parameters to initial values
     */
    void reset_parameters() override;
    
    // ========================================================================
    // Getters
    // ========================================================================
    
    size_t in_features() const { return in_features_; }
    size_t out_features() const { return out_features_; }
    bool has_bias() const { return use_bias_; }

private:
    // Configuration
    size_t in_features_;
    size_t out_features_;
    bool use_bias_;
    std::string weight_init_;
    
    // Parameters (learned during training)
    Tensor weights_;     // Shape: (in_features, out_features)
    Tensor bias_;        // Shape: (out_features,)
    
    // Gradients (computed during backward pass)
    Tensor grad_weights_;
    Tensor grad_bias_;
    
    /**
     * @brief Initialize weights based on the specified method
     */
    void initialize_weights();
};

} // namespace nn

#endif // NN_DENSE_HPP
