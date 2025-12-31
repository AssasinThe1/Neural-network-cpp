#ifndef NN_CONV2D_HPP
#define NN_CONV2D_HPP

/**
 * @file conv2d.hpp
 * @brief 2D Convolutional Layer
 * 
 * Convolutional layers are the core building blocks of CNNs (Convolutional
 * Neural Networks) used for image processing. They learn spatial features
 * by sliding filters (kernels) across the input.
 * 
 * Key concepts:
 * - Kernel/Filter: Small weight matrix that slides across input
 * - Stride: Step size when sliding the kernel
 * - Padding: Adding zeros around input to control output size
 * - Channels: Color channels (e.g., RGB=3) or feature maps
 * 
 * Shape conventions (we use NCHW format):
 *   Input:  (batch_size, in_channels, height, width)
 *   Kernel: (out_channels, in_channels, kernel_h, kernel_w)
 *   Output: (batch_size, out_channels, out_h, out_w)
 * 
 * Where:
 *   out_h = (height + 2*padding - kernel_h) / stride + 1
 *   out_w = (width + 2*padding - kernel_w) / stride + 1
 */

#include "nn/layers/layer.hpp"

namespace nn {

/**
 * @brief 2D Convolutional Layer
 */
class Conv2D : public Layer {
public:
    /**
     * @brief Construct a Conv2D layer
     * 
     * @param in_channels Number of input channels (e.g., 1 for grayscale, 3 for RGB)
     * @param out_channels Number of output channels (number of filters)
     * @param kernel_size Size of the square kernel (e.g., 3 for 3x3)
     * @param stride Step size (default: 1)
     * @param padding Zero-padding on each side (default: 0)
     * @param use_bias Whether to include bias (default: true)
     */
    Conv2D(size_t in_channels,
           size_t out_channels,
           size_t kernel_size,
           size_t stride = 1,
           size_t padding = 0,
           bool use_bias = true);
    
    /**
     * @brief Construct with separate height/width kernel
     */
    Conv2D(size_t in_channels,
           size_t out_channels,
           size_t kernel_h,
           size_t kernel_w,
           size_t stride_h,
           size_t stride_w,
           size_t padding_h,
           size_t padding_w,
           bool use_bias = true);
    
    // ========================================================================
    // Core Operations
    // ========================================================================
    
    /**
     * @brief Forward pass: convolution operation
     * 
     * @param input Tensor of shape (batch, in_channels, height, width)
     * @return Tensor of shape (batch, out_channels, out_h, out_w)
     */
    Tensor forward(const Tensor& input) override;
    
    /**
     * @brief Backward pass: compute gradients
     * 
     * @param grad_output Gradient from next layer
     * @return Gradient to pass to previous layer
     */
    Tensor backward(const Tensor& grad_output) override;
    
    // ========================================================================
    // Parameter Access
    // ========================================================================
    
    std::vector<Tensor*> parameters() override;
    std::vector<Tensor*> gradients() override;
    size_t num_parameters() const override;
    
    const Tensor& weights() const { return weights_; }
    Tensor& weights() { return weights_; }
    
    const Tensor& bias() const { return bias_; }
    Tensor& bias() { return bias_; }
    
    // ========================================================================
    // Layer Info
    // ========================================================================
    
    std::string name() const override { return "Conv2D"; }
    Tensor::Shape output_shape(const Tensor::Shape& input_shape) const override;
    std::string summary() const override;
    void reset_parameters() override;
    
    // Getters for layer configuration
    size_t in_channels() const { return in_channels_; }
    size_t out_channels() const { return out_channels_; }
    std::pair<size_t, size_t> kernel_size() const { return {kernel_h_, kernel_w_}; }
    std::pair<size_t, size_t> stride() const { return {stride_h_, stride_w_}; }
    std::pair<size_t, size_t> padding() const { return {padding_h_, padding_w_}; }

private:
    // Configuration
    size_t in_channels_;
    size_t out_channels_;
    size_t kernel_h_, kernel_w_;
    size_t stride_h_, stride_w_;
    size_t padding_h_, padding_w_;
    bool use_bias_;
    
    // Parameters
    Tensor weights_;     // (out_channels, in_channels, kernel_h, kernel_w)
    Tensor bias_;        // (out_channels,)
    
    // Gradients
    Tensor grad_weights_;
    Tensor grad_bias_;
    
    // Cached for backward
    size_t cached_batch_size_;
    size_t cached_in_h_, cached_in_w_;
    
    void initialize_weights();
    
    /**
     * @brief Compute output dimensions
     */
    std::pair<size_t, size_t> compute_output_size(size_t in_h, size_t in_w) const;
};

} // namespace nn

#endif // NN_CONV2D_HPP
