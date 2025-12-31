/**
 * @file conv2d.cpp
 * @brief Implementation of 2D Convolutional Layer
 * 
 * The convolution operation slides a kernel (filter) across the input
 * and computes the dot product at each position.
 * 
 * For beginners: Think of it like a magnifying glass that looks at
 * small patches of an image, extracting features like edges, textures, etc.
 */

#include "nn/layers/conv2d.hpp"
#include <sstream>
#include <cmath>

namespace nn {

// ============================================================================
// Constructors
// ============================================================================

Conv2D::Conv2D(size_t in_channels,
               size_t out_channels,
               size_t kernel_size,
               size_t stride,
               size_t padding,
               bool use_bias)
    : Conv2D(in_channels, out_channels, 
             kernel_size, kernel_size,    // kernel_h, kernel_w
             stride, stride,              // stride_h, stride_w
             padding, padding,            // padding_h, padding_w
             use_bias)
{
}

Conv2D::Conv2D(size_t in_channels,
               size_t out_channels,
               size_t kernel_h,
               size_t kernel_w,
               size_t stride_h,
               size_t stride_w,
               size_t padding_h,
               size_t padding_w,
               bool use_bias)
    : in_channels_(in_channels)
    , out_channels_(out_channels)
    , kernel_h_(kernel_h)
    , kernel_w_(kernel_w)
    , stride_h_(stride_h)
    , stride_w_(stride_w)
    , padding_h_(padding_h)
    , padding_w_(padding_w)
    , use_bias_(use_bias)
    , weights_(Tensor::Shape{out_channels, in_channels, kernel_h, kernel_w})
    , bias_(use_bias ? Tensor(Tensor::Shape{out_channels}) : Tensor())
    , grad_weights_(Tensor::Shape{out_channels, in_channels, kernel_h, kernel_w})
    , grad_bias_(use_bias ? Tensor(Tensor::Shape{out_channels}) : Tensor())
    , cached_batch_size_(0)
    , cached_in_h_(0)
    , cached_in_w_(0)
{
    initialize_weights();
}

// ============================================================================
// Forward Pass
// ============================================================================

Tensor Conv2D::forward(const Tensor& input) {
    /*
     * Convolution operation visualization (2D for simplicity):
     * 
     * Input (padded):          Kernel:         Output:
     * [0 0 0 0 0]              [w1 w2]         For each position,
     * [0 a b c 0]      *       [w3 w4]    =    compute dot product
     * [0 d e f 0]                              of overlapping region
     * [0 g h i 0]
     * [0 0 0 0 0]
     * 
     * Example: output[0,0] = 0*w1 + 0*w2 + 0*w3 + a*w4
     */
    
    // Validate input shape: (batch, channels, height, width)
    if (input.ndim() != 4) {
        throw std::invalid_argument(
            "Conv2D expects 4D input (batch, channels, height, width), got " +
            std::to_string(input.ndim()) + "D"
        );
    }
    
    if (input.dim(1) != in_channels_) {
        throw std::invalid_argument(
            "Conv2D: Expected " + std::to_string(in_channels_) + 
            " input channels, got " + std::to_string(input.dim(1))
        );
    }
    
    // Cache input for backward pass
    input_cache_ = input;
    
    size_t batch_size = input.dim(0);
    size_t in_h = input.dim(2);
    size_t in_w = input.dim(3);
    
    // Cache dimensions for backward pass
    cached_batch_size_ = batch_size;
    cached_in_h_ = in_h;
    cached_in_w_ = in_w;
    
    // Compute output dimensions
    auto [out_h, out_w] = compute_output_size(in_h, in_w);
    
    // Allocate output tensor
    Tensor output({batch_size, out_channels_, out_h, out_w}, 0.0f);
    
    // Perform convolution
    // For each sample in batch
    for (size_t b = 0; b < batch_size; ++b) {
        // For each output channel (filter)
        for (size_t oc = 0; oc < out_channels_; ++oc) {
            // For each output position
            for (size_t oh = 0; oh < out_h; ++oh) {
                for (size_t ow = 0; ow < out_w; ++ow) {
                    float sum = 0.0f;
                    
                    // For each input channel
                    for (size_t ic = 0; ic < in_channels_; ++ic) {
                        // For each kernel position
                        for (size_t kh = 0; kh < kernel_h_; ++kh) {
                            for (size_t kw = 0; kw < kernel_w_; ++kw) {
                                // Compute input position (considering stride and padding)
                                int ih = static_cast<int>(oh * stride_h_ + kh) - 
                                         static_cast<int>(padding_h_);
                                int iw = static_cast<int>(ow * stride_w_ + kw) - 
                                         static_cast<int>(padding_w_);
                                
                                // Check bounds (padding gives 0)
                                if (ih >= 0 && ih < static_cast<int>(in_h) &&
                                    iw >= 0 && iw < static_cast<int>(in_w)) {
                                    
                                    float input_val = input.at({b, ic, 
                                        static_cast<size_t>(ih), 
                                        static_cast<size_t>(iw)});
                                    float kernel_val = weights_.at({oc, ic, kh, kw});
                                    sum += input_val * kernel_val;
                                }
                            }
                        }
                    }
                    
                    // Add bias if present
                    if (use_bias_) {
                        sum += bias_[oc];
                    }
                    
                    output.at({b, oc, oh, ow}) = sum;
                }
            }
        }
    }
    
    return output;
}

// ============================================================================
// Backward Pass
// ============================================================================

Tensor Conv2D::backward(const Tensor& grad_output) {
    /*
     * Backward pass for convolution involves:
     * 1. Gradient w.r.t. weights: correlate input with grad_output
     * 2. Gradient w.r.t. bias: sum of grad_output over spatial dimensions
     * 3. Gradient w.r.t. input: "full" convolution of grad_output with flipped weights
     */
    
    size_t batch_size = grad_output.dim(0);
    auto [out_h, out_w] = std::make_pair(grad_output.dim(2), grad_output.dim(3));
    
    // Initialize gradients
    grad_weights_.fill(0.0f);
    if (use_bias_) {
        grad_bias_.fill(0.0f);
    }
    
    Tensor grad_input({batch_size, in_channels_, cached_in_h_, cached_in_w_}, 0.0f);
    
    // Compute gradients
    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t oc = 0; oc < out_channels_; ++oc) {
            // Gradient w.r.t. bias: sum over spatial positions
            if (use_bias_) {
                for (size_t oh = 0; oh < out_h; ++oh) {
                    for (size_t ow = 0; ow < out_w; ++ow) {
                        grad_bias_[oc] += grad_output.at({b, oc, oh, ow});
                    }
                }
            }
            
            for (size_t oh = 0; oh < out_h; ++oh) {
                for (size_t ow = 0; ow < out_w; ++ow) {
                    float grad_out_val = grad_output.at({b, oc, oh, ow});
                    
                    for (size_t ic = 0; ic < in_channels_; ++ic) {
                        for (size_t kh = 0; kh < kernel_h_; ++kh) {
                            for (size_t kw = 0; kw < kernel_w_; ++kw) {
                                int ih = static_cast<int>(oh * stride_h_ + kh) - 
                                         static_cast<int>(padding_h_);
                                int iw = static_cast<int>(ow * stride_w_ + kw) - 
                                         static_cast<int>(padding_w_);
                                
                                if (ih >= 0 && ih < static_cast<int>(cached_in_h_) &&
                                    iw >= 0 && iw < static_cast<int>(cached_in_w_)) {
                                    
                                    // Gradient w.r.t. weights
                                    float input_val = input_cache_.at({b, ic,
                                        static_cast<size_t>(ih),
                                        static_cast<size_t>(iw)});
                                    grad_weights_.at({oc, ic, kh, kw}) += 
                                        input_val * grad_out_val;
                                    
                                    // Gradient w.r.t. input
                                    grad_input.at({b, ic, 
                                        static_cast<size_t>(ih),
                                        static_cast<size_t>(iw)}) +=
                                        weights_.at({oc, ic, kh, kw}) * grad_out_val;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    
    return grad_input;
}

// ============================================================================
// Parameter Access
// ============================================================================

std::vector<Tensor*> Conv2D::parameters() {
    if (use_bias_) {
        return {&weights_, &bias_};
    }
    return {&weights_};
}

std::vector<Tensor*> Conv2D::gradients() {
    if (use_bias_) {
        return {&grad_weights_, &grad_bias_};
    }
    return {&grad_weights_};
}

size_t Conv2D::num_parameters() const {
    size_t count = out_channels_ * in_channels_ * kernel_h_ * kernel_w_;
    if (use_bias_) {
        count += out_channels_;
    }
    return count;
}

// ============================================================================
// Layer Info
// ============================================================================

Tensor::Shape Conv2D::output_shape(const Tensor::Shape& input_shape) const {
    if (input_shape.size() != 4) {
        throw std::invalid_argument("Conv2D expects 4D input shape");
    }
    
    auto [out_h, out_w] = compute_output_size(input_shape[2], input_shape[3]);
    return {input_shape[0], out_channels_, out_h, out_w};
}

std::string Conv2D::summary() const {
    std::ostringstream oss;
    oss << "Conv2D(" << in_channels_ << " -> " << out_channels_;
    oss << ", kernel=" << kernel_h_ << "x" << kernel_w_;
    if (stride_h_ != 1 || stride_w_ != 1) {
        oss << ", stride=" << stride_h_;
    }
    if (padding_h_ != 0 || padding_w_ != 0) {
        oss << ", padding=" << padding_h_;
    }
    oss << ") [" << num_parameters() << " params]";
    return oss.str();
}

void Conv2D::reset_parameters() {
    initialize_weights();
}

// ============================================================================
// Private Helpers
// ============================================================================

void Conv2D::initialize_weights() {
    // He initialization (good for ReLU)
    size_t fan_in = in_channels_ * kernel_h_ * kernel_w_;
    float std = std::sqrt(2.0f / static_cast<float>(fan_in));
    
    weights_ = Tensor::random_normal(
        {out_channels_, in_channels_, kernel_h_, kernel_w_}, 0.0f, std);
    
    if (use_bias_) {
        bias_.fill(0.0f);
    }
    
    grad_weights_.fill(0.0f);
    if (use_bias_) {
        grad_bias_.fill(0.0f);
    }
}

std::pair<size_t, size_t> Conv2D::compute_output_size(size_t in_h, size_t in_w) const {
    size_t out_h = (in_h + 2 * padding_h_ - kernel_h_) / stride_h_ + 1;
    size_t out_w = (in_w + 2 * padding_w_ - kernel_w_) / stride_w_ + 1;
    return {out_h, out_w};
}

} // namespace nn
