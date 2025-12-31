#ifndef NN_POOLING_HPP
#define NN_POOLING_HPP

/**
 * @file pooling.hpp
 * @brief Pooling layers for CNNs
 * 
 * Pooling layers reduce the spatial dimensions of feature maps while
 * retaining important information. They help with:
 * - Reducing computation
 * - Providing translation invariance
 * - Preventing overfitting
 * 
 * Common types:
 * - MaxPool: Takes maximum value in each window (most common)
 * - AvgPool: Takes average value in each window
 */

#include "nn/layers/layer.hpp"
#include <limits>

namespace nn {

// ============================================================================
// MaxPool2D
// ============================================================================

/**
 * @brief 2D Max Pooling Layer
 * 
 * Slides a window over the input and outputs the maximum value
 * in each window. This helps detect features regardless of their
 * exact position (translation invariance).
 * 
 * Example with 2x2 pool:
 * [1 2 5 6]       [4 6]
 * [3 4 7 8]  -->  [7 8]  (taking max of each 2x2 region)
 * [1 2 3 4]
 * [5 6 7 8]
 */
class MaxPool2D : public Layer {
public:
    /**
     * @brief Construct MaxPool2D
     * @param pool_size Size of pooling window (default: 2)
     * @param stride Stride of pooling (default: same as pool_size)
     */
    explicit MaxPool2D(size_t pool_size = 2, size_t stride = 0)
        : pool_h_(pool_size)
        , pool_w_(pool_size)
        , stride_h_(stride == 0 ? pool_size : stride)
        , stride_w_(stride == 0 ? pool_size : stride)
    {}
    
    MaxPool2D(size_t pool_h, size_t pool_w, size_t stride_h, size_t stride_w)
        : pool_h_(pool_h)
        , pool_w_(pool_w)
        , stride_h_(stride_h)
        , stride_w_(stride_w)
    {}
    
    Tensor forward(const Tensor& input) override {
        if (input.ndim() != 4) {
            throw std::invalid_argument("MaxPool2D expects 4D input");
        }
        
        input_cache_ = input;
        
        size_t batch_size = input.dim(0);
        size_t channels = input.dim(1);
        size_t in_h = input.dim(2);
        size_t in_w = input.dim(3);
        
        // Compute output dimensions
        size_t out_h = (in_h - pool_h_) / stride_h_ + 1;
        size_t out_w = (in_w - pool_w_) / stride_w_ + 1;
        
        Tensor output(Tensor::Shape{batch_size, channels, out_h, out_w});
        
        // Store max indices for backward pass
        max_indices_.resize(batch_size * channels * out_h * out_w);
        
        size_t idx = 0;
        for (size_t b = 0; b < batch_size; ++b) {
            for (size_t c = 0; c < channels; ++c) {
                for (size_t oh = 0; oh < out_h; ++oh) {
                    for (size_t ow = 0; ow < out_w; ++ow) {
                        float max_val = std::numeric_limits<float>::lowest();
                        size_t max_ih = 0, max_iw = 0;
                        
                        // Find max in pooling window
                        for (size_t ph = 0; ph < pool_h_; ++ph) {
                            for (size_t pw = 0; pw < pool_w_; ++pw) {
                                size_t ih = oh * stride_h_ + ph;
                                size_t iw = ow * stride_w_ + pw;
                                
                                float val = input.at({b, c, ih, iw});
                                if (val > max_val) {
                                    max_val = val;
                                    max_ih = ih;
                                    max_iw = iw;
                                }
                            }
                        }
                        
                        output.at({b, c, oh, ow}) = max_val;
                        max_indices_[idx++] = {max_ih, max_iw};
                    }
                }
            }
        }
        
        return output;
    }
    
    Tensor backward(const Tensor& grad_output) override {
        // Gradient flows only through the maximum values
        size_t batch_size = input_cache_.dim(0);
        size_t channels = input_cache_.dim(1);
        size_t in_h = input_cache_.dim(2);
        size_t in_w = input_cache_.dim(3);
        size_t out_h = grad_output.dim(2);
        size_t out_w = grad_output.dim(3);
        
        Tensor grad_input(Tensor::Shape{batch_size, channels, in_h, in_w}, 0.0f);
        
        size_t idx = 0;
        for (size_t b = 0; b < batch_size; ++b) {
            for (size_t c = 0; c < channels; ++c) {
                for (size_t oh = 0; oh < out_h; ++oh) {
                    for (size_t ow = 0; ow < out_w; ++ow) {
                        auto [max_ih, max_iw] = max_indices_[idx++];
                        grad_input.at({b, c, max_ih, max_iw}) += 
                            grad_output.at({b, c, oh, ow});
                    }
                }
            }
        }
        
        return grad_input;
    }
    
    std::string name() const override { return "MaxPool2D"; }
    
    Tensor::Shape output_shape(const Tensor::Shape& input_shape) const override {
        size_t out_h = (input_shape[2] - pool_h_) / stride_h_ + 1;
        size_t out_w = (input_shape[3] - pool_w_) / stride_w_ + 1;
        return {input_shape[0], input_shape[1], out_h, out_w};
    }
    
    std::string summary() const override {
        return "MaxPool2D(" + std::to_string(pool_h_) + "x" + 
               std::to_string(pool_w_) + ")";
    }

private:
    size_t pool_h_, pool_w_;
    size_t stride_h_, stride_w_;
    std::vector<std::pair<size_t, size_t>> max_indices_;
};

// ============================================================================
// AvgPool2D
// ============================================================================

/**
 * @brief 2D Average Pooling Layer
 */
class AvgPool2D : public Layer {
public:
    explicit AvgPool2D(size_t pool_size = 2, size_t stride = 0)
        : pool_h_(pool_size)
        , pool_w_(pool_size)
        , stride_h_(stride == 0 ? pool_size : stride)
        , stride_w_(stride == 0 ? pool_size : stride)
    {}
    
    Tensor forward(const Tensor& input) override {
        if (input.ndim() != 4) {
            throw std::invalid_argument("AvgPool2D expects 4D input");
        }
        
        input_cache_ = input;
        
        size_t batch_size = input.dim(0);
        size_t channels = input.dim(1);
        size_t in_h = input.dim(2);
        size_t in_w = input.dim(3);
        
        size_t out_h = (in_h - pool_h_) / stride_h_ + 1;
        size_t out_w = (in_w - pool_w_) / stride_w_ + 1;
        
        Tensor output(Tensor::Shape{batch_size, channels, out_h, out_w});
        
        float pool_area = static_cast<float>(pool_h_ * pool_w_);
        
        for (size_t b = 0; b < batch_size; ++b) {
            for (size_t c = 0; c < channels; ++c) {
                for (size_t oh = 0; oh < out_h; ++oh) {
                    for (size_t ow = 0; ow < out_w; ++ow) {
                        float sum = 0.0f;
                        
                        for (size_t ph = 0; ph < pool_h_; ++ph) {
                            for (size_t pw = 0; pw < pool_w_; ++pw) {
                                size_t ih = oh * stride_h_ + ph;
                                size_t iw = ow * stride_w_ + pw;
                                sum += input.at({b, c, ih, iw});
                            }
                        }
                        
                        output.at({b, c, oh, ow}) = sum / pool_area;
                    }
                }
            }
        }
        
        return output;
    }
    
    Tensor backward(const Tensor& grad_output) override {
        // Each input contributes equally to the average
        size_t batch_size = input_cache_.dim(0);
        size_t channels = input_cache_.dim(1);
        size_t in_h = input_cache_.dim(2);
        size_t in_w = input_cache_.dim(3);
        size_t out_h = grad_output.dim(2);
        size_t out_w = grad_output.dim(3);
        
        Tensor grad_input(Tensor::Shape{batch_size, channels, in_h, in_w}, 0.0f);
        float pool_area = static_cast<float>(pool_h_ * pool_w_);
        
        for (size_t b = 0; b < batch_size; ++b) {
            for (size_t c = 0; c < channels; ++c) {
                for (size_t oh = 0; oh < out_h; ++oh) {
                    for (size_t ow = 0; ow < out_w; ++ow) {
                        float grad = grad_output.at({b, c, oh, ow}) / pool_area;
                        
                        for (size_t ph = 0; ph < pool_h_; ++ph) {
                            for (size_t pw = 0; pw < pool_w_; ++pw) {
                                size_t ih = oh * stride_h_ + ph;
                                size_t iw = ow * stride_w_ + pw;
                                grad_input.at({b, c, ih, iw}) += grad;
                            }
                        }
                    }
                }
            }
        }
        
        return grad_input;
    }
    
    std::string name() const override { return "AvgPool2D"; }
    
    Tensor::Shape output_shape(const Tensor::Shape& input_shape) const override {
        size_t out_h = (input_shape[2] - pool_h_) / stride_h_ + 1;
        size_t out_w = (input_shape[3] - pool_w_) / stride_w_ + 1;
        return {input_shape[0], input_shape[1], out_h, out_w};
    }

private:
    size_t pool_h_, pool_w_;
    size_t stride_h_, stride_w_;
};

// ============================================================================
// Flatten Layer
// ============================================================================

/**
 * @brief Flatten layer - converts multi-dimensional input to 1D
 * 
 * Typically used between convolutional layers and dense layers.
 * Reshapes (batch, channels, height, width) to (batch, channels*height*width)
 */
class Flatten : public Layer {
public:
    Tensor forward(const Tensor& input) override {
        input_shape_ = input.shape();
        
        // Keep batch dimension, flatten the rest
        size_t batch_size = input.dim(0);
        size_t flat_size = input.size() / batch_size;
        
        return input.reshape({batch_size, flat_size});
    }
    
    Tensor backward(const Tensor& grad_output) override {
        // Simply reshape back to original shape
        return grad_output.reshape(input_shape_);
    }
    
    std::string name() const override { return "Flatten"; }
    
    Tensor::Shape output_shape(const Tensor::Shape& input_shape) const override {
        size_t batch_size = input_shape[0];
        size_t flat_size = 1;
        for (size_t i = 1; i < input_shape.size(); ++i) {
            flat_size *= input_shape[i];
        }
        return {batch_size, flat_size};
    }

private:
    Tensor::Shape input_shape_;
};

// ============================================================================
// Dropout Layer (for regularization)
// ============================================================================

/**
 * @brief Dropout layer for regularization
 * 
 * During training, randomly sets elements to zero with probability p.
 * This helps prevent overfitting by forcing the network to not rely
 * on any single neuron.
 * 
 * During inference (evaluation), dropout is disabled and outputs are
 * scaled by (1-p) to maintain expected values.
 */
class Dropout : public Layer {
public:
    /**
     * @param p Probability of dropping a neuron (default: 0.5)
     */
    explicit Dropout(float p = 0.5f) : p_(p) {
        if (p < 0.0f || p > 1.0f) {
            throw std::invalid_argument("Dropout probability must be in [0, 1]");
        }
    }
    
    Tensor forward(const Tensor& input) override {
        if (!training_) {
            // During inference, just pass through (scaling is handled differently)
            return input;
        }
        
        // Generate mask
        mask_ = Tensor::random_uniform(input.shape(), 0.0f, 1.0f);
        Tensor output(input.shape());
        
        // Scale by 1/(1-p) to maintain expected values
        float scale = 1.0f / (1.0f - p_);
        
        for (size_t i = 0; i < input.size(); ++i) {
            if (mask_[i] > p_) {
                output[i] = input[i] * scale;
                mask_[i] = scale;  // Store scale for backward
            } else {
                output[i] = 0.0f;
                mask_[i] = 0.0f;
            }
        }
        
        return output;
    }
    
    Tensor backward(const Tensor& grad_output) override {
        if (!training_) {
            return grad_output;
        }
        
        // Gradient is masked the same way
        Tensor grad_input(grad_output.shape());
        for (size_t i = 0; i < grad_output.size(); ++i) {
            grad_input[i] = grad_output[i] * mask_[i];
        }
        return grad_input;
    }
    
    std::string name() const override { return "Dropout"; }
    
    Tensor::Shape output_shape(const Tensor::Shape& input_shape) const override {
        return input_shape;
    }
    
    std::string summary() const override {
        return "Dropout(p=" + std::to_string(p_) + ")";
    }

private:
    float p_;
    Tensor mask_;
};

} // namespace nn

#endif // NN_POOLING_HPP
