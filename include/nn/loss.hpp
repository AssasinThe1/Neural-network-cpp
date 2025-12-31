#ifndef NN_LOSS_HPP
#define NN_LOSS_HPP

/**
 * @file loss.hpp
 * @brief Loss functions for neural network training
 * 
 * Loss functions measure how well the network's predictions match
 * the target values. During training, we try to minimize the loss.
 * 
 * Common loss functions:
 * - MSE (Mean Squared Error): For regression problems
 * - Cross-Entropy: For classification problems
 * - Binary Cross-Entropy: For binary classification
 */

#include "nn/tensor.hpp"
#include <cmath>
#include <algorithm>
#include <memory>

namespace nn {

/**
 * @brief Abstract base class for loss functions
 */
class Loss {
public:
    virtual ~Loss() = default;
    
    /**
     * @brief Compute loss value
     * @param prediction Network output
     * @param target Ground truth
     * @return Scalar loss value
     */
    virtual float forward(const Tensor& prediction, const Tensor& target) = 0;
    
    /**
     * @brief Compute gradient of loss w.r.t. prediction
     * @return Gradient tensor (same shape as prediction)
     */
    virtual Tensor backward() = 0;
    
    /**
     * @brief Get loss name
     */
    virtual std::string name() const = 0;

protected:
    Tensor prediction_cache_;
    Tensor target_cache_;
};

using LossPtr = std::unique_ptr<Loss>;

// ============================================================================
// Mean Squared Error (MSE)
// ============================================================================

/**
 * @brief Mean Squared Error loss
 * 
 * L = (1/n) Σ (prediction - target)²
 * 
 * Good for regression problems where you want to predict continuous values.
 * Penalizes large errors more than small ones due to squaring.
 */
class MSELoss : public Loss {
public:
    float forward(const Tensor& prediction, const Tensor& target) override {
        if (!prediction.same_shape(target)) {
            throw std::invalid_argument("Prediction and target must have same shape");
        }
        
        prediction_cache_ = prediction;
        target_cache_ = target;
        
        // L = (1/n) Σ (pred - target)²
        float sum = 0.0f;
        for (size_t i = 0; i < prediction.size(); ++i) {
            float diff = prediction[i] - target[i];
            sum += diff * diff;
        }
        
        return sum / static_cast<float>(prediction.size());
    }
    
    Tensor backward() override {
        // ∂L/∂pred = (2/n) * (pred - target)
        Tensor grad(prediction_cache_.shape());
        float scale = 2.0f / static_cast<float>(prediction_cache_.size());
        
        for (size_t i = 0; i < prediction_cache_.size(); ++i) {
            grad[i] = scale * (prediction_cache_[i] - target_cache_[i]);
        }
        
        return grad;
    }
    
    std::string name() const override { return "MSELoss"; }
};

// ============================================================================
// Cross-Entropy Loss (for multi-class classification)
// ============================================================================

/**
 * @brief Cross-Entropy loss with softmax
 * 
 * L = -(1/n) Σ target_i * log(prediction_i)
 * 
 * For multi-class classification where target is a probability distribution
 * (usually one-hot encoded).
 * 
 * Note: Assumes predictions are already softmax outputs (probabilities).
 * For numerical stability, we clip predictions to avoid log(0).
 */
class CrossEntropyLoss : public Loss {
public:
    float forward(const Tensor& prediction, const Tensor& target) override {
        if (!prediction.same_shape(target)) {
            throw std::invalid_argument("Prediction and target must have same shape");
        }
        
        prediction_cache_ = prediction;
        target_cache_ = target;
        
        // Number of samples (batch size)
        size_t batch_size = prediction.ndim() == 2 ? prediction.dim(0) : 1;
        
        // L = -(1/n) Σ target * log(prediction)
        float sum = 0.0f;
        for (size_t i = 0; i < prediction.size(); ++i) {
            // Clip prediction to avoid log(0)
            float p = std::max(prediction[i], 1e-7f);
            p = std::min(p, 1.0f - 1e-7f);
            sum -= target[i] * std::log(p);
        }
        
        return sum / static_cast<float>(batch_size);
    }
    
    Tensor backward() override {
        /*
         * For softmax + cross-entropy, the gradient simplifies nicely:
         * ∂L/∂z = softmax(z) - target = prediction - target
         * 
         * This is one of the reasons this combination is popular!
         * The gradient is simply the difference between prediction and target.
         */
        
        size_t batch_size = prediction_cache_.ndim() == 2 ? 
                            prediction_cache_.dim(0) : 1;
        
        Tensor grad(prediction_cache_.shape());
        float scale = 1.0f / static_cast<float>(batch_size);
        
        for (size_t i = 0; i < prediction_cache_.size(); ++i) {
            grad[i] = scale * (prediction_cache_[i] - target_cache_[i]);
        }
        
        return grad;
    }
    
    std::string name() const override { return "CrossEntropyLoss"; }
};

// ============================================================================
// Binary Cross-Entropy Loss
// ============================================================================

/**
 * @brief Binary Cross-Entropy loss
 * 
 * L = -(1/n) Σ [target * log(pred) + (1-target) * log(1-pred)]
 * 
 * For binary classification (two classes, e.g., cat vs dog).
 * Assumes predictions are sigmoid outputs in range (0, 1).
 */
class BCELoss : public Loss {
public:
    float forward(const Tensor& prediction, const Tensor& target) override {
        if (!prediction.same_shape(target)) {
            throw std::invalid_argument("Prediction and target must have same shape");
        }
        
        prediction_cache_ = prediction;
        target_cache_ = target;
        
        float sum = 0.0f;
        for (size_t i = 0; i < prediction.size(); ++i) {
            float p = std::max(prediction[i], 1e-7f);
            p = std::min(p, 1.0f - 1e-7f);
            
            sum -= target[i] * std::log(p) + 
                   (1.0f - target[i]) * std::log(1.0f - p);
        }
        
        return sum / static_cast<float>(prediction.size());
    }
    
    Tensor backward() override {
        Tensor grad(prediction_cache_.shape());
        float scale = 1.0f / static_cast<float>(prediction_cache_.size());
        
        for (size_t i = 0; i < prediction_cache_.size(); ++i) {
            float p = std::max(prediction_cache_[i], 1e-7f);
            p = std::min(p, 1.0f - 1e-7f);
            float t = target_cache_[i];
            
            // ∂L/∂p = -t/p + (1-t)/(1-p)
            grad[i] = scale * (-t / p + (1.0f - t) / (1.0f - p));
        }
        
        return grad;
    }
    
    std::string name() const override { return "BCELoss"; }
};

// ============================================================================
// Huber Loss (smooth L1)
// ============================================================================

/**
 * @brief Huber Loss (Smooth L1)
 * 
 * Combines MSE for small errors and L1 for large errors.
 * Less sensitive to outliers than MSE.
 * 
 * L = 0.5 * (pred - target)² if |pred - target| < delta
 *     delta * |pred - target| - 0.5 * delta² otherwise
 */
class HuberLoss : public Loss {
public:
    explicit HuberLoss(float delta = 1.0f) : delta_(delta) {}
    
    float forward(const Tensor& prediction, const Tensor& target) override {
        if (!prediction.same_shape(target)) {
            throw std::invalid_argument("Prediction and target must have same shape");
        }
        
        prediction_cache_ = prediction;
        target_cache_ = target;
        
        float sum = 0.0f;
        for (size_t i = 0; i < prediction.size(); ++i) {
            float diff = std::abs(prediction[i] - target[i]);
            if (diff < delta_) {
                sum += 0.5f * diff * diff;
            } else {
                sum += delta_ * diff - 0.5f * delta_ * delta_;
            }
        }
        
        return sum / static_cast<float>(prediction.size());
    }
    
    Tensor backward() override {
        Tensor grad(prediction_cache_.shape());
        float scale = 1.0f / static_cast<float>(prediction_cache_.size());
        
        for (size_t i = 0; i < prediction_cache_.size(); ++i) {
            float diff = prediction_cache_[i] - target_cache_[i];
            if (std::abs(diff) < delta_) {
                grad[i] = scale * diff;
            } else {
                grad[i] = scale * delta_ * (diff > 0 ? 1.0f : -1.0f);
            }
        }
        
        return grad;
    }
    
    std::string name() const override { return "HuberLoss"; }

private:
    float delta_;
};

// ============================================================================
// Factory function
// ============================================================================

/**
 * @brief Create a loss function by name
 */
inline LossPtr create_loss(const std::string& loss_name) {
    if (loss_name == "mse" || loss_name == "mean_squared_error") {
        return std::make_unique<MSELoss>();
    } else if (loss_name == "cross_entropy" || loss_name == "categorical_crossentropy") {
        return std::make_unique<CrossEntropyLoss>();
    } else if (loss_name == "bce" || loss_name == "binary_crossentropy") {
        return std::make_unique<BCELoss>();
    } else if (loss_name == "huber") {
        return std::make_unique<HuberLoss>();
    } else {
        throw std::invalid_argument("Unknown loss function: " + loss_name);
    }
}

} // namespace nn

#endif // NN_LOSS_HPP
