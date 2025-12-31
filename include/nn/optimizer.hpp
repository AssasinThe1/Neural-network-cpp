#ifndef NN_OPTIMIZER_HPP
#define NN_OPTIMIZER_HPP

/**
 * @file optimizer.hpp
 * @brief Optimization algorithms for neural network training
 * 
 * Optimizers update the network's parameters (weights, biases) based on
 * the computed gradients. The goal is to minimize the loss function.
 * 
 * Basic idea: new_weight = old_weight - learning_rate * gradient
 * 
 * Different optimizers use different strategies:
 * - SGD: Simple gradient descent
 * - Momentum: Accumulates velocity to escape local minima
 * - Adam: Adaptive learning rates per parameter
 */

#include "nn/tensor.hpp"
#include <vector>
#include <memory>
#include <cmath>
#include <unordered_map>

namespace nn {

/**
 * @brief Abstract base class for optimizers
 */
class Optimizer {
public:
    explicit Optimizer(float learning_rate = 0.01f) 
        : learning_rate_(learning_rate) {}
    
    virtual ~Optimizer() = default;
    
    /**
     * @brief Update parameters using their gradients
     * @param params Vector of pointers to parameter tensors
     * @param grads Vector of pointers to gradient tensors (same order as params)
     */
    virtual void step(const std::vector<Tensor*>& params,
                      const std::vector<Tensor*>& grads) = 0;
    
    /**
     * @brief Reset optimizer state (momentum, etc.)
     */
    virtual void reset() {}
    
    /**
     * @brief Get optimizer name
     */
    virtual std::string name() const = 0;
    
    /**
     * @brief Set learning rate
     */
    void set_learning_rate(float lr) { learning_rate_ = lr; }
    
    /**
     * @brief Get current learning rate
     */
    float learning_rate() const { return learning_rate_; }

protected:
    float learning_rate_;
};

using OptimizerPtr = std::unique_ptr<Optimizer>;

// ============================================================================
// Stochastic Gradient Descent (SGD)
// ============================================================================

/**
 * @brief Stochastic Gradient Descent optimizer
 * 
 * The simplest optimizer: w = w - lr * grad
 * 
 * With momentum: 
 *   v = momentum * v - lr * grad
 *   w = w + v
 * 
 * Momentum helps:
 * - Accelerate in consistent gradient directions
 * - Dampen oscillations
 * - Escape shallow local minima
 */
class SGD : public Optimizer {
public:
    /**
     * @param learning_rate Step size for updates
     * @param momentum Momentum factor (0 = no momentum)
     * @param weight_decay L2 regularization factor
     */
    SGD(float learning_rate = 0.01f, 
        float momentum = 0.0f,
        float weight_decay = 0.0f)
        : Optimizer(learning_rate)
        , momentum_(momentum)
        , weight_decay_(weight_decay)
    {}
    
    void step(const std::vector<Tensor*>& params,
              const std::vector<Tensor*>& grads) override {
        
        if (params.size() != grads.size()) {
            throw std::invalid_argument("Number of parameters and gradients must match");
        }
        
        // Initialize velocity buffers if needed
        if (velocities_.size() != params.size()) {
            velocities_.clear();
            for (const auto* p : params) {
                velocities_.push_back(Tensor::zeros(p->shape()));
            }
        }
        
        for (size_t i = 0; i < params.size(); ++i) {
            Tensor* param = params[i];
            const Tensor* grad = grads[i];
            
            // Apply weight decay (L2 regularization)
            // grad_wd = grad + weight_decay * param
            Tensor grad_with_decay = *grad;
            if (weight_decay_ > 0) {
                for (size_t j = 0; j < param->size(); ++j) {
                    grad_with_decay[j] += weight_decay_ * (*param)[j];
                }
            }
            
            if (momentum_ > 0) {
                // v = momentum * v + grad
                // param = param - lr * v
                for (size_t j = 0; j < param->size(); ++j) {
                    velocities_[i][j] = momentum_ * velocities_[i][j] + 
                                        grad_with_decay[j];
                    (*param)[j] -= learning_rate_ * velocities_[i][j];
                }
            } else {
                // Simple SGD: param = param - lr * grad
                for (size_t j = 0; j < param->size(); ++j) {
                    (*param)[j] -= learning_rate_ * grad_with_decay[j];
                }
            }
        }
    }
    
    void reset() override {
        velocities_.clear();
    }
    
    std::string name() const override { 
        return "SGD(lr=" + std::to_string(learning_rate_) + 
               ", momentum=" + std::to_string(momentum_) + ")";
    }

private:
    float momentum_;
    float weight_decay_;
    std::vector<Tensor> velocities_;
};

// ============================================================================
// Adam (Adaptive Moment Estimation)
// ============================================================================

/**
 * @brief Adam optimizer
 * 
 * Combines ideas from momentum and RMSprop:
 * - Maintains per-parameter learning rates
 * - Uses exponential moving averages of gradients (momentum)
 * - Uses exponential moving averages of squared gradients (adaptive lr)
 * 
 * Often works well out of the box with default parameters.
 * 
 * Algorithm:
 *   m = β1 * m + (1 - β1) * grad          (first moment estimate)
 *   v = β2 * v + (1 - β2) * grad²         (second moment estimate)
 *   m_hat = m / (1 - β1^t)                (bias correction)
 *   v_hat = v / (1 - β2^t)
 *   param = param - lr * m_hat / (√v_hat + ε)
 */
class Adam : public Optimizer {
public:
    /**
     * @param learning_rate Step size (default: 0.001, works well for most cases)
     * @param beta1 Exponential decay rate for first moment (default: 0.9)
     * @param beta2 Exponential decay rate for second moment (default: 0.999)
     * @param epsilon Small constant for numerical stability (default: 1e-8)
     * @param weight_decay L2 regularization (default: 0)
     */
    Adam(float learning_rate = 0.001f,
         float beta1 = 0.9f,
         float beta2 = 0.999f,
         float epsilon = 1e-8f,
         float weight_decay = 0.0f)
        : Optimizer(learning_rate)
        , beta1_(beta1)
        , beta2_(beta2)
        , epsilon_(epsilon)
        , weight_decay_(weight_decay)
        , t_(0)
    {}
    
    void step(const std::vector<Tensor*>& params,
              const std::vector<Tensor*>& grads) override {
        
        if (params.size() != grads.size()) {
            throw std::invalid_argument("Number of parameters and gradients must match");
        }
        
        // Initialize moment buffers if needed
        if (m_.size() != params.size()) {
            m_.clear();
            v_.clear();
            for (const auto* p : params) {
                m_.push_back(Tensor::zeros(p->shape()));
                v_.push_back(Tensor::zeros(p->shape()));
            }
        }
        
        t_++;
        
        // Bias correction factors
        float bias_correction1 = 1.0f - std::pow(beta1_, static_cast<float>(t_));
        float bias_correction2 = 1.0f - std::pow(beta2_, static_cast<float>(t_));
        
        for (size_t i = 0; i < params.size(); ++i) {
            Tensor* param = params[i];
            const Tensor* grad = grads[i];
            
            for (size_t j = 0; j < param->size(); ++j) {
                float g = (*grad)[j];
                
                // Add weight decay
                if (weight_decay_ > 0) {
                    g += weight_decay_ * (*param)[j];
                }
                
                // Update biased first moment estimate
                m_[i][j] = beta1_ * m_[i][j] + (1.0f - beta1_) * g;
                
                // Update biased second raw moment estimate
                v_[i][j] = beta2_ * v_[i][j] + (1.0f - beta2_) * g * g;
                
                // Bias-corrected estimates
                float m_hat = m_[i][j] / bias_correction1;
                float v_hat = v_[i][j] / bias_correction2;
                
                // Update parameter
                (*param)[j] -= learning_rate_ * m_hat / (std::sqrt(v_hat) + epsilon_);
            }
        }
    }
    
    void reset() override {
        m_.clear();
        v_.clear();
        t_ = 0;
    }
    
    std::string name() const override {
        return "Adam(lr=" + std::to_string(learning_rate_) + ")";
    }

private:
    float beta1_, beta2_, epsilon_, weight_decay_;
    size_t t_;                    // Time step (number of updates)
    std::vector<Tensor> m_;       // First moment estimates
    std::vector<Tensor> v_;       // Second moment estimates
};

// ============================================================================
// RMSprop
// ============================================================================

/**
 * @brief RMSprop optimizer
 * 
 * Adapts learning rate for each parameter based on the magnitude
 * of recent gradients. Divides learning rate by a running average
 * of gradient magnitudes.
 * 
 * Good for recurrent neural networks and non-stationary objectives.
 */
class RMSprop : public Optimizer {
public:
    RMSprop(float learning_rate = 0.01f,
            float alpha = 0.99f,
            float epsilon = 1e-8f)
        : Optimizer(learning_rate)
        , alpha_(alpha)
        , epsilon_(epsilon)
    {}
    
    void step(const std::vector<Tensor*>& params,
              const std::vector<Tensor*>& grads) override {
        
        if (params.size() != grads.size()) {
            throw std::invalid_argument("Number of parameters and gradients must match");
        }
        
        // Initialize squared gradient average buffers
        if (sq_avg_.size() != params.size()) {
            sq_avg_.clear();
            for (const auto* p : params) {
                sq_avg_.push_back(Tensor::zeros(p->shape()));
            }
        }
        
        for (size_t i = 0; i < params.size(); ++i) {
            Tensor* param = params[i];
            const Tensor* grad = grads[i];
            
            for (size_t j = 0; j < param->size(); ++j) {
                float g = (*grad)[j];
                
                // Update running average of squared gradients
                sq_avg_[i][j] = alpha_ * sq_avg_[i][j] + (1.0f - alpha_) * g * g;
                
                // Update parameter
                (*param)[j] -= learning_rate_ * g / (std::sqrt(sq_avg_[i][j]) + epsilon_);
            }
        }
    }
    
    void reset() override {
        sq_avg_.clear();
    }
    
    std::string name() const override {
        return "RMSprop(lr=" + std::to_string(learning_rate_) + ")";
    }

private:
    float alpha_, epsilon_;
    std::vector<Tensor> sq_avg_;
};

// ============================================================================
// Factory function
// ============================================================================

/**
 * @brief Create an optimizer by name
 */
inline OptimizerPtr create_optimizer(const std::string& name, float learning_rate = 0.01f) {
    if (name == "sgd") {
        return std::make_unique<SGD>(learning_rate);
    } else if (name == "sgd_momentum") {
        return std::make_unique<SGD>(learning_rate, 0.9f);
    } else if (name == "adam") {
        return std::make_unique<Adam>(learning_rate);
    } else if (name == "rmsprop") {
        return std::make_unique<RMSprop>(learning_rate);
    } else {
        throw std::invalid_argument("Unknown optimizer: " + name);
    }
}

} // namespace nn

#endif // NN_OPTIMIZER_HPP
