#ifndef NN_NETWORK_HPP
#define NN_NETWORK_HPP

/**
 * @file network.hpp
 * @brief Neural Network container class
 * 
 * The Network class provides a high-level interface for building,
 * training, and using neural networks. It manages a sequence of
 * layers and handles forward/backward passes through the entire network.
 * 
 * Example usage:
 * @code
 *   Network net;
 *   net.add<Dense>(784, 128);
 *   net.add<ReLU>();
 *   net.add<Dense>(128, 10);
 *   net.add<Softmax>();
 *   
 *   net.compile("cross_entropy", "adam", 0.001f);
 *   net.fit(train_x, train_y, 32, 10);  // batch_size=32, epochs=10
 * @endcode
 */

#include "nn/layers/layer.hpp"
#include "nn/layers/dense.hpp"
#include "nn/layers/activations.hpp"
#include "nn/layers/conv2d.hpp"
#include "nn/layers/pooling.hpp"
#include "nn/loss.hpp"
#include "nn/optimizer.hpp"

#include <vector>
#include <memory>
#include <iostream>
#include <chrono>
#include <iomanip>

namespace nn {

/**
 * @brief Training history - stores metrics during training
 */
struct TrainingHistory {
    std::vector<float> train_loss;
    std::vector<float> train_accuracy;
    std::vector<float> val_loss;
    std::vector<float> val_accuracy;
    
    void clear() {
        train_loss.clear();
        train_accuracy.clear();
        val_loss.clear();
        val_accuracy.clear();
    }
};

/**
 * @brief Neural Network container
 * 
 * A sequential container that chains layers together.
 * Supports training with backpropagation and inference.
 */
class Network {
public:
    Network() = default;
    
    // Non-copyable (layers contain unique_ptrs)
    Network(const Network&) = delete;
    Network& operator=(const Network&) = delete;
    
    // Movable
    Network(Network&&) = default;
    Network& operator=(Network&&) = default;
    
    // ========================================================================
    // Building the Network
    // ========================================================================
    
    /**
     * @brief Add a layer to the network
     * 
     * Template version allows easy layer creation:
     *   net.add<Dense>(784, 128);
     *   net.add<ReLU>();
     * 
     * @tparam LayerType Type of layer to add
     * @tparam Args Constructor argument types
     * @param args Arguments to forward to the layer constructor
     * @return Reference to this network (for chaining)
     */
    template<typename LayerType, typename... Args>
    Network& add(Args&&... args) {
        layers_.push_back(std::make_unique<LayerType>(std::forward<Args>(args)...));
        return *this;
    }
    
    /**
     * @brief Add a pre-constructed layer
     * @param layer Unique pointer to the layer
     */
    Network& add(LayerPtr layer) {
        layers_.push_back(std::move(layer));
        return *this;
    }
    
    /**
     * @brief Configure the network for training
     * 
     * @param loss_name Name of loss function ("mse", "cross_entropy", etc.)
     * @param optimizer_name Name of optimizer ("sgd", "adam", etc.)
     * @param learning_rate Learning rate for optimizer
     */
    void compile(const std::string& loss_name,
                 const std::string& optimizer_name,
                 float learning_rate = 0.01f) {
        loss_ = create_loss(loss_name);
        optimizer_ = create_optimizer(optimizer_name, learning_rate);
        compiled_ = true;
    }
    
    /**
     * @brief Set custom loss and optimizer
     */
    void compile(LossPtr loss, OptimizerPtr optimizer) {
        loss_ = std::move(loss);
        optimizer_ = std::move(optimizer);
        compiled_ = true;
    }
    
    // ========================================================================
    // Forward and Backward Passes
    // ========================================================================
    
    /**
     * @brief Forward pass through all layers
     * 
     * @param input Input tensor
     * @return Output tensor from the last layer
     */
    Tensor forward(const Tensor& input) {
        Tensor current = input;
        for (auto& layer : layers_) {
            current = layer->forward(current);
        }
        return current;
    }
    
    /**
     * @brief Backward pass through all layers
     * 
     * Propagates gradients from output to input, updating
     * gradient buffers in each layer.
     * 
     * @param grad_output Gradient from the loss function
     */
    void backward(const Tensor& grad_output) {
        Tensor current_grad = grad_output;
        // Iterate through layers in reverse order
        for (auto it = layers_.rbegin(); it != layers_.rend(); ++it) {
            current_grad = (*it)->backward(current_grad);
        }
    }
    
    /**
     * @brief Update all parameters using the optimizer
     */
    void update_parameters() {
        if (!optimizer_) {
            throw std::runtime_error("Network not compiled - call compile() first");
        }
        
        std::vector<Tensor*> all_params;
        std::vector<Tensor*> all_grads;
        
        for (auto& layer : layers_) {
            auto params = layer->parameters();
            auto grads = layer->gradients();
            
            all_params.insert(all_params.end(), params.begin(), params.end());
            all_grads.insert(all_grads.end(), grads.begin(), grads.end());
        }
        
        optimizer_->step(all_params, all_grads);
    }
    
    // ========================================================================
    // Training
    // ========================================================================
    
    /**
     * @brief Train the network
     * 
     * @param X Training inputs, shape (num_samples, ...)
     * @param y Training targets, shape (num_samples, ...)
     * @param batch_size Number of samples per batch
     * @param epochs Number of passes through the training data
     * @param validation_split Fraction of data to use for validation
     * @param verbose Whether to print progress
     * @return Training history
     */
    TrainingHistory fit(const Tensor& X, const Tensor& y,
                        size_t batch_size = 32,
                        size_t epochs = 10,
                        float validation_split = 0.0f,
                        bool verbose = true) {
        
        if (!compiled_) {
            throw std::runtime_error("Network not compiled - call compile() first");
        }
        
        set_training(true);
        
        TrainingHistory history;
        size_t num_samples = X.dim(0);
        
        // Split into train and validation
        size_t val_samples = static_cast<size_t>(num_samples * validation_split);
        size_t train_samples = num_samples - val_samples;
        
        // Create index array for shuffling
        std::vector<size_t> indices(num_samples);
        std::iota(indices.begin(), indices.end(), 0);
        
        for (size_t epoch = 0; epoch < epochs; ++epoch) {
            auto epoch_start = std::chrono::high_resolution_clock::now();
            
            // Shuffle training indices
            std::shuffle(indices.begin(), indices.begin() + train_samples,
                        std::mt19937{std::random_device{}()});
            
            float epoch_loss = 0.0f;
            size_t correct = 0;
            size_t num_batches = (train_samples + batch_size - 1) / batch_size;
            
            for (size_t batch = 0; batch < num_batches; ++batch) {
                size_t start = batch * batch_size;
                size_t end = std::min(start + batch_size, train_samples);
                size_t actual_batch_size = end - start;
                
                // Create batch tensors
                auto [batch_X, batch_y] = create_batch(X, y, indices, start, end);
                
                // Forward pass
                Tensor output = forward(batch_X);
                
                // Compute loss
                float loss = loss_->forward(output, batch_y);
                epoch_loss += loss * static_cast<float>(actual_batch_size);
                
                // Compute accuracy (for classification)
                correct += compute_accuracy_count(output, batch_y);
                
                // Backward pass
                Tensor grad = loss_->backward();
                backward(grad);
                
                // Update weights
                update_parameters();
            }
            
            // Record training metrics
            epoch_loss /= static_cast<float>(train_samples);
            float epoch_acc = static_cast<float>(correct) / static_cast<float>(train_samples);
            history.train_loss.push_back(epoch_loss);
            history.train_accuracy.push_back(epoch_acc);
            
            // Validation
            float val_loss = 0.0f, val_acc = 0.0f;
            if (val_samples > 0) {
                set_training(false);
                auto [vl, va] = evaluate_internal(X, y, train_samples, num_samples);
                val_loss = vl;
                val_acc = va;
                history.val_loss.push_back(val_loss);
                history.val_accuracy.push_back(val_acc);
                set_training(true);
            }
            
            auto epoch_end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                epoch_end - epoch_start).count();
            
            if (verbose) {
                std::cout << "Epoch " << std::setw(3) << (epoch + 1) << "/" << epochs;
                std::cout << " - " << std::setw(4) << duration << "ms";
                std::cout << " - loss: " << std::fixed << std::setprecision(4) << epoch_loss;
                std::cout << " - acc: " << std::setprecision(4) << epoch_acc;
                if (val_samples > 0) {
                    std::cout << " - val_loss: " << val_loss;
                    std::cout << " - val_acc: " << val_acc;
                }
                std::cout << std::endl;
            }
        }
        
        set_training(false);
        return history;
    }
    
    /**
     * @brief Evaluate the network on test data
     * 
     * @param X Test inputs
     * @param y Test targets
     * @return Pair of (loss, accuracy)
     */
    std::pair<float, float> evaluate(const Tensor& X, const Tensor& y) {
        set_training(false);
        return evaluate_internal(X, y, 0, X.dim(0));
    }
    
    /**
     * @brief Make predictions
     * 
     * @param X Input data
     * @return Network output
     */
    Tensor predict(const Tensor& X) {
        set_training(false);
        return forward(X);
    }
    
    /**
     * @brief Predict class labels (argmax of output)
     */
    Tensor predict_classes(const Tensor& X) {
        Tensor output = predict(X);
        if (output.ndim() == 2) {
            return output.argmax(1);
        }
        return output;
    }
    
    // ========================================================================
    // Utilities
    // ========================================================================
    
    /**
     * @brief Set training mode for all layers
     */
    void set_training(bool training) {
        for (auto& layer : layers_) {
            layer->set_training(training);
        }
    }
    
    /**
     * @brief Get total number of trainable parameters
     */
    size_t num_parameters() const {
        size_t total = 0;
        for (const auto& layer : layers_) {
            total += layer->num_parameters();
        }
        return total;
    }
    
    /**
     * @brief Print network summary
     */
    void summary() const {
        std::cout << "\n========================================\n";
        std::cout << "     Network Summary\n";
        std::cout << "========================================\n";
        
        size_t total_params = 0;
        for (size_t i = 0; i < layers_.size(); ++i) {
            std::cout << "[" << std::setw(2) << i << "] " 
                      << layers_[i]->summary() << "\n";
            total_params += layers_[i]->num_parameters();
        }
        
        std::cout << "----------------------------------------\n";
        std::cout << "Total parameters: " << total_params << "\n";
        std::cout << "========================================\n\n";
    }
    
    /**
     * @brief Get number of layers
     */
    size_t num_layers() const { return layers_.size(); }
    
    /**
     * @brief Access a layer by index
     */
    Layer& layer(size_t index) { return *layers_.at(index); }
    const Layer& layer(size_t index) const { return *layers_.at(index); }

private:
    std::vector<LayerPtr> layers_;
    LossPtr loss_;
    OptimizerPtr optimizer_;
    bool compiled_ = false;
    
    /**
     * @brief Create a batch from data using indices
     */
    std::pair<Tensor, Tensor> create_batch(const Tensor& X, const Tensor& y,
                                           const std::vector<size_t>& indices,
                                           size_t start, size_t end) {
        size_t batch_size = end - start;
        
        // Get sample shape (everything except batch dimension)
        Tensor::Shape x_sample_shape(X.shape().begin() + 1, X.shape().end());
        Tensor::Shape y_sample_shape(y.shape().begin() + 1, y.shape().end());
        
        size_t x_sample_size = 1;
        for (size_t s : x_sample_shape) x_sample_size *= s;
        
        size_t y_sample_size = 1;
        for (size_t s : y_sample_shape) y_sample_size *= s;
        
        // Create batch shapes
        Tensor::Shape x_batch_shape = {batch_size};
        x_batch_shape.insert(x_batch_shape.end(), x_sample_shape.begin(), x_sample_shape.end());
        
        Tensor::Shape y_batch_shape = {batch_size};
        y_batch_shape.insert(y_batch_shape.end(), y_sample_shape.begin(), y_sample_shape.end());
        
        Tensor batch_X(x_batch_shape);
        Tensor batch_y(y_batch_shape);
        
        // Copy data
        for (size_t i = 0; i < batch_size; ++i) {
            size_t idx = indices[start + i];
            for (size_t j = 0; j < x_sample_size; ++j) {
                batch_X[i * x_sample_size + j] = X[idx * x_sample_size + j];
            }
            for (size_t j = 0; j < y_sample_size; ++j) {
                batch_y[i * y_sample_size + j] = y[idx * y_sample_size + j];
            }
        }
        
        return {batch_X, batch_y};
    }
    
    /**
     * @brief Count correct predictions (for accuracy)
     */
    size_t compute_accuracy_count(const Tensor& output, const Tensor& target) {
        if (output.ndim() != 2 || target.ndim() != 2) {
            return 0;  // Only compute for 2D classification
        }
        
        size_t batch_size = output.dim(0);
        size_t num_classes = output.dim(1);
        size_t correct = 0;
        
        for (size_t b = 0; b < batch_size; ++b) {
            // Find predicted class (argmax of output)
            size_t pred_class = 0;
            float max_val = output.at({b, 0});
            for (size_t c = 1; c < num_classes; ++c) {
                if (output.at({b, c}) > max_val) {
                    max_val = output.at({b, c});
                    pred_class = c;
                }
            }
            
            // Find true class (argmax of target)
            size_t true_class = 0;
            max_val = target.at({b, 0});
            for (size_t c = 1; c < num_classes; ++c) {
                if (target.at({b, c}) > max_val) {
                    max_val = target.at({b, c});
                    true_class = c;
                }
            }
            
            if (pred_class == true_class) {
                correct++;
            }
        }
        
        return correct;
    }
    
    /**
     * @brief Internal evaluation helper
     */
    std::pair<float, float> evaluate_internal(const Tensor& X, const Tensor& y,
                                               size_t start, size_t end) {
        size_t num_samples = end - start;
        std::vector<size_t> indices(num_samples);
        std::iota(indices.begin(), indices.end(), start);
        
        // Process in batches
        size_t batch_size = 64;
        float total_loss = 0.0f;
        size_t correct = 0;
        
        for (size_t i = 0; i < num_samples; i += batch_size) {
            size_t batch_end = std::min(i + batch_size, num_samples);
            size_t actual_batch = batch_end - i;
            
            auto [batch_X, batch_y] = create_batch(X, y, indices, i, batch_end);
            
            Tensor output = forward(batch_X);
            float loss = loss_->forward(output, batch_y);
            total_loss += loss * static_cast<float>(actual_batch);
            correct += compute_accuracy_count(output, batch_y);
        }
        
        float avg_loss = total_loss / static_cast<float>(num_samples);
        float accuracy = static_cast<float>(correct) / static_cast<float>(num_samples);
        
        return {avg_loss, accuracy};
    }
};

} // namespace nn

#endif // NN_NETWORK_HPP
