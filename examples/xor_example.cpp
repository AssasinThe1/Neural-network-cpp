/**
 * @file xor_example.cpp
 * @brief XOR Problem - A Classic Neural Network Demo
 * 
 * The XOR (exclusive or) problem is historically significant because
 * it was one of the first problems to demonstrate the need for
 * hidden layers in neural networks. A single layer perceptron cannot
 * solve XOR because it's not linearly separable.
 * 
 * XOR Truth Table:
 *   Input1 | Input2 | Output
 *   -------|--------|-------
 *      0   |   0    |   0
 *      0   |   1    |   1
 *      1   |   0    |   1
 *      1   |   1    |   0
 */

#include "nn/network.hpp"
#include <iostream>
#include <iomanip>

using namespace nn;

int main() {
    std::cout << "\n";
    std::cout << "========================================\n";
    std::cout << "     XOR Neural Network Demo\n";
    std::cout << "========================================\n\n";
    
    // ========================================================================
    // Step 1: Create Training Data
    // ========================================================================
    
    std::cout << "Creating XOR training data...\n\n";
    
    // Input: 4 samples, 2 features each
    Tensor X(Tensor::Shape{4, 2});
    X.at({0, 0}) = 0.0f; X.at({0, 1}) = 0.0f;  // [0, 0]
    X.at({1, 0}) = 0.0f; X.at({1, 1}) = 1.0f;  // [0, 1]
    X.at({2, 0}) = 1.0f; X.at({2, 1}) = 0.0f;  // [1, 0]
    X.at({3, 0}) = 1.0f; X.at({3, 1}) = 1.0f;  // [1, 1]
    
    // Target: 4 samples, 1 output each
    Tensor y(Tensor::Shape{4, 1});
    y.at({0, 0}) = 0.0f;  // 0 XOR 0 = 0
    y.at({1, 0}) = 1.0f;  // 0 XOR 1 = 1
    y.at({2, 0}) = 1.0f;  // 1 XOR 0 = 1
    y.at({3, 0}) = 0.0f;  // 1 XOR 1 = 0
    
    std::cout << "Training data:\n";
    std::cout << "  Input      | Target\n";
    std::cout << "  -----------|-------\n";
    for (size_t i = 0; i < 4; ++i) {
        std::cout << "  [" << X.at({i, 0}) << ", " << X.at({i, 1}) << "]   |   " 
                  << y.at({i, 0}) << "\n";
    }
    std::cout << "\n";
    
    // ========================================================================
    // Step 2: Build the Network
    // ========================================================================
    
    std::cout << "Building neural network...\n\n";
    
    Network net;
    
    // Input layer -> Hidden layer (2 -> 8 neurons)
    // Using He initialization (good for ReLU)
    net.add<Dense>(2, 8, true, "he");
    net.add<ReLU>();
    
    // Hidden layer -> Output layer (8 -> 1 neuron)
    // Using Xavier initialization (good for sigmoid)
    net.add<Dense>(8, 1, true, "xavier");
    net.add<Sigmoid>();  // Output between 0 and 1
    
    // Print network architecture
    net.summary();
    
    // ========================================================================
    // Step 3: Configure Training
    // ========================================================================
    
    // Use Mean Squared Error loss and Adam optimizer
    net.compile("mse", "adam", 0.1f);
    
    std::cout << "Training configuration:\n";
    std::cout << "  Loss function: MSE\n";
    std::cout << "  Optimizer: Adam (lr=0.1)\n";
    std::cout << "  Epochs: 1000\n";
    std::cout << "  Batch size: 4 (full batch)\n\n";
    
    // ========================================================================
    // Step 4: Train the Network
    // ========================================================================
    
    std::cout << "Training...\n\n";
    
    auto history = net.fit(X, y, 
                           4,      // batch_size (full batch for this small dataset)
                           1000,   // epochs
                           0.0f,   // validation_split
                           false); // verbose (we'll print our own progress)
    
    // Print selected epochs
    std::cout << "Training progress:\n";
    for (size_t e = 0; e < history.train_loss.size(); e += 200) {
        std::cout << "  Epoch " << std::setw(4) << (e + 1) 
                  << " - loss: " << std::fixed << std::setprecision(6) 
                  << history.train_loss[e] << "\n";
    }
    std::cout << "  Epoch " << history.train_loss.size() 
              << " - loss: " << history.train_loss.back() << "\n\n";
    
    // ========================================================================
    // Step 5: Test the Network
    // ========================================================================
    
    std::cout << "Testing predictions:\n";
    std::cout << "  Input      | Target | Prediction | Rounded\n";
    std::cout << "  -----------|--------|------------|--------\n";
    
    Tensor predictions = net.predict(X);
    
    int correct = 0;
    for (size_t i = 0; i < 4; ++i) {
        float pred = predictions.at({i, 0});
        int rounded = pred >= 0.5f ? 1 : 0;
        int target = static_cast<int>(y.at({i, 0}));
        
        std::cout << "  [" << X.at({i, 0}) << ", " << X.at({i, 1}) << "]   |   " 
                  << target << "    |   " 
                  << std::fixed << std::setprecision(4) << pred << "    |    "
                  << rounded << "\n";
        
        if (rounded == target) correct++;
    }
    
    std::cout << "\nAccuracy: " << correct << "/4 = " 
              << (100.0f * correct / 4.0f) << "%\n";
    
    // ========================================================================
    // Step 6: Test with New Inputs (Same as Training, but Demonstrates Usage)
    // ========================================================================
    
    std::cout << "\n========================================\n";
    std::cout << "  Demo Complete!\n";
    std::cout << "========================================\n";
    
    if (correct == 4) {
        std::cout << "\n✓ Network successfully learned XOR!\n";
        std::cout << "  This demonstrates that hidden layers can solve\n";
        std::cout << "  non-linearly separable problems.\n\n";
    } else {
        std::cout << "\n⚠ Network didn't perfectly learn XOR.\n";
        std::cout << "  Try running again (different random initialization)\n";
        std::cout << "  or increase training epochs.\n\n";
    }
    
    return correct == 4 ? 0 : 1;
}
