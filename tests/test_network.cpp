/**
 * @file test_network.cpp
 * @brief Unit tests for the Network class
 */

#include "nn/network.hpp"
#include <iostream>
#include <cmath>

using namespace nn;

int tests_passed = 0;
int tests_failed = 0;

#define TEST(name) void test_##name()
#define RUN_TEST(name) do { \
    std::cout << "Running " << #name << "... "; \
    try { \
        test_##name(); \
        std::cout << "PASSED\n"; \
        tests_passed++; \
    } catch (const std::exception& e) { \
        std::cout << "FAILED: " << e.what() << "\n"; \
        tests_failed++; \
    } \
} while(0)

#define ASSERT_TRUE(cond) do { \
    if (!(cond)) throw std::runtime_error("Assertion failed: " #cond); \
} while(0)

#define ASSERT_EQ(a, b) do { \
    if ((a) != (b)) throw std::runtime_error("Assertion failed: " #a " == " #b); \
} while(0)

#define ASSERT_NEAR(a, b, tol) do { \
    if (std::abs((a) - (b)) > (tol)) \
        throw std::runtime_error("Assertion failed: " #a " ≈ " #b); \
} while(0)

// ============================================================================
// Network Building Tests
// ============================================================================

TEST(network_add_layers) {
    Network net;
    net.add<Dense>(10, 20);
    net.add<ReLU>();
    net.add<Dense>(20, 5);
    
    ASSERT_EQ(net.num_layers(), 3u);
}

TEST(network_forward) {
    Network net;
    net.add<Dense>(10, 20);
    net.add<ReLU>();
    net.add<Dense>(20, 5);
    
    Tensor input = Tensor::random_uniform({4, 10});
    Tensor output = net.forward(input);
    
    ASSERT_EQ(output.dim(0), 4u);
    ASSERT_EQ(output.dim(1), 5u);
}

TEST(network_parameters) {
    Network net;
    net.add<Dense>(10, 20);   // 10*20 + 20 = 220
    net.add<ReLU>();          // 0
    net.add<Dense>(20, 5);    // 20*5 + 5 = 105
    
    size_t expected = 220 + 105;
    ASSERT_EQ(net.num_parameters(), expected);
}

TEST(network_compile) {
    Network net;
    net.add<Dense>(10, 5);
    net.add<Softmax>();
    
    net.compile("cross_entropy", "adam", 0.001f);
    
    // Should not throw
    Tensor x = Tensor::random_uniform({2, 10});
    Tensor y = Tensor::zeros({2, 5});
    y.at({0, 0}) = 1.0f;
    y.at({1, 1}) = 1.0f;
    
    // Forward and backward should work
    Tensor output = net.forward(x);
    ASSERT_EQ(output.dim(1), 5u);
}

// ============================================================================
// Training Tests
// ============================================================================

TEST(network_train_xor) {
    // XOR is a classic test: non-linearly separable, requires hidden layer
    Network net;
    net.add<Dense>(2, 8, true, "he");
    net.add<ReLU>();
    net.add<Dense>(8, 1, true, "xavier");
    net.add<Sigmoid>();
    
    net.compile("mse", "adam", 0.1f);
    
    // XOR training data
    // 0 XOR 0 = 0
    // 0 XOR 1 = 1
    // 1 XOR 0 = 1
    // 1 XOR 1 = 0
    
    Tensor X(Tensor::Shape{4, 2});
    X.at({0, 0}) = 0.0f; X.at({0, 1}) = 0.0f;
    X.at({1, 0}) = 0.0f; X.at({1, 1}) = 1.0f;
    X.at({2, 0}) = 1.0f; X.at({2, 1}) = 0.0f;
    X.at({3, 0}) = 1.0f; X.at({3, 1}) = 1.0f;
    
    Tensor y(Tensor::Shape{4, 1});
    y.at({0, 0}) = 0.0f;
    y.at({1, 0}) = 1.0f;
    y.at({2, 0}) = 1.0f;
    y.at({3, 0}) = 0.0f;
    
    // Train for a few epochs
    auto history = net.fit(X, y, 4, 500, 0.0f, false);
    
    // Test predictions
    Tensor output = net.predict(X);
    
    // Check that network learned XOR
    // Allow some tolerance since we're using a simple network
    ASSERT_TRUE(output.at({0, 0}) < 0.3f);  // 0 XOR 0 = 0
    ASSERT_TRUE(output.at({1, 0}) > 0.7f);  // 0 XOR 1 = 1
    ASSERT_TRUE(output.at({2, 0}) > 0.7f);  // 1 XOR 0 = 1
    ASSERT_TRUE(output.at({3, 0}) < 0.3f);  // 1 XOR 1 = 0
    
    // Check that loss decreased
    ASSERT_TRUE(history.train_loss.back() < history.train_loss.front());
}

TEST(network_classification) {
    // Simple 3-class classification
    Network net;
    net.add<Dense>(4, 8);
    net.add<ReLU>();
    net.add<Dense>(8, 3);
    net.add<Softmax>();
    
    net.compile("cross_entropy", "adam", 0.01f);
    
    // Create simple linearly separable data
    size_t samples_per_class = 20;
    Tensor X(Tensor::Shape{3 * samples_per_class, 4});
    Tensor y(Tensor::Shape{3 * samples_per_class, 3}, 0.0f);
    
    for (size_t i = 0; i < samples_per_class; ++i) {
        // Class 0: features around [1, 0, 0, 0]
        X.at({i, 0}) = 0.8f + 0.2f * static_cast<float>(i % 10) / 10.0f;
        X.at({i, 1}) = 0.2f * static_cast<float>(i % 10) / 10.0f;
        X.at({i, 2}) = 0.1f;
        X.at({i, 3}) = 0.1f;
        y.at({i, 0}) = 1.0f;
        
        // Class 1: features around [0, 1, 0, 0]
        size_t j = i + samples_per_class;
        X.at({j, 0}) = 0.2f * static_cast<float>(i % 10) / 10.0f;
        X.at({j, 1}) = 0.8f + 0.2f * static_cast<float>(i % 10) / 10.0f;
        X.at({j, 2}) = 0.1f;
        X.at({j, 3}) = 0.1f;
        y.at({j, 1}) = 1.0f;
        
        // Class 2: features around [0, 0, 1, 0]
        size_t k = i + 2 * samples_per_class;
        X.at({k, 0}) = 0.1f;
        X.at({k, 1}) = 0.1f;
        X.at({k, 2}) = 0.8f + 0.2f * static_cast<float>(i % 10) / 10.0f;
        X.at({k, 3}) = 0.2f * static_cast<float>(i % 10) / 10.0f;
        y.at({k, 2}) = 1.0f;
    }
    
    auto history = net.fit(X, y, 16, 100, 0.0f, false);
    
    // Evaluate
    auto [loss, acc] = net.evaluate(X, y);
    
    // Should achieve good accuracy on this simple task
    ASSERT_TRUE(acc > 0.8f);
}

TEST(network_validation_split) {
    Network net;
    net.add<Dense>(10, 5);
    net.add<Softmax>();
    
    net.compile("cross_entropy", "adam");
    
    // Create dummy data
    Tensor X = Tensor::random_uniform({100, 10});
    Tensor y = Tensor::zeros({100, 5});
    for (size_t i = 0; i < 100; ++i) {
        y.at({i, i % 5}) = 1.0f;
    }
    
    auto history = net.fit(X, y, 10, 5, 0.2f, false);
    
    // Should have validation metrics
    ASSERT_EQ(history.val_loss.size(), 5u);
    ASSERT_EQ(history.val_accuracy.size(), 5u);
}

// ============================================================================
// Optimizer Tests
// ============================================================================

TEST(sgd_optimizer) {
    Network net;
    net.add<Dense>(2, 4);
    net.add<ReLU>();
    net.add<Dense>(4, 1);
    net.add<Sigmoid>();
    
    net.compile("mse", "sgd", 0.5f);
    
    Tensor X(Tensor::Shape{4, 2});
    X.at({0, 0}) = 0.0f; X.at({0, 1}) = 0.0f;
    X.at({1, 0}) = 0.0f; X.at({1, 1}) = 1.0f;
    X.at({2, 0}) = 1.0f; X.at({2, 1}) = 0.0f;
    X.at({3, 0}) = 1.0f; X.at({3, 1}) = 1.0f;
    
    Tensor y(Tensor::Shape{4, 1});
    y.at({0, 0}) = 0.0f;
    y.at({1, 0}) = 1.0f;
    y.at({2, 0}) = 1.0f;
    y.at({3, 0}) = 0.0f;
    
    auto history = net.fit(X, y, 4, 100, 0.0f, false);
    
    // SGD should make progress
    ASSERT_TRUE(history.train_loss.back() < history.train_loss.front());
}

TEST(adam_optimizer) {
    Network net;
    net.add<Dense>(2, 4);
    net.add<ReLU>();
    net.add<Dense>(4, 1);
    net.add<Sigmoid>();
    
    net.compile("mse", "adam", 0.1f);
    
    Tensor X(Tensor::Shape{4, 2});
    X.at({0, 0}) = 0.0f; X.at({0, 1}) = 0.0f;
    X.at({1, 0}) = 0.0f; X.at({1, 1}) = 1.0f;
    X.at({2, 0}) = 1.0f; X.at({2, 1}) = 0.0f;
    X.at({3, 0}) = 1.0f; X.at({3, 1}) = 1.0f;
    
    Tensor y(Tensor::Shape{4, 1});
    y.at({0, 0}) = 0.0f;
    y.at({1, 0}) = 1.0f;
    y.at({2, 0}) = 1.0f;
    y.at({3, 0}) = 0.0f;
    
    auto history = net.fit(X, y, 4, 100, 0.0f, false);
    
    // Adam typically converges faster
    ASSERT_TRUE(history.train_loss.back() < 0.2f);
}

// ============================================================================
// CNN Tests
// ============================================================================

TEST(cnn_forward) {
    Network net;
    net.add<Conv2D>(1, 8, 3, 1, 1);  // (1, 28, 28) -> (8, 28, 28)
    net.add<ReLU>();
    net.add<MaxPool2D>(2);            // (8, 28, 28) -> (8, 14, 14)
    net.add<Conv2D>(8, 16, 3, 1, 1);  // (8, 14, 14) -> (16, 14, 14)
    net.add<ReLU>();
    net.add<MaxPool2D>(2);            // (16, 14, 14) -> (16, 7, 7)
    net.add<Flatten>();               // (16, 7, 7) -> (784,)
    net.add<Dense>(16 * 7 * 7, 10);
    net.add<Softmax>();
    
    Tensor input = Tensor::random_uniform({2, 1, 28, 28});
    Tensor output = net.forward(input);
    
    ASSERT_EQ(output.dim(0), 2u);
    ASSERT_EQ(output.dim(1), 10u);
    
    // Check softmax output sums to 1
    float sum = 0;
    for (size_t i = 0; i < 10; ++i) {
        sum += output.at({0, i});
    }
    ASSERT_NEAR(sum, 1.0f, 1e-5f);
}

// ============================================================================
// Main
// ============================================================================

int main() {
    std::cout << "\n========================================\n";
    std::cout << "     Network Tests\n";
    std::cout << "========================================\n\n";
    
    // Building
    RUN_TEST(network_add_layers);
    RUN_TEST(network_forward);
    RUN_TEST(network_parameters);
    RUN_TEST(network_compile);
    
    // Training
    RUN_TEST(network_train_xor);
    RUN_TEST(network_classification);
    RUN_TEST(network_validation_split);
    
    // Optimizers
    RUN_TEST(sgd_optimizer);
    RUN_TEST(adam_optimizer);
    
    // CNN
    RUN_TEST(cnn_forward);
    
    // Summary
    std::cout << "\n========================================\n";
    std::cout << "  Results: " << tests_passed << " passed, " 
              << tests_failed << " failed\n";
    std::cout << "========================================\n\n";
    
    return tests_failed > 0 ? 1 : 0;
}
