/**
 * @file test_layers.cpp
 * @brief Unit tests for neural network layers
 */

#include "nn/tensor.hpp"
#include "nn/layers/dense.hpp"
#include "nn/layers/activations.hpp"
#include "nn/layers/conv2d.hpp"
#include "nn/layers/pooling.hpp"
#include "nn/loss.hpp"

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
// Dense Layer Tests
// ============================================================================

TEST(dense_forward_shape) {
    Dense layer(10, 5);  // 10 inputs, 5 outputs
    
    // Single sample
    Tensor input = Tensor::random_uniform({10});
    Tensor output = layer.forward(input);
    ASSERT_EQ(output.ndim(), 1u);
    ASSERT_EQ(output.size(), 5u);
    
    // Batch
    Tensor batch_input = Tensor::random_uniform({32, 10});
    Tensor batch_output = layer.forward(batch_input);
    ASSERT_EQ(batch_output.ndim(), 2u);
    ASSERT_EQ(batch_output.dim(0), 32u);
    ASSERT_EQ(batch_output.dim(1), 5u);
}

TEST(dense_forward_computation) {
    Dense layer(2, 2, false, "zeros");  // No bias, zero weights
    
    // Set weights manually for testing
    layer.weights().at({0, 0}) = 1.0f;
    layer.weights().at({0, 1}) = 2.0f;
    layer.weights().at({1, 0}) = 3.0f;
    layer.weights().at({1, 1}) = 4.0f;
    
    Tensor input = {1.0f, 1.0f};  // [1, 1]
    Tensor output = layer.forward(input);
    
    // output = [1, 1] x [[1, 2], [3, 4]] = [1*1+1*3, 1*2+1*4] = [4, 6]
    ASSERT_NEAR(output[0], 4.0f, 1e-5f);
    ASSERT_NEAR(output[1], 6.0f, 1e-5f);
}

TEST(dense_backward) {
    Dense layer(3, 2);
    
    Tensor input = Tensor::random_uniform({4, 3});  // Batch of 4
    Tensor output = layer.forward(input);
    
    Tensor grad_output = Tensor::ones({4, 2});
    Tensor grad_input = layer.backward(grad_output);
    
    ASSERT_EQ(grad_input.dim(0), 4u);
    ASSERT_EQ(grad_input.dim(1), 3u);
    
    // Check gradients were computed
    auto grads = layer.gradients();
    ASSERT_EQ(grads.size(), 2u);  // weights and bias
}

TEST(dense_num_parameters) {
    Dense layer1(10, 5, true);   // With bias
    Dense layer2(10, 5, false);  // Without bias
    
    ASSERT_EQ(layer1.num_parameters(), 10u * 5u + 5u);  // 55
    ASSERT_EQ(layer2.num_parameters(), 10u * 5u);       // 50
}

// ============================================================================
// Activation Tests
// ============================================================================

TEST(relu_forward) {
    ReLU relu;
    
    Tensor input = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f};
    Tensor output = relu.forward(input);
    
    ASSERT_NEAR(output[0], 0.0f, 1e-6f);
    ASSERT_NEAR(output[1], 0.0f, 1e-6f);
    ASSERT_NEAR(output[2], 0.0f, 1e-6f);
    ASSERT_NEAR(output[3], 1.0f, 1e-6f);
    ASSERT_NEAR(output[4], 2.0f, 1e-6f);
}

TEST(relu_backward) {
    ReLU relu;
    
    Tensor input = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f};
    relu.forward(input);
    
    Tensor grad_output = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
    Tensor grad_input = relu.backward(grad_output);
    
    ASSERT_NEAR(grad_input[0], 0.0f, 1e-6f);  // Input was negative
    ASSERT_NEAR(grad_input[1], 0.0f, 1e-6f);
    ASSERT_NEAR(grad_input[2], 0.0f, 1e-6f);  // Input was zero
    ASSERT_NEAR(grad_input[3], 1.0f, 1e-6f);  // Input was positive
    ASSERT_NEAR(grad_input[4], 1.0f, 1e-6f);
}

TEST(sigmoid_forward) {
    Sigmoid sigmoid;
    
    Tensor input = {-10.0f, 0.0f, 10.0f};
    Tensor output = sigmoid.forward(input);
    
    ASSERT_TRUE(output[0] < 0.01f);  // Very close to 0
    ASSERT_NEAR(output[1], 0.5f, 1e-5f);
    ASSERT_TRUE(output[2] > 0.99f);  // Very close to 1
}

TEST(sigmoid_backward) {
    Sigmoid sigmoid;
    
    Tensor input = {0.0f};  // sigmoid(0) = 0.5
    sigmoid.forward(input);
    
    Tensor grad_output = {1.0f};
    Tensor grad_input = sigmoid.backward(grad_output);
    
    // dsigmoid/dx at x=0 is 0.5 * (1 - 0.5) = 0.25
    ASSERT_NEAR(grad_input[0], 0.25f, 1e-5f);
}

TEST(tanh_forward) {
    Tanh tanh_layer;
    
    Tensor input = {-10.0f, 0.0f, 10.0f};
    Tensor output = tanh_layer.forward(input);
    
    ASSERT_TRUE(output[0] < -0.99f);  // Close to -1
    ASSERT_NEAR(output[1], 0.0f, 1e-5f);
    ASSERT_TRUE(output[2] > 0.99f);   // Close to 1
}

TEST(softmax_forward) {
    Softmax softmax;
    
    // 1D case
    Tensor input1d = {1.0f, 2.0f, 3.0f};
    Tensor output1d = softmax.forward(input1d);
    
    // Should sum to 1
    ASSERT_NEAR(output1d.sum(), 1.0f, 1e-5f);
    // Larger input -> larger output
    ASSERT_TRUE(output1d[2] > output1d[1]);
    ASSERT_TRUE(output1d[1] > output1d[0]);
    
    // 2D case (batch)
    Tensor input2d = {{1.0f, 2.0f, 3.0f},
                      {3.0f, 2.0f, 1.0f}};
    Tensor output2d = softmax.forward(input2d);
    
    // Each row should sum to 1
    ASSERT_NEAR(output2d.at({0, 0}) + output2d.at({0, 1}) + output2d.at({0, 2}), 
                1.0f, 1e-5f);
    ASSERT_NEAR(output2d.at({1, 0}) + output2d.at({1, 1}) + output2d.at({1, 2}), 
                1.0f, 1e-5f);
}

TEST(leaky_relu) {
    LeakyReLU lrelu(0.1f);
    
    Tensor input = {-10.0f, -1.0f, 0.0f, 1.0f, 10.0f};
    Tensor output = lrelu.forward(input);
    
    ASSERT_NEAR(output[0], -1.0f, 1e-5f);   // -10 * 0.1
    ASSERT_NEAR(output[1], -0.1f, 1e-5f);   // -1 * 0.1
    ASSERT_NEAR(output[2], 0.0f, 1e-5f);
    ASSERT_NEAR(output[3], 1.0f, 1e-5f);
    ASSERT_NEAR(output[4], 10.0f, 1e-5f);
}

// ============================================================================
// Conv2D Tests
// ============================================================================

TEST(conv2d_forward_shape) {
    // Input: 1 channel, output: 4 channels, 3x3 kernel
    Conv2D conv(1, 4, 3);
    
    // Single image: (batch=1, channels=1, h=8, w=8)
    Tensor input = Tensor::random_uniform({1, 1, 8, 8});
    Tensor output = conv.forward(input);
    
    // Output shape: (1, 4, 6, 6)  -- (8-3+1 = 6)
    ASSERT_EQ(output.dim(0), 1u);
    ASSERT_EQ(output.dim(1), 4u);
    ASSERT_EQ(output.dim(2), 6u);
    ASSERT_EQ(output.dim(3), 6u);
}

TEST(conv2d_with_padding) {
    Conv2D conv(1, 4, 3, 1, 1);  // padding=1 for same output size
    
    Tensor input = Tensor::random_uniform({1, 1, 8, 8});
    Tensor output = conv.forward(input);
    
    // With padding=1 and kernel=3, output size = input size
    ASSERT_EQ(output.dim(2), 8u);
    ASSERT_EQ(output.dim(3), 8u);
}

TEST(conv2d_with_stride) {
    Conv2D conv(1, 4, 3, 2, 0);  // stride=2
    
    Tensor input = Tensor::random_uniform({1, 1, 8, 8});
    Tensor output = conv.forward(input);
    
    // Output: (8-3)/2 + 1 = 3
    ASSERT_EQ(output.dim(2), 3u);
    ASSERT_EQ(output.dim(3), 3u);
}

TEST(conv2d_backward) {
    Conv2D conv(1, 2, 3);
    
    Tensor input = Tensor::random_uniform({2, 1, 6, 6});  // Batch of 2
    Tensor output = conv.forward(input);
    
    Tensor grad_output = Tensor::ones(output.shape());
    Tensor grad_input = conv.backward(grad_output);
    
    ASSERT_TRUE(grad_input.same_shape(input));
}

// ============================================================================
// Pooling Tests
// ============================================================================

TEST(maxpool_forward) {
    MaxPool2D pool(2);  // 2x2 pooling
    
    // Create input with known values
    Tensor input(Tensor::Shape{1, 1, 4, 4});
    // Fill with increasing values
    for (size_t i = 0; i < 16; ++i) {
        input[i] = static_cast<float>(i);
    }
    
    Tensor output = pool.forward(input);
    
    ASSERT_EQ(output.dim(2), 2u);
    ASSERT_EQ(output.dim(3), 2u);
    
    // Should pick max of each 2x2 region
    ASSERT_NEAR(output.at({0, 0, 0, 0}), 5.0f, 1e-5f);  // max(0,1,4,5)
    ASSERT_NEAR(output.at({0, 0, 0, 1}), 7.0f, 1e-5f);  // max(2,3,6,7)
}

TEST(avgpool_forward) {
    AvgPool2D pool(2);
    
    Tensor input(Tensor::Shape{1, 1, 4, 4}, 4.0f);  // All 4s
    Tensor output = pool.forward(input);
    
    ASSERT_EQ(output.dim(2), 2u);
    ASSERT_EQ(output.dim(3), 2u);
    
    // Average of 4s should be 4
    ASSERT_NEAR(output.at({0, 0, 0, 0}), 4.0f, 1e-5f);
}

TEST(flatten_forward) {
    Flatten flatten;
    
    Tensor input = Tensor::random_uniform({4, 3, 8, 8});
    Tensor output = flatten.forward(input);
    
    ASSERT_EQ(output.dim(0), 4u);
    ASSERT_EQ(output.dim(1), 3u * 8u * 8u);
}

TEST(dropout) {
    Dropout dropout(0.5f);
    
    Tensor input = Tensor::ones({1000});
    
    // Training mode: should drop some values
    dropout.set_training(true);
    Tensor train_output = dropout.forward(input);
    
    // Count zeros (approximately half should be zero)
    int zeros = 0;
    for (size_t i = 0; i < train_output.size(); ++i) {
        if (train_output[i] == 0.0f) zeros++;
    }
    ASSERT_TRUE(zeros > 300 && zeros < 700);  // Should be around 500
    
    // Eval mode: no dropout
    dropout.set_training(false);
    Tensor eval_output = dropout.forward(input);
    ASSERT_NEAR(eval_output.sum(), 1000.0f, 1e-5f);  // All ones
}

// ============================================================================
// Loss Function Tests
// ============================================================================

TEST(mse_loss) {
    MSELoss mse;
    
    Tensor pred = {1.0f, 2.0f, 3.0f};
    Tensor target = {1.0f, 2.0f, 3.0f};
    
    float loss = mse.forward(pred, target);
    ASSERT_NEAR(loss, 0.0f, 1e-6f);
    
    // Test with difference
    Tensor pred2 = {2.0f, 3.0f, 4.0f};  // All +1 from target
    float loss2 = mse.forward(pred2, target);
    ASSERT_NEAR(loss2, 1.0f, 1e-6f);  // (1² + 1² + 1²) / 3 = 1
}

TEST(cross_entropy_loss) {
    CrossEntropyLoss ce;
    
    // Perfect prediction
    Tensor pred = {{1.0f, 0.0f, 0.0f},
                   {0.0f, 1.0f, 0.0f}};
    Tensor target = {{1.0f, 0.0f, 0.0f},
                     {0.0f, 1.0f, 0.0f}};
    
    float loss = ce.forward(pred, target);
    ASSERT_TRUE(loss < 0.01f);  // Should be very small
    
    // Check gradient shape
    Tensor grad = ce.backward();
    ASSERT_TRUE(grad.same_shape(pred));
}

TEST(bce_loss) {
    BCELoss bce;
    
    Tensor pred = {0.9f, 0.1f, 0.9f};
    Tensor target = {1.0f, 0.0f, 1.0f};
    
    float loss = bce.forward(pred, target);
    ASSERT_TRUE(loss < 0.5f);  // Should be reasonably small
}

// ============================================================================
// Integration Tests
// ============================================================================

TEST(dense_relu_dense) {
    // Simple forward pass through multiple layers
    Dense dense1(10, 20);
    ReLU relu;
    Dense dense2(20, 5);
    
    Tensor input = Tensor::random_uniform({4, 10});  // Batch of 4
    
    Tensor x = dense1.forward(input);
    x = relu.forward(x);
    Tensor output = dense2.forward(x);
    
    ASSERT_EQ(output.dim(0), 4u);
    ASSERT_EQ(output.dim(1), 5u);
}

// ============================================================================
// Main
// ============================================================================

int main() {
    std::cout << "\n========================================\n";
    std::cout << "     Layer Tests\n";
    std::cout << "========================================\n\n";
    
    // Dense layer
    RUN_TEST(dense_forward_shape);
    RUN_TEST(dense_forward_computation);
    RUN_TEST(dense_backward);
    RUN_TEST(dense_num_parameters);
    
    // Activations
    RUN_TEST(relu_forward);
    RUN_TEST(relu_backward);
    RUN_TEST(sigmoid_forward);
    RUN_TEST(sigmoid_backward);
    RUN_TEST(tanh_forward);
    RUN_TEST(softmax_forward);
    RUN_TEST(leaky_relu);
    
    // Conv2D
    RUN_TEST(conv2d_forward_shape);
    RUN_TEST(conv2d_with_padding);
    RUN_TEST(conv2d_with_stride);
    RUN_TEST(conv2d_backward);
    
    // Pooling
    RUN_TEST(maxpool_forward);
    RUN_TEST(avgpool_forward);
    RUN_TEST(flatten_forward);
    RUN_TEST(dropout);
    
    // Loss functions
    RUN_TEST(mse_loss);
    RUN_TEST(cross_entropy_loss);
    RUN_TEST(bce_loss);
    
    // Integration
    RUN_TEST(dense_relu_dense);
    
    // Summary
    std::cout << "\n========================================\n";
    std::cout << "  Results: " << tests_passed << " passed, " 
              << tests_failed << " failed\n";
    std::cout << "========================================\n\n";
    
    return tests_failed > 0 ? 1 : 0;
}
