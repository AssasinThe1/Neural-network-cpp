/**
 * @file test_tensor.cpp
 * @brief Unit tests for the Tensor class
 * 
 * This file contains tests to verify that the Tensor class works correctly.
 * Testing is crucial for a portfolio project - it shows:
 * 1. You understand edge cases
 * 2. Your code is reliable
 * 3. You follow professional practices
 */

#include "nn/tensor.hpp"
#include <iostream>
#include <cassert>
#include <cmath>

using namespace nn;

// ============================================================================
// Test Utilities
// ============================================================================

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
// Construction Tests
// ============================================================================

TEST(default_constructor) {
    Tensor t;
    ASSERT_TRUE(t.empty());
    ASSERT_EQ(t.size(), 0u);
    ASSERT_EQ(t.ndim(), 0u);
}

TEST(shape_constructor) {
    Tensor t(Tensor::Shape{2, 3, 4});
    ASSERT_EQ(t.ndim(), 3u);
    ASSERT_EQ(t.dim(0), 2u);
    ASSERT_EQ(t.dim(1), 3u);
    ASSERT_EQ(t.dim(2), 4u);
    ASSERT_EQ(t.size(), 24u);
}

TEST(fill_constructor) {
    Tensor t(Tensor::Shape{3, 3}, 5.0f);
    ASSERT_EQ(t.size(), 9u);
    for (size_t i = 0; i < t.size(); ++i) {
        ASSERT_NEAR(t[i], 5.0f, 1e-6f);
    }
}

TEST(initializer_list_1d) {
    Tensor t = {1.0f, 2.0f, 3.0f, 4.0f};
    ASSERT_EQ(t.ndim(), 1u);
    ASSERT_EQ(t.size(), 4u);
    ASSERT_NEAR(t[0], 1.0f, 1e-6f);
    ASSERT_NEAR(t[3], 4.0f, 1e-6f);
}

TEST(initializer_list_2d) {
    Tensor t = {{1.0f, 2.0f, 3.0f},
                {4.0f, 5.0f, 6.0f}};
    ASSERT_EQ(t.ndim(), 2u);
    ASSERT_EQ(t.dim(0), 2u);
    ASSERT_EQ(t.dim(1), 3u);
    ASSERT_NEAR(t.at({0, 0}), 1.0f, 1e-6f);
    ASSERT_NEAR(t.at({1, 2}), 6.0f, 1e-6f);
}

// ============================================================================
// Factory Method Tests
// ============================================================================

TEST(zeros) {
    Tensor t = Tensor::zeros({2, 3});
    for (size_t i = 0; i < t.size(); ++i) {
        ASSERT_NEAR(t[i], 0.0f, 1e-6f);
    }
}

TEST(ones) {
    Tensor t = Tensor::ones({2, 3});
    for (size_t i = 0; i < t.size(); ++i) {
        ASSERT_NEAR(t[i], 1.0f, 1e-6f);
    }
}

TEST(random_uniform) {
    Tensor t = Tensor::random_uniform({100}, 0.0f, 1.0f);
    for (size_t i = 0; i < t.size(); ++i) {
        ASSERT_TRUE(t[i] >= 0.0f && t[i] < 1.0f);
    }
}

TEST(eye) {
    Tensor t = Tensor::eye(3);
    ASSERT_NEAR(t.at({0, 0}), 1.0f, 1e-6f);
    ASSERT_NEAR(t.at({1, 1}), 1.0f, 1e-6f);
    ASSERT_NEAR(t.at({2, 2}), 1.0f, 1e-6f);
    ASSERT_NEAR(t.at({0, 1}), 0.0f, 1e-6f);
    ASSERT_NEAR(t.at({1, 0}), 0.0f, 1e-6f);
}

// ============================================================================
// Shape Operation Tests
// ============================================================================

TEST(reshape) {
    Tensor t(Tensor::Shape{2, 3}, 0.0f);
    for (size_t i = 0; i < 6; ++i) t[i] = static_cast<float>(i);
    
    Tensor r = t.reshape(Tensor::Shape{3, 2});
    ASSERT_EQ(r.dim(0), 3u);
    ASSERT_EQ(r.dim(1), 2u);
    ASSERT_NEAR(r.at({0, 0}), 0.0f, 1e-6f);
    ASSERT_NEAR(r.at({2, 1}), 5.0f, 1e-6f);
}

TEST(flatten) {
    Tensor t(Tensor::Shape{2, 3}, 0.0f);
    Tensor f = t.flatten();
    ASSERT_EQ(f.ndim(), 1u);
    ASSERT_EQ(f.size(), 6u);
}

TEST(transpose_2d) {
    Tensor t = {{1.0f, 2.0f, 3.0f},
                {4.0f, 5.0f, 6.0f}};
    Tensor tr = t.transpose();
    ASSERT_EQ(tr.dim(0), 3u);
    ASSERT_EQ(tr.dim(1), 2u);
    ASSERT_NEAR(tr.at({0, 0}), 1.0f, 1e-6f);
    ASSERT_NEAR(tr.at({0, 1}), 4.0f, 1e-6f);
    ASSERT_NEAR(tr.at({2, 0}), 3.0f, 1e-6f);
    ASSERT_NEAR(tr.at({2, 1}), 6.0f, 1e-6f);
}

// ============================================================================
// Arithmetic Tests
// ============================================================================

TEST(addition) {
    Tensor a = {1.0f, 2.0f, 3.0f};
    Tensor b = {4.0f, 5.0f, 6.0f};
    Tensor c = a + b;
    ASSERT_NEAR(c[0], 5.0f, 1e-6f);
    ASSERT_NEAR(c[1], 7.0f, 1e-6f);
    ASSERT_NEAR(c[2], 9.0f, 1e-6f);
}

TEST(subtraction) {
    Tensor a = {4.0f, 5.0f, 6.0f};
    Tensor b = {1.0f, 2.0f, 3.0f};
    Tensor c = a - b;
    ASSERT_NEAR(c[0], 3.0f, 1e-6f);
    ASSERT_NEAR(c[1], 3.0f, 1e-6f);
    ASSERT_NEAR(c[2], 3.0f, 1e-6f);
}

TEST(element_wise_multiplication) {
    Tensor a = {2.0f, 3.0f, 4.0f};
    Tensor b = {5.0f, 6.0f, 7.0f};
    Tensor c = a * b;
    ASSERT_NEAR(c[0], 10.0f, 1e-6f);
    ASSERT_NEAR(c[1], 18.0f, 1e-6f);
    ASSERT_NEAR(c[2], 28.0f, 1e-6f);
}

TEST(scalar_operations) {
    Tensor a = {1.0f, 2.0f, 3.0f};
    
    Tensor b = a + 10.0f;
    ASSERT_NEAR(b[0], 11.0f, 1e-6f);
    
    Tensor c = a * 2.0f;
    ASSERT_NEAR(c[0], 2.0f, 1e-6f);
    ASSERT_NEAR(c[2], 6.0f, 1e-6f);
    
    Tensor d = 3.0f * a;  // Scalar on left
    ASSERT_NEAR(d[1], 6.0f, 1e-6f);
}

TEST(inplace_operations) {
    Tensor a = {1.0f, 2.0f, 3.0f};
    a += 1.0f;
    ASSERT_NEAR(a[0], 2.0f, 1e-6f);
    
    a *= 2.0f;
    ASSERT_NEAR(a[0], 4.0f, 1e-6f);
    ASSERT_NEAR(a[2], 8.0f, 1e-6f);
}

// ============================================================================
// Matrix Operation Tests
// ============================================================================

TEST(matmul_simple) {
    // [1, 2] x [5, 6]   = [1*5+2*7, 1*6+2*8]   = [19, 22]
    // [3, 4]   [7, 8]     [3*5+4*7, 3*6+4*8]     [43, 50]
    
    Tensor a = {{1.0f, 2.0f},
                {3.0f, 4.0f}};
    Tensor b = {{5.0f, 6.0f},
                {7.0f, 8.0f}};
    
    Tensor c = a.matmul(b);
    
    ASSERT_EQ(c.dim(0), 2u);
    ASSERT_EQ(c.dim(1), 2u);
    ASSERT_NEAR(c.at({0, 0}), 19.0f, 1e-6f);
    ASSERT_NEAR(c.at({0, 1}), 22.0f, 1e-6f);
    ASSERT_NEAR(c.at({1, 0}), 43.0f, 1e-6f);
    ASSERT_NEAR(c.at({1, 1}), 50.0f, 1e-6f);
}

TEST(matmul_different_shapes) {
    // (2, 3) x (3, 4) = (2, 4)
    Tensor a({2, 3}, 1.0f);
    Tensor b({3, 4}, 2.0f);
    
    Tensor c = a.matmul(b);
    
    ASSERT_EQ(c.dim(0), 2u);
    ASSERT_EQ(c.dim(1), 4u);
    // Each element should be 1*2 + 1*2 + 1*2 = 6
    ASSERT_NEAR(c.at({0, 0}), 6.0f, 1e-6f);
}

TEST(dot_product) {
    Tensor a = {1.0f, 2.0f, 3.0f};
    Tensor b = {4.0f, 5.0f, 6.0f};
    
    float dot = a.dot(b);
    // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
    ASSERT_NEAR(dot, 32.0f, 1e-6f);
}

// ============================================================================
// Reduction Tests
// ============================================================================

TEST(sum) {
    Tensor t = {1.0f, 2.0f, 3.0f, 4.0f};
    ASSERT_NEAR(t.sum(), 10.0f, 1e-6f);
}

TEST(sum_axis) {
    Tensor t = {{1.0f, 2.0f, 3.0f},
                {4.0f, 5.0f, 6.0f}};
    
    // Sum along axis 0 (columns): [5, 7, 9]
    Tensor s0 = t.sum(0);
    ASSERT_EQ(s0.size(), 3u);
    ASSERT_NEAR(s0[0], 5.0f, 1e-6f);
    ASSERT_NEAR(s0[1], 7.0f, 1e-6f);
    ASSERT_NEAR(s0[2], 9.0f, 1e-6f);
    
    // Sum along axis 1 (rows): [6, 15]
    Tensor s1 = t.sum(1);
    ASSERT_EQ(s1.size(), 2u);
    ASSERT_NEAR(s1[0], 6.0f, 1e-6f);
    ASSERT_NEAR(s1[1], 15.0f, 1e-6f);
}

TEST(mean) {
    Tensor t = {2.0f, 4.0f, 6.0f, 8.0f};
    ASSERT_NEAR(t.mean(), 5.0f, 1e-6f);
}

TEST(max_min) {
    Tensor t = {3.0f, 1.0f, 4.0f, 1.0f, 5.0f, 9.0f};
    ASSERT_NEAR(t.max(), 9.0f, 1e-6f);
    ASSERT_NEAR(t.min(), 1.0f, 1e-6f);
}

TEST(argmax) {
    Tensor t = {{1.0f, 3.0f, 2.0f},
                {4.0f, 2.0f, 5.0f}};
    
    // Argmax along axis 1 (find max in each row)
    Tensor am = t.argmax(1);
    ASSERT_EQ(am.size(), 2u);
    ASSERT_NEAR(am[0], 1.0f, 1e-6f);  // Index 1 (value 3)
    ASSERT_NEAR(am[1], 2.0f, 1e-6f);  // Index 2 (value 5)
}

// ============================================================================
// Mathematical Function Tests
// ============================================================================

TEST(exp_log) {
    Tensor t = {0.0f, 1.0f, 2.0f};
    
    Tensor e = t.exp();
    ASSERT_NEAR(e[0], 1.0f, 1e-5f);
    ASSERT_NEAR(e[1], std::exp(1.0f), 1e-5f);
    
    Tensor l = e.log();
    ASSERT_NEAR(l[0], 0.0f, 1e-5f);
    ASSERT_NEAR(l[1], 1.0f, 1e-5f);
}

TEST(sqrt_pow) {
    Tensor t = {4.0f, 9.0f, 16.0f};
    
    Tensor s = t.sqrt();
    ASSERT_NEAR(s[0], 2.0f, 1e-5f);
    ASSERT_NEAR(s[1], 3.0f, 1e-5f);
    
    Tensor p = t.pow(0.5f);
    ASSERT_NEAR(p[0], 2.0f, 1e-5f);
}

TEST(clip) {
    Tensor t = {-2.0f, 0.5f, 3.0f};
    Tensor c = t.clip(0.0f, 1.0f);
    ASSERT_NEAR(c[0], 0.0f, 1e-6f);
    ASSERT_NEAR(c[1], 0.5f, 1e-6f);
    ASSERT_NEAR(c[2], 1.0f, 1e-6f);
}

TEST(apply) {
    Tensor t = {1.0f, 2.0f, 3.0f};
    Tensor squared = t.apply([](float x) { return x * x; });
    ASSERT_NEAR(squared[0], 1.0f, 1e-6f);
    ASSERT_NEAR(squared[1], 4.0f, 1e-6f);
    ASSERT_NEAR(squared[2], 9.0f, 1e-6f);
}

// ============================================================================
// Comparison Tests
// ============================================================================

TEST(equals) {
    Tensor a = {1.0f, 2.0f, 3.0f};
    Tensor b = {1.0f, 2.0f, 3.0f};
    Tensor c = {1.0f, 2.0f, 3.1f};
    
    ASSERT_TRUE(a.equals(b));
    ASSERT_TRUE(!a.equals(c));
    ASSERT_TRUE(a.equals(c, 0.2f));  // With tolerance
}

TEST(same_shape) {
    Tensor a(Tensor::Shape{2, 3});
    Tensor b(Tensor::Shape{2, 3});
    Tensor c(Tensor::Shape{3, 2});
    
    ASSERT_TRUE(a.same_shape(b));
    ASSERT_TRUE(!a.same_shape(c));
}

// ============================================================================
// Main
// ============================================================================

int main() {
    std::cout << "\n========================================\n";
    std::cout << "     Tensor Tests\n";
    std::cout << "========================================\n\n";
    
    // Construction
    RUN_TEST(default_constructor);
    RUN_TEST(shape_constructor);
    RUN_TEST(fill_constructor);
    RUN_TEST(initializer_list_1d);
    RUN_TEST(initializer_list_2d);
    
    // Factory methods
    RUN_TEST(zeros);
    RUN_TEST(ones);
    RUN_TEST(random_uniform);
    RUN_TEST(eye);
    
    // Shape operations
    RUN_TEST(reshape);
    RUN_TEST(flatten);
    RUN_TEST(transpose_2d);
    
    // Arithmetic
    RUN_TEST(addition);
    RUN_TEST(subtraction);
    RUN_TEST(element_wise_multiplication);
    RUN_TEST(scalar_operations);
    RUN_TEST(inplace_operations);
    
    // Matrix operations
    RUN_TEST(matmul_simple);
    RUN_TEST(matmul_different_shapes);
    RUN_TEST(dot_product);
    
    // Reductions
    RUN_TEST(sum);
    RUN_TEST(sum_axis);
    RUN_TEST(mean);
    RUN_TEST(max_min);
    RUN_TEST(argmax);
    
    // Math functions
    RUN_TEST(exp_log);
    RUN_TEST(sqrt_pow);
    RUN_TEST(clip);
    RUN_TEST(apply);
    
    // Comparisons
    RUN_TEST(equals);
    RUN_TEST(same_shape);
    
    // Summary
    std::cout << "\n========================================\n";
    std::cout << "  Results: " << tests_passed << " passed, " 
              << tests_failed << " failed\n";
    std::cout << "========================================\n\n";
    
    return tests_failed > 0 ? 1 : 0;
}
