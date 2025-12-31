#ifndef NN_TENSOR_HPP
#define NN_TENSOR_HPP

/**
 * @file tensor.hpp
 * @brief Multi-dimensional tensor class for neural network computations
 * 
 * This file defines the Tensor class, which is the fundamental data structure
 * for storing and manipulating multi-dimensional arrays of floating-point numbers.
 * It supports common operations needed for neural networks: element-wise operations,
 * matrix multiplication, reshaping, and broadcasting.
 * 
 * @author [Your Name]
 * @date 2024
 */

#include <vector>
#include <initializer_list>
#include <stdexcept>
#include <cmath>
#include <random>
#include <algorithm>
#include <numeric>
#include <functional>
#include <iostream>
#include <iomanip>
#include <cassert>
#include <string>
#include <sstream>

namespace nn {

/**
 * @brief Multi-dimensional tensor class
 * 
 * The Tensor class provides a flexible container for n-dimensional arrays.
 * It uses row-major ordering (C-style) for memory layout.
 * 
 * Key concepts for C++ beginners:
 * - std::vector: A dynamic array that can grow/shrink at runtime
 * - Templates: Allow the same code to work with different data types
 * - Operator overloading: Custom behavior for operators like +, -, *, []
 * 
 * Example usage:
 * @code
 *   Tensor a({2, 3});           // 2x3 matrix filled with zeros
 *   Tensor b({2, 3}, 1.0f);     // 2x3 matrix filled with ones
 *   Tensor c = a + b;           // Element-wise addition
 *   float val = c.at({0, 1});   // Access element at row 0, col 1
 * @endcode
 */
class Tensor {
public:
    // ========================================================================
    // Type Aliases (make code more readable)
    // ========================================================================
    
    /** @brief Type used for tensor dimensions */
    using Shape = std::vector<size_t>;
    
    /** @brief Type used for tensor data storage */
    using DataType = float;
    
    // ========================================================================
    // Constructors and Destructor
    // ========================================================================
    
    /**
     * @brief Default constructor - creates an empty tensor
     * 
     * In C++, the default constructor is called when you create an object
     * without any arguments: Tensor t;
     */
    Tensor() = default;
    
    /**
     * @brief Create a tensor with given shape, filled with a value
     * 
     * @param shape Dimensions of the tensor (e.g., {batch, height, width, channels})
     * @param fill_value Value to fill the tensor with (default: 0)
     * 
     * Example: Tensor({2, 3, 4}) creates a 2x3x4 tensor with 24 elements
     */
    explicit Tensor(const Shape& shape, DataType fill_value = 0.0f);
    
    /**
     * @brief Create a tensor from a flat data vector with given shape
     * 
     * @param shape Dimensions of the tensor
     * @param data Flat vector of data (must match total size)
     * @throws std::invalid_argument if data size doesn't match shape
     */
    Tensor(const Shape& shape, const std::vector<DataType>& data);
    
    /**
     * @brief Create a 1D tensor from an initializer list
     * 
     * Allows convenient syntax: Tensor t = {1.0f, 2.0f, 3.0f};
     */
    Tensor(std::initializer_list<DataType> data);
    
    /**
     * @brief Create a 2D tensor from nested initializer lists
     * 
     * Allows matrix-like syntax:
     * Tensor t = {{1, 2, 3}, {4, 5, 6}};
     */
    Tensor(std::initializer_list<std::initializer_list<DataType>> data);
    
    // Copy constructor and assignment (Rule of Five in C++)
    Tensor(const Tensor& other) = default;
    Tensor& operator=(const Tensor& other) = default;
    
    // Move constructor and assignment (efficient for temporary objects)
    Tensor(Tensor&& other) noexcept = default;
    Tensor& operator=(Tensor&& other) noexcept = default;
    
    ~Tensor() = default;
    
    // ========================================================================
    // Static Factory Methods (Alternative ways to create tensors)
    // ========================================================================
    
    /**
     * @brief Create a tensor filled with zeros
     * @param shape Dimensions of the tensor
     */
    static Tensor zeros(const Shape& shape);
    
    /**
     * @brief Create a tensor filled with ones
     * @param shape Dimensions of the tensor
     */
    static Tensor ones(const Shape& shape);
    
    /**
     * @brief Create a tensor with random values from uniform distribution
     * @param shape Dimensions of the tensor
     * @param min Minimum value (inclusive)
     * @param max Maximum value (exclusive)
     */
    static Tensor random_uniform(const Shape& shape, 
                                  DataType min = 0.0f, 
                                  DataType max = 1.0f);
    
    /**
     * @brief Create a tensor with random values from normal distribution
     * @param shape Dimensions of the tensor
     * @param mean Mean of the distribution
     * @param std Standard deviation
     */
    static Tensor random_normal(const Shape& shape,
                                 DataType mean = 0.0f,
                                 DataType std = 1.0f);
    
    /**
     * @brief Xavier/Glorot initialization (good for sigmoid/tanh)
     * @param shape Dimensions (typically {fan_in, fan_out})
     * 
     * This initialization helps with training deep networks by keeping
     * the variance of activations roughly the same across layers.
     */
    static Tensor xavier_uniform(const Shape& shape);
    
    /**
     * @brief He/Kaiming initialization (good for ReLU)
     * @param shape Dimensions (typically {fan_in, fan_out})
     */
    static Tensor he_uniform(const Shape& shape);
    
    /**
     * @brief Create an identity matrix
     * @param size Size of the square matrix
     */
    static Tensor eye(size_t size);
    
    // ========================================================================
    // Shape and Size Operations
    // ========================================================================
    
    /** @brief Get the shape of the tensor */
    const Shape& shape() const noexcept { return shape_; }
    
    /** @brief Get the number of dimensions (rank) */
    size_t ndim() const noexcept { return shape_.size(); }
    
    /** @brief Get total number of elements */
    size_t size() const noexcept { return data_.size(); }
    
    /** @brief Check if tensor is empty */
    bool empty() const noexcept { return data_.empty(); }
    
    /** @brief Get size of specific dimension */
    size_t dim(size_t index) const;
    
    /**
     * @brief Reshape the tensor (must have same total size)
     * @param new_shape New dimensions
     * @return New tensor with reshaped view
     * @throws std::invalid_argument if sizes don't match
     * 
     * Note: This creates a copy. A more advanced implementation might
     * use views/strides to avoid copying.
     */
    Tensor reshape(const Shape& new_shape) const;
    
    /**
     * @brief Flatten to 1D tensor
     * @return 1D tensor with same data
     */
    Tensor flatten() const;
    
    /**
     * @brief Transpose a 2D matrix
     * @return Transposed matrix
     * @throws std::invalid_argument if not 2D
     */
    Tensor transpose() const;
    
    /**
     * @brief Transpose with specified axes
     * @param axes New order of axes
     * @return Transposed tensor
     */
    Tensor transpose(const std::vector<size_t>& axes) const;
    
    // ========================================================================
    // Element Access
    // ========================================================================
    
    /**
     * @brief Access element at given indices (with bounds checking)
     * @param indices Multi-dimensional index
     * @return Reference to element
     * @throws std::out_of_range if indices are out of bounds
     */
    DataType& at(const std::vector<size_t>& indices);
    const DataType& at(const std::vector<size_t>& indices) const;
    
    /**
     * @brief Access element with single flat index
     * @param index Flat index (0 to size()-1)
     */
    DataType& operator[](size_t index) { return data_[index]; }
    const DataType& operator[](size_t index) const { return data_[index]; }
    
    /**
     * @brief Access raw data pointer (for performance-critical code)
     * 
     * Be careful with raw pointers! They bypass bounds checking.
     */
    DataType* data() noexcept { return data_.data(); }
    const DataType* data() const noexcept { return data_.data(); }
    
    /**
     * @brief Get underlying data vector
     */
    std::vector<DataType>& data_vector() noexcept { return data_; }
    const std::vector<DataType>& data_vector() const noexcept { return data_; }
    
    // ========================================================================
    // Arithmetic Operations (Element-wise)
    // ========================================================================
    
    // Tensor-Tensor operations
    Tensor operator+(const Tensor& other) const;
    Tensor operator-(const Tensor& other) const;
    Tensor operator*(const Tensor& other) const;  // Element-wise multiplication
    Tensor operator/(const Tensor& other) const;
    
    // In-place operations (more efficient, modifies this tensor)
    Tensor& operator+=(const Tensor& other);
    Tensor& operator-=(const Tensor& other);
    Tensor& operator*=(const Tensor& other);
    Tensor& operator/=(const Tensor& other);
    
    // Scalar operations
    Tensor operator+(DataType scalar) const;
    Tensor operator-(DataType scalar) const;
    Tensor operator*(DataType scalar) const;
    Tensor operator/(DataType scalar) const;
    
    Tensor& operator+=(DataType scalar);
    Tensor& operator-=(DataType scalar);
    Tensor& operator*=(DataType scalar);
    Tensor& operator/=(DataType scalar);
    
    // Unary minus
    Tensor operator-() const;
    
    // ========================================================================
    // Matrix Operations
    // ========================================================================
    
    /**
     * @brief Matrix multiplication
     * @param other Right-hand matrix
     * @return Result of matrix multiplication
     * @throws std::invalid_argument if dimensions don't match
     * 
     * For 2D tensors A (m x n) and B (n x p), returns C (m x p)
     * where C[i][j] = sum(A[i][k] * B[k][j]) for k = 0 to n-1
     */
    Tensor matmul(const Tensor& other) const;
    
    /**
     * @brief Dot product (for 1D tensors)
     * @param other Another 1D tensor of same size
     * @return Scalar result
     */
    DataType dot(const Tensor& other) const;
    
    // ========================================================================
    // Reduction Operations
    // ========================================================================
    
    /** @brief Sum of all elements */
    DataType sum() const;
    
    /** @brief Sum along specified axis */
    Tensor sum(size_t axis, bool keepdims = false) const;
    
    /** @brief Mean of all elements */
    DataType mean() const;
    
    /** @brief Mean along specified axis */
    Tensor mean(size_t axis, bool keepdims = false) const;
    
    /** @brief Maximum element */
    DataType max() const;
    
    /** @brief Maximum along specified axis */
    Tensor max(size_t axis, bool keepdims = false) const;
    
    /** @brief Index of maximum element along axis */
    Tensor argmax(size_t axis) const;
    
    /** @brief Minimum element */
    DataType min() const;
    
    // ========================================================================
    // Mathematical Functions (Element-wise)
    // ========================================================================
    
    /** @brief Apply function to each element */
    Tensor apply(std::function<DataType(DataType)> func) const;
    
    /** @brief Element-wise exponential */
    Tensor exp() const;
    
    /** @brief Element-wise natural logarithm */
    Tensor log() const;
    
    /** @brief Element-wise power */
    Tensor pow(DataType exponent) const;
    
    /** @brief Element-wise square root */
    Tensor sqrt() const;
    
    /** @brief Element-wise absolute value */
    Tensor abs() const;
    
    /** @brief Clip values to range [min_val, max_val] */
    Tensor clip(DataType min_val, DataType max_val) const;
    
    // ========================================================================
    // Comparison Operations
    // ========================================================================
    
    /** @brief Check if tensors have same shape and values */
    bool equals(const Tensor& other, DataType tolerance = 1e-6f) const;
    
    /** @brief Check if shapes match */
    bool same_shape(const Tensor& other) const;
    
    // ========================================================================
    // Utility Functions
    // ========================================================================
    
    /** @brief Print tensor to stream */
    void print(std::ostream& os = std::cout, int precision = 4) const;
    
    /** @brief Get string representation */
    std::string to_string(int precision = 4) const;
    
    /** @brief Get shape as string (e.g., "(2, 3, 4)") */
    std::string shape_string() const;
    
    /** @brief Fill all elements with a value */
    void fill(DataType value);
    
    /** @brief Create a deep copy */
    Tensor clone() const;

private:
    // ========================================================================
    // Private Helper Methods
    // ========================================================================
    
    /**
     * @brief Convert multi-dimensional indices to flat index
     * 
     * For a 2x3 matrix, index (1, 2) becomes 1*3 + 2 = 5
     * This is row-major ordering.
     */
    size_t compute_flat_index(const std::vector<size_t>& indices) const;
    
    /**
     * @brief Compute total size from shape
     */
    static size_t compute_size(const Shape& shape);
    
    /**
     * @brief Validate indices are within bounds
     */
    void validate_indices(const std::vector<size_t>& indices) const;
    
    /**
     * @brief Check if shapes are compatible for broadcasting
     */
    static bool broadcast_compatible(const Shape& a, const Shape& b);
    
    // ========================================================================
    // Member Variables
    // ========================================================================
    
    Shape shape_;                    ///< Dimensions of the tensor
    std::vector<DataType> data_;     ///< Actual data storage (row-major)
};

// ============================================================================
// Non-member operator overloads (for scalar on left side)
// ============================================================================

inline Tensor operator+(Tensor::DataType scalar, const Tensor& t) { return t + scalar; }
inline Tensor operator*(Tensor::DataType scalar, const Tensor& t) { return t * scalar; }

// ============================================================================
// Stream output operator
// ============================================================================

std::ostream& operator<<(std::ostream& os, const Tensor& t);

} // namespace nn

#endif // NN_TENSOR_HPP
