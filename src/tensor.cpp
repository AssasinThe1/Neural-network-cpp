/**
 * @file tensor.cpp
 * @brief Implementation of the Tensor class
 */

#include "nn/tensor.hpp"
#include <thread>

namespace nn {

// ============================================================================
// Random Number Generator (thread-local for safety)
// ============================================================================

namespace {
    // Thread-local random engine ensures thread safety
    thread_local std::mt19937 rng(std::random_device{}());
}

// ============================================================================
// Constructors
// ============================================================================

Tensor::Tensor(const Shape& shape, DataType fill_value)
    : shape_(shape)
    , data_(compute_size(shape), fill_value)
{
    // Empty tensors are valid (size = 0)
}

Tensor::Tensor(const Shape& shape, const std::vector<DataType>& data)
    : shape_(shape)
    , data_(data)
{
    size_t expected_size = compute_size(shape);
    if (data.size() != expected_size) {
        throw std::invalid_argument(
            "Data size (" + std::to_string(data.size()) + 
            ") doesn't match shape size (" + std::to_string(expected_size) + ")"
        );
    }
}

Tensor::Tensor(std::initializer_list<DataType> data)
    : shape_({data.size()})
    , data_(data)
{
}

Tensor::Tensor(std::initializer_list<std::initializer_list<DataType>> data) {
    if (data.size() == 0) {
        shape_ = {0, 0};
        return;
    }
    
    size_t rows = data.size();
    size_t cols = data.begin()->size();
    
    // Verify all rows have same length
    for (const auto& row : data) {
        if (row.size() != cols) {
            throw std::invalid_argument("All rows must have the same number of columns");
        }
    }
    
    shape_ = {rows, cols};
    data_.reserve(rows * cols);
    
    for (const auto& row : data) {
        for (DataType val : row) {
            data_.push_back(val);
        }
    }
}

// ============================================================================
// Static Factory Methods
// ============================================================================

Tensor Tensor::zeros(const Shape& shape) {
    return Tensor(shape, 0.0f);
}

Tensor Tensor::ones(const Shape& shape) {
    return Tensor(shape, 1.0f);
}

Tensor Tensor::random_uniform(const Shape& shape, DataType min, DataType max) {
    Tensor result(shape);
    std::uniform_real_distribution<DataType> dist(min, max);
    
    for (auto& val : result.data_) {
        val = dist(rng);
    }
    
    return result;
}

Tensor Tensor::random_normal(const Shape& shape, DataType mean, DataType std) {
    Tensor result(shape);
    std::normal_distribution<DataType> dist(mean, std);
    
    for (auto& val : result.data_) {
        val = dist(rng);
    }
    
    return result;
}

Tensor Tensor::xavier_uniform(const Shape& shape) {
    // Xavier initialization: uniform(-sqrt(6/(fan_in+fan_out)), sqrt(6/(fan_in+fan_out)))
    if (shape.size() < 2) {
        throw std::invalid_argument("Xavier initialization requires at least 2D tensor");
    }
    
    size_t fan_in = shape[0];
    size_t fan_out = shape[1];
    
    // For conv layers, multiply by kernel size
    for (size_t i = 2; i < shape.size(); ++i) {
        fan_in *= shape[i];
        fan_out *= shape[i];
    }
    
    DataType limit = std::sqrt(6.0f / static_cast<DataType>(fan_in + fan_out));
    return random_uniform(shape, -limit, limit);
}

Tensor Tensor::he_uniform(const Shape& shape) {
    // He initialization: uniform(-sqrt(6/fan_in), sqrt(6/fan_in))
    if (shape.size() < 2) {
        throw std::invalid_argument("He initialization requires at least 2D tensor");
    }
    
    size_t fan_in = shape[0];
    for (size_t i = 2; i < shape.size(); ++i) {
        fan_in *= shape[i];
    }
    
    DataType limit = std::sqrt(6.0f / static_cast<DataType>(fan_in));
    return random_uniform(shape, -limit, limit);
}

Tensor Tensor::eye(size_t size) {
    Tensor result({size, size}, 0.0f);
    for (size_t i = 0; i < size; ++i) {
        result.at({i, i}) = 1.0f;
    }
    return result;
}

// ============================================================================
// Shape Operations
// ============================================================================

size_t Tensor::dim(size_t index) const {
    if (index >= shape_.size()) {
        throw std::out_of_range("Dimension index out of range");
    }
    return shape_[index];
}

Tensor Tensor::reshape(const Shape& new_shape) const {
    // Allow one -1 dimension (infer from total size)
    Shape actual_shape = new_shape;
    size_t infer_idx = SIZE_MAX;
    size_t known_product = 1;
    
    for (size_t i = 0; i < actual_shape.size(); ++i) {
        if (actual_shape[i] == static_cast<size_t>(-1)) {
            if (infer_idx != SIZE_MAX) {
                throw std::invalid_argument("Can only have one inferred dimension (-1)");
            }
            infer_idx = i;
        } else {
            known_product *= actual_shape[i];
        }
    }
    
    if (infer_idx != SIZE_MAX) {
        if (size() % known_product != 0) {
            throw std::invalid_argument("Cannot infer dimension: sizes don't divide evenly");
        }
        actual_shape[infer_idx] = size() / known_product;
    }
    
    if (compute_size(actual_shape) != size()) {
        throw std::invalid_argument(
            "Cannot reshape: total size must remain the same (" + 
            std::to_string(size()) + " vs " + std::to_string(compute_size(actual_shape)) + ")"
        );
    }
    
    return Tensor(actual_shape, data_);
}

Tensor Tensor::flatten() const {
    return reshape({size()});
}

Tensor Tensor::transpose() const {
    if (ndim() != 2) {
        throw std::invalid_argument("Simple transpose only works on 2D tensors");
    }
    
    size_t rows = shape_[0];
    size_t cols = shape_[1];
    
    Tensor result(Shape{cols, rows});
    
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            result.at({j, i}) = at({i, j});
        }
    }
    
    return result;
}

Tensor Tensor::transpose(const std::vector<size_t>& axes) const {
    if (axes.size() != ndim()) {
        throw std::invalid_argument("Axes must have same length as tensor dimensions");
    }
    
    // Verify axes is a permutation of [0, ndim)
    std::vector<bool> seen(ndim(), false);
    for (size_t ax : axes) {
        if (ax >= ndim()) {
            throw std::invalid_argument("Axis out of range");
        }
        if (seen[ax]) {
            throw std::invalid_argument("Duplicate axis in permutation");
        }
        seen[ax] = true;
    }
    
    // Compute new shape
    Shape new_shape(ndim());
    for (size_t i = 0; i < ndim(); ++i) {
        new_shape[i] = shape_[axes[i]];
    }
    
    // Create result tensor
    Tensor result(new_shape);
    
    // Iterate through all indices of the result
    std::vector<size_t> old_indices(ndim());
    std::vector<size_t> new_indices(ndim(), 0);
    
    for (size_t flat = 0; flat < size(); ++flat) {
        // Convert new_indices to old_indices
        for (size_t i = 0; i < ndim(); ++i) {
            old_indices[axes[i]] = new_indices[i];
        }
        
        result.at(new_indices) = at(old_indices);
        
        // Increment new_indices
        for (int i = static_cast<int>(ndim()) - 1; i >= 0; --i) {
            if (++new_indices[static_cast<size_t>(i)] < new_shape[static_cast<size_t>(i)]) {
                break;
            }
            new_indices[static_cast<size_t>(i)] = 0;
        }
    }
    
    return result;
}

// ============================================================================
// Element Access
// ============================================================================

Tensor::DataType& Tensor::at(const std::vector<size_t>& indices) {
    validate_indices(indices);
    return data_[compute_flat_index(indices)];
}

const Tensor::DataType& Tensor::at(const std::vector<size_t>& indices) const {
    validate_indices(indices);
    return data_[compute_flat_index(indices)];
}

// ============================================================================
// Arithmetic Operations
// ============================================================================

Tensor Tensor::operator+(const Tensor& other) const {
    if (!same_shape(other)) {
        throw std::invalid_argument("Shape mismatch for addition: " + 
            shape_string() + " vs " + other.shape_string());
    }
    
    Tensor result(shape_);
    for (size_t i = 0; i < size(); ++i) {
        result.data_[i] = data_[i] + other.data_[i];
    }
    return result;
}

Tensor Tensor::operator-(const Tensor& other) const {
    if (!same_shape(other)) {
        throw std::invalid_argument("Shape mismatch for subtraction");
    }
    
    Tensor result(shape_);
    for (size_t i = 0; i < size(); ++i) {
        result.data_[i] = data_[i] - other.data_[i];
    }
    return result;
}

Tensor Tensor::operator*(const Tensor& other) const {
    if (!same_shape(other)) {
        throw std::invalid_argument("Shape mismatch for multiplication");
    }
    
    Tensor result(shape_);
    for (size_t i = 0; i < size(); ++i) {
        result.data_[i] = data_[i] * other.data_[i];
    }
    return result;
}

Tensor Tensor::operator/(const Tensor& other) const {
    if (!same_shape(other)) {
        throw std::invalid_argument("Shape mismatch for division");
    }
    
    Tensor result(shape_);
    for (size_t i = 0; i < size(); ++i) {
        result.data_[i] = data_[i] / other.data_[i];
    }
    return result;
}

Tensor& Tensor::operator+=(const Tensor& other) {
    if (!same_shape(other)) {
        throw std::invalid_argument("Shape mismatch for addition");
    }
    for (size_t i = 0; i < size(); ++i) {
        data_[i] += other.data_[i];
    }
    return *this;
}

Tensor& Tensor::operator-=(const Tensor& other) {
    if (!same_shape(other)) {
        throw std::invalid_argument("Shape mismatch for subtraction");
    }
    for (size_t i = 0; i < size(); ++i) {
        data_[i] -= other.data_[i];
    }
    return *this;
}

Tensor& Tensor::operator*=(const Tensor& other) {
    if (!same_shape(other)) {
        throw std::invalid_argument("Shape mismatch for multiplication");
    }
    for (size_t i = 0; i < size(); ++i) {
        data_[i] *= other.data_[i];
    }
    return *this;
}

Tensor& Tensor::operator/=(const Tensor& other) {
    if (!same_shape(other)) {
        throw std::invalid_argument("Shape mismatch for division");
    }
    for (size_t i = 0; i < size(); ++i) {
        data_[i] /= other.data_[i];
    }
    return *this;
}

Tensor Tensor::operator+(DataType scalar) const {
    Tensor result(shape_);
    for (size_t i = 0; i < size(); ++i) {
        result.data_[i] = data_[i] + scalar;
    }
    return result;
}

Tensor Tensor::operator-(DataType scalar) const {
    Tensor result(shape_);
    for (size_t i = 0; i < size(); ++i) {
        result.data_[i] = data_[i] - scalar;
    }
    return result;
}

Tensor Tensor::operator*(DataType scalar) const {
    Tensor result(shape_);
    for (size_t i = 0; i < size(); ++i) {
        result.data_[i] = data_[i] * scalar;
    }
    return result;
}

Tensor Tensor::operator/(DataType scalar) const {
    Tensor result(shape_);
    for (size_t i = 0; i < size(); ++i) {
        result.data_[i] = data_[i] / scalar;
    }
    return result;
}

Tensor& Tensor::operator+=(DataType scalar) {
    for (auto& val : data_) val += scalar;
    return *this;
}

Tensor& Tensor::operator-=(DataType scalar) {
    for (auto& val : data_) val -= scalar;
    return *this;
}

Tensor& Tensor::operator*=(DataType scalar) {
    for (auto& val : data_) val *= scalar;
    return *this;
}

Tensor& Tensor::operator/=(DataType scalar) {
    for (auto& val : data_) val /= scalar;
    return *this;
}

Tensor Tensor::operator-() const {
    Tensor result(shape_);
    for (size_t i = 0; i < size(); ++i) {
        result.data_[i] = -data_[i];
    }
    return result;
}

// ============================================================================
// Matrix Operations
// ============================================================================

Tensor Tensor::matmul(const Tensor& other) const {
    // Handle batched matrix multiplication for 2D case
    if (ndim() != 2 || other.ndim() != 2) {
        throw std::invalid_argument("matmul requires 2D tensors (matrices)");
    }
    
    size_t m = shape_[0];      // Rows of A
    size_t k = shape_[1];      // Cols of A = Rows of B
    size_t n = other.shape_[1]; // Cols of B
    
    if (k != other.shape_[0]) {
        throw std::invalid_argument(
            "Matrix dimensions don't match for multiplication: " +
            shape_string() + " x " + other.shape_string()
        );
    }
    
    Tensor result({m, n}, 0.0f);
    
    // Standard O(n³) matrix multiplication
    // Note: For production, you'd want optimizations like:
    // - Cache blocking (tiling)
    // - SIMD (AVX/SSE)
    // - OpenMP parallelization
    // - BLAS library calls
    
    const DataType* A = data_.data();
    const DataType* B = other.data_.data();
    DataType* C = result.data_.data();
    
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            DataType sum = 0.0f;
            for (size_t l = 0; l < k; ++l) {
                sum += A[i * k + l] * B[l * n + j];
            }
            C[i * n + j] = sum;
        }
    }
    
    return result;
}

Tensor::DataType Tensor::dot(const Tensor& other) const {
    if (ndim() != 1 || other.ndim() != 1) {
        throw std::invalid_argument("Dot product requires 1D tensors");
    }
    if (size() != other.size()) {
        throw std::invalid_argument("Vectors must have same length for dot product");
    }
    
    DataType result = 0.0f;
    for (size_t i = 0; i < size(); ++i) {
        result += data_[i] * other.data_[i];
    }
    return result;
}

// ============================================================================
// Reduction Operations
// ============================================================================

Tensor::DataType Tensor::sum() const {
    DataType result = 0.0f;
    for (const auto& val : data_) {
        result += val;
    }
    return result;
}

Tensor Tensor::sum(size_t axis, bool keepdims) const {
    if (axis >= ndim()) {
        throw std::out_of_range("Axis out of range for sum");
    }
    
    // Compute output shape
    Shape out_shape;
    for (size_t i = 0; i < ndim(); ++i) {
        if (i == axis) {
            if (keepdims) out_shape.push_back(1);
        } else {
            out_shape.push_back(shape_[i]);
        }
    }
    if (out_shape.empty()) out_shape.push_back(1);
    
    Tensor result(out_shape, 0.0f);
    
    // Compute strides for the axis
    size_t axis_size = shape_[axis];
    size_t outer_size = 1;
    size_t inner_size = 1;
    
    for (size_t i = 0; i < axis; ++i) outer_size *= shape_[i];
    for (size_t i = axis + 1; i < ndim(); ++i) inner_size *= shape_[i];
    
    // Sum along axis
    for (size_t outer = 0; outer < outer_size; ++outer) {
        for (size_t inner = 0; inner < inner_size; ++inner) {
            DataType sum = 0.0f;
            for (size_t a = 0; a < axis_size; ++a) {
                size_t idx = outer * (axis_size * inner_size) + a * inner_size + inner;
                sum += data_[idx];
            }
            result.data_[outer * inner_size + inner] = sum;
        }
    }
    
    return result;
}

Tensor::DataType Tensor::mean() const {
    if (empty()) return 0.0f;
    return sum() / static_cast<DataType>(size());
}

Tensor Tensor::mean(size_t axis, bool keepdims) const {
    Tensor s = sum(axis, keepdims);
    s /= static_cast<DataType>(shape_[axis]);
    return s;
}

Tensor::DataType Tensor::max() const {
    if (empty()) {
        throw std::runtime_error("Cannot compute max of empty tensor");
    }
    return *std::max_element(data_.begin(), data_.end());
}

Tensor Tensor::max(size_t axis, bool keepdims) const {
    if (axis >= ndim()) {
        throw std::out_of_range("Axis out of range for max");
    }
    
    // Compute output shape
    Shape out_shape;
    for (size_t i = 0; i < ndim(); ++i) {
        if (i == axis) {
            if (keepdims) out_shape.push_back(1);
        } else {
            out_shape.push_back(shape_[i]);
        }
    }
    if (out_shape.empty()) out_shape.push_back(1);
    
    Tensor result(out_shape);
    
    size_t axis_size = shape_[axis];
    size_t outer_size = 1;
    size_t inner_size = 1;
    
    for (size_t i = 0; i < axis; ++i) outer_size *= shape_[i];
    for (size_t i = axis + 1; i < ndim(); ++i) inner_size *= shape_[i];
    
    for (size_t outer = 0; outer < outer_size; ++outer) {
        for (size_t inner = 0; inner < inner_size; ++inner) {
            DataType max_val = std::numeric_limits<DataType>::lowest();
            for (size_t a = 0; a < axis_size; ++a) {
                size_t idx = outer * (axis_size * inner_size) + a * inner_size + inner;
                max_val = std::max(max_val, data_[idx]);
            }
            result.data_[outer * inner_size + inner] = max_val;
        }
    }
    
    return result;
}

Tensor Tensor::argmax(size_t axis) const {
    if (axis >= ndim()) {
        throw std::out_of_range("Axis out of range for argmax");
    }
    
    // Compute output shape (axis dimension removed)
    Shape out_shape;
    for (size_t i = 0; i < ndim(); ++i) {
        if (i != axis) {
            out_shape.push_back(shape_[i]);
        }
    }
    if (out_shape.empty()) out_shape.push_back(1);
    
    Tensor result(out_shape);
    
    size_t axis_size = shape_[axis];
    size_t outer_size = 1;
    size_t inner_size = 1;
    
    for (size_t i = 0; i < axis; ++i) outer_size *= shape_[i];
    for (size_t i = axis + 1; i < ndim(); ++i) inner_size *= shape_[i];
    
    for (size_t outer = 0; outer < outer_size; ++outer) {
        for (size_t inner = 0; inner < inner_size; ++inner) {
            DataType max_val = std::numeric_limits<DataType>::lowest();
            size_t max_idx = 0;
            for (size_t a = 0; a < axis_size; ++a) {
                size_t idx = outer * (axis_size * inner_size) + a * inner_size + inner;
                if (data_[idx] > max_val) {
                    max_val = data_[idx];
                    max_idx = a;
                }
            }
            result.data_[outer * inner_size + inner] = static_cast<DataType>(max_idx);
        }
    }
    
    return result;
}

Tensor::DataType Tensor::min() const {
    if (empty()) {
        throw std::runtime_error("Cannot compute min of empty tensor");
    }
    return *std::min_element(data_.begin(), data_.end());
}

// ============================================================================
// Mathematical Functions
// ============================================================================

Tensor Tensor::apply(std::function<DataType(DataType)> func) const {
    Tensor result(shape_);
    for (size_t i = 0; i < size(); ++i) {
        result.data_[i] = func(data_[i]);
    }
    return result;
}

Tensor Tensor::exp() const {
    return apply([](DataType x) { return std::exp(x); });
}

Tensor Tensor::log() const {
    return apply([](DataType x) { return std::log(x); });
}

Tensor Tensor::pow(DataType exponent) const {
    return apply([exponent](DataType x) { return std::pow(x, exponent); });
}

Tensor Tensor::sqrt() const {
    return apply([](DataType x) { return std::sqrt(x); });
}

Tensor Tensor::abs() const {
    return apply([](DataType x) { return std::abs(x); });
}

Tensor Tensor::clip(DataType min_val, DataType max_val) const {
    return apply([min_val, max_val](DataType x) {
        return std::max(min_val, std::min(max_val, x));
    });
}

// ============================================================================
// Comparison Operations
// ============================================================================

bool Tensor::equals(const Tensor& other, DataType tolerance) const {
    if (!same_shape(other)) return false;
    
    for (size_t i = 0; i < size(); ++i) {
        if (std::abs(data_[i] - other.data_[i]) > tolerance) {
            return false;
        }
    }
    return true;
}

bool Tensor::same_shape(const Tensor& other) const {
    return shape_ == other.shape_;
}

// ============================================================================
// Utility Functions
// ============================================================================

void Tensor::print(std::ostream& os, int precision) const {
    os << to_string(precision);
}

std::string Tensor::to_string(int precision) const {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(precision);
    
    oss << "Tensor" << shape_string() << ":\n";
    
    if (ndim() == 1) {
        oss << "[";
        for (size_t i = 0; i < size(); ++i) {
            if (i > 0) oss << ", ";
            oss << data_[i];
        }
        oss << "]";
    } else if (ndim() == 2) {
        oss << "[";
        for (size_t i = 0; i < shape_[0]; ++i) {
            if (i > 0) oss << " ";
            oss << "[";
            for (size_t j = 0; j < shape_[1]; ++j) {
                if (j > 0) oss << ", ";
                oss << std::setw(precision + 4) << at({i, j});
            }
            oss << "]";
            if (i < shape_[0] - 1) oss << ",\n";
        }
        oss << "]";
    } else {
        // For higher dimensions, just print flat with shape info
        oss << "[";
        size_t max_print = std::min(size(), size_t(10));
        for (size_t i = 0; i < max_print; ++i) {
            if (i > 0) oss << ", ";
            oss << data_[i];
        }
        if (size() > max_print) oss << ", ...";
        oss << "]";
    }
    
    return oss.str();
}

std::string Tensor::shape_string() const {
    std::ostringstream oss;
    oss << "(";
    for (size_t i = 0; i < shape_.size(); ++i) {
        if (i > 0) oss << ", ";
        oss << shape_[i];
    }
    oss << ")";
    return oss.str();
}

void Tensor::fill(DataType value) {
    std::fill(data_.begin(), data_.end(), value);
}

Tensor Tensor::clone() const {
    return Tensor(shape_, data_);
}

// ============================================================================
// Private Helper Methods
// ============================================================================

size_t Tensor::compute_flat_index(const std::vector<size_t>& indices) const {
    size_t flat = 0;
    size_t stride = 1;
    
    // Row-major: rightmost index changes fastest
    for (int i = static_cast<int>(indices.size()) - 1; i >= 0; --i) {
        flat += indices[static_cast<size_t>(i)] * stride;
        stride *= shape_[static_cast<size_t>(i)];
    }
    
    return flat;
}

size_t Tensor::compute_size(const Shape& shape) {
    if (shape.empty()) return 0;
    return std::accumulate(shape.begin(), shape.end(), 
                           size_t(1), std::multiplies<size_t>());
}

void Tensor::validate_indices(const std::vector<size_t>& indices) const {
    if (indices.size() != ndim()) {
        throw std::out_of_range(
            "Wrong number of indices: expected " + std::to_string(ndim()) +
            ", got " + std::to_string(indices.size())
        );
    }
    
    for (size_t i = 0; i < indices.size(); ++i) {
        if (indices[i] >= shape_[i]) {
            throw std::out_of_range(
                "Index " + std::to_string(indices[i]) + 
                " out of range for dimension " + std::to_string(i) +
                " with size " + std::to_string(shape_[i])
            );
        }
    }
}

bool Tensor::broadcast_compatible(const Shape& a, const Shape& b) {
    // Broadcasting rules: dimensions must match or one must be 1
    size_t max_dims = std::max(a.size(), b.size());
    
    for (size_t i = 0; i < max_dims; ++i) {
        size_t dim_a = i < a.size() ? a[a.size() - 1 - i] : 1;
        size_t dim_b = i < b.size() ? b[b.size() - 1 - i] : 1;
        
        if (dim_a != dim_b && dim_a != 1 && dim_b != 1) {
            return false;
        }
    }
    return true;
}

// ============================================================================
// Stream Output
// ============================================================================

std::ostream& operator<<(std::ostream& os, const Tensor& t) {
    t.print(os);
    return os;
}

} // namespace nn
