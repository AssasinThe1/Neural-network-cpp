# Neural Network Library in C++

**A lightweight, educational neural network library built from scratch in modern C++17**

![C++ 17](https://img.shields.io/badge/C%2B%2B-17-blue.svg)
![MIT License](https://img.shields.io/badge/License-MIT-green.svg)
![CMake](https://img.shields.io/badge/Build-CMake-red.svg)

## Overview

This project implements a complete neural network library in pure C++ with no external machine learning dependencies. It demonstrates proficiency in:

- **Modern C++17** features (smart pointers, auto, structured bindings, etc.)
- **Object-Oriented Design** (inheritance, polymorphism, abstract classes)
- **Template Programming** (variadic templates, type deduction)
- **Memory Management** (RAII, move semantics)
- **Mathematical Computing** (linear algebra, calculus for backpropagation)
- **Software Engineering** (CMake, testing, documentation)

## Features

### Core Components
- **Tensor Class**: N-dimensional array with broadcasting, slicing, and mathematical operations
- **Layer System**: Extensible base class with multiple implementations
- **Automatic Differentiation**: Backpropagation through computational graph
- **Optimizers**: SGD (with momentum), Adam, RMSprop
- **Loss Functions**: MSE, Cross-Entropy, Binary Cross-Entropy, Huber

### Supported Layers
| Layer Type | Description |
|------------|-------------|
| `Dense` | Fully connected layer |
| `Conv2D` | 2D Convolutional layer |
| `MaxPool2D` | Max pooling |
| `AvgPool2D` | Average pooling |
| `Flatten` | Reshape for FC layers |
| `Dropout` | Regularization layer |
| `ReLU` | Rectified Linear Unit |
| `LeakyReLU` | Leaky ReLU variant |
| `Sigmoid` | Sigmoid activation |
| `Tanh` | Hyperbolic tangent |
| `Softmax` | Softmax activation |

## Quick Start

### Prerequisites
- C++17 compatible compiler (GCC 7+, Clang 5+, MSVC 2017+)
- CMake 3.16+
- (Optional) CLion IDE

### Building

```bash
# Clone the repository
git clone https://github.com/yourusername/neural-network-cpp.git
cd neural-network-cpp

# Create build directory
mkdir build && cd build

# Configure and build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j4

# Run tests
ctest --output-on-failure

# Run examples
./xor_example
./mnist_example
```

### Basic Usage

```cpp
#include "nn/network.hpp"

int main() {
    using namespace nn;
    
    // Build a simple network
    Network net;
    net.add<Dense>(784, 128);  // Input -> Hidden
    net.add<ReLU>();
    net.add<Dense>(128, 10);   // Hidden -> Output
    net.add<Softmax>();
    
    // Configure training
    net.compile("cross_entropy", "adam", 0.001f);
    
    // Train
    auto history = net.fit(train_X, train_y, 
                           /*batch_size=*/32, 
                           /*epochs=*/10);
    
    // Evaluate
    auto [loss, accuracy] = net.evaluate(test_X, test_y);
    
    // Predict
    Tensor predictions = net.predict(new_data);
    
    return 0;
}
```

## Project Structure

```
neural_network_cpp/
├── CMakeLists.txt          # Build configuration
├── README.md               # This file
├── include/nn/             # Header files
│   ├── tensor.hpp          # N-dimensional array
│   ├── network.hpp         # Network container
│   ├── loss.hpp            # Loss functions
│   ├── optimizer.hpp       # Optimization algorithms
│   ├── serialization.hpp   # Save/load models
│   └── layers/
│       ├── layer.hpp       # Abstract base class
│       ├── dense.hpp       # Fully connected
│       ├── conv2d.hpp      # Convolutional
│       ├── pooling.hpp     # Pooling layers
│       └── activations.hpp # Activation functions
├── src/                    # Implementation files
├── tests/                  # Unit tests
└── examples/               # Demo applications
    ├── xor_example.cpp     # Classic XOR problem
    └── mnist_example.cpp   # Digit classification
```

## Architecture

### Class Hierarchy

```
Layer (abstract)
├── Dense
├── Conv2D
├── MaxPool2D
├── AvgPool2D
├── Flatten
├── Dropout
├── ReLU
├── LeakyReLU
├── Sigmoid
├── Tanh
└── Softmax

Loss (abstract)
├── MSELoss
├── CrossEntropyLoss
├── BCELoss
└── HuberLoss

Optimizer (abstract)
├── SGD
├── Adam
└── RMSprop
```

### Design Patterns Used
- **Template Method**: Layer base class defines interface
- **Factory Method**: `create_activation()`, `create_loss()`, `create_optimizer()`
- **Strategy Pattern**: Interchangeable optimizers and loss functions
- **Builder Pattern**: Network configuration with method chaining

## Examples

### 1. XOR Problem
The classic non-linearly separable problem that requires hidden layers:

```cpp
Network net;
net.add<Dense>(2, 8);
net.add<ReLU>();
net.add<Dense>(8, 1);
net.add<Sigmoid>();

net.compile("mse", "adam", 0.1f);
net.fit(X, y, 4, 1000);
```

### 2. MNIST Classification
Handwritten digit recognition:

```cpp
Network net;
net.add<Dense>(784, 128);
net.add<ReLU>();
net.add<Dense>(128, 64);
net.add<ReLU>();
net.add<Dense>(64, 10);
net.add<Softmax>();

net.compile("cross_entropy", "adam", 0.001f);
net.fit(train_X, train_y, 32, 20, 0.1f);  // 10% validation
```

### 3. Convolutional Network
For image processing:

```cpp
Network net;
net.add<Conv2D>(1, 32, 3, 1, 1);  // 32 filters, 3x3 kernel
net.add<ReLU>();
net.add<MaxPool2D>(2);
net.add<Conv2D>(32, 64, 3, 1, 1);
net.add<ReLU>();
net.add<MaxPool2D>(2);
net.add<Flatten>();
net.add<Dense>(64 * 7 * 7, 128);
net.add<ReLU>();
net.add<Dense>(128, 10);
net.add<Softmax>();
```

## Performance Considerations

This library prioritizes readability and education over raw performance. For production use, consider:

- **BLAS Libraries**: Replace matrix multiplication with optimized BLAS calls
- **SIMD Instructions**: Use AVX/SSE for element-wise operations
- **Parallelization**: OpenMP for multi-threading
- **GPU Support**: CUDA for GPU acceleration

## Testing

The project includes comprehensive unit tests:

```bash
# Run all tests
cd build
ctest --output-on-failure

# Run specific test suite
./test_tensor
./test_layers
./test_network
```

Test coverage includes:
- Tensor operations (arithmetic, reshaping, reductions)
- Layer forward/backward passes
- Gradient computation verification
- Network training convergence

## Future Enhancements

- [ ] Batch Normalization layer
- [ ] LSTM/GRU recurrent layers
- [ ] Model serialization (save/load trained models)
- [ ] ONNX format support
- [ ] GPU acceleration with CUDA
- [ ] Python bindings with pybind11

## Learning Resources

If you're learning from this project, here are key concepts to study:

1. **Backpropagation**: How gradients flow backward through the network
2. **Weight Initialization**: Why Xavier/He initialization matters
3. **Optimization**: How Adam improves on basic SGD
4. **Regularization**: Preventing overfitting with dropout
5. **Numerical Stability**: Softmax stability, gradient clipping
