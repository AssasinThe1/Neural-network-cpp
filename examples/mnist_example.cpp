/**
 * @file mnist_example.cpp
 * @brief MNIST Handwritten Digit Classification
 * 
 * MNIST is the "Hello World" of deep learning - a dataset of 28x28
 * grayscale images of handwritten digits (0-9).
 * 
 * This example demonstrates:
 * 1. Loading binary data files
 * 2. Building a neural network for image classification
 * 3. Training and evaluating the model
 * 
 * Download MNIST from: http://yann.lecun.com/exdb/mnist/
 * Files needed:
 *   - train-images-idx3-ubyte (training images)
 *   - train-labels-idx1-ubyte (training labels)
 *   - t10k-images-idx3-ubyte  (test images)
 *   - t10k-labels-idx1-ubyte  (test labels)
 */

#include "nn/network.hpp"
#include <iostream>
#include <fstream>
#include <vector>
#include <iomanip>

using namespace nn;

// ============================================================================
// MNIST Data Loader
// ============================================================================

namespace mnist {

/**
 * @brief Reverse bytes (MNIST uses big-endian)
 */
uint32_t reverse_int(uint32_t i) {
    unsigned char c1, c2, c3, c4;
    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;
    return ((uint32_t)c1 << 24) + ((uint32_t)c2 << 16) + 
           ((uint32_t)c3 << 8) + c4;
}

/**
 * @brief Load MNIST images from idx3-ubyte file
 */
Tensor load_images(const std::string& filepath, size_t max_samples = 0) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Cannot open image file: " + filepath);
    }
    
    uint32_t magic, num_images, rows, cols;
    file.read(reinterpret_cast<char*>(&magic), 4);
    file.read(reinterpret_cast<char*>(&num_images), 4);
    file.read(reinterpret_cast<char*>(&rows), 4);
    file.read(reinterpret_cast<char*>(&cols), 4);
    
    magic = reverse_int(magic);
    num_images = reverse_int(num_images);
    rows = reverse_int(rows);
    cols = reverse_int(cols);
    
    if (magic != 2051) {
        throw std::runtime_error("Invalid MNIST image file magic number");
    }
    
    if (max_samples > 0 && max_samples < num_images) {
        num_images = static_cast<uint32_t>(max_samples);
    }
    
    size_t image_size = rows * cols;
    Tensor images({num_images, image_size});
    
    std::vector<unsigned char> buffer(image_size);
    for (uint32_t i = 0; i < num_images; ++i) {
        file.read(reinterpret_cast<char*>(buffer.data()), 
                  static_cast<std::streamsize>(image_size));
        for (size_t j = 0; j < image_size; ++j) {
            // Normalize to [0, 1]
            images.at({i, j}) = static_cast<float>(buffer[j]) / 255.0f;
        }
    }
    
    return images;
}

/**
 * @brief Load MNIST labels from idx1-ubyte file
 */
Tensor load_labels(const std::string& filepath, size_t max_samples = 0) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Cannot open label file: " + filepath);
    }
    
    uint32_t magic, num_labels;
    file.read(reinterpret_cast<char*>(&magic), 4);
    file.read(reinterpret_cast<char*>(&num_labels), 4);
    
    magic = reverse_int(magic);
    num_labels = reverse_int(num_labels);
    
    if (magic != 2049) {
        throw std::runtime_error("Invalid MNIST label file magic number");
    }
    
    if (max_samples > 0 && max_samples < num_labels) {
        num_labels = static_cast<uint32_t>(max_samples);
    }
    
    // One-hot encode labels
    Tensor labels({num_labels, 10}, 0.0f);
    
    for (uint32_t i = 0; i < num_labels; ++i) {
        unsigned char label;
        file.read(reinterpret_cast<char*>(&label), 1);
        labels.at({i, static_cast<size_t>(label)}) = 1.0f;
    }
    
    return labels;
}

/**
 * @brief Print an MNIST digit to console
 */
void print_digit(const Tensor& images, size_t index) {
    std::cout << "┌" << std::string(28, '─') << "┐\n";
    for (size_t row = 0; row < 28; ++row) {
        std::cout << "│";
        for (size_t col = 0; col < 28; ++col) {
            float pixel = images.at({index, row * 28 + col});
            if (pixel > 0.75f) std::cout << "█";
            else if (pixel > 0.5f) std::cout << "▓";
            else if (pixel > 0.25f) std::cout << "▒";
            else if (pixel > 0.0f) std::cout << "░";
            else std::cout << " ";
        }
        std::cout << "│\n";
    }
    std::cout << "└" << std::string(28, '─') << "┘\n";
}

/**
 * @brief Get label from one-hot encoded tensor
 */
int get_label(const Tensor& labels, size_t index) {
    for (size_t i = 0; i < 10; ++i) {
        if (labels.at({index, i}) > 0.5f) {
            return static_cast<int>(i);
        }
    }
    return -1;
}

} // namespace mnist

// ============================================================================
// Create Synthetic MNIST-like Data (for testing without real MNIST)
// ============================================================================

std::pair<Tensor, Tensor> create_synthetic_data(size_t num_samples) {
    /*
     * Create synthetic data that looks vaguely like digits.
     * Each "digit" is just a simple pattern.
     * This is useful for testing the network structure without needing
     * the actual MNIST dataset.
     */
    
    Tensor X({num_samples, 784}, 0.0f);
    Tensor y({num_samples, 10}, 0.0f);
    
    for (size_t i = 0; i < num_samples; ++i) {
        int digit = static_cast<int>(i % 10);
        y.at({i, static_cast<size_t>(digit)}) = 1.0f;
        
        // Create simple patterns for each digit
        for (size_t r = 0; r < 28; ++r) {
            for (size_t c = 0; c < 28; ++c) {
                float val = 0.0f;
                
                switch (digit) {
                    case 0:  // Circle
                        if (r >= 4 && r < 24 && c >= 8 && c < 20) {
                            int dr = static_cast<int>(r) - 14;
                            int dc = static_cast<int>(c) - 14;
                            int dist = dr * dr + dc * dc;
                            if (dist > 36 && dist < 100) val = 1.0f;
                        }
                        break;
                    case 1:  // Vertical line
                        if (c >= 12 && c < 16 && r >= 4 && r < 24) val = 1.0f;
                        break;
                    case 2:  // S-shape
                        if (r >= 4 && r < 8 && c >= 8 && c < 20) val = 1.0f;
                        if (r >= 12 && r < 16 && c >= 8 && c < 20) val = 1.0f;
                        if (r >= 20 && r < 24 && c >= 8 && c < 20) val = 1.0f;
                        if (r >= 4 && r < 14 && c >= 16 && c < 20) val = 1.0f;
                        if (r >= 14 && r < 24 && c >= 8 && c < 12) val = 1.0f;
                        break;
                    case 3:  // E-shape
                        if (c >= 8 && c < 12 && r >= 4 && r < 24) val = 1.0f;
                        if (r >= 4 && r < 8 && c >= 8 && c < 20) val = 1.0f;
                        if (r >= 12 && r < 16 && c >= 8 && c < 18) val = 1.0f;
                        if (r >= 20 && r < 24 && c >= 8 && c < 20) val = 1.0f;
                        break;
                    default:  // Simple block patterns
                        if (r >= 8 && r < 20 && c >= 8 && c < 20) {
                            val = (r + c + digit) % 3 == 0 ? 1.0f : 0.0f;
                        }
                }
                
                // Add noise
                val += 0.1f * static_cast<float>(rand()) / RAND_MAX;
                val = std::min(1.0f, std::max(0.0f, val));
                
                X.at({i, r * 28 + c}) = val;
            }
        }
    }
    
    return {X, y};
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char* argv[]) {
    std::cout << "\n";
    std::cout << "========================================\n";
    std::cout << "     MNIST Digit Classification\n";
    std::cout << "========================================\n\n";
    
    // Configuration
    std::string data_dir = "data/mnist/";
    bool use_synthetic = false;
    size_t train_samples = 1000;  // Limit for faster training
    size_t test_samples = 200;
    
    // Parse command line
    if (argc > 1) {
        data_dir = argv[1];
    }
    
    Tensor train_X, train_y, test_X, test_y;
    
    // ========================================================================
    // Load Data
    // ========================================================================
    
    std::cout << "Loading data...\n";
    
    try {
        train_X = mnist::load_images(data_dir + "train-images-idx3-ubyte", train_samples);
        train_y = mnist::load_labels(data_dir + "train-labels-idx1-ubyte", train_samples);
        test_X = mnist::load_images(data_dir + "t10k-images-idx3-ubyte", test_samples);
        test_y = mnist::load_labels(data_dir + "t10k-labels-idx1-ubyte", test_samples);
        
        std::cout << "Loaded real MNIST data:\n";
        std::cout << "  Training: " << train_X.dim(0) << " samples\n";
        std::cout << "  Test: " << test_X.dim(0) << " samples\n\n";
        
        // Show sample digit
        std::cout << "Sample training digit (label: " 
                  << mnist::get_label(train_y, 0) << "):\n";
        mnist::print_digit(train_X, 0);
        std::cout << "\n";
        
    } catch (const std::exception& e) {
        std::cout << "Could not load MNIST: " << e.what() << "\n";
        std::cout << "Using synthetic data instead...\n\n";
        
        use_synthetic = true;
        auto [synth_train_X, synth_train_y] = create_synthetic_data(train_samples);
        auto [synth_test_X, synth_test_y] = create_synthetic_data(test_samples);
        
        train_X = std::move(synth_train_X);
        train_y = std::move(synth_train_y);
        test_X = std::move(synth_test_X);
        test_y = std::move(synth_test_y);
        
        std::cout << "Created synthetic data:\n";
        std::cout << "  Training: " << train_X.dim(0) << " samples\n";
        std::cout << "  Test: " << test_X.dim(0) << " samples\n\n";
    }
    
    // ========================================================================
    // Build Network
    // ========================================================================
    
    std::cout << "Building neural network...\n\n";
    
    Network net;
    
    // Simple MLP for MNIST
    net.add<Dense>(784, 128, true, "he");  // 28*28 = 784 input features
    net.add<ReLU>();
    net.add<Dense>(128, 64, true, "he");
    net.add<ReLU>();
    net.add<Dense>(64, 10, true, "xavier");  // 10 output classes
    net.add<Softmax>();
    
    net.summary();
    
    // ========================================================================
    // Train Network
    // ========================================================================
    
    net.compile("cross_entropy", "adam", 0.001f);
    
    std::cout << "Training configuration:\n";
    std::cout << "  Loss: Cross-Entropy\n";
    std::cout << "  Optimizer: Adam (lr=0.001)\n";
    std::cout << "  Epochs: 20\n";
    std::cout << "  Batch size: 32\n\n";
    
    std::cout << "Training...\n\n";
    
    auto history = net.fit(train_X, train_y, 
                           32,    // batch_size
                           20,    // epochs
                           0.1f,  // validation_split
                           true); // verbose
    
    // ========================================================================
    // Evaluate
    // ========================================================================
    
    std::cout << "\nEvaluating on test set...\n";
    auto [test_loss, test_acc] = net.evaluate(test_X, test_y);
    std::cout << "Test Loss: " << std::fixed << std::setprecision(4) << test_loss << "\n";
    std::cout << "Test Accuracy: " << (test_acc * 100.0f) << "%\n\n";
    
    // ========================================================================
    // Show Predictions
    // ========================================================================
    
    std::cout << "Sample predictions:\n";
    std::cout << "Index | True | Pred | Correct\n";
    std::cout << "------|------|------|--------\n";
    
    Tensor predictions = net.predict(test_X);
    int correct = 0;
    
    for (size_t i = 0; i < std::min(size_t(10), test_X.dim(0)); ++i) {
        int true_label = mnist::get_label(test_y, i);
        
        // Find predicted label (argmax)
        int pred_label = 0;
        float max_val = predictions.at({i, 0});
        for (size_t j = 1; j < 10; ++j) {
            if (predictions.at({i, j}) > max_val) {
                max_val = predictions.at({i, j});
                pred_label = static_cast<int>(j);
            }
        }
        
        bool is_correct = (true_label == pred_label);
        if (is_correct) correct++;
        
        std::cout << std::setw(5) << i << " | " 
                  << std::setw(4) << true_label << " | "
                  << std::setw(4) << pred_label << " | "
                  << (is_correct ? "  ✓" : "  ✗") << "\n";
    }
    
    // ========================================================================
    // Summary
    // ========================================================================
    
    std::cout << "\n========================================\n";
    std::cout << "  Training Complete!\n";
    std::cout << "========================================\n\n";
    
    std::cout << "Final Results:\n";
    std::cout << "  Training Loss: " << history.train_loss.back() << "\n";
    std::cout << "  Training Acc:  " << (history.train_accuracy.back() * 100.0f) << "%\n";
    std::cout << "  Test Acc:      " << (test_acc * 100.0f) << "%\n\n";
    
    if (use_synthetic) {
        std::cout << "Note: Used synthetic data. Download real MNIST for better results.\n";
        std::cout << "Place files in: " << data_dir << "\n\n";
    }
    
    return 0;
}
