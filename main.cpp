#include <cstddef>
#include <cstdint>
#include <algorithm>
#include <stdexcept>
#include <vector>
#include <iostream>
#include <cmath>
#include <fstream>
#include <string>
#include <cassert>
#include <chrono>
#include <iomanip>


class Tensor{  // The actual tensor
public:
    Tensor(
        std::vector<std::size_t> shape,
        std::vector<double> data
    ) : shape_(std::move(shape)), data_(std::move(data)){
        if(shape_.empty()){
            throw std::invalid_argument("Tensor must have at least one dimension");
        }
        std::size_t expected_elements = 1;

        for(const std::size_t dimension: shape_){
            if(dimension == 0){
                throw std::invalid_argument("Tensor dimensions must be positive");
            }
            expected_elements *= dimension;
            // Fix the overflow later
        }
        if(expected_elements != data_.size()){
            throw std::invalid_argument("Tensor shape does not match its data");
        }
    }

    [[nodiscard]] const std::vector<std::size_t>& shape() const noexcept { // caller can read the data but cannot change it 
        return shape_;
    }

    [[nodiscard]] const std::vector<double>& data() const noexcept {
        return data_;
    }

    [[nodiscard]] std::size_t rank() const noexcept {
        return shape_.size();
    }

    [[nodiscard]] std::size_t numel() const noexcept {
        return data_.size();
    }

    [[nodiscard]] double at(const std::vector<std::size_t> idx) const {
        if(idx.size() != rank()) {
            throw std::invalid_argument("Number of indices must match tensor rank");
        }
        
        std::size_t flat_index = 0;
        std::size_t stride = 1;
        for(std::size_t axis = rank(); axis > 0; --axis) {
            const std::size_t current_axis = axis - 1;
            
            if(idx[current_axis] >= shape_[current_axis]){
                throw std::out_of_range("Tensor index is outside its dimension");
            }
            
            flat_index += idx[current_axis] * stride;
            stride *= shape_[current_axis];
            // Fix the overflow later
        }
        return data_[flat_index];
    }

private:
    std::vector<std::size_t> shape_;
    std::vector<double> data_;

};

[[nodiscard]] Tensor matmul(const Tensor& a, const Tensor& b) {
    // For our MNIST engine, we only care about 2D matrices
    if(a.rank() != 2 || b.rank() != 2) {
        throw std::invalid_argument("Matmul requires 2D tensors");
    }

    std::size_t left_rows = a.shape()[0]; // Rows of A
    std::size_t left_cols = a.shape()[1]; // Cols of A
    std::size_t right_rows = b.shape()[0]; // Rows of B
    std::size_t right_cols = b.shape()[1]; // Cols of B

    if(left_cols != right_rows) {
        throw std::invalid_argument("Inner dimensions must match for matrix multiplication");
    }
    
    // Prepare the output flat vector with zeros
    std::vector<double> result_data(left_rows * right_cols, 0.0);
    // We already know how the result will be at the end so we can just prepare it 
    std::vector<std::size_t> result_shape{left_rows, right_cols};

    // The core mathematical engine (this is where INT8 logic will be implemented later)
    for (std::size_t i = 0; i < left_rows; ++i) {
        for (std::size_t j = 0; j < right_cols; ++j) {
            double sum = 0.0;

            // grabbing the raw arrays before the n^3 loop for maximum speed
            const auto& a_data = a.data();
            const auto& b_data = b.data();
            for (std::size_t k = 0; k < left_cols; ++k) {
                // A's row elements * B's column elements
                // Fast way
                sum += a_data[i * left_cols + k] * b_data[k * right_cols + j];
                // Works but slower because having to run to at function millions of times so we can skip it directly
                // sum += a.at({i, k}) * b.at({k, j}); 
            }
            result_data[i * right_cols + j] = sum;
        }
    }

    // Return a new Tensor with shape {M, N}
    return Tensor (
      std::move(result_shape),
      std::move(result_data)
    );
}

[[nodiscard]] Tensor add(const Tensor& a, const Tensor& b) {
    if(a.shape() != b.shape()) {
        throw std::invalid_argument("Shapes must match for addition");
    }
    
    std::vector<double> result_data(a.numel());
    const auto& a_data = a.data();
    const auto& b_data = b.data();
    
    for(std::size_t i = 0; i < a.numel(); ++i) {
        result_data[i] = a_data[i] + b_data[i];
    }
    
    return Tensor(a.shape(), std::move(result_data));
}

[[nodiscard]] Tensor relu(const Tensor& t) {
    // relu function that keeps the positive value and makes all the negative value to be turned into 0
    std::vector<double> result_data(t.numel());
    const auto& t_data = t.data();
    
    for (std::size_t i = 0; i < t.numel(); ++i) {
        // If greater than 0, keep it. Otherwise, clamp to 0.
        result_data[i] = t_data[i] > 0.0 ? t_data[i] : 0.0;
    }
    
    return Tensor(t.shape(), std::move(result_data));
}

[[nodiscard]] Tensor sigmoid(const Tensor& t) {
    // Sometimes we want more control over output range thats where sigmoid function comes it makes the result more 
    // smooth in between 0 to 1, extremely negative tends to be toward zero and extremely positive tends to be toward 1
    std::vector<double> result_data(t.numel());
    const auto& t_data = t.data();
    
    for (std::size_t i = 0; i < t.numel(); ++i) {
        result_data[i] = 1.0 / (1.0 + std::exp(-t_data[i]));
    }
    
    return Tensor(t.shape(), std::move(result_data));
}

[[nodiscard]] Tensor tanh(const Tensor& t) {
    // Sometimes we want to have -1 to +1 range as the output range like for example how in case of stearing wheel
    // we'll need -1 to represent wheel at the maximum left and +1 represent the maximum right and 0 being straigt ahead
    // thats where tanh or the hyperbole of tangent is used for
    std::vector<double> result_data(t.numel());
    const auto& t_data = t.data();

    for (std::size_t i = 0; i < t.numel(); ++i) {
        result_data[i] = std::tanh(t_data[i]);
    }
    return Tensor(t.shape(), std::move(result_data));   
}

[[nodiscard]] std::size_t argmax(const Tensor& t){
    // The purpose of this function is to boil of probabilities down to a single final answer (the winning index)
    if(t.numel() == 0){
        throw std::invalid_argument("cannot find argmax of an empty tensor");
    }
    const auto& t_data = t.data();
    std::size_t max_idx = 0;
    double max_val = t_data[0];

    // loop through the flat array to find the highest value's position
    for (std::size_t i = 1; i < t.numel(); ++i)
    {
        if(t_data[i] > max_val) {
            max_val = t_data[i];
            max_idx = i;
        }
    }
    return max_idx;
}

void save_tensor(const Tensor& t, const std::string& filepath) {
    std::ofstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file for writing: " + filepath);
    }
    
    // Write the raw bytes of the vector directly to the hard drive
    file.write(reinterpret_cast<const char*>(t.data().data()), t.numel() * sizeof(double));
}

[[nodiscard]] Tensor load_tensor(const std::string& filepath, std::vector<std::size_t> shape) {
    // 1. Calculate how many numbers we expect based on the requested shape
    std::size_t expected_elements = 1;
    for (const std::size_t dim : shape) {
        expected_elements *= dim;
    }

    // 2. Open the file in raw binary mode
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + filepath);
    }

    // 3. Pre-allocate the exact amount of memory needed
    std::vector<double> data(expected_elements);

    // 4. Dump the binary data straight from the hard drive into RAM
    if (!file.read(reinterpret_cast<char*>(data.data()), expected_elements * sizeof(double))) {
        throw std::runtime_error("Failed to read the correct amount of data from: " + filepath);
    }

    return Tensor(std::move(shape), std::move(data));
}


struct QuantizedTensor {
    std::vector<std::size_t> shape;
    std::vector<int8_t> data;
    double scale;
    int32_t zero_point;
};

[[nodiscard]] QuantizedTensor quantize(const Tensor& t) {
    const auto& fp_data = t.data();
    
    // 1. Find the minimum and maximum values in the float tensor
    auto [min_it, max_it] = std::minmax_element(fp_data.begin(), fp_data.end());
    // Ensure 0.0 is always included in our range so zero is accurately represented
    double min_val = std::min(*min_it, 0.0);
    double max_val = std::max(*max_it, 0.0);

    // 2. Define the INT8 hardware limits (-128 to 127)
    const double q_min = -128.0;
    const double q_max = 127.0;

    // 3. Calculate Scale and Zero-Point
    double scale = (max_val - min_val) / (q_max - q_min);
    if (scale == 0.0) {
        scale = 1.0; // Prevent division by zero if all numbers in the tensor are identical
    }

    int32_t zero_point = static_cast<int32_t>(std::round(q_min - (min_val / scale)));
    zero_point = std::clamp(zero_point, -128, 127);

    // 4. Compress every 64-bit double into an 8-bit integer
    std::vector<int8_t> q_data(t.numel());
    for (std::size_t i = 0; i < t.numel(); ++i) {
        int32_t quantized_val = static_cast<int32_t>(std::round(fp_data[i] / scale)) + zero_point;
        // Clamp ensures we never overflow the 8-bit boundaries
        q_data[i] = static_cast<int8_t>(std::clamp(quantized_val, -128, 127));
    }

    return QuantizedTensor{
        t.shape(),
        std::move(q_data),
        scale,
        zero_point
    };
}


[[nodiscard]] Tensor matmul_int8(const QuantizedTensor& A, const QuantizedTensor& B) {
    if (A.shape.size() != 2 || B.shape.size() != 2) {
        throw std::invalid_argument("matmul_int8 requires 2D tensors");
    }
    if (A.shape[1] != B.shape[0]) {
        throw std::invalid_argument("Incompatible dimensions for matmul_int8");
    }

    std::size_t M = A.shape[0];
    std::size_t K = A.shape[1];
    std::size_t N = B.shape[1];

    std::vector<double> result_data(M * N);
    double combined_scale = A.scale * B.scale;

    for (std::size_t i = 0; i < M; ++i) {
        for (std::size_t j = 0; j < N; ++j) {
            int32_t acc = 0; // 32-bit accumulator prevents 8-bit overflow
            for (std::size_t k = 0; k < K; ++k) {
                int32_t a_val = static_cast<int32_t>(A.data[i * K + k]) - A.zero_point;
                int32_t b_val = static_cast<int32_t>(B.data[k * N + j]) - B.zero_point;
                acc += a_val * b_val;
            }
            // Convert the accumulated integer result back to real scale
            result_data[i * N + j] = static_cast<double>(acc) * combined_scale;
        }
    }

    return Tensor({M, N}, std::move(result_data));
}


// Forward pass function for 64-bit floating point for benchmark pupose 
[[nodiscard]] std::size_t forward_pass_fp64( 
    const Tensor& input,
    const Tensor& W1, const Tensor& b1,
    const Tensor& W2, const Tensor& b2) 
{
    Tensor hidden = relu(add(matmul(input, W1), b1));
    Tensor logits = add(matmul(hidden, W2), b2);
    return argmax(logits);
}


[[nodiscard]] std::size_t forward_pass(
    const Tensor& input,
    const QuantizedTensor& q_W1, const Tensor& b1,
    const QuantizedTensor& q_W2, const Tensor& b2) 
{
    // Layer 1: Input -> Hidden Layer
    QuantizedTensor q_input = quantize(input);
    Tensor hidden = matmul_int8(q_input, q_W1); // multiply by layer1 weights
    hidden = add(hidden, b1); // add layer 1 bias
    hidden = relu(hidden); // call relu to zero out the negatives

    // Layer 2: Hidden Layer -> Output Logits
    QuantizedTensor q_hidden = quantize(hidden);
    Tensor logits = matmul_int8(q_hidden, q_W2); // multiply by layer2 weight
    logits = add(logits, b2); // add layer 2 bias

    // Return the index of the highest score
    return argmax(logits);
}


int main(){

    std::vector<std::size_t> shape_1 {2,3};
    std::vector<double> data_1 {0,1,2,3,4,5};

    Tensor t{
        shape_1,
        data_1
    };


    std::cout << "Test 1\n";
    auto& actual_shape_1 = t.shape();
    assert(actual_shape_1 == shape_1);


    std::cout << "Test 2\n";
    auto& actual_data_1 = t.data();
    assert(actual_data_1 == data_1);


    std::cout << "Test 3\n";
    std::size_t actual_rank_1 = t.rank();
    assert(actual_rank_1 == 2);


    std::cout << "Test 4\n";
    std::size_t actual_numel_1 = t.numel();
    assert(actual_numel_1 == 6);



    std::cout<< "Test 5\n";
    bool test_5_threw = false;
    try {
        Tensor({}, {});
    } catch (std::invalid_argument& error) {
        assert(std::string(error.what()) == "Tensor must have at least one dimension");
        test_5_threw = true;
    }
    assert(test_5_threw);



    std::cout << "Test 6\n";
    bool test_6_threw = false;
    try {
        Tensor({1, 0}, {});
    } catch (std::invalid_argument& error) {
        assert(std::string(error.what()) == "Tensor dimensions must be positive");
        test_6_threw = true;
    }
    assert(test_6_threw);


    std::cout << "Test 7\n";
    bool test_7_threw = false;
    try {
        Tensor({2, 3}, {1, 2, 3});
    } catch (std::invalid_argument& error) {
        assert(std::string(error.what()) == "Tensor shape does not match its data");
        test_7_threw = true;
    }
    assert(test_7_threw);


    std::cout << "Test 8\n";
    bool test_8_threw = false;
    try {
        double val_8 = t.at({1,2,3});
    } catch (std::invalid_argument& error) {
        assert(std::string(error.what()) == "Number of indices must match tensor rank");
        test_8_threw = true;
    }
    assert(test_8_threw);



    std::cout << "Test 9\n";
    bool test_9_threw = false;
    try {
        double val_9 = t.at({2,3});
    } catch (std::out_of_range& error) {
        assert(std::string(error.what()) == "Tensor index is outside its dimension");
        test_9_threw = true;
    }
    assert(test_9_threw);



    std::cout << "Test 10: Matrix Multiplication\n";
    // Matrix A: 2x2
    // [1, 2]
    // [3, 4]
    Tensor A({2, 2}, {1, 2, 3, 4});
    // Matrix B: 2x2
    // [2, 0]
    // [1, 2]
    Tensor B({2, 2}, {2, 0, 1, 2});
    // Expected Result C:
    // [ (1*2 + 2*1), (1*0 + 2*2) ] = [ 4, 4 ]
    // [ (3*2 + 4*1), (3*0 + 4*2) ] = [10, 8 ]
    Tensor C = matmul(A, B);

    assert(C.data()[0] == 4);
    assert(C.data()[1] == 4);
    assert(C.data()[2] == 10);
    assert(C.data()[3] == 8);
    assert(C.shape()[0] == 2 && C.shape()[1] == 2);
    std::cout << "Matmul Success!\n";



    std::cout << "Test 11: Element-wise Addition\n";
    // Matrix A: 2x2
    // [1, 2]
    // [3, 4]
    Tensor add_A({2, 2}, {1, 2, 3, 4});
    // Matrix B (Bias): 2x2
    // [10, 20]
    // [30, 40]
    Tensor add_B({2, 2}, {10, 20, 30, 40});
    // Expected Result C:
    // [1+10,  2+20] = [11, 22]
    // [3+30,  4+40] = [33, 44]
    
    Tensor add_C = add(add_A, add_B);
    assert(add_C.data()[0] == 11);
    assert(add_C.data()[1] == 22);
    assert(add_C.data()[2] == 33);
    assert(add_C.data()[3] == 44);
    assert(add_C.shape()[0] == 2 && add_C.shape()[1] == 2);
    std::cout << "Addition Success!\n";

    // Relu test
    std::cout << "Test 12: ReLU Activation\n";
    // Matrix: 2x2 with negative and positive numbers
    Tensor relu_in({2, 2}, {-1.5, 2.0, -0.01, 5.5});
    
    Tensor relu_out = relu(relu_in); 
    assert(relu_out.data()[0] == 0.0);
    assert(relu_out.data()[1] == 2.0);
    assert(relu_out.data()[2] == 0.0);
    assert(relu_out.data()[3] == 5.5); 
    std::cout << "ReLU Success!\n";


    // Sigmoid test
    std::cout << "Test 13: Sigmoid Activation\n";
    Tensor sig_in({2}, {0.0, 20.0});
    Tensor sig_out = sigmoid(sig_in);
    assert(sig_out.data()[0] == 0.5);
    assert(sig_out.data()[1] > 0.999);
    std::cout << "Sigmoid Success!\n";



    // Tanh test
    std::cout << "Test 14: Tanh Activation\n";
    Tensor tanh_in({3}, {0.0, 10.0, -10.0});
    Tensor tanh_out = tanh(tanh_in);
    assert(tanh_out.data()[0] == 0.0);
    assert(tanh_out.data()[1] > 0.999);
    assert(tanh_out.data()[2] < -0.999);
    std::cout << "Tanh Success!\n";


    // argmax test
    std::cout << "Test 15: Argmax\n";
    // Simulating network output for 10 digits (0-9)
    // The highest confidence is 9.5 at indext 7
    Tensor logits({10}, {0.1, 0.0, 0.05, 0.8, 0.1, 0.0, 0.0, 9.5, 0.2, 0.0});
    std::size_t predicted_digit = argmax(logits);
    assert(predicted_digit == 7);
    std::cout << "Argmax Success!\n";


    // Load tensor test
    std::cout << "Test 16: Binary File I/O\n";
    // 1. Create a tensor with specific test values
    Tensor io_original({2, 3}, {1.1, 2.2, 3.3, 4.4, 5.5, 6.6});
    std::string test_filepath = "dummy_weights.bin";
    // 2. Save it to disk
    save_tensor(io_original, test_filepath);
    // 3. Load it back from disk into a brand new tensor
    Tensor io_loaded = load_tensor(test_filepath, {2, 3});
    // 4. Verify the data matches exactly
    assert(io_loaded.shape()[0] == 2 && io_loaded.shape()[1] == 3);
    assert(io_loaded.data()[0] == 1.1);
    assert(io_loaded.data()[2] == 3.3);
    assert(io_loaded.data()[5] == 6.6);
    std::cout << "File I/O Success!\n";


    // Qunatization to INT8 test
    std::cout << "Test 17: INT8 Quantization\n";
    
    // Create a float tensor with negative, zero, and positive values
    Tensor fp_tensor({4}, {-10.0, 0.0, 5.0, 10.0});
    
    QuantizedTensor q_tensor = quantize(fp_tensor);
    
    // Minimum value (-10.0) should map to the bottom of INT8 (-128)
    assert(q_tensor.data[0] == -128);
    // 0.0 should map exactly to our calculated zero_point
    assert(q_tensor.data[1] == q_tensor.zero_point);
    // Maximum value (10.0) should map to the top of INT8 (127)
    assert(q_tensor.data[3] == 127);
    
    std::cout << "Quantization Success! (Scale: " << q_tensor.scale 
              << ", Zero-Point: " << q_tensor.zero_point << ")\n";


    // INT8 matrix multiplication test
    std::cout << "Test 18: INT8 Matrix Multiplication\n";
    
    Tensor fp_A({2, 2}, {1.0, 2.0, 3.0, 4.0});
    Tensor fp_B({2, 2}, {5.0, 6.0, 7.0, 8.0});
    
    // Exact FP64 result: [19.0, 22.0, 43.0, 50.0]
    Tensor exact_out = matmul(fp_A, fp_B);
    
    // Quantize both inputs and run INT8 matmul
    QuantizedTensor q_A = quantize(fp_A);
    QuantizedTensor q_B = quantize(fp_B);
    Tensor approx_out = matmul_int8(q_A, q_B);
    
    // Verify the INT8 approximation is very close to the exact FP64 math
    for (std::size_t i = 0; i < exact_out.numel(); ++i) {
        assert(std::abs(exact_out.data()[i] - approx_out.data()[i]) < 0.5);
    }
    
    std::cout << "INT8 Matmul Success! (Exact[0]: " << exact_out.data()[0] 
              << " vs INT8[0]: " << approx_out.data()[0] << ")\n";



    // Forward pass test
    std::cout << "Test 19: Full INT8 Forward Pass\n";
    
    // 1x4 Input
    Tensor net_input({1, 4}, {1.0, 2.0, 3.0, 4.0});
    
    // Layer 1: 4x3 Weights and 1x3 Bias
    Tensor W1({4, 3}, {
        0.1, 0.2, -0.1,
        0.4, -0.5, 0.2,
        -0.3, 0.1, 0.5,
        0.2, 0.3, -0.4
    });
    Tensor b1({1, 3}, {0.1, 0.1, 0.1});
    
    // Layer 2: 3x2 Weights and 1x2 Bias ( rigged so index 1 wins )
    Tensor W2({3, 2}, {
        -0.5, 1.5,
        -0.2, 1.2,
        -0.1, 0.8
    });
    Tensor b2({1, 2}, {0.0, 0.5});
    
    // Pre-quantize the weights
    QuantizedTensor q_W1 = quantize(W1);
    QuantizedTensor q_W2 = quantize(W2);
    
    // Run inference
    std::size_t prediction = forward_pass(net_input, q_W1, b1, q_W2, b2);
    
    assert(prediction == 1);
    std::cout << "Forward Pass Success! Predicted Class: " << prediction << "\n";
    

    // Live MNIST inference test
    std::cout << "Test 20: Live MNIST INT8 Inference\n";
    
    // 1. Load the exported PyTorch weights, biases, and test image
    Tensor mnist_img = load_tensor("test_img.bin", {1, 784});
    Tensor mnist_w1  = load_tensor("w1.bin", {784, 128});
    Tensor mnist_b1  = load_tensor("b1.bin", {1, 128});
    Tensor mnist_w2  = load_tensor("w2.bin", {128, 10});
    Tensor mnist_b2  = load_tensor("b2.bin", {1, 10});
    
    // 2. Quantize the 64-bit weights down to 8-bit integers
    QuantizedTensor q_mnist_w1 = quantize(mnist_w1);
    QuantizedTensor q_mnist_w2 = quantize(mnist_w2);
    
    // 3. Run the full INT8 forward pass
    std::size_t mnist_pred = forward_pass(mnist_img, q_mnist_w1, mnist_b1, q_mnist_w2, mnist_b2);
    
    std::cout << "C++ INT8 Engine Prediction: " << mnist_pred << " (Expected: 7)\n";
    assert(mnist_pred == 7);
    std::cout << "MNIST Inference Success!\n";


    // Benchmark suite
    std::cout << "\n--- Test 21: 1,000-Image MNIST Benchmark (FP64 vs INT8) ---\n";
    
    const std::size_t num_samples = 1000;
    Tensor batch_imgs = load_tensor("test_images_1k.bin", {num_samples, 784});
    Tensor batch_lbls = load_tensor("test_labels_1k.bin", {num_samples});

    // Slice the flat 1000x784 tensor into 1,000 individual 1x784 image tensors
    std::vector<Tensor> inputs;
    inputs.reserve(num_samples);
    for (std::size_t i = 0; i < num_samples; ++i) {
        auto start_it = batch_imgs.data().begin() + i * 784;
        inputs.emplace_back(std::vector<std::size_t>{1, 784}, std::vector<double>(start_it, start_it + 784));
    }

    // 1. Benchmark FP64
    std::size_t fp64_correct = 0;
    auto t0 = std::chrono::high_resolution_clock::now();
    for (std::size_t i = 0; i < num_samples; ++i) {
        if (forward_pass_fp64(inputs[i], mnist_w1, mnist_b1, mnist_w2, mnist_b2) == static_cast<std::size_t>(batch_lbls.data()[i])) {
            fp64_correct++;
        }
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double fp64_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // 2. Benchmark INT8
    std::size_t int8_correct = 0;
    auto t2 = std::chrono::high_resolution_clock::now();
    for (std::size_t i = 0; i < num_samples; ++i) {
        if (forward_pass(inputs[i], q_mnist_w1, mnist_b1, q_mnist_w2, mnist_b2) == static_cast<std::size_t>(batch_lbls.data()[i])) {
            int8_correct++;
        }
    }
    auto t3 = std::chrono::high_resolution_clock::now();
    double int8_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();

    // 3. Calculate Weight Memory Footprint (W1 + W2)
    double fp64_kb = (mnist_w1.numel() + mnist_w2.numel()) * sizeof(double) / 1024.0;
    double int8_kb = (q_mnist_w1.data.size() + q_mnist_w2.data.size()) * sizeof(int8_t) / 1024.0;

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "FP64 -> Accuracy: " << (fp64_correct / 10.0) << "% | Weight Memory: " << fp64_kb << " KB | Time: " << fp64_ms << " ms\n";
    std::cout << "INT8 -> Accuracy: " << (int8_correct / 10.0) << "% | Weight Memory: " << int8_kb << " KB | Time: " << int8_ms << " ms\n";
    std::cout << "Memory Reduction: 8.00x (" << fp64_kb - int8_kb << " KB saved)\n";


    // without nodiscard just calling t.shape() which is practically useless so we need a warning when we write something like this which is done by nodiscard.
    auto& shape = t.shape();
    std::cout << "Success!\n";

    return 0;
}