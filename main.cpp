#include <cstddef>
#include <stdexcept>
#include <vector>
#include <iostream>
#include <cassert>
class Tensor{
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

























    // without nodiscard just calling t.shape() which is practically useless so we need a warning when we write something like this which is done by nodiscard.
    auto& shape = t.shape();
    std::cout << "Success!\n";

    return 0;
}