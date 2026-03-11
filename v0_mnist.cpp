#include <vector>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <random>
#include <cassert>
#include <algorithm>
#include <chrono>

// ============================================================
// TENSOR: just data + shape, nothing else
// ============================================================
struct Tensor {
    std::vector<float> data;
    size_t rows, cols;

    Tensor() : rows(0), cols(0) {}
    Tensor(size_t r, size_t c) : data(r * c, 0.f), rows(r), cols(c) {}

    float& at(size_t i, size_t j) { return data[i * cols + j]; }
    float  at(size_t i, size_t j) const { return data[i * cols + j]; }
    size_t size() const { return data.size(); }

    void zero() { std::fill(data.begin(), data.end(), 0.f); }

    static Tensor randn(size_t r, size_t c, float std = 0.01f) {
        Tensor t(r, c);
        static std::mt19937 gen(42);
        std::normal_distribution<float> dist(0.f, std);
        for (auto& v : t.data) v = dist(gen);
        return t;
    }
};

// ============================================================
// FORWARD OPS (no graph, just compute)
// ============================================================

// C = A @ B  |  A:[M,K] B:[K,N] -> C:[M,N]
Tensor matmul(const Tensor& A, const Tensor& B) {
    assert(A.cols == B.rows);
    Tensor C(A.rows, B.cols);
    for (size_t i = 0; i < A.rows; ++i)
        for (size_t k = 0; k < A.cols; ++k) {
            float a = A.at(i, k);
            for (size_t j = 0; j < B.cols; ++j)
                C.at(i, j) += a * B.at(k, j);
        }
    return C;
}

// out = X + bias (broadcast bias [1,N] over [M,N])
Tensor add_bias(const Tensor& X, const Tensor& bias) {
    assert(bias.rows == 1 && bias.cols == X.cols);
    Tensor out = X; // copy
    for (size_t i = 0; i < X.rows; ++i)
        for (size_t j = 0; j < X.cols; ++j)
            out.at(i, j) += bias.at(0, j);
    return out;
}

// element-wise relu
Tensor relu(const Tensor& X) {
    Tensor out(X.rows, X.cols);
    for (size_t i = 0; i < X.size(); ++i)
        out.data[i] = X.data[i] > 0.f ? X.data[i] : 0.f;
    return out;
}

// softmax per row (numerically stable)
Tensor softmax(const Tensor& X) {
    Tensor out(X.rows, X.cols);
    for (size_t i = 0; i < X.rows; ++i) {
        float maxv = *std::max_element(&X.data[i * X.cols],
                                        &X.data[(i + 1) * X.cols]);
        float sum = 0.f;
        for (size_t j = 0; j < X.cols; ++j) {
            out.at(i, j) = std::exp(X.at(i, j) - maxv);
            sum += out.at(i, j);
        }
        for (size_t j = 0; j < X.cols; ++j)
            out.at(i, j) /= sum;
    }
    return out;
}

// cross-entropy loss (takes softmax output and one-hot targets)
float cross_entropy(const Tensor& probs, const Tensor& targets) {
    float loss = 0.f;
    for (size_t i = 0; i < probs.rows; ++i)
        for (size_t j = 0; j < probs.cols; ++j)
            loss -= targets.at(i, j) * std::log(probs.at(i, j) + 1e-9f);
    return loss / probs.rows;
}

// ============================================================
// BACKWARD OPS (manual, no graph)
// ============================================================

// For cross_entropy + softmax combined:
// d_logits = (softmax - targets) / batch_size
Tensor softmax_cross_entropy_backward(const Tensor& probs, const Tensor& targets) {
    Tensor grad(probs.rows, probs.cols);
    for (size_t i = 0; i < probs.size(); ++i)
        grad.data[i] = (probs.data[i] - targets.data[i]) / probs.rows;
    return grad;
}

// For C = A @ B:
//   dA = dC @ B^T
//   dB = A^T @ dC
Tensor matmul_dA(const Tensor& dC, const Tensor& B) {
    // dC:[M,N] @ B^T:[N,K] -> [M,K]
    Tensor dA(dC.rows, B.rows);
    for (size_t i = 0; i < dC.rows; ++i)
        for (size_t j = 0; j < B.rows; ++j) {
            float sum = 0.f;
            for (size_t k = 0; k < dC.cols; ++k)
                sum += dC.at(i, k) * B.at(j, k); // B^T[k,j] = B[j,k]
            dA.at(i, j) = sum;
        }
    return dA;
}

Tensor matmul_dB(const Tensor& A, const Tensor& dC) {
    // A^T:[K,M] @ dC:[M,N] -> [K,N]
    Tensor dB(A.cols, dC.cols);
    for (size_t k = 0; k < A.cols; ++k)
        for (size_t j = 0; j < dC.cols; ++j) {
            float sum = 0.f;
            for (size_t i = 0; i < A.rows; ++i)
                sum += A.at(i, k) * dC.at(i, j); // A^T[k,i] = A[i,k]
            dB.at(k, j) = sum;
        }
    return dB;
}

// For out = X + bias:  d_bias = sum(dout, axis=0)
Tensor bias_backward(const Tensor& dout) {
    Tensor db(1, dout.cols);
    for (size_t i = 0; i < dout.rows; ++i)
        for (size_t j = 0; j < dout.cols; ++j)
            db.at(0, j) += dout.at(i, j);
    return db;
}

// For relu: grad * (input > 0)
Tensor relu_backward(const Tensor& grad, const Tensor& input) {
    Tensor out(grad.rows, grad.cols);
    for (size_t i = 0; i < grad.size(); ++i)
        out.data[i] = input.data[i] > 0.f ? grad.data[i] : 0.f;
    return out;
}

// ============================================================
// SGD
// ============================================================
void sgd_update(Tensor& param, const Tensor& grad, float lr) {
    for (size_t i = 0; i < param.size(); ++i)
        param.data[i] -= lr * grad.data[i];
}

// ============================================================
// MNIST LOADER
// ============================================================
uint32_t read_u32_be(std::ifstream& f) {
    uint8_t b[4]; f.read((char*)b, 4);
    return (b[0]<<24)|(b[1]<<16)|(b[2]<<8)|b[3];
}

struct MNISTData {
    Tensor images; // [N, 784]
    Tensor labels; // [N, 10] one-hot
    size_t N;
};

MNISTData load_mnist(const std::string& img_path, const std::string& lbl_path) {
    std::ifstream imgf(img_path, std::ios::binary);
    assert(imgf.is_open());
    assert(read_u32_be(imgf) == 2051);
    uint32_t N = read_u32_be(imgf);
    read_u32_be(imgf); read_u32_be(imgf); // rows, cols

    Tensor images(N, 784);
    std::vector<uint8_t> raw(N * 784);
    imgf.read((char*)raw.data(), raw.size());
    for (size_t i = 0; i < raw.size(); ++i)
        images.data[i] = raw[i] / 255.f;

    std::ifstream lblf(lbl_path, std::ios::binary);
    assert(lblf.is_open());
    assert(read_u32_be(lblf) == 2049);
    assert(read_u32_be(lblf) == N);

    Tensor labels(N, 10);
    for (size_t i = 0; i < N; ++i) {
        uint8_t l; lblf.read((char*)&l, 1);
        labels.at(i, l) = 1.f;
    }

    std::cout << "Loaded " << N << " images\n";
    return {images, labels, N};
}

// ============================================================
// MAIN: Training loop
// ============================================================
int main() {
    // Load data
    auto train = load_mnist("data/train-images-idx3-ubyte",
                             "data/train-labels-idx1-ubyte");
    auto test  = load_mnist("data/t10k-images-idx3-ubyte",
                             "data/t10k-labels-idx1-ubyte");

    // Model: 784 -> 128 (relu) -> 10
    float init_std = std::sqrt(2.f / 784.f); // He init
    Tensor W1 = Tensor::randn(784, 128, init_std);
    Tensor b1(1, 128);
    Tensor W2 = Tensor::randn(128, 10, std::sqrt(2.f / 128.f));
    Tensor b2(1, 10);

    float lr = 0.1f;
    size_t batch_size = 64;
    size_t epochs = 5;

    for (size_t epoch = 0; epoch < epochs; ++epoch) {
        auto t0 = std::chrono::high_resolution_clock::now();
        float total_loss = 0.f;
        size_t num_batches = 0;

        for (size_t start = 0; start < train.N; start += batch_size) {
            size_t end = std::min(start + batch_size, train.N);
            size_t B = end - start;

            // -- Extract batch --
            Tensor x(B, 784), y(B, 10);
            std::memcpy(x.data.data(), &train.images.data[start * 784],
                        B * 784 * sizeof(float));
            std::memcpy(y.data.data(), &train.labels.data[start * 10],
                        B * 10 * sizeof(float));

            // -- Forward --
            Tensor z1 = matmul(x, W1);        // [B, 128]
            Tensor a1 = add_bias(z1, b1);      // [B, 128]
            Tensor h1 = relu(a1);              // [B, 128]
            Tensor z2 = matmul(h1, W2);        // [B, 10]
            Tensor a2 = add_bias(z2, b2);      // [B, 10]
            Tensor probs = softmax(a2);        // [B, 10]
            float loss = cross_entropy(probs, y);
            total_loss += loss;

            // -- Backward --
            // d_a2 = (softmax - targets) / B
            Tensor d_a2 = softmax_cross_entropy_backward(probs, y);

            // d_b2 = sum(d_a2, axis=0)
            Tensor d_b2 = bias_backward(d_a2);

            // d_z2 = d_a2 (bias add is identity w.r.t. input)
            // d_W2 = h1^T @ d_z2
            Tensor d_W2 = matmul_dB(h1, d_a2);

            // d_h1 = d_z2 @ W2^T
            Tensor d_h1 = matmul_dA(d_a2, W2);

            // d_a1 = d_h1 * relu'(a1)
            Tensor d_a1 = relu_backward(d_h1, a1);

            // d_b1 = sum(d_a1, axis=0)
            Tensor d_b1 = bias_backward(d_a1);

            // d_W1 = x^T @ d_a1
            Tensor d_W1 = matmul_dB(x, d_a1);

            // -- Update --
            sgd_update(W1, d_W1, lr);
            sgd_update(b1, d_b1, lr);
            sgd_update(W2, d_W2, lr);
            sgd_update(b2, d_b2, lr);

            ++num_batches;
        }

        // -- Test accuracy --
        size_t correct = 0;
        for (size_t i = 0; i < test.N; ++i) {
            // Forward single sample (or batch, but single is simpler here)
            Tensor x(1, 784);
            std::memcpy(x.data.data(), &test.images.data[i * 784],
                        784 * sizeof(float));
            Tensor z1 = matmul(x, W1);
            Tensor a1 = add_bias(z1, b1);
            Tensor h1 = relu(a1);
            Tensor z2 = matmul(h1, W2);
            Tensor a2 = add_bias(z2, b2);

            size_t pred = 0;
            for (size_t j = 1; j < 10; ++j)
                if (a2.at(0, j) > a2.at(0, pred)) pred = j;

            size_t truth = 0;
            for (size_t j = 0; j < 10; ++j)
                if (test.labels.at(i, j) == 1.f) { truth = j; break; }

            if (pred == truth) ++correct;
        }

        auto t1 = std::chrono::high_resolution_clock::now();
        float elapsed = std::chrono::duration<float>(t1 - t0).count();

        std::cout << "Epoch " << epoch + 1
                  << " | Loss: " << total_loss / num_batches
                  << " | Acc: " << 100.f * correct / test.N << "%"
                  << " | Time: " << elapsed << "s\n";
    }
}