# ml-cpp

A machine learning library built from scratch in C++ with no external dependencies.

Currently implements a two-layer perceptron that trains on MNIST, including matrix math, backpropagation, and SGD — all written from raw C++17.

## Results

```
Epoch 1 | Loss: 0.3589 | Acc: 93.14% | Time: 5.36s
Epoch 2 | Loss: 0.1954 | Acc: 94.88% | Time: 5.35s
Epoch 3 | Loss: 0.1483 | Acc: 95.78% | Time: 5.34s
Epoch 4 | Loss: 0.1201 | Acc: 96.42% | Time: 5.36s
Epoch 5 | Loss: 0.1010 | Acc: 96.75% | Time: 5.38s
```

96.75% test accuracy on MNIST with a 784 → 128 (ReLU) → 10 network.

## What's implemented

- Tensor struct with row-major storage
- Forward ops: matmul, bias add, ReLU, softmax (numerically stable)
- Backward ops: manual gradients for all of the above
- Fused softmax + cross-entropy backward
- SGD optimizer
- MNIST IDX binary format parser
- Batched training loop (batch size 64)

Everything lives in a single `v0_mnist.cpp` - no frameworks, no BLAS, no dependencies beyond the standard library.

## Build

```bash
g++ -O2 -std=c++17 v0_mnist.cpp -o v0_mnist
./v0_mnist
```

Requires MNIST data files in `data/`:

```bash
mkdir -p data && cd data
wget http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
wget http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
wget http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
wget http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz
gunzip *.gz
```

## Roadmap

- [ ] Autograd engine (reverse-mode automatic differentiation)
- [ ] Module/layer abstractions (Linear, Sequential)
- [ ] Op dispatch system for backend extensibility
- [ ] Performance: tiled matmul, OpenMP
- [ ] Conv2d (im2col approach)
- [ ] CUDA backend
