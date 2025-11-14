# Machine Learning Based Calibration of Heston Model

## 🔍 Overview

This project develops a **machine-learning-driven calibration framework** for the Heston stochastic volatility model. Traditional analytical calibration methods are computationally heavy, especially in iterative pricing workflows. By training a **feedforward neural network** to approximate the model's analytical pricing function, the framework delivers fast forward evaluation and global-optimization-based calibration. Results show the ML approach reaches **near-analytical accuracy** while reducing computation time by **~60%**, proving its value for real-world derivatives pricing.

## 🛠 Tools

- **Language**: Python.
- **Packages**: TensorFlow/Keras; NumPy, Pandas, SciPy; QuantLib.
- GPU (AWS DLAMI + NVIDIA T4) used for training.

## 🧠 Methodology

The project follows a two-stage framework combining neural-network
approximation with global optimization.

### **Forward Pass: Neural Network Approximation**

A feedforward neural network is trained using millions of synthetic
samples generated from the analytical Heston formula.\
Two formulations are evaluated:
- **Parameters → IV Surface (IVS)**
- **Parameters → Single IV**

The models use ReLU activation, batch normalization, MSE loss, and Adam
optimization. Multiple architectures are tested to identify the optimal
model.

### **Backward Pass: Calibration with Differential Evolution**

The trained network replaces the analytical solution inside a
calibration routine.\
Calibration minimizes the error between observed IV surfaces and
NN-generated IVs using **differential evolution**, a global and
derivative-free optimizer.\
Evaluation metrics include parameter error, IV surface error, and
computation time, benchmarked against analytical QuantLib calibration.

## 📊 Results

The ML-based calibration framework shows strong performance:

### **Accuracy**

-   Best model ("Model E") achieves **\~0.1% IV surface error**, close
    to the benchmark of \~0.05% using analytical calibration. 
-   Real market IVS (AAPL, FB): **3.8% average error**, matching the performance using
    analytical calibration.

### **Speed**

-   Per-iteration calibration time: **25~75%** of analytical solution.
-   Synthetic data: **\~38%** of analytical solution runtime.
-   Real data: **\~62%** of analytical solution runtime.
-   Neural networks achieve speedup via parallelized inference.

### **Implications**

The framework is accurate, fast, and extendable to other financial
models, making it suitable for intraday and production-level calibration
workflows.
