#include "mlp.h"


Matrix::Matrix(int r, int c) : rows(r), cols(c), data(r * c, 0.0) {}

double& Matrix::operator()(int i, int j)             { return data[i * cols + j]; }
const double& Matrix::operator()(int i, int j) const { return data[i * cols + j]; }

Matrix Matrix::dot(const Matrix& other) const {
    Matrix result(rows, other.cols);
    for (int i = 0; i < rows; ++i) {
        for (int k = 0; k < cols; ++k) {
            double val = (*this)(i, k);
            for (int j = 0; j < other.cols; ++j)
                result(i, j) += val * other(k, j);
        }
    }
    return result;
}

Matrix Matrix::add(const Matrix& other) const {
    Matrix result(rows, cols);
    for (int i = 0; i < data.size(); ++i)
        result.data[i] = data[i] + other.data[i];
    return result;
}

Matrix Matrix::sub(const Matrix& other) const {
    Matrix result(rows, cols);
    for (int i = 0; i < data.size(); ++i)
        result.data[i] = data[i] - other.data[i];
    return result;
}

Matrix Matrix::multiply(const Matrix& other) const {
    Matrix result(rows, cols);
    for (int i = 0; i < data.size(); ++i)
        result.data[i] = data[i] * other.data[i];
    return result;
}

Matrix Matrix::transpose() const {
    Matrix result(cols, rows);
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            result(j, i) = (*this)(i, j);
    return result;
}

Matrix Matrix::multiplyScalar(double scalar) const {
    Matrix result(rows, cols);
    for (int i = 0; i < data.size(); ++i)
        result.data[i] = data[i] * scalar;
    return result;
}

Matrix Matrix::map(double (*func)(double)) const {
    Matrix result(rows, cols);
    for (int i = 0; i < data.size(); ++i)
        result.data[i] = func(data[i]);
    return result;
}

void Matrix::randomize() {
    
    std::mt19937 gen(42);
    std::uniform_real_distribution<> dist(-0.5, 0.5);
    for (int i = 0; i < data.size(); ++i)
        data[i] = dist(gen);
}



double sigmoid(double x)  { return 1.0 / (1.0 + std::exp(-x)); }
double dsigmoid(double y) { return y * (1.0 - y); }

// ============================================================
// NeuralNetwork — implementation
// ============================================================

NeuralNetwork::NeuralNetwork(int in, int hidden, int out, double lr)
    : inputNodes(in), hiddenNodes(hidden), outputNodes(out), learningRate(lr)
{
    weights_ih = Matrix(hiddenNodes, inputNodes);
    weights_ho = Matrix(outputNodes, hiddenNodes);
    weights_ih.randomize();
    weights_ho.randomize();

    bias_h = Matrix(hiddenNodes, 1);
    bias_o = Matrix(outputNodes, 1);
    bias_h.randomize();
    bias_o.randomize();
}

Matrix NeuralNetwork::feedForward(const Matrix& input) {
    // Hidden layer
    Matrix hidden = weights_ih.dot(input).add(bias_h).map(sigmoid);
    // Output layer
    Matrix output = weights_ho.dot(hidden).add(bias_o).map(sigmoid);
    return output;
}

void NeuralNetwork::train(const Matrix& input, const Matrix& target) {
    // --- Forward pass ---
    Matrix hidden = weights_ih.dot(input).add(bias_h).map(sigmoid);
    Matrix output = weights_ho.dot(hidden).add(bias_o).map(sigmoid);

    // --- Backward pass ---

    // Output layer: error = (target - output)
    Matrix output_errors = target.sub(output);

    // Output gradient: delta_k = error * sigmoid'(output)
    Matrix out_grad = output.map(dsigmoid).multiply(output_errors).multiplyScalar(learningRate);

    // Update output weights and biases
    weights_ho = weights_ho.add(out_grad.dot(hidden.transpose()));
    bias_o     = bias_o.add(out_grad);

    // Hidden layer: backprop the error through the output weights
    Matrix hidden_errors = weights_ho.transpose().dot(output_errors);

    // Hidden gradient: delta_j = hidden_errors * sigmoid'(hidden)
    Matrix hidden_grad = hidden.map(dsigmoid).multiply(hidden_errors).multiplyScalar(learningRate);

    // Update hidden weights and biases
    weights_ih = weights_ih.add(hidden_grad.dot(input.transpose()));
    bias_h     = bias_h.add(hidden_grad);
}
