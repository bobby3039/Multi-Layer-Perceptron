#pragma once

#include <vector>
#include <cmath>
#include <random>


struct Matrix {
    int rows, cols;
    std::vector<double> data;

    Matrix(int r = 0, int c = 0);

    double&       operator()(int i, int j);
    const double& operator()(int i, int j) const;

    Matrix dot(const Matrix& other) const;         
    Matrix add(const Matrix& other) const;          
    Matrix sub(const Matrix& other) const;          
    Matrix multiply(const Matrix& other) const;     
    Matrix transpose() const;
    Matrix multiplyScalar(double scalar) const;
    Matrix map(double (*func)(double)) const;  

    void randomize();
};

double sigmoid(double x);
double dsigmoid(double y);   


class NeuralNetwork {
public:
    NeuralNetwork(int inputNodes, int hiddenNodes, int outputNodes, double learningRate);

    
    Matrix feedForward(const Matrix& input);

    void train(const Matrix& input, const Matrix& target);

private:
    int inputNodes, hiddenNodes, outputNodes;
    double learningRate;

    Matrix weights_ih;   
    Matrix weights_ho;   
    Matrix bias_h;       
    Matrix bias_o;      
};
