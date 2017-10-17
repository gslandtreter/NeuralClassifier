#include <iostream>

#include "matplotlib/matplotlibcpp.h"
#include "NeuralNetwork/Neuron.h"

namespace plt = matplotlibcpp;

double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

double derivativeSigmoid(double x) {
    double result = sigmoid(x);
    return result * (1.0 - result);
}

int main() {

    Neuron *n = new Neuron((vFunctionCall) sigmoid, (vFunctionCall) derivativeSigmoid);
    Neuron *n2 = new Neuron((vFunctionCall) sigmoid, (vFunctionCall) derivativeSigmoid);

    n->link(n2);

    std::vector<double> dots;
    std::vector<double> dots2;
    std::vector<double> axis;
    for(double i = -10; i < 10 ; i+= 0.1) {
        axis.push_back(i);
        dots.push_back(n->activate(i));
        dots2.push_back(n->derivativeActivation(i));
    }
    plt::plot(axis, dots);
    plt::plot(axis, dots2);
    plt::show();

    delete n;
    delete n2;
    return 0;
}
