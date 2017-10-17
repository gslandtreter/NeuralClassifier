#include <iostream>

#include "matplotlib/matplotlibcpp.h"
#include "NeuralNetwork/NeuralNetwork.h"

namespace plt = matplotlibcpp;


int main() {

    std::vector<int> topology;

    topology.push_back(2);
    topology.push_back(5);
    topology.push_back(1);

    NeuralNetwork *myNet = new NeuralNetwork(topology);
    delete myNet;

    /*Neuron *n = new Neuron((vFunctionCall) sigmoid, (vFunctionCall) derivativeSigmoid);
    Neuron *n2 = new Neuron((vFunctionCall) sigmoid, (vFunctionCall) derivativeSigmoid);*/

    std::vector<double> dots;
    std::vector<double> dots2;
    std::vector<double> axis;
    for(double i = -10; i < 10 ; i+= 0.1) {
        axis.push_back(i);
        //dots.push_back(n->activate(i));
        //dots2.push_back(n->derivativeActivation(i));
    }
    //plt::plot(axis, dots);
    //plt::plot(axis, dots2);
    plt::show();

    //delete n;
    //delete n2;
    return 0;
}
