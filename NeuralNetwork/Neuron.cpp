//
// Created by gslandtreter on 10/17/17.
//

#include <cstdlib>
#include "Neuron.h"
#include "Layer.h"

double Neuron::theta = 0.15;
double Neuron::alpha = 0.5;

Neuron::Neuron(vFunctionCall activationFunction, vFunctionCall derivativeActivationFunction) {
    this->activationFunction = activationFunction;
    this->derivativeActivationFunction = derivativeActivationFunction;
}

double Neuron::activate(double x) {
    return activationFunction(x);
}

double Neuron::derivativeActivation(double x) {
    return derivativeActivationFunction(x);
}


double Neuron::initRandomWeight() {
    return rand() / (double) RAND_MAX;
}

void Neuron::setMyLayer(Layer *layer) {
    this->myLayer = layer;
}

double Neuron::getOutputVal() const {
    return outputVal;
}

void Neuron::setOutputVal(double outputVal) {
    this->outputVal = outputVal;
}
