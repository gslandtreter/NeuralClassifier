//
// Created by gslandtreter on 10/17/17.
//

#include "Neuron.h"
#include "Synapse.h"

Neuron::Neuron(vFunctionCall activationFunction, vFunctionCall derivativeActivationFunction) {
    this->activationFunction = activationFunction;
    this->derivativeActivationFunction = derivativeActivationFunction;

    this->inputNeurons.clear();
    this->outputNeurons.clear();
}

double Neuron::activate(double x) {
    return activationFunction(x);
}

double Neuron::derivativeActivation(double x) {
    return derivativeActivationFunction(x);
}

Synapse* Neuron::link(Neuron *linkTo) {
    Synapse *synapse = new Synapse(this, linkTo);

    this->addOutputNeuron(synapse);
    linkTo->addInputNeuron(synapse);

    return synapse;
}

void Neuron::addInputNeuron(Synapse* synapse) {
    this->inputNeurons.push_back(synapse);
}

void Neuron::addOutputNeuron(Synapse* synapse) {
    this->outputNeurons.push_back(synapse);
}
