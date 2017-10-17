//
// Created by gslandtreter on 10/17/17.
//

#include "Layer.h"

Layer::Layer(int outputs) {
    this->numOutputs = outputs;
}

void Layer::addNeuron(Neuron *neuron) {
    this->neurons.push_back(neuron);
}

int Layer::getNumOutputs() const {
    return numOutputs;
}

void Layer::setNumOutputs(int numOutputs) {
    this->numOutputs = numOutputs;
}

const std::vector<Neuron *> &Layer::getNeurons() const {
    return neurons;
}
