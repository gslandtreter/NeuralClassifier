//
// Created by gslandtreter on 10/17/17.
//

#include <cstdlib>
#include "Neuron.h"

Neuron::Neuron(vFunctionCall activationFunction, vFunctionCall derivativeActivationFunction, Layer* previousLayer, double bias) {
    this->activationFunction = activationFunction;
    this->derivativeActivationFunction = derivativeActivationFunction;
    this->setPreviousLayer(previousLayer);
    this->setBias(bias);

    if(this->previousLayer != nullptr) {
        for(int i = 0; i < previousLayer->getNeurons().size(); i++) {
            Neuron* sourceNeuron = previousLayer->getNeurons()[i];

            Synapse* newSynapse = new Synapse(sourceNeuron, this);
            newSynapse->initRandomWeight();

            this->inputSynapses.push_back(newSynapse);
            sourceNeuron->outputSynapses.push_back(newSynapse);
        }
    }
}

double Neuron::activate(double x) {
    return activationFunction(x);
}

double Neuron::derivativeActivation(double x) {
    return derivativeActivationFunction(x);
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

Layer *Neuron::getPreviousLayer() const {
    return previousLayer;
}

void Neuron::setPreviousLayer(Layer *previousLayer) {
    Neuron::previousLayer = previousLayer;
}

double Neuron::evaluateOutput() {
    this->inputValue = this->bias;

    for(int i = 0; i < this->inputSynapses.size(); i++) {
        Synapse* synapse = this->inputSynapses[i];
        this->inputValue += synapse->getSource()->getOutputVal() * synapse->getWeight();
    }

    this->outputVal = this->activate(this->inputValue);
    return this->outputVal;
}

double Neuron::getBias() const {
    return bias;
}

void Neuron::setBias(double bias) {
    Neuron::bias = bias;
}

const std::vector<Synapse *> &Neuron::getInputSynapses() const {
    return inputSynapses;
}

const std::vector<Synapse *> &Neuron::getOutputSynapses() const {
    return outputSynapses;
}

double Neuron::getError() const {
    return error;
}

void Neuron::setError(double error) {
    Neuron::error = error;
}

double Neuron::getInputError() const {
    return inputError;
}

void Neuron::setInputError(double inputError) {
    Neuron::inputError = inputError;
}

double Neuron::getTotalInputError() const {
    return totalInputError;
}

void Neuron::setTotalInputError(double totalInputError) {
    Neuron::totalInputError = totalInputError;
}

int Neuron::getProcessedErrors() const {
    return processedErrors;
}

void Neuron::setProcessedErrors(int processedErrors) {
    Neuron::processedErrors = processedErrors;
}

double Neuron::getInputValue() const {
    return inputValue;
}

void Neuron::increaseProcessedErrors() {
    this->processedErrors++;
}

void Synapse::initRandomWeight() {
    this->weight = rand() / (double) RAND_MAX;
}

Synapse::Synapse(Neuron *source, Neuron *destination) {
    this->source = source;
    this->destination = destination;
}

Neuron *Synapse::getSource() const {
    return source;
}

Neuron *Synapse::getDestination() const {
    return destination;
}

double Synapse::getError() const {
    return error;
}

void Synapse::setError(double error) {
    Synapse::error = error;
}

double Synapse::getTotalError() const {
    return totalError;
}

void Synapse::setTotalError(double totalError) {
    Synapse::totalError = totalError;
}

int Synapse::getProcessedErrors() const {
    return processedErrors;
}

void Synapse::setProcessedErrors(int processedErrors) {
    Synapse::processedErrors = processedErrors;
}

void Synapse::increaseProcessedErrors() {
    this->processedErrors++;
}

double Synapse::getWeight() const {
    return weight;
}

void Synapse::setWeight(double weight) {
    Synapse::weight = weight;
}

