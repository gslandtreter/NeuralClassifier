//
// Created by gustavo_landtreter on 10/17/17.
//

#include <cmath>
#include <cstdio>
#include "NeuralNetwork.h"


double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

double derivativeSigmoid(double x) {
    double result = sigmoid(x);
    return result * (1.0 - result);
}

NeuralNetwork::NeuralNetwork(const std::vector<int> &topology) {

    this->alpha = 0.1;

    int numLayers = (int)topology.size();
    Layer *previousLayer = nullptr;

    for(int i = 0; i < numLayers; i++) {

        int numNeurons = topology[i];
        int numOutputs = 0;

        if(i != topology.size() -1) {
            numOutputs = topology[i+1];
        }

        Layer *newLayer = new Layer(numOutputs);
        this->layers.push_back(newLayer);

        for(int j = 0; j < numNeurons; j++) {
            Neuron *newNeuron = new Neuron((vFunctionCall) sigmoid, (vFunctionCall) derivativeSigmoid, previousLayer, 0.1);
            newLayer->addNeuron(newNeuron);
        }

        previousLayer = newLayer;
    }
}

NeuralNetwork::~NeuralNetwork() {

    int numLayers = (int)this->layers.size();
    for(int i = 0; i < numLayers; i++) {
        Layer *layer = this->layers[i];

        int numNeurons = (int)layer->getNeurons().size();

        for(int j = 0; j < numNeurons; j++) {

            for(int l = 0; l < layer->getNeurons()[j]->getOutputSynapses().size(); l++) {
                delete layer->getNeurons()[j]->getOutputSynapses()[l];
            }

            delete layer->getNeurons()[j];
        }

        delete layer;
    }
}


double NeuralNetwork::getAlpha() const {
    return alpha;
}

void NeuralNetwork::setAlpha(double alpha) {
    NeuralNetwork::alpha = alpha;
}


double NeuralNetwork::evaluate(std::vector<double> inputs) {

    Layer* inputLayer = this->layers[0];

    //Check if input size is correct
    if(inputs.size() != inputLayer->getNeurons().size()) {
        printf("Error! Input size not equal no first layer size!!\n");
        return 0;
    }

    //Update first layer
    for(int i = 0; i < inputLayer->getNeurons().size(); i++) {
        inputLayer->getNeurons()[i]->setOutputVal(inputs[i]);
    }

    for(int i = 1; i < this->layers.size(); i++) {
        Layer* currentLayer = this->layers[i];

        for(int j = 0; j < currentLayer->getNeurons().size(); j++) {
            currentLayer->getNeurons()[j]->evaluateOutput();
        }
    }

    //Network Output
    return this->layers[this->layers.size() - 1]->getNeurons()[0]->getOutputVal();
}

void NeuralNetwork::backPropagate(double targetValue) {

    Neuron* outputNeuron = this->layers[layers.size() - 1]->getNeurons()[0];
    outputNeuron->setError(outputNeuron->getOutputVal() - targetValue);

    for(unsigned long layerN = layers.size() - 1; layerN > 0; layerN--) {
        Layer* layer = this->layers[layerN];

        for(int i = 0; i < layer->getNeurons().size(); i++) {
            Neuron* neuron = layer->getNeurons()[i];

            double inputError = neuron->getError() * neuron->derivativeActivation(neuron->getInputValue());

            neuron->setInputError(inputError);
            neuron->setTotalInputError(neuron->getTotalInputError() + inputError);
            neuron->increaseProcessedErrors();

            for(int j = 0; j < neuron->getInputSynapses().size(); j++) {

                Synapse* synapse = neuron->getInputSynapses()[j];

                //TODO: Regularizacao

                double synError = neuron->getInputError() * synapse->getSource()->getOutputVal();
                synapse->setError(synError);
                synapse->setTotalError(synapse->getTotalError() + synError);
                synapse->increaseProcessedErrors();
            }
        }

        //Input layer
        if(layerN == 1)
            continue;

        Layer* previousLayer = this->layers[layerN - 1];

        //Do the math
        for(int i = 0; i < previousLayer->getNeurons().size(); i++) {
            Neuron* neuron = previousLayer->getNeurons()[i];

            double error = 0;

            for(int j = 0; j < neuron->getOutputSynapses().size(); j++) {
                Synapse* synapse = neuron->getOutputSynapses()[j];
                error += synapse->getWeight() * synapse->getDestination()->getInputError();
            }

            neuron->setError(error);
        }

    }

}

void NeuralNetwork::update() {

    for(unsigned long layerN = 1; layerN < this->layers.size(); layerN++) {

        Layer* layer = this->layers[layerN];

        for(int i = 0; i < layer->getNeurons().size(); i++) {
            Neuron* neuron = layer->getNeurons()[i];

            if(neuron->getProcessedErrors() > 0) {
                double newBiasDelta = (this->alpha * neuron->getTotalInputError()) / neuron->getProcessedErrors();
                neuron->setBias(neuron->getBias() - newBiasDelta);

                neuron->setProcessedErrors(0);
                neuron->setTotalInputError(0);
            }

            for(int j = 0; j < neuron->getInputSynapses().size(); j++) {
                Synapse* synapse = neuron->getInputSynapses()[j];

                //TODO: Regularizacao

                if(synapse->getProcessedErrors() > 0) {
                    double newWeightDelta = (this->alpha / synapse->getProcessedErrors()) * synapse->getTotalError();
                    synapse->setWeight(synapse->getWeight() - newWeightDelta);

                    synapse->setProcessedErrors(0);
                    synapse->setTotalError(0);
                }
            }
        }


    }
}

double NeuralNetwork::learn(std::vector<double> inputs, double expectedOutput) {

    double output = this->evaluate(inputs);
    this->backPropagate(expectedOutput);
    this->update();

    //Return error, aka, euclidean distance
    return sqrt(pow(output - expectedOutput, 2));
}


