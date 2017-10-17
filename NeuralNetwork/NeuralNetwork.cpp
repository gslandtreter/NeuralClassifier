//
// Created by gustavo_landtreter on 10/17/17.
//

#include <cmath>
#include "NeuralNetwork.h"


double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

double derivativeSigmoid(double x) {
    double result = sigmoid(x);
    return result * (1.0 - result);
}

NeuralNetwork::NeuralNetwork(const std::vector<int> &topology) {

    int numLayers = (int)topology.size();
    for(int i = 0; i < numLayers; i++) {

        int numNeurons = topology[i];
        int numOutputs = 0;

        if(i != topology.size() -1) {
            numOutputs = topology[i+1];
        }

        Layer *newLayer = new Layer(numOutputs);
        this->layers.push_back(newLayer);

        for(int j = 0; j < numNeurons; j++) {
            Neuron *newNeuron = new Neuron((vFunctionCall) sigmoid, (vFunctionCall) derivativeSigmoid);
            newNeuron->setMyLayer(newLayer);
            newLayer->addNeuron(newNeuron);
        }

        newLayer->getNeurons().back()->setOutputVal(1.0);
    }
}

NeuralNetwork::~NeuralNetwork() {

    int numLayers = (int)this->layers.size();
    for(int i = 0; i < numLayers; i++) {
        Layer *layer = this->layers[i];

        int numNeurons = (int)layer->getNeurons().size();

        for(int j = 0; j < numNeurons; j++) {
            delete layer->getNeurons()[j];
        }

        delete layer;
    }
}