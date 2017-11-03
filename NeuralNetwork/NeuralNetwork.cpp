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

double derivativeTanh(double x) {
    double result = tanh(x);
    return 1.0 - (result * result);
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
            Neuron *newNeuron = new Neuron((vFunctionCall) tanh, (vFunctionCall) derivativeTanh, previousLayer, 1);
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


std::vector<double> NeuralNetwork::evaluate(std::vector<double> inputs) {

    std::vector<double> outputs;

    Layer* inputLayer = this->layers[0];

    //Check if input size is correct
    if(inputs.size() != inputLayer->getNeurons().size()) {
        printf("Error! Input size not equal no first layer size!!\n");
        return outputs;
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

    Layer* outputLayer = this->layers[this->layers.size() - 1];

    for(int i = 0; i < outputLayer->getNeurons().size(); i++) {
        outputs.push_back(outputLayer->getNeurons()[i]->getOutputVal());
    }

    return outputs;
}

void NeuralNetwork::backPropagate(std::vector<double> targetValues) {

    Layer* outputLayer = this->layers[this->layers.size() - 1];

    for(int i = 0; i < outputLayer->getNeurons().size(); i++) {
        Neuron* outputNeuron = this->layers[layers.size() - 1]->getNeurons()[i];
        outputNeuron->setError(outputNeuron->getOutputVal() - targetValues[i]);
    }

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

std::vector<double> NeuralNetwork::learn(std::vector<double> inputs, std::vector<double> expectedOutputs) {

    std::vector<double> error;
    Layer* outputLayer = this->layers[this->layers.size() - 1];

    if(expectedOutputs.size() != outputLayer->getNeurons().size()) {
        printf("Error! Output size not equal to output layer size!!\n");
        return error;
    }

    std::vector<double> outputs = this->evaluate(inputs);
    this->backPropagate(expectedOutputs);
    this->update();

    //Return error, aka, euclidean distance
    for(int i = 0; i < outputs.size(); i++){
        error.push_back(sqrt(pow(outputs[i] - expectedOutputs[i], 2)));
    }

    return error;
}

int NeuralNetwork::classifyEvaluation(std::vector<double> output) {

    int predictedClass = 1;
    double predictedValue = output[0];

    for(int i = 1; i < output.size(); i++) {
        if(output[i] > predictedValue) {
            predictedValue = output[i];
            predictedClass = i + 1;
        }
    }

    return predictedClass;
}



