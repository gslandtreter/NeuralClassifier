//
// Created by gslandtreter on 10/17/17.
//
#pragma once

#include <list>
#include "Layer.h"
class Layer;
class Neuron;

typedef double (* vFunctionCall)(double);

class Synapse {
protected:
    Neuron* source;
    Neuron* destination;
    double weight;

    double error;
    double totalError;
    int processedErrors;


public:
    void initRandomWeight();
    Synapse(Neuron* source, Neuron* destination);

    Neuron *getSource() const;
    Neuron *getDestination() const;

    double getWeight() const;

    void setWeight(double weight);

    double getError() const;

    void setError(double error);

    double getTotalError() const;

    void setTotalError(double totalError);

    int getProcessedErrors() const;

    void setProcessedErrors(int processedErrors);
    void increaseProcessedErrors();

};

class Neuron {

public:
    Neuron(vFunctionCall activationFunction, vFunctionCall derivativeActivationFunction, Layer*, double);
    double activate(double x);
    double derivativeActivation(double x);

    void setMyLayer(Layer *layer);
    double evaluateOutput();

    double getOutputVal() const;
    void setOutputVal(double outputVal);
    double getBias() const;
    void setBias(double bias);
    Layer *getPreviousLayer() const;
    void setPreviousLayer(Layer *previousLayer);

    const std::vector<Synapse *> &getInputSynapses() const;

    const std::vector<Synapse *> &getOutputSynapses() const;

    double getError() const;

    void setError(double error);

    double getInputError() const;

    void setInputError(double inputError);

    double getTotalInputError() const;

    void setTotalInputError(double totalInputError);

    int getProcessedErrors() const;

    void setProcessedErrors(int processedErrors);

    double getInputValue() const;
    void increaseProcessedErrors();

protected:

    double bias;
    double inputValue;
    double outputVal;
    double error;

    double inputError;
    double totalInputError;

    int processedErrors;


    vFunctionCall activationFunction;
    vFunctionCall derivativeActivationFunction;

    Layer* myLayer;
    Layer* previousLayer;

    std::vector<Synapse*> inputSynapses;
    std::vector<Synapse*> outputSynapses;

};



