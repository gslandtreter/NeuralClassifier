//
// Created by gslandtreter on 10/17/17.
//
#pragma once

#include <list>
#include "NeuralNetwork.h"

typedef double (* vFunctionCall)(double);

class Neuron {

public:
    Neuron(vFunctionCall activationFunction, vFunctionCall derivativeActivationFunction);
    double activate(double x);
    double derivativeActivation(double x);

    Synapse* link(Neuron* linkTo);
    void addInputNeuron(Synapse* who);
    void addOutputNeuron(Synapse* who);


protected:
    vFunctionCall activationFunction;
    vFunctionCall derivativeActivationFunction;

    std::list<Synapse*> inputNeurons;
    std::list<Synapse*> outputNeurons;

};

