//
// Created by gslandtreter on 10/17/17.
//

#pragma once

#include "NeuralNetwork.h"

class Synapse {

public:
    Synapse(Neuron *source, Neuron *destination);

private:
    Neuron *source;
    Neuron *destination;
};

