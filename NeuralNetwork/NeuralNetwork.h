//
// Created by gslandtreter on 10/17/17.
//

#pragma once


#include "Neuron.h"
#include "Layer.h"

#include <vector>

class NeuralNetwork {

public:
    NeuralNetwork(const std::vector<int> &topology);
    ~NeuralNetwork();

protected:

    std::vector<Layer*> layers;
};