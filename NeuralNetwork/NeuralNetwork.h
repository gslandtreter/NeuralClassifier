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

    double getAlpha() const;
    void setAlpha(double alpha);


    double evaluate(std::vector<double> inputs);
    void backPropagate(double target);
    void update();

    double learn(std::vector<double> inputs, double expectedOutput);


protected:

    std::vector<Layer*> layers;
    double alpha;
};