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


    std::vector<double> evaluate(std::vector<double> inputs);
    void backPropagate(std::vector<double> targets);
    void update();

    std::vector<double> learn(std::vector<double> inputs, std::vector<double> expectedOutput);

    int classifyEvaluation(std::vector<double> outputs);

protected:

    std::vector<Layer*> layers;
    double alpha;
};