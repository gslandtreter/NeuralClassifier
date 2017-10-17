//
// Created by gslandtreter on 10/17/17.
//

#pragma once

#include <vector>
#include "Neuron.h"

class Neuron;

class Layer {

public:
    Layer(int);
    void addNeuron(Neuron* neuron);


private:
    std::vector<Neuron *> neurons;
public:
    const std::vector<Neuron *> &getNeurons() const;

private:
    int numOutputs;
public:
    int getNumOutputs() const;

    void setNumOutputs(int numOutputs);
};

