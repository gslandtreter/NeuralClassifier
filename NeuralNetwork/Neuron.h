//
// Created by gslandtreter on 10/17/17.
//
#pragma once

#include <list>
#include "Layer.h"
class Layer;

typedef double (* vFunctionCall)(double);
class Neuron {

public:
    Neuron(vFunctionCall activationFunction, vFunctionCall derivativeActivationFunction);
    double activate(double x);
    double derivativeActivation(double x);

    void setMyLayer(Layer *layer);


protected:

    static double theta;
    static double alpha;

    static double initRandomWeight();
    double outputVal;
public:
    double getOutputVal() const;

    void setOutputVal(double outputVal);

protected:

    vFunctionCall activationFunction;
    vFunctionCall derivativeActivationFunction;

    Layer* myLayer;

};



