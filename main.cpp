#include <iostream>
#include <fstream>

#include "matplotlib/matplotlibcpp.h"
#include "NeuralNetwork/NeuralNetwork.h"

namespace plt = matplotlibcpp;

double sigmoide(double x) {
    return 1.0 / (1.0 + exp(-x));
}


int main() {


    srand(time(NULL));
    int numLines = 0;

    std::vector<int> topology;

    topology.push_back(3);
    topology.push_back(3);
    topology.push_back(3);
    topology.push_back(1);

    NeuralNetwork *myNet = new NeuralNetwork(topology);

    std::ifstream infile("/home/gustavo_landtreter/Documents/neuraldata/haberman.bin");
    std::string line;

    while(std::getline(infile, line)) {
        numLines++;
    }

    infile.clear();
    infile.seekg(0, infile.beg);

    int trainingPhase = floor(numLines * 0.8);

    int totalErros = 0, totalAcertos = 0;

    while(std::getline(infile, line)) {

        int age, year, auxwtf, dead;
        sscanf(line.c_str(), "%d,%d,%d,%d", &age, &year, &auxwtf, &dead);

        if(dead == 1)
            dead = 0;
        else dead = 1;

        if(trainingPhase-- > 0) {

            myNet->learn({sigmoide(age), sigmoide(year), sigmoide(auxwtf)}, dead);

        }
        else {

            double output = myNet->evaluate({sigmoide(age), sigmoide(year), sigmoide(auxwtf)});
            int isDead;

            if(output >= 0.5) {
                isDead = 1;
            } else {
                isDead = 0;
            }

            if(isDead == dead) {
                printf("Acertou!\n");
                totalAcertos++;
            } else {
                printf("Errou!\n");
                totalErros++;
            }

        }
    }

    printf("Total Acertos: %.02f%%\n", (double(totalAcertos) / double(totalAcertos + totalErros)) * 100);

    delete myNet;

    /*Neuron *n = new Neuron((vFunctionCall) sigmoid, (vFunctionCall) derivativeSigmoid);
    Neuron *n2 = new Neuron((vFunctionCall) sigmoid, (vFunctionCall) derivativeSigmoid);*/

    std::vector<double> dots;
    std::vector<double> dots2;
    std::vector<double> axis;
    for(double i = -10; i < 10 ; i+= 0.1) {
        axis.push_back(i);
        //dots.push_back(n->activate(i));
        //dots2.push_back(n->derivativeActivation(i));
    }
    //plt::plot(axis, dots);
    //plt::plot(axis, dots2);
    plt::show();

    //delete n;
    //delete n2;
    return 0;
}
