#include <iostream>

#include "matplotlib/matplotlibcpp.h"
namespace plt = matplotlibcpp;

double func(double x) {
    return sin(x);
}
int main() {

    std::vector<double> dots;
    std::vector<double> axis;
    for(double i = 0; i < 2 * Py_MATH_PI; i+= 0.1) {
        axis.push_back(i);
        dots.push_back(func(i));
    }
    plt::plot(axis, dots);
    plt::show();
    return 0;
}
