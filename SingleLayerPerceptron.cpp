//
// Created by lev on 20.06.23.
//

#include "SingleLayerPerceptron.h"
#include <iostream>
#include <utility>

SingleLayerPerceptron::SingleLayerPerceptron(std::vector<std::vector<double>> inputs,
                                             std::vector<double> targets) {
    std::cout << "Single Layer Perceptron" << std::endl;
    this->inputs = std::move(inputs);
    this->targets = std::move(targets);
}

double SingleLayerPerceptron::getY(std::vector<double> X) {
    return weights[0] * X[0] + weights[1] * X[1] + weights[2] * bias;
}

double SingleLayerPerceptron::sigmoidfunc(double y) {
    return 1 / exp(-1 * y);
}

double SingleLayerPerceptron::thresholdOut(double out) {
    return out < 0.5 ? 0 : 1;
}

double SingleLayerPerceptron::calculateError(double target, double res) {
    return target - res;
}

void SingleLayerPerceptron::updateWeights(std::vector<double> X, double error) {
    weights[0]  = weights[0] + lr * error * X[0];
    weights[1]  = weights[1] + lr * error * X[1];
    weights[2]  = weights[2] + lr * error * bias;
}
