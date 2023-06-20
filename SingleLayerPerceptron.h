//
// Created by lev on 20.06.23.
//

#ifndef LINEAR_MODEL_CPP_IMPL_SINGLELAYERPERCEPTRON_H
#define LINEAR_MODEL_CPP_IMPL_SINGLELAYERPERCEPTRON_H

#include <vector>
#include <cmath>
#include <iostream>

class SingleLayerPerceptron {
public:
    std::vector<std::vector<double>> inputs;
    std::vector<double> targets;
    SingleLayerPerceptron (std::vector<std::vector<double>> inputs,  std::vector<double> targets);
    double weights[3] = {0.0, 0.0, 0.0};
    double lr = 0.2, bias = 1;

    // (forward pass)
    double getY (std::vector<double> X);
    double sigmoidfunc (double y);
    double thresholdOut (double out);
    // (backward pass)
    double calculateError (double target, double res);
    void updateWeights (std::vector<double> X, double error);

};


#endif //LINEAR_MODEL_CPP_IMPL_SINGLELAYERPERCEPTRON_H
