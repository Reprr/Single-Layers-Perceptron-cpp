//
// Created by lev on 20.06.23.
//

#include "SingleLayerPerceptron.h"
#include <iostream>

using namespace std;

int main() {
    std::vector<std::vector<double>> inputs { {100,200}, {200,-100}, {50,50}, {50,10},
                                              {-50,-20}, {-70,-50}, {200,200}, {200,-50},
                                              {150,-50}, {180,220}, {0,0}, {-100,-220}
    };
    std::vector<double> targets { 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1};
    SingleLayerPerceptron slp (inputs, targets);
    int epoch_num = 1, input_size = slp.inputs.size(), epochs = 1000000;
    while (true) {
        for (int i = 0; i < input_size; ++i) {
            double y = slp.getY(slp.inputs[i]);
            double error = slp.calculateError(slp.targets[i], slp.thresholdOut(slp.sigmoidfunc(y)));
            slp.updateWeights(slp.inputs[i], error);
        }
        if (epoch_num == epochs) {
            cout << "training done" << endl;
            cout << "weights: " << endl;
            for (double weight : slp.weights) cout << weight << " ";
            break;
        }
        epoch_num++;
    }
}