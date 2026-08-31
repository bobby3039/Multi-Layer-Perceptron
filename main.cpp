#include<bits/stdc++.h>
#include "mlp.h"
#include "data.h"

using namespace std;



int main() {
    // Network topology: 784 inputs (28x28 pixels) -> 128 hidden -> 10 outputs (digits 0-9)
    const int    INPUT_SIZE   = 784;
    const int    HIDDEN_SIZE  = 128;
    const int    OUTPUT_SIZE  = 10;
    const double LEARNING_RATE = 0.1;
    const int    EPOCHS        = 3;

    cout << "Initializing neural network (" << INPUT_SIZE
              << " -> " << HIDDEN_SIZE << " -> " << OUTPUT_SIZE << ")...\n";

    NeuralNetwork nn(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, LEARNING_RATE);

    vector<Matrix> train_inputs, train_targets;
    cout << "Loading training data from train.csv...\n";
    loadTrainData("train.csv", train_inputs, train_targets);
    cout << "Loaded " << train_inputs.size() << " samples.\n";

    cout << "Training for " << EPOCHS << " epoch(s)...\n";
    for (int epoch = 1; epoch <= EPOCHS; ++epoch) {
        cout << "  Epoch " << epoch << "/" << EPOCHS << " " << flush;

        for (size_t i = 0; i < train_inputs.size(); ++i) {
            nn.train(train_inputs[i], train_targets[i]);
            if (i % 10000 == 0 && i != 0)
                cout << "." << flush;
        }

        cout << " done\n";
    }

    cout << "Generating predictions -> submission1.csv...\n";
    processTestData("test.csv", "submission1.csv", nn);

    cout << "All done. Upload submission1.csv to Kaggle.\n";
    return 0;
}
