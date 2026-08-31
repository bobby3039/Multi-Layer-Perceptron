
#include<bits/stdc++.h>
#include "data.h"
using namespace std;

void loadTrainData(const string& filename,
                   vector<Matrix>& inputs,
                   vector<Matrix>& targets)
{
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Error: could not open " << filename << "\n";
        return;
    }

    string line, val;
    getline(file, line); // skip header row

    while (getline(file, line)) {
        stringstream ss(line);

        Matrix input(784, 1);
        Matrix target(10, 1);

        // First column is the digit label
        getline(ss, val, ',');
        target(stoi(val), 0) = 1.0;   // one-hot encode

        // Remaining 784 columns are pixel values 0-255
        int i = 0;
        while (getline(ss, val, ','))
            input(i++, 0) = stod(val) / 255.0;   // normalize to [0, 1]

        inputs.push_back(input);
        targets.push_back(target);
    }
}

void processTestData(const string& inputFilename,
                     const string& outputFilename,
                     NeuralNetwork& nn)
{
    ifstream file(inputFilename);
    ofstream out(outputFilename);

    if (!file.is_open()) {
        cerr << "Error: could not open " << inputFilename << "\n";
        return;
    }

    out << "ImageId,Label\n";

    string line, val;
    getline(file, line); // skip header row

    int imageId = 1;
    while (getline(file, line)) {
        stringstream ss(line);

        Matrix input(784, 1);
        int i = 0;
        while (getline(ss, val, ','))
            input(i++, 0) = stod(val) / 255.0;

        Matrix prediction = nn.feedForward(input);

        // Pick the output neuron with the highest activation
        int best = 0;
        double best_val = prediction(0, 0);
        for (int j = 1; j < 10; ++j) {
            if (prediction(j, 0) > best_val) {
                best_val = prediction(j, 0);
                best = j;
            }
        }

        out << imageId++ << "," << best << "\n";
    }
}
