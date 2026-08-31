#pragma once

#include <string>
#include <vector>
#include "mlp.h"


void loadTrainData(const std::string& filename,
                   std::vector<Matrix>& inputs,
                   std::vector<Matrix>& targets);


void processTestData(const std::string& inputFilename,
                     const std::string& outputFilename,
                     NeuralNetwork& nn);
