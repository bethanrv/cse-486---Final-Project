#pragma once

#include <vector>
#include <set>
#include "Eigen/Dense"

using namespace std;

static const int IMAGE_SIZE = 64;
static const int IMAGE_SIZE2 = IMAGE_SIZE*IMAGE_SIZE;
static const int NUMB_EIGENVECTORS = 20;


//I got sick of writing out the whole thing, so renamed Eigenpair.
typedef	std::pair<float, Eigen::VectorXf> EigenPair;

bool test();

std::vector<EigenPair> CreateEigenvectors(std::vector<Eigen::VectorXf> faceMatrixes, Eigen::VectorXf& averageFace);

Eigen::VectorXf TurnImageIntoWeights(Eigen::VectorXf faceVectorMinusAverageFace, std::vector<EigenPair> eigenStuff);

Eigen::VectorXf TurnImageIntoWeights(Eigen::VectorXf faceVector, Eigen::VectorXf AverageFace, std::vector<EigenPair> eigenStuff);

Eigen::VectorXf TurnWeightsIntoImage(Eigen::VectorXf weights, std::vector<EigenPair> eigenStuff);

Eigen::VectorXf TurnWeightsIntoImage(Eigen::VectorXf weights, Eigen::VectorXf AverageFace, std::vector<EigenPair> eigenStuff);

Eigen::VectorXf ParseImage(std::string filename);

void SaveImage(Eigen::VectorXf faceVector, std::string filename);

void PrintImageMatrix(Eigen::VectorXf faceVector);
