#pragma once

#include <vector>
#include <set>
#include "Eigen/Dense"

using namespace std;

static const int IMAGE_SIZE = 128;
static const int IMAGE_SIZE2 = IMAGE_SIZE*IMAGE_SIZE;
static const int NUMB_EIGENVECTORS = 30;


//I got sick of writing out the whole thing, so renamed Eigenpair.
typedef	std::pair<float, Eigen::VectorXf> EigenPair;

bool test();

std::vector<EigenPair> CreateEigenvectors(const std::vector<Eigen::VectorXf> &faceMatrixes, Eigen::VectorXf& averageFace);

Eigen::VectorXf TurnImageIntoWeights(const Eigen::VectorXf& faceVectorMinusAverageFace, const std::vector<EigenPair>& eigenStuff);

Eigen::VectorXf TurnImageIntoWeights(const Eigen::VectorXf& faceVector, const Eigen::VectorXf& AverageFace, const std::vector<EigenPair>& eigenStuff);

Eigen::VectorXf TurnWeightsIntoImage(const Eigen::VectorXf& weights, const std::vector<EigenPair>& eigenStuff);

Eigen::VectorXf TurnWeightsIntoImage(const Eigen::VectorXf &weights, const Eigen::VectorXf &AverageFace, const std::vector<EigenPair> &eigenStuff);

Eigen::VectorXf ParseImage(std::string filename);

float CompareFaceWeights(const Eigen::VectorXf& weights1, const Eigen::VectorXf& weights2);

std::pair<size_t, float> MatchFace(Eigen::VectorXf unknownWeight, std::vector<Eigen::VectorXf> knownFaceWeights);

void SaveImage(const Eigen::VectorXf& faceVector, std::string filename);

void PrintImageMatrix(const Eigen::VectorXf& faceVector);
