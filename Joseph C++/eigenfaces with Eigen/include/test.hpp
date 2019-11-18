#pragma once
#define EIGEN_DONT_ALIGN

#include <vector>
#include <set>
#include "Eigen/Dense"
#include <Eigen/StdVector>

using namespace std;

static const int IMAGE_SIZE = 256;
static const int IMAGE_SIZE2 = IMAGE_SIZE*IMAGE_SIZE;
static const int NUMB_EIGENVECTORS = 5;


//I got sick of writing out the whole thing, so renamed Eigenpair.
typedef	std::pair<float, Eigen::VectorXf> EigenPair;
typedef std::vector<Eigen::VectorXf, Eigen::aligned_allocator<Eigen::VectorXf>> VectorOfVectors;
typedef std::vector<EigenPair, Eigen::aligned_allocator<EigenPair>> VectorOfEigenPairs;

bool test();

VectorOfEigenPairs CreateEigenvectors(const VectorOfVectors &faceMatrixes, Eigen::VectorXf& averageFace);

Eigen::VectorXf TurnImageIntoWeights(const Eigen::VectorXf& faceVectorMinusAverageFace, const VectorOfEigenPairs& eigenStuff);

Eigen::VectorXf TurnImageIntoWeights(const Eigen::VectorXf& faceVector, const Eigen::VectorXf& AverageFace, const VectorOfEigenPairs& eigenStuff);

Eigen::VectorXf TurnWeightsIntoImage(const Eigen::VectorXf& weights, const VectorOfEigenPairs& eigenStuff);

Eigen::VectorXf TurnWeightsIntoImage(const Eigen::VectorXf &weights, const Eigen::VectorXf &AverageFace, const VectorOfEigenPairs &eigenStuff);

Eigen::VectorXf ParseImage(std::string filename);

float CompareFaceWeights(const Eigen::VectorXf& weights1, const Eigen::VectorXf& weights2);

std::pair<size_t, float> MatchFace(const Eigen::VectorXf& unknownWeight, const VectorOfVectors& knownFaceWeights);

void SaveImage(const Eigen::VectorXf& faceVector, std::string filename);

void PrintImageMatrix(const Eigen::VectorXf& faceVector);
