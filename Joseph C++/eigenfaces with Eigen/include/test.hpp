#pragma once

#include <vector>
#include <set>
#include <array>
#include "Eigen/Dense"
#include <Eigen/StdVector>

using namespace std;

static const int IMAGE_SIZE = 128;
static const int IMAGE_SIZE2 = IMAGE_SIZE*IMAGE_SIZE;

//40 for faces
//static const int NUMB_EIGENVECTORS = 40;
//10 for trees
static const int NUMB_EIGENVECTORS = 10;


//I got sick of writing out the whole thing, so renamed Eigenpair.
typedef	std::pair<float, Eigen::VectorXf> EigenPair;
typedef std::vector<EigenPair, Eigen::aligned_allocator<EigenPair>> VectorOfEigenPairs;
//typedef Eigen::VectorXf FaceVector;
typedef Eigen::VectorXf FaceVector;
typedef Eigen::VectorXf WeightsVector;
typedef std::vector<FaceVector, Eigen::aligned_allocator<FaceVector>> ListOfFaces;
typedef std::vector<WeightsVector, Eigen::aligned_allocator<WeightsVector>> ListOfWeights;

bool test();

template<typename T>
void setZero(T& thing1) {
	for (size_t i=0; i<thing1.size(); i++) {
		thing1[i] = 0.0f;
	}
}

VectorOfEigenPairs CreateEigenvectors(const ListOfFaces &faceMatrixes, FaceVector& averageFace);

WeightsVector TurnImageIntoWeights(const FaceVector& faceVectorMinusAverageFace, const VectorOfEigenPairs& eigenStuff);

WeightsVector TurnImageIntoWeights(const FaceVector& faceVector, const FaceVector& AverageFace, const VectorOfEigenPairs& eigenStuff);

FaceVector TurnWeightsIntoImage(const WeightsVector& weights, const VectorOfEigenPairs& eigenStuff);

FaceVector TurnWeightsIntoImage(const WeightsVector &weights, const FaceVector &AverageFace, const VectorOfEigenPairs &eigenStuff);

FaceVector ParseImage(std::string filename);


/**
 * @brief CompareFaceWeights Gets the "distance" between faces.
 * @param weights1
 * @param weights2
 * @return 
 */
template<typename T>
float CompareFaceWeights(const T &weights1, const T &weights2) {
	float compare = 0;
	assert(weights1.size() == weights2.size());
	for (int i=0; i<weights1.size(); i++) {
		compare += pow(weights1[i]-weights2[i],2);
	}
	return sqrt(compare);
}

std::pair<size_t, float> MatchFace(const WeightsVector& unknownWeight, const ListOfWeights& knownFaceWeights);

void SaveImage(const FaceVector& faceVector, std::string filename);

void PrintImageMatrix(const FaceVector& faceVector);

void SaveWeights(const WeightsVector& weightsVector, std::string filename);

WeightsVector LoadWeights(std::string filename);
