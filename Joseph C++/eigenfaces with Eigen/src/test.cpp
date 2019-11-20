#include "test.hpp"
#include "CImg.h"
#include "Eigen/Eigenvalues"
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <fstream>


bool test() {
	std::cout << "TEST" << std::endl;
	return true;
}

VectorOfEigenPairs CreateEigenvectors(const ListOfFaces &faceMatrixes, FaceVector& averageFace) {
	const size_t NUMB_FACES = faceMatrixes.size();
	
	averageFace = Eigen::VectorXf(IMAGE_SIZE2);
	//Well setting it to zero would have helped!
	std::cout << averageFace.size() << std::endl;
	setZero(averageFace);
	averageFace[0] = 0;
	
	for (const FaceVector& faceMatrix : faceMatrixes) {
		averageFace += faceMatrix;
	}
	
	averageFace /= static_cast<float>(NUMB_FACES);
	
	Eigen::MatrixXf normalizedFaceMatrix(IMAGE_SIZE2, NUMB_FACES);
	for (size_t i = 0; i < NUMB_FACES; i++) {
		const FaceVector& faceMatrix = faceMatrixes[i];
		normalizedFaceMatrix.col(static_cast<long>(i)) = (faceMatrix-averageFace);
	}
	
	
	std::cout << "normalizedFaceMatrix: " << std::endl;
	std::cout << normalizedFaceMatrix.rows() << " " << normalizedFaceMatrix.cols() << " " << std::endl;
	std::cout << normalizedFaceMatrix.row(0) << std::endl;
	
	
	/*
	Eigen::MatrixXf covariance;
	
	do {
		std::cout << "Calculating covariance " << covariance(0,0) << std::endl;
	} while (covariance(0,0) > 100000 || covariance(0,0) == INFINITY);
	*/
	
	Eigen::MatrixXf covariance(NUMB_FACES, NUMB_FACES);
	covariance = normalizedFaceMatrix.transpose() * normalizedFaceMatrix;
	/*
	for (size_t i = 0; i < NUMB_FACES; i++) {
		for (size_t j = 0; j < NUMB_FACES; j++) {
			//covariance(i,j) = normalizedFaceVectors[i].dot(normalizedFaceVectors[j]);
			float dotProduct = 0;
			for (size_t dotA = 0; dotA < IMAGE_SIZE2; dotA++) {
				dotProduct += normalizedFaceMatrix(dotA, i) * normalizedFaceMatrix(dotA, j);
			}
			covariance(i, j) = dotProduct;
			
			//covariance(i,j) = normalizedFaceMatrix.col(i).dot(normalizedFaceMatrix.col(j));
		}
	}
	*/
	
	std::cout << "Covariance: " << std::endl;
	std::cout << covariance << std::endl;
		
	//Find all the eigenvectors and eigenvalues
	Eigen::EigenSolver<Eigen::MatrixXf> solver(covariance, true);
	VectorOfEigenPairs eigenStuff;
	Eigen::VectorXf fullEigenvalues = solver.eigenvalues().real();
	Eigen::MatrixXf fullEigenVectors = solver.eigenvectors().real();
	
	//Store in pairs (so we can sort all at once)
	for (int i=0; i<fullEigenvalues.size(); i++) {
		Eigen::VectorXf eigenVectorOfAAt = fullEigenVectors.col(i);
		eigenStuff.push_back(EigenPair(fullEigenvalues[i], (normalizedFaceMatrix*eigenVectorOfAAt).normalized()));
	}
	
	//Sort the eigenpairs by highest eigenvalues
	std::sort(eigenStuff.begin(),eigenStuff.end(), [](EigenPair a, EigenPair b){return a.first > b.first;});
	
	//Truncate Eigenpairs.
	eigenStuff.resize(NUMB_EIGENVECTORS);
	
	return eigenStuff;
}

WeightsVector TurnImageIntoWeights(const FaceVector &faceVectorMinusAverageFace, const VectorOfEigenPairs &eigenStuff) {
	WeightsVector weights(NUMB_EIGENVECTORS);
	for (size_t i=0; i<NUMB_EIGENVECTORS; i++) {
		weights[static_cast<long>(i)] = eigenStuff[i].second.dot(faceVectorMinusAverageFace);
	}
	return weights;
}

WeightsVector TurnImageIntoWeights(const FaceVector &faceVector, const FaceVector &AverageFace, const VectorOfEigenPairs& eigenStuff) {
	return TurnImageIntoWeights((faceVector - AverageFace), eigenStuff);
}

FaceVector TurnWeightsIntoImage(const WeightsVector &weights, const VectorOfEigenPairs &eigenStuff) {
	
	FaceVector image = Eigen::VectorXf::Zero(IMAGE_SIZE2);
	setZero(image);
	for (size_t i=0; i<NUMB_EIGENVECTORS; i++) {
		image += weights[static_cast<long>(i)]*eigenStuff[i].second;
	}
	return image;
}

FaceVector TurnWeightsIntoImage(const WeightsVector &weights, const FaceVector &AverageFace, const VectorOfEigenPairs& eigenStuff) {
	return (TurnWeightsIntoImage(weights, eigenStuff) + AverageFace);
}

std::pair<size_t, float> MatchFace(const WeightsVector &unknownWeight, const ListOfWeights &knownFaceWeights) {
	float minFaceDistance = std::numeric_limits<float>::max();
	size_t minFaceIndex = 0;
	//Find the face with the MINIMUM distance
	for (size_t i=0; i<knownFaceWeights.size(); i++) {
		float faceDistance = CompareFaceWeights(unknownWeight, knownFaceWeights[i]);
		if (faceDistance < minFaceDistance) {
			minFaceDistance = faceDistance;
			minFaceIndex = i;
		}
	}
	return {minFaceIndex, minFaceDistance};
}

FaceVector ParseImage(std::string filename) {
	using namespace cimg_library;
	
	CImg<float> image(filename.c_str());
	image.resize(IMAGE_SIZE, IMAGE_SIZE);
	
	FaceVector imageMatrix(IMAGE_SIZE2);
	
	for (size_t i=0; i<image.width(); i++) {
		for (size_t j=0; j < image.height(); j++) {
			//Fill the matrix with image data.
			imageMatrix[i*IMAGE_SIZE+j] = image(static_cast<unsigned int>(i),static_cast<unsigned int>(j))/256.0f;
		}
	}
	return imageMatrix;
}

void SaveImage(const FaceVector &faceVector, std::string filename) {
	using namespace cimg_library;
	
	CImg<char> image(IMAGE_SIZE, IMAGE_SIZE);
	
	for (int i=0; i<image.width(); i++) {
		for (int j = 0; j < image.height(); j++) {
			//Fill the matrix with image data.
			image(static_cast<unsigned int>(i),static_cast<unsigned int>(j)) = static_cast<char>(std::max(std::min(faceVector[i*IMAGE_SIZE+j], 0.99f)*256.0f, 0.0f));
		}
	}
	image.save(filename.c_str());
}

void PrintImageMatrix(const FaceVector &faceVector) {
	using namespace cimg_library;
	
	
	for (int i=0; i<IMAGE_SIZE; i++) {
		for (int j = 0; j < IMAGE_SIZE; j++) {
			//Fill the matrix with image data.
			std::cout << std::setw(5) << (faceVector[i*IMAGE_SIZE+j]) << " ";
		}
		std::cout << std::endl;
	}
}

void SaveWeights(const WeightsVector &weightsVector, string filename)
{
	std::ofstream saveFile(filename);
	
	for (int i=0; i<NUMB_EIGENVECTORS; i++) {
		saveFile << weightsVector[i] << " ";
	}
}

WeightsVector LoadWeights(string filename)
{
	std::ifstream loadFile(filename);
	WeightsVector weightsVector(NUMB_EIGENVECTORS);
	
	for (int i=0; i<NUMB_EIGENVECTORS; i++) {
		loadFile >> weightsVector[i];
	}
	return weightsVector;
}
