#include "test.hpp"
#include "CImg.h"
#include "Eigen/Eigenvalues"
#include <iostream>
#include <iomanip>
#include <algorithm>


bool test() {
	std::cout << "TEST" << std::endl;
	return true;
}

VectorOfEigenPairs CreateEigenvectors(const std::vector<Eigen::VectorXf, Eigen::aligned_allocator<Eigen::VectorXf> > &faceMatrixes, Eigen::VectorXf& averageFace) {
	const size_t NUMB_FACES = faceMatrixes.size();
	
	averageFace = Eigen::VectorXf(IMAGE_SIZE2);
	for (const Eigen::VectorXf& faceMatrix : faceMatrixes) {
		
		for (long i=0; i<IMAGE_SIZE2; i++) {
			averageFace[i] += faceMatrix[i];
			if (abs(faceMatrix[i]) > 1) {
				std::cout << "BAD VALUE IN FACE MATRIX at " << i << ": " << faceMatrix[i] << std::endl;
			}
		}
		//averageFace += faceMatrix;
	}
	
	for (long i=0; i<IMAGE_SIZE2; i++) {
		assert(averageFace[i] != INFINITY);
		if (abs(averageFace[i]) > NUMB_FACES) {
			std::cout << "BAD VALUE IN PREDIVIDE MATRIX at " << i << ": " << averageFace[i] << std::endl;
			//assert(false);
		}
	}
	
	for (long i=0; i<IMAGE_SIZE2; i++) {
		averageFace[i] /= static_cast<float>(NUMB_FACES);
	}
	//averageFace /= static_cast<float>(NUMB_FACES);
	
	
	for (long i=0; i<averageFace.size(); i++) {
		assert(averageFace[i] != INFINITY);
		if (abs(averageFace[i]) > 1) {
			std::cout << "BAD VALUE IN AVERAGE MATRIX: " << i << ":" << averageFace[i] << std::endl;
			//assert(false);
		}
	}
	
	Eigen::MatrixXf normalizedFaceMatrix(IMAGE_SIZE2, NUMB_FACES);
	for (size_t i = 0; i < NUMB_FACES; i++) {
		const Eigen::VectorXf& faceMatrix = faceMatrixes[i];
		for (int j=0; j<IMAGE_SIZE2; j++) {
			normalizedFaceMatrix(j, static_cast<long>(i)) = faceMatrix[j]-averageFace[j];
		}
		//normalizedFaceMatrix.col(static_cast<long>(i)) = (faceMatrix-averageFace);
	}
	
	
	std::cout << "normalizedFaceMatrix: " << std::endl;
	std::cout << normalizedFaceMatrix.rows() << " " << normalizedFaceMatrix.cols() << " " << std::endl;
	std::cout << normalizedFaceMatrix.row(0) << std::endl;
	
	for (long i=0; i<normalizedFaceMatrix.rows(); i++) {
		for (long j=0; j<normalizedFaceMatrix.cols(); j++) {
			assert(normalizedFaceMatrix(i, j) != INFINITY);
			if (abs(normalizedFaceMatrix(i, j)) > 1) {
				std::cout << "BAD VALUE: " << i << "," << j << ":" << normalizedFaceMatrix(i, j) << std::endl;
				assert(false);
			}
		}
	}
	
	/*
	Eigen::MatrixXf covariance;
	
	do {
		covariance = normalizedFaceMatrix.transpose() * normalizedFaceMatrix;
		std::cout << "Calculating covariance " << covariance(0,0) << std::endl;
	} while (covariance(0,0) > 100000 || covariance(0,0) == INFINITY);
	*/
	
	Eigen::MatrixXf covariance(NUMB_FACES, NUMB_FACES);
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

Eigen::VectorXf TurnImageIntoWeights(const Eigen::VectorXf &faceVectorMinusAverageFace, const VectorOfEigenPairs &eigenStuff) {
	
	Eigen::VectorXf weights(NUMB_EIGENVECTORS);
	for (size_t i=0; i<NUMB_EIGENVECTORS; i++) {
		weights[static_cast<long>(i)] = eigenStuff[i].second.dot(faceVectorMinusAverageFace);
	}
	return weights;
}

Eigen::VectorXf TurnImageIntoWeights(const Eigen::VectorXf& faceVector, const Eigen::VectorXf& AverageFace, const VectorOfEigenPairs& eigenStuff) {
	return TurnImageIntoWeights(faceVector-AverageFace, eigenStuff);
}

Eigen::VectorXf TurnWeightsIntoImage(const Eigen::VectorXf &weights, const VectorOfEigenPairs &eigenStuff) {
	
	Eigen::VectorXf image = Eigen::VectorXf::Zero(IMAGE_SIZE2);
	for (size_t i=0; i<NUMB_EIGENVECTORS; i++) {
		image += weights[static_cast<long>(i)]*eigenStuff[i].second;
	}
	return image;
}

Eigen::VectorXf TurnWeightsIntoImage(const Eigen::VectorXf& weights, const Eigen::VectorXf& AverageFace, const VectorOfEigenPairs& eigenStuff) {
	return (TurnWeightsIntoImage(weights, eigenStuff)+AverageFace);
}

/**
 * @brief CompareFaceWeights Gets the "distance" between faces.
 * @param weights1
 * @param weights2
 * @return 
 */
float CompareFaceWeights(const Eigen::VectorXf& weights1, const Eigen::VectorXf& weights2) {
	return (weights1-weights2).norm();
}

std::pair<size_t, float> MatchFace(const Eigen::VectorXf &unknownWeight, const VectorOfVectors &knownFaceWeights) {
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

Eigen::VectorXf ParseImage(std::string filename) {
	using namespace cimg_library;
	
	CImg<float> image(filename.c_str());
	image.resize(IMAGE_SIZE, IMAGE_SIZE);
	
	Eigen::VectorXf imageMatrix(IMAGE_SIZE2);
	
	for (int i=0; i<image.width(); i++) {
		for (int j = 0; j < image.height(); j++) {
			//Fill the matrix with image data.
			imageMatrix(i*IMAGE_SIZE+j) = image(static_cast<unsigned int>(i),static_cast<unsigned int>(j))/256.0f;
		}
	}
	return imageMatrix;
}

void SaveImage(const Eigen::VectorXf &faceVector, std::string filename) {
	using namespace cimg_library;
	
	CImg<char> image(IMAGE_SIZE, IMAGE_SIZE);
	//image.resize(IMAGE_SIZE, IMAGE_SIZE);
	
	for (int i=0; i<image.width(); i++) {
		for (int j = 0; j < image.height(); j++) {
			//Fill the matrix with image data.
			image(static_cast<unsigned int>(i),static_cast<unsigned int>(j)) = static_cast<char>(std::max(std::min(faceVector(i*IMAGE_SIZE+j), 0.99f)*256.0f, 0.0f));
		}
	}
	image.save(filename.c_str());
}

void PrintImageMatrix(const Eigen::VectorXf& faceVector) {
	using namespace cimg_library;
	
	
	for (int i=0; i<IMAGE_SIZE; i++) {
		for (int j = 0; j < IMAGE_SIZE; j++) {
			//Fill the matrix with image data.
			std::cout << std::setw(5) << (faceVector(i*IMAGE_SIZE+j)) << " ";
		}
		std::cout << std::endl;
	}
}
