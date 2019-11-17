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

std::vector<EigenPair> CreateEigenvectors(std::vector<Eigen::VectorXf> faceMatrixes, Eigen::VectorXf& averageFace) {
	std::cout << "TEST" << std::endl;
	const size_t NUMB_FACES = faceMatrixes.size();
	
	averageFace = Eigen::VectorXf(IMAGE_SIZE2);
	for (Eigen::VectorXf& faceMatrix : faceMatrixes) {
		averageFace += faceMatrix;
	}
	averageFace /= static_cast<float>(NUMB_FACES);
	
	//std::vector<Eigen::VectorXf> normalizedFaceVectors;
	Eigen::MatrixXf normalizedFaceMatrix(IMAGE_SIZE2, NUMB_FACES);// = normalizedFaceMatrix.transpose() * normalizedFaceMatrix;
	for (size_t i = 0; i < NUMB_FACES; i++) {
		Eigen::VectorXf& faceMatrix = faceMatrixes[i];
		normalizedFaceMatrix.col(i) = (faceMatrix-averageFace);
	}
	
	//Eigen::MatrixXf normalizedFaceMatrix(IMAGE_SIZE2, NUMB_FACES);
	//Eigen::MatrixXf& A = normalizedFaceMatrix;
	Eigen::MatrixXf covariance = normalizedFaceMatrix.transpose() * normalizedFaceMatrix;
	/*
	Eigen::MatrixXf covariance(NUMB_FACES, NUMB_FACES);
	for (size_t i = 0; i < NUMB_FACES; i++) {
		for (size_t j = 0; i < NUMB_FACES; i++) {
			//covariance(i,j) = normalizedFaceVectors[i].dot(normalizedFaceVectors[j]);
			//normalizedFaceMatrix.col(0) = faceMatrixes[i]-averageFace;
		}
	}
	*/
	
	std::cout << covariance << std::endl;
	
	//assert(covariance == covariance.transpose());
	
	//Find all the eigenvectors and eigenvalues
	Eigen::EigenSolver<Eigen::MatrixXf> solver(covariance, true);
	std::vector<EigenPair> eigenStuff;
	Eigen::VectorXf fullEigenvalues = solver.eigenvalues().real();
	Eigen::MatrixXf fullEigenVectors = solver.eigenvectors().real();
	
	//Store in pairs (so we can sort all at once)
	for (int i=0; i<fullEigenvalues.size(); i++) {
		Eigen::VectorXf eigenVectorOfAAt = fullEigenVectors.col(i);
		eigenStuff.push_back(EigenPair(fullEigenvalues[i], normalizedFaceMatrix*eigenVectorOfAAt));
	}
	
	//Sort the eigenpairs by highest eigenvalues
	std::sort(eigenStuff.begin(),eigenStuff.end(), [](EigenPair a, EigenPair b){return abs(a.first) > abs(b.first);});
	
	//Truncate Eigenpairs.
	eigenStuff.resize(NUMB_EIGENVECTORS);
	
	return eigenStuff;
}

Eigen::VectorXf TurnImageIntoWeights(Eigen::VectorXf faceVectorMinusAverageFace, std::vector<EigenPair> eigenStuff) {
	
	Eigen::VectorXf weights(NUMB_EIGENVECTORS);
	for (size_t i=0; i<NUMB_EIGENVECTORS; i++) {
		weights[i] = eigenStuff[i].second.dot(faceVectorMinusAverageFace);
	}
	return weights;
}

Eigen::VectorXf TurnImageIntoWeights(Eigen::VectorXf faceVector, Eigen::VectorXf AverageFace, std::vector<EigenPair> eigenStuff) {
	return TurnImageIntoWeights(faceVector-AverageFace, eigenStuff);
}

Eigen::VectorXf TurnWeightsIntoImage(Eigen::VectorXf weights, std::vector<EigenPair> eigenStuff) {
	
	Eigen::VectorXf image = Eigen::VectorXf::Zero(IMAGE_SIZE2);
	for (size_t i=0; i<NUMB_EIGENVECTORS; i++) {
		image += weights[i]*eigenStuff[i].second;
	}
	return image;
}

Eigen::VectorXf TurnWeightsIntoImage(Eigen::VectorXf weights, Eigen::VectorXf AverageFace, std::vector<EigenPair> eigenStuff) {
	return (TurnWeightsIntoImage(weights, eigenStuff)+AverageFace);
}

Eigen::VectorXf ParseImage(std::string filename) {
	using namespace cimg_library;
	
	CImg<float> image = CImg<>(filename.c_str());
	image.resize(IMAGE_SIZE, IMAGE_SIZE);
	
	Eigen::VectorXf imageMatrix(IMAGE_SIZE2);
	
	for (int i=0; i<image.width(); i++) {
		for (int j = 0; j < image.height(); j++) {
			//Fill the matrix with image data.
			imageMatrix(i*IMAGE_SIZE+j) = image(static_cast<unsigned int>(i),static_cast<unsigned int>(j));
		}
	}
	return imageMatrix;
}

void SaveImage(Eigen::VectorXf faceVector, std::string filename) {
	using namespace cimg_library;
	
	CImg<float> image = CImg<>(IMAGE_SIZE, IMAGE_SIZE);
	//image.resize(IMAGE_SIZE, IMAGE_SIZE);
	
	for (int i=0; i<image.width(); i++) {
		for (int j = 0; j < image.height(); j++) {
			//Fill the matrix with image data.
			image(static_cast<unsigned int>(i),static_cast<unsigned int>(j)) = faceVector(i*IMAGE_SIZE+j)/256.0f;
		}
	}
	image.save(filename.c_str());
}

void PrintImageMatrix(Eigen::VectorXf faceVector) {
	using namespace cimg_library;
	
	
	for (int i=0; i<IMAGE_SIZE; i++) {
		for (int j = 0; j < IMAGE_SIZE; j++) {
			//Fill the matrix with image data.
			std::cout << std::setw(5) << (faceVector(i*IMAGE_SIZE+j)*256.0f) << " ";
		}
		std::cout << std::endl;
	}
}
