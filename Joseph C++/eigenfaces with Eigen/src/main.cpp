
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <iomanip>


#include "test.hpp"
#include "Eigen/Dense"

using namespace std;

static int NUMB_FACES_TO_PARSE = 10;
static int NUMB_FACES_TO_ANALYZE = 3;
static std::string FILENAME_APPEND = "tree_";

Eigen::VectorXf loadImageI(std::string relPath, int i) {
	std::stringstream saveString;
	saveString << SOURCE_DIR << relPath << FILENAME_APPEND << std::setfill('0') << std::setw(6) << i << ".jpg";
	return ParseImage(saveString.str());
}

void analyzeFace(int i, const VectorOfVectors& currentFaceWeights, const Eigen::VectorXf& averageFace, const VectorOfEigenPairs& eigenCrap) {
	std::cout << "Analyzing face " << i << std::endl;
	
	Eigen::VectorXf unknownFace = loadImageI("/../begin_images/", i);
	Eigen::VectorXf unknownFaceWeights = TurnImageIntoWeights(unknownFace, averageFace, eigenCrap);
	Eigen::VectorXf unknownFaceReconstructed = TurnWeightsIntoImage(unknownFaceWeights, averageFace, eigenCrap);
	
	float chanceItsAFace = CompareFaceWeights(unknownFace, unknownFaceReconstructed);
	std::cout << "    Chance the mystery face is a face: " << chanceItsAFace << std::endl;
	
	int matchedFace;
	float matchChance;
	std::tie(matchedFace, matchChance) = MatchFace(unknownFaceWeights, currentFaceWeights);
	std::cout << "    Face matches face: " << matchedFace << " with confidence " << matchChance << std::endl;
	
	std::stringstream saveString;
	saveString << SOURCE_DIR << "/../end_images/" << FILENAME_APPEND << std::setfill('0') << std::setw(6) << i << ".jpg";
	SaveImage(unknownFaceReconstructed, saveString.str());
	
}

int main(int argc, char *argv[]) {	
	VectorOfVectors vectorOfFaces((Eigen::aligned_allocator<Eigen::VectorXf>()));
	
	for (int i=0; i<NUMB_FACES_TO_PARSE; i++) {
		Eigen::VectorXf face = loadImageI("/../begin_images/", i+1);
		vectorOfFaces.push_back(face);	
		if (i == 1) {
			//PrintImageMatrix(face);
		}
		
		std::stringstream saveStringTest;
		saveStringTest << SOURCE_DIR << "/../begin_images_shrunk/" << FILENAME_APPEND << std::setfill('0') << std::setw(6) << (i+1) << ".jpg";
		SaveImage(face, saveStringTest.str());
	}
	
	Eigen::VectorXf averageFace;
	VectorOfEigenPairs eigenCrap = CreateEigenvectors(vectorOfFaces, averageFace);
	std::cout << "Average Face:" << std::endl;
	//PrintImageMatrix(averageFace);
	
	
	std::cout << "Got eigen crap, size " << eigenCrap[0].second.size() << std::endl;
	
	//Test face
	VectorOfVectors currentFaceWeights;
	for (int i=0; i<NUMB_FACES_TO_PARSE; i++) {
		currentFaceWeights.push_back(TurnImageIntoWeights(vectorOfFaces[i], averageFace, eigenCrap));
	}
	
	
	
	//std::cout << "Got test weights, size " << testWeights.size() << std::endl;
	
	
	std::cout << "Got new image:" << std::endl;
	//PrintImageMatrix(testImageReconstructed);
	
	for (int i=0; i<NUMB_FACES_TO_PARSE; i++) {
		Eigen::VectorXf testImageReconstructed = TurnWeightsIntoImage(currentFaceWeights[i], averageFace, eigenCrap);
		std::stringstream saveString;
		saveString << SOURCE_DIR << "/../end_images/" << FILENAME_APPEND << std::setfill('0') << std::setw(6) << (i+1) << ".jpg";
		SaveImage(testImageReconstructed, saveString.str());
	}
	
	//analyzeFace(NUMB_FACES_TO_PARSE+1, currentFaceWeights, averageFace, eigenCrap);
	for (int i=100000; i<100000+NUMB_FACES_TO_ANALYZE ; i++) {
		analyzeFace(i, currentFaceWeights, averageFace, eigenCrap);
	}
	/*
	float CompareFaceWeights(const Eigen::VectorXf& weights1, const Eigen::VectorXf& weights2) {
		return (weights1-weights2).norm();
	}
	*/
	
	//MatchFace(Eigen::VectorXf unknownWeight, VectorOfVectors knownFaceWeights);
	
	return 0;
}
