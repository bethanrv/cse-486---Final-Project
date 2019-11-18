
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <iomanip>


#include "test.hpp"
#include "Eigen/Dense"

using namespace std;

static int NUMB_FACES_TO_PARSE = 100;

Eigen::VectorXf loadImageI(std::string relPath, int i) {
	std::stringstream saveString;
	saveString << SOURCE_DIR << relPath << std::setfill('0') << std::setw(6) << i << ".jpg";
	return ParseImage(saveString.str());
}

void analyzeFace(int i, std::vector<Eigen::VectorXf> currentFaceWeights, Eigen::VectorXf averageFace, std::vector<EigenPair> eigenCrap) {
	std::cout << "Analyzing face " << i << std::endl;
	
	Eigen::VectorXf unknownFace = loadImageI("/../img_align_celeba/", i);
	Eigen::VectorXf unknownFaceWeights = TurnImageIntoWeights(unknownFace, averageFace, eigenCrap);
	Eigen::VectorXf unknownFaceReconstructed = TurnWeightsIntoImage(unknownFaceWeights, averageFace, eigenCrap);
	
	float chanceItsAFace = CompareFaceWeights(unknownFace, unknownFaceReconstructed);
	std::cout << "    Chance the mystery face is a face: " << chanceItsAFace << std::endl;
	
	int matchedFace;
	float matchChance;
	std::tie(matchedFace, matchChance) = MatchFace(unknownFaceWeights, currentFaceWeights);
	std::cout << "    Face matches face: " << matchedFace << " with confidence " << matchChance << std::endl;
	
	std::stringstream saveString;
	saveString << SOURCE_DIR << "/../img_test_2/" << std::setfill('0') << std::setw(6) << i << ".jpg";
	SaveImage(unknownFaceReconstructed, saveString.str());
	
}

int main(int argc, char *argv[]) {	
	std::vector<Eigen::VectorXf> vectorOfFaces;
	
	for (int i=1; i<NUMB_FACES_TO_PARSE; i++) {
		Eigen::VectorXf face = loadImageI("/../img_align_celeba/", i);
		vectorOfFaces.push_back(face);	
		if (i == 1) {
			//PrintImageMatrix(face);
		}
		
		std::stringstream saveStringTest;
		saveStringTest << SOURCE_DIR << "/../img_test/" << std::setfill('0') << std::setw(6) << i << ".jpg";
		SaveImage(face, saveStringTest.str());
	}
	
	Eigen::VectorXf averageFace;
	std::vector<EigenPair> eigenCrap = CreateEigenvectors(vectorOfFaces, averageFace);
	std::cout << "Average Face:" << std::endl;
	//PrintImageMatrix(averageFace);
	
	
	std::cout << "Got eigen crap, size " << eigenCrap[0].second.size() << std::endl;
	
	//Test face
	std::vector<Eigen::VectorXf> currentFaceWeights;
	for (int i=1; i<NUMB_FACES_TO_PARSE; i++) {
//		std::stringstream saveString;
//		saveString << SOURCE_DIR << "/../img_align_celeba/" << std::setfill('0') << std::setw(6) << (i) << ".jpg";
//		testFace = ParseImage(saveString.str());
		currentFaceWeights.push_back(TurnImageIntoWeights(vectorOfFaces[i-1], averageFace, eigenCrap));
	}
	
	
	
	//std::cout << "Got test weights, size " << testWeights.size() << std::endl;
	
	
	std::cout << "Got new image:" << std::endl;
	//PrintImageMatrix(testImageReconstructed);
	
	for (int i=1; i<NUMB_FACES_TO_PARSE; i++) {
		Eigen::VectorXf testImageReconstructed = TurnWeightsIntoImage(currentFaceWeights[i-1], averageFace, eigenCrap);
		std::stringstream saveString;
		saveString << SOURCE_DIR << "/../img_test_2/" << std::setfill('0') << std::setw(6) << i << ".jpg";
		SaveImage(testImageReconstructed, saveString.str());
	}
	
	analyzeFace(NUMB_FACES_TO_PARSE+1, currentFaceWeights, averageFace, eigenCrap);
	analyzeFace(100000, currentFaceWeights, averageFace, eigenCrap);
	analyzeFace(100001, currentFaceWeights, averageFace, eigenCrap);
	/*
	float CompareFaceWeights(const Eigen::VectorXf& weights1, const Eigen::VectorXf& weights2) {
		return (weights1-weights2).norm();
	}
	*/
	
	//MatchFace(Eigen::VectorXf unknownWeight, std::vector<Eigen::VectorXf> knownFaceWeights);
	
	return 0;
}
