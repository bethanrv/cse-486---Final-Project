
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <iomanip>


#include "test.hpp"
#include "Eigen/Dense"

using namespace std;

static int NUMB_FACES_TO_PARSE = 100;

int main(int argc, char *argv[]) {	
	std::vector<Eigen::VectorXf> vectorOfFaces;
	
	for (int i=1; i<NUMB_FACES_TO_PARSE; i++) {
		std::stringstream saveString;
		saveString << SOURCE_DIR << "/../img_align_celeba/" << std::setfill('0') << std::setw(6) << i << ".jpg";
		//fprintf(saveString, "test");
		Eigen::VectorXf face = ParseImage(saveString.str());
		//std::cout << "[ " << face << " ] " << std::endl;
		vectorOfFaces.push_back(face);	
		//009999.jpg
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
	
	
	return 0;
}
