
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <iomanip>


#include "test.hpp"
#include "Eigen/Dense"

using namespace std;

static int NUMB_FACES_TO_PARSE = 50;

int main(int argc, char *argv[]) {
	test();
	
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
			PrintImageMatrix(face);
		}
		
		std::stringstream saveStringTest;
		saveStringTest << SOURCE_DIR << "/../img_test/" << std::setfill('0') << std::setw(6) << i << ".jpg";
		SaveImage(face, saveStringTest.str());
	}
	
	Eigen::VectorXf averageFace;
	std::vector<EigenPair> eigenCrap = CreateEigenvectors(vectorOfFaces, averageFace);
	
	std::cout << "Got eigen crap, size " << eigenCrap[0].second.size() << std::endl;
	
	//Test face
	Eigen::VectorXf testFace;
	{
		std::stringstream saveString;
		saveString << SOURCE_DIR << "/../img_align_celeba/" << std::setfill('0') << std::setw(6) << (NUMB_FACES_TO_PARSE-1) << ".jpg";
		testFace = ParseImage(saveString.str());
	}
	
	
	Eigen::VectorXf testWeights = TurnImageIntoWeights(testFace, averageFace, eigenCrap);
	
	std::cout << "Got test weights, size " << testWeights.size() << std::endl;
	
	Eigen::VectorXf testImageReconstructed = TurnWeightsIntoImage(testWeights, averageFace, eigenCrap);
	
	std::cout << "Got new image, size " << testImageReconstructed.size() << " Element 0 " << testImageReconstructed[0] << std::endl;
	
	{
		std::stringstream saveString;
		saveString << SOURCE_DIR << "/testImage" << std::setfill('0') << std::setw(6) << (NUMB_FACES_TO_PARSE-1) << ".jpg";
		SaveImage(testImageReconstructed, saveString.str());
	}
	
	
	return 0;
}
