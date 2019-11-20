
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <iomanip>


#include "test.hpp"
#include "Eigen/Dense"

using namespace std;

static int NUMB_FACES_TO_PARSE = 99;
static int NUMB_FACES_TO_ANALYZE = 3;
static int NUMB_RANDOM_FACES = 50;
static std::string FILENAME_APPEND = "tree_";

FaceVector loadImageI(std::string relPath, int i) {
	std::stringstream saveString;
	saveString << SOURCE_DIR << relPath << FILENAME_APPEND << std::setfill('0') << std::setw(6) << i << ".jpg";
	return ParseImage(saveString.str());
}

void analyzeFace(int i, const ListOfWeights& currentFaceWeights, const FaceVector& averageFace, const VectorOfEigenPairs& eigenCrap) {
	std::cout << "Analyzing face " << i << std::endl;
	
	FaceVector unknownFace = loadImageI("/../begin_images/", i);
	WeightsVector unknownFaceWeights = TurnImageIntoWeights(unknownFace, averageFace, eigenCrap);
	FaceVector unknownFaceReconstructed = TurnWeightsIntoImage(unknownFaceWeights, averageFace, eigenCrap);
	
	float chanceItsAFace = CompareFaceWeights(unknownFace, unknownFaceReconstructed);
	std::cout << "    Chance the mystery face is a face: " << chanceItsAFace << std::endl;
	
	int matchedFace;
	float matchChance;
	std::tie(matchedFace, matchChance) = MatchFace(unknownFaceWeights, currentFaceWeights);
	std::cout << "    Face matches face: " << matchedFace << " with confidence " << matchChance << std::endl;
	
	std::stringstream saveString;
	saveString << SOURCE_DIR << "/../end_images/" << FILENAME_APPEND << std::setfill('0') << std::setw(6) << i << ".jpg";
	SaveImage(unknownFaceReconstructed, saveString.str());
	
	{
		std::stringstream saveString;
		saveString << SOURCE_DIR << "/../begin_images_shrunk/" << FILENAME_APPEND << std::setfill('0') << std::setw(6) << i << ".jpg";
		SaveImage(unknownFace, saveString.str());	
	}
	
	{
		std::stringstream saveString;
		saveString << SOURCE_DIR << "/../weights/" << FILENAME_APPEND << std::setfill('0') << std::setw(6) << i << ".txt";
		SaveWeights(unknownFaceWeights, saveString.str());	
	}
	
}

int main(int argc, char *argv[]) {	
	ListOfFaces vectorOfFaces((Eigen::aligned_allocator<Eigen::VectorXf>()));
	
	for (int i=0; i<NUMB_FACES_TO_PARSE; i++) {
		FaceVector face = loadImageI("/../begin_images/", i+1);
		vectorOfFaces.push_back(face);
		
		std::stringstream saveStringTest;
		saveStringTest << SOURCE_DIR << "/../begin_images_shrunk/" << FILENAME_APPEND << std::setfill('0') << std::setw(6) << (i+1) << ".jpg";
		SaveImage(face, saveStringTest.str());
	}
	
	FaceVector averageFace;
	VectorOfEigenPairs eigenCrap = CreateEigenvectors(vectorOfFaces, averageFace);
	std::cout << "Average Face:" << std::endl;
	//PrintImageMatrix(averageFace);
	
	
	std::cout << "Got eigen crap, size " << eigenCrap[0].second.size() << std::endl;
	
	//Test face
	ListOfWeights currentFaceWeights;
	for (int i=0; i<NUMB_FACES_TO_PARSE; i++) {
		currentFaceWeights.push_back(TurnImageIntoWeights(vectorOfFaces[i], averageFace, eigenCrap));
	}
	
	
	
	//std::cout << "Got test weights, size " << testWeights.size() << std::endl;
	
	
	std::cout << "Got new image:" << std::endl;
	//PrintImageMatrix(testImageReconstructed);
	
	for (int i=0; i<NUMB_FACES_TO_PARSE; i++) {
		FaceVector testImageReconstructed = TurnWeightsIntoImage(currentFaceWeights[i], averageFace, eigenCrap);
		{
			std::stringstream saveString;
			saveString << SOURCE_DIR << "/../end_images/" << FILENAME_APPEND << std::setfill('0') << std::setw(6) << (i+1) << ".jpg";
			SaveImage(testImageReconstructed, saveString.str());
		}
		{
			std::stringstream saveString;
			saveString << SOURCE_DIR << "/../weights/" << FILENAME_APPEND << std::setfill('0') << std::setw(6) << i << ".txt";
			SaveWeights(currentFaceWeights[static_cast<unsigned long>(i)], saveString.str());	
		}
	}
	
	//analyzeFace(NUMB_FACES_TO_PARSE+1, currentFaceWeights, averageFace, eigenCrap);
	for (int i=100000; i<100000+NUMB_FACES_TO_ANALYZE ; i++) {
		analyzeFace(i, currentFaceWeights, averageFace, eigenCrap);
	}
	
	for (int i=1; i<1+NUMB_RANDOM_FACES; i++) {
		WeightsVector randomWeights(NUMB_EIGENVECTORS);
		for (int j=0; j<randomWeights.size(); j++) {
			randomWeights[j] = static_cast<float>(rand()) / RAND_MAX*20-10;
		}
		FaceVector randomImage = TurnWeightsIntoImage(randomWeights, averageFace, eigenCrap);
		{
			std::stringstream saveString;
			saveString << SOURCE_DIR << "/../generated_images/" << FILENAME_APPEND << std::setfill('0') << std::setw(6) << (i+1) << ".jpg";
			SaveImage(randomImage, saveString.str());
		}
		{
			std::stringstream saveString;
			saveString << SOURCE_DIR << "/../generating_weights/" << FILENAME_APPEND << std::setfill('0') << std::setw(6) << i << ".txt";
			SaveWeights(randomWeights, saveString.str());	
		}
		
		//analyzeFace(i, currentFaceWeights, averageFace, eigenCrap);
	}
	/*
	float CompareFaceWeights(const Eigen::VectorXf& weights1, const Eigen::VectorXf& weights2) {
		return (weights1-weights2).norm();
	}
	*/
	
	//MatchFace(Eigen::VectorXf unknownWeight, VectorOfVectors knownFaceWeights);
	
	return 0;
}
