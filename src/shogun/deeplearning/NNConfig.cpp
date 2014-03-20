#include "NNConfig.h"

//Basic parameters of NN
int32_t CNNConfig::epoch = 0;
int32_t CNNConfig::batchsize = 0;
int32_t CNNConfig::layers = 0;
int32_t* CNNConfig::length = NULL;
RuntimeConfig CNNConfig::rtc;
TrainingOptions CNNConfig::opt;

//Datasets for train/test
const char* CNNConfig::trainx = NULL;
const char* CNNConfig::trainy = NULL;
const char* CNNConfig::testx = NULL;
const char* CNNConfig::testy = NULL;

TrainingOptions::TrainingOptions()
{
	act_type = FuncType::RECTIFIED;
	out_type = FuncType::SOFTMAX;
	weightPenaltyL2 = 0.001;
	isGradChecking = false; //Flag for gradient checking
}

void CNNConfig::Initialize()
{
	//To be further updated as parameters setting interfaces
	epoch = 1;
	batchsize = 100;

	std::stringstream layer_config;
	//config string for layer setting: layer#, followed by unit# in each layer
	layer_config << "4 784 1024 1024 10";
	layer_config >> layers;
	length = (int32_t*)malloc(sizeof(int32_t) * layers);
	for (int32_t i = 0; i < layers; ++i)
		layer_config >> length[i];

	trainx = "Dataset/MNIST_TrainX.dat";
	trainy = "Dataset/MNIST_TrainY.dat";
	testx = "Dataset/MNIST_TestX.dat";
	testy = "Dataset/MNIST_TestY.dat";
}
