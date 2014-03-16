#include "NNConfig.h"

int32_t NNConfig::epoch = 0;
int32_t NNConfig::batchsize = 0;
const char* NNConfig::trainx = NULL;
const char* NNConfig::trainy = NULL;
const char* NNConfig::testx = NULL;
const char* NNConfig::testy = NULL;
int32_t NNConfig::layers = 0;
int32_t* NNConfig::length = NULL;
RuntimeConfig NNConfig::rtc;
TrainingOptions NNConfig::opt;
TaskType NNConfig::task;


TrainingOptions::TrainingOptions()
{
	act_type = FuncType::SIGM;
	out_type = FuncType::SIGM;	
	weightPenaltyL2 = 0;
	inputZeroMaskedFraction = 0;
	learningRateScaling = 0.8;
	isGradChecking = false;
}

void NNConfig::Initialize()
{

}
