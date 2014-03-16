#ifndef NN_CONFIG
#define NN_CONFIG

#include "Enums.h"
#include "TypeDefine.h"
//#include <vector>

using namespace shogun::deeplearning::typedefine;
using namespace shogun::deeplearning::enums;

const int32_t MAXLAYERS = 15;

struct RuntimeConfig
{
	float32_t learning_rate;
	float32_t momentum;
	bool is_testing;
	
	RuntimeConfig()
	{
		learning_rate = 1;
		momentum = 0.5;
		is_testing = false;
	}
};

struct TrainingOptions
{
	FuncType act_type;
    FuncType out_type;
    double weightPenaltyL2;
    double learningRateScaling;
    double inputZeroMaskedFraction;
    bool isGradChecking;
    TrainingOptions();
};

class NNConfig
{
public:
    static void Initialize();
	static int32_t epoch, batchsize;
    static const char *trainx, *trainy, *testx, *testy;
	static int32_t layers;
	static int32_t* length;
    static RuntimeConfig rtc;
    static TrainingOptions opt;
    static TaskType task;

private:
    static void ParseLayers(const char* layers_config);
    static TaskType GetTaskType(const char* task_type);
    static FuncType GetFuncType(const char* func_type);
};

#endif