#ifndef NEURAL_NETS
#define NEURAL_NETS

#include "NNConfig.h"
#include "Maths.h"
#include "DataAdapter.h"

using namespace shogun::deeplearning::typedefine;
using namespace shogun::deeplearning::enums;
using namespace shogun::deeplearning::maths;

class NeuralNets
{
public:
	NeuralNets();
	//Implemented in NeuralNetsUtils.cpp
	void SetDataAdapter(DataAdapter* _data_adapter);
	void SetDenseOutputLabels(std::vector<int32_t> &outputLabels);	
	float32_t TrainMiniBatch(void * samples);
	void TestMiniBatch(void* samples, void*& result);
	void InitEpoch(int32_t cur_epoch);
	void TrainEpoch(int32_t cur_epoch);
	void TrainAll();
	void TestAll();

	EigenDenseMat ground_truth;
	DataAdapter* nn_data_adapter;

protected:
	//Implemented in NeuralNetsCore.cpp
	float32_t FeedForward(EigenDenseMat &inputs, const EigenDenseMat& outputs);
	void BackPropogation(EigenDenseMat &inputs);
	void ApplyGradients();
	float32_t CalcErr(EigenDenseMat& output, const EigenDenseMat& true_outputs, EigenDenseMat& err);

	EigenDenseMat* m_weights;
	EigenDenseRowVec* m_bias;
	EigenDenseMat m_dw[MAXLAYERS];
	EigenDenseMat m_vw[MAXLAYERS];
	EigenDenseVec m_db[MAXLAYERS];
	EigenDenseVec m_vb[MAXLAYERS];
	EigenDenseMat m_err;
	EigenDenseMat m_activations[MAXLAYERS];
};

#endif
