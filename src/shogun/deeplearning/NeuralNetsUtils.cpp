#include "NeuralNets.h"
#include "DataAdapter.h"
#include <iostream>

void NeuralNets::SetDataAdapter(DataAdapter* _data_adapter)
{
	nn_data_adapter = _data_adapter;
}

void NeuralNets::SetDenseOutputLabels(std::vector<int32_t> &outputLabels)
{
	std_output.resize(outputLabels.size(), NNConfig::length[NNConfig::layers - 1]);
	memset(std_output.data(), 0, sizeof(float32_t)* std_output.size());
	for (int i = 0; i < outputLabels.size(); ++i)
		std_output(i, outputLabels[i]) = 1;
}

void NeuralNets::InitEpoch(int32_t cur_epoch)
{
	NNConfig::rtc.learning_rate = 0.1 / (1 + cur_epoch / 20.0);
	NNConfig::rtc.momentum = 0.5 + (0.999 - 0.5) * cur_epoch / 1000.0;
	for (int32_t cur_layer = 0; cur_layer < NNConfig::layers - 1; ++cur_layer)
	{
		m_vw[cur_layer].setZero();
		m_vb[cur_layer].setZero();
	}
	std::cout << "Epoch:" << cur_epoch + 1 << ", LearningRate:" << NNConfig::rtc.learning_rate << ", Momentum:" << NNConfig::rtc.momentum << std::endl;
}

float32_t NeuralNets::TrainMiniBatch(void* samples)
{
	FeatLabelPair* data = (FeatLabelPair*)samples;
	SetDenseOutputLabels(data->labels);
	float32_t loss = FeedForward(data->feats, std_output);
	BackPropogation(data->feats);
	ApplyGradients();
	return loss;
}

void NeuralNets::TestMiniBatch(void* samples, void*& result)
{
	
}