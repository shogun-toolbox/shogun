#include "NeuralNets.h"
#include "DataAdapter.h"
#include <iostream>
#include <ctime>

void NeuralNets::SetDataAdapter(DataAdapter* _data_adapter)
{
	nn_data_adapter = _data_adapter;
}

void NeuralNets::SetDenseOutputLabels(std::vector<int32_t> &outputLabels)
{
	ground_truth.resize(outputLabels.size(), NNConfig::length[NNConfig::layers - 1]);
	memset(ground_truth.data(), 0, sizeof(float32_t)* ground_truth.size());
	for (int i = 0; i < outputLabels.size(); ++i)
		ground_truth(i, outputLabels[i]) = 1;
}

float32_t NeuralNets::TrainMiniBatch(void* samples)
{
	FeatLabelPair* data = (FeatLabelPair*)samples;
	SetDenseOutputLabels(data->labels);
	float32_t loss = FeedForward(data->feats, ground_truth);
	BackPropogation(data->feats);
	ApplyGradients();
	return loss;
}

void NeuralNets::TestMiniBatch(void* samples, void*& result)
{
	FeatLabelPair* data = (FeatLabelPair*)samples;
	SetDenseOutputLabels(data->labels);
	FeedForward(data->feats, ground_truth);

	int32_t errNum = 0;
	for (int32_t i = 0; i < ground_truth.rows(); ++i)
	{
		int32_t maxIdx = -1;
		float32_t maxV = -1;
		for (int32_t j = 0; j < ground_truth.cols(); ++j)
		{
			if (m_activations[NNConfig::layers - 1](i, j) > maxV)
			{
				maxIdx = j;
				maxV = m_activations[NNConfig::layers - 1](i, j);
			}
		}
		if (ground_truth(i, maxIdx) < 0.01)
			errNum++;
	}
	*(int32_t*)result = errNum;
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

void NeuralNets::TrainEpoch(int32_t cur_epoch)
{
	InitEpoch(cur_epoch);
	
	nn_data_adapter->Open();
	int64_t total = 0;
	void* samples = NULL;
	int32_t total_batch = 0;
	float32_t total_loss = 0.0;
	while (1)
	{
		int cnt = nn_data_adapter->GetBatchSamples(NNConfig::batchsize, samples);
		if (cnt == 0)
			break;
		total += cnt;
		float32_t loss = TrainMiniBatch(samples);
		total_loss += loss;
		total_batch++;
		printf("Learning Rate: %.4f\tSamples: %I64d\tLoss: %.4f\n", NNConfig::rtc.learning_rate, total, loss);
	}
	printf("Training Loss:%f\n", total_loss / total_batch);
	nn_data_adapter->Close();
}

void NeuralNets::TrainAll()
{
	nn_data_adapter->Init(NNConfig::trainx, NNConfig::trainy);
	time_t begin = clock();
	for (int cur_epoch = 0; cur_epoch < NNConfig::epoch; ++cur_epoch)
	{
		TrainEpoch(cur_epoch);
	}
	time_t end = clock();
	std::cout << "total training time: " << (double)(end - begin) / CLOCKS_PER_SEC << std::endl;
	nn_data_adapter->Destroy();
}