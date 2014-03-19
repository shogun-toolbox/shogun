#include "NeuralNets.h"
#include "DataAdapter.h"
#include <iostream>
#include <ctime>

void CNeuralNets::SetDataAdapter(CDataAdapter* _data_adapter)
{
	nn_data_adapter = _data_adapter;
}

void CNeuralNets::SetDenseOutputLabels(std::vector<int32_t> &outputLabels)
{
	ground_truth.resize(outputLabels.size(), CNNConfig::length[CNNConfig::layers - 1]);
	memset(ground_truth.data(), 0, sizeof(float32_t)* ground_truth.size());
	for (int32_t i = 0; i < outputLabels.size(); ++i)
		ground_truth(i, outputLabels[i]) = 1;
}

float32_t CNeuralNets::TrainMiniBatch(void* samples)
{
	FeatLabelPair* data = (FeatLabelPair*)samples;
	SetDenseOutputLabels(data->labels);
	float32_t loss = FeedForward(data->feats, ground_truth);
	BackPropogation(data->feats);
	ApplyGradients();
	return loss;
}

void CNeuralNets::TestMiniBatch(void* samples, void*& result)
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
			if (m_activations[CNNConfig::layers - 1](i, j) > maxV)
			{
				maxIdx = j;
				maxV = m_activations[CNNConfig::layers - 1](i, j);
			}
		}
		if (ground_truth(i, maxIdx) < 0.01)
			errNum++;
	}
	*(int32_t*)result = errNum;
}

void CNeuralNets::InitEpoch(int32_t cur_epoch)
{
	CNNConfig::rtc.learning_rate = 0.1 / (1 + cur_epoch / 20.0);
	CNNConfig::rtc.momentum = 0.5 + (0.999 - 0.5) * cur_epoch / 1000.0;
	for (int32_t cur_layer = 0; cur_layer < CNNConfig::layers - 1; ++cur_layer)
	{
		m_vw[cur_layer].setZero();
		m_vb[cur_layer].setZero();
	}
	std::cout << "Epoch:" << cur_epoch + 1 << ", LearningRate:" << CNNConfig::rtc.learning_rate << ", Momentum:" << CNNConfig::rtc.momentum << std::endl;
}

void CNeuralNets::TrainEpoch(int32_t cur_epoch)
{
	InitEpoch(cur_epoch);
	
	nn_data_adapter->Open();
	int64_t total = 0;
	void* samples = NULL;
	int32_t total_batch = 0;
	float32_t total_loss = 0.0;
	while (1)
	{
		int32_t cnt = nn_data_adapter->GetBatchSamples(CNNConfig::batchsize, samples);
		if (cnt == 0)
			break;

		total += cnt;
		float32_t loss = TrainMiniBatch(samples);
		total_loss += loss;
		total_batch++;
		printf("Learning Rate: %.4f\tSamples: %I64d\tLoss: %.4f\n", CNNConfig::rtc.learning_rate, total, loss);
	}
	printf("Training Loss:%f\n", total_loss / total_batch);
	nn_data_adapter->Close();
}

void CNeuralNets::TrainAll()
{
	time_t begin = clock();

	nn_data_adapter->Init(CNNConfig::trainx, CNNConfig::trainy);
	for (int32_t cur_epoch = 0; cur_epoch < CNNConfig::epoch; ++cur_epoch)
	{
		TrainEpoch(cur_epoch);
	}

	time_t end = clock();
	std::cout << "total training time: " << (double)(end - begin) / CLOCKS_PER_SEC << std::endl;

	nn_data_adapter->Destroy();
}

void CNeuralNets::TestAll()
{
	nn_data_adapter->Init(CNNConfig::testx, CNNConfig::testy);
	nn_data_adapter->Open();
	
	int total = 0, err_num = 0;
	void* res = malloc(sizeof(int32_t));
	void* samples = NULL;
	while (1)
	{
		int32_t cnt = nn_data_adapter->GetBatchSamples(CNNConfig::batchsize, samples);
		if (cnt == 0) break;
		total += cnt;
		TestMiniBatch(samples, res);
		err_num += *(int32_t*)res;
	}

	if (total)
		std::cout << "Accuracy: " << (double)(total - err_num) / total * 100.0 << "\%" << std::endl;

	nn_data_adapter->Close();
	nn_data_adapter->Destroy();
}
