#include "DataAdapter.h"
#include <iostream>

DataAdapter::DataAdapter()
{
	fid_feat = fid_label = NULL;
	feature_file = label_file = NULL;
}

void DataAdapter::Init(const char* _feature_file, const char* _label_file)
{
	feature_file = _feature_file;
	label_file = _label_file;
}

void DataAdapter::Open()
{
	//Currently for MNIST data, and should be edited more generally later
	int32_t row_cnt;
	fopen_s(&fid_feat, feature_file, "rb");
	fread_s(&row_cnt, sizeof(int32_t), sizeof(int32_t), 1, fid_feat);			//For MNIST: 60000
	fread_s(&feat_cnt, sizeof(int32_t), sizeof(int32_t), 1, fid_feat);			//For MNIST: 784
	feat_buffer.resize(NNConfig::batchsize, feat_cnt);

	fopen_s(&fid_label, label_file, "rb");
	fread_s(&row_cnt, sizeof(int32_t), sizeof(int32_t), 1, fid_label);			//For MNIST: 60000
	fread_s(&label_cnt, sizeof(int32_t), sizeof(int32_t), 1, fid_label);		//For MNIST: 10
	label_buffer.resize(NNConfig::batchsize, label_cnt);
}

int32_t DataAdapter::GetBatchSamples(int32_t batch_size, void*& samples)
{
	FeatLabelPair* data;
	if (samples)
		data = (FeatLabelPair*)samples;
	else 
	{
		data = new FeatLabelPair();
		samples = data;
	}

	data->feats.resize(batch_size, feat_cnt);
	data->labels.clear();

	int32_t lines = LoadBuffer();
	data->feats = feat_buffer;
	for (int li = 0; li < lines; ++li)
	{
		//For MNIST, label file is recorded in 1-hot fashion, i.e. 0000100000
		//Fetch the only one true label by simple linear search
		int label = -1, max_dim = -1;
		for (int i = 0; i < label_cnt; ++i)
		{
			if (label_buffer(li, i) > max_dim)
			{
				max_dim = label_buffer(li, i);
				label = i;
			}
		}
		data->labels.push_back(label);
	}

	return 0;
}

void DataAdapter::Close()
{
	if (fid_feat)
		fclose(fid_feat);
	if (fid_label)
		fclose(fid_label);

	fid_feat = fid_label = NULL;
}

void DataAdapter::Destroy()
{
	Close();
	feature_file = label_file = NULL;
}

int32_t DataAdapter::LoadMatrix(EigenDenseMat &m, int rows, int cols, FILE* fid)
{
	m.resize(rows, cols);
	int32_t loaded_row_num = 0;
	float32_t data;
	for (int32_t i = 0; i < rows; ++i)
	{
		for (int32_t j = 0; j < cols; ++j)
		{
			if (!fread(&data, sizeof(float64_t), 1, fid))
				return loaded_row_num;
			m(i, j) = data;
		}
		loaded_row_num++;
	}

	//The actual loaded row# can be equal or smaller than the requested rows
	//The case of being smaller should occur only once, at the last time of loading
	return loaded_row_num;
}

int32_t DataAdapter::LoadBuffer()
{
	int32_t feat_row_num = LoadMatrix(feat_buffer, NNConfig::batchsize, feat_cnt, fid_feat);
	int32_t label_row_num = LoadMatrix(label_buffer, NNConfig::batchsize, label_cnt, fid_label);
	//Ensure the loaded row# of feature and label files keep the same
	assert(feat_row_num == label_row_num);

	return label_row_num;
}