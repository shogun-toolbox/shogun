#ifndef DATA_ADAPTER
#define DATA_ADAPTER

#include "TypeDefine.h"
#include "NNConfig.h"
#include <vector>

using namespace shogun::deeplearning::typedefine;

struct FeatLabelPair
{
	EigenDenseMat feats;
	std::vector<int32_t> labels;

	FeatLabelPair()
	{
		labels.clear();
	}
};

class DataAdapter
{
public:
	DataAdapter();

	//Call before training
	void Init(const char* _feature_file, const char* _label_file);
	//Call before each epoch training begins
	void Open();
	//Call before each batch training begins
	int32_t GetBatchSamples(int32_t batch_size, void*& samples);
	//Call after each epoch training ends (prepare for the next epoch)
	void Close();
	//Call after all training done
	void Destroy();

protected:
	int32_t LoadBuffer();
	int32_t LoadMatrix(EigenDenseMat &m, int32_t rows, int32_t cols, FILE* fid);

	int32_t feat_cnt, label_cnt;
	EigenDenseMat feat_buffer, label_buffer;

	FILE* fid_feat, *fid_label;
	const char* feature_file;
	const char* label_file;
};

#endif