#include <shogun/features/StreamingDotFeatures.h>

using namespace shogun;

CStreamingDotFeatures::CStreamingDotFeatures() : CStreamingFeatures()
{
	init();
	set_property(FP_STREAMING_DOT);
}

CStreamingDotFeatures::CStreamingDotFeatures(CStreamingFile* file, bool is_labelled,
		int32_t size) : CStreamingFeatures()
{
	init(file, is_labelled, size);
	set_property(FP_STREAMING_DOT);
}

CStreamingDotFeatures::CStreamingDotFeatures(CDotFeatures* dot_features,
		float64_t* lab)
{
	SG_NOTIMPLEMENTED;
	return;
}

CStreamingDotFeatures::~CStreamingDotFeatures()
{
}

void CStreamingDotFeatures::dense_dot_range(float32_t* output, float32_t* alphas,
		float32_t* vec, int32_t dim, float32_t b, int32_t num_vec)
{
	ASSERT(num_vec>=0);

	int32_t counter=0;
	start_parser();
	while (get_next_example())
	{
		if (alphas)
			output[counter]=alphas[counter]*dense_dot(vec, dim)+b;
		else
			output[counter]=dense_dot(vec, dim)+b;

		release_example();

		counter++;
		if ((counter>=num_vec) && (num_vec>0))
			break;
	}
	end_parser();
}

void CStreamingDotFeatures::expand_if_required(float32_t*& vec, int32_t &len)
{
	int32_t dim = get_dim_feature_space();
	if (dim > len)
	{
		vec = SG_REALLOC(float32_t, vec, dim);
		memset(&vec[len], 0, (dim-len) * sizeof(float32_t));
		len = dim;
	}
}

void CStreamingDotFeatures::expand_if_required(float64_t*& vec, int32_t &len)
{
	int32_t dim = get_dim_feature_space();
	if (dim > len)
	{
		vec = SG_REALLOC(float64_t, vec, dim);
		memset(&vec[len], 0, (dim-len) * sizeof(float64_t));
		len = dim;
	}
}

void* CStreamingDotFeatures::get_feature_iterator()
{
	SG_NOTIMPLEMENTED;
	return NULL;
}

int32_t CStreamingDotFeatures::get_nnz_features_for_vector()
{
	SG_NOTIMPLEMENTED;
	return -1;
}

bool CStreamingDotFeatures::get_next_feature(int32_t& index, float32_t& value, void* iterator)
{
	SG_NOTIMPLEMENTED;
	return false;
}

void CStreamingDotFeatures::free_feature_iterator(void* iterator)
{
	SG_NOTIMPLEMENTED;
	return;
}

void CStreamingDotFeatures::init()
{
}

void CStreamingDotFeatures::init(CStreamingFile *file, bool is_labelled, int32_t size)
{
}
