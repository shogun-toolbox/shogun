/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Sergey Lisitsyn, Viktor Gal
 */
#include <shogun/features/streaming/StreamingDotFeatures.h>

using namespace shogun;

StreamingDotFeatures::StreamingDotFeatures() : StreamingFeatures()
{
	set_property(FP_STREAMING_DOT);
}

StreamingDotFeatures::StreamingDotFeatures(DotFeatures* dot_features,
		float64_t* lab)
{
	not_implemented(SOURCE_LOCATION);
	return;
}

StreamingDotFeatures::~StreamingDotFeatures()
{
}

void StreamingDotFeatures::dense_dot_range(float32_t* output, float32_t* alphas,
		float32_t* vec, int32_t dim, float32_t b, int32_t num_vec)
{
	ASSERT(num_vec>=0)

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

void StreamingDotFeatures::expand_if_required(float32_t*& vec, int32_t &len)
{
	int32_t dim = get_dim_feature_space();
	if (dim > len)
	{
		vec = SG_REALLOC(float32_t, vec, len, dim);
		memset(&vec[len], 0, (dim-len) * sizeof(float32_t));
		len = dim;
	}
}

void StreamingDotFeatures::expand_if_required(float64_t*& vec, int32_t &len)
{
	int32_t dim = get_dim_feature_space();
	if (dim > len)
	{
		vec = SG_REALLOC(float64_t, vec, len, dim);
		memset(&vec[len], 0, (dim-len) * sizeof(float64_t));
		len = dim;
	}
}

void* StreamingDotFeatures::get_feature_iterator()
{
	not_implemented(SOURCE_LOCATION);
	return NULL;
}

int32_t StreamingDotFeatures::get_nnz_features_for_vector()
{
	not_implemented(SOURCE_LOCATION);
	return -1;
}

bool StreamingDotFeatures::get_next_feature(int32_t& index, float32_t& value, void* iterator)
{
	not_implemented(SOURCE_LOCATION);
	return false;
}

void StreamingDotFeatures::free_feature_iterator(void* iterator)
{
	not_implemented(SOURCE_LOCATION);
	return;
}
