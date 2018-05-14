/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg
 */

#include <shogun/preprocessor/SortUlongString.h>
#include <shogun/features/Features.h>
#include <shogun/features/StringFeatures.h>
#include <shogun/mathematics/Math.h>

using namespace shogun;

CSortUlongString::CSortUlongString()
: CStringPreprocessor<uint64_t>()
{
}

CSortUlongString::~CSortUlongString()
{
}

/// clean up allocated memory
void CSortUlongString::cleanup()
{
}

/// initialize preprocessor from file
bool CSortUlongString::load(FILE* f)
{
	SG_SET_LOCALE_C;
	SG_RESET_LOCALE;
	return false;
}

/// save preprocessor init-data to file
bool CSortUlongString::save(FILE* f)
{
	SG_SET_LOCALE_C;
	SG_RESET_LOCALE;
	return false;
}

/// apply preproc on feature matrix
/// result in feature matrix
/// return pointer to feature_matrix, i.e. f->get_feature_matrix();
bool CSortUlongString::apply_to_string_features(CFeatures* f)
{
	int32_t i;
	auto sf = f->as<CStringFeatures<uint64_t>>();
	int32_t num_vec = sf->get_num_vectors();

	for (i=0; i<num_vec; i++)
	{
		int32_t len=0;
		bool free_vec;
		uint64_t* vec = sf->get_feature_vector(i, len, free_vec);
		ASSERT(!free_vec) // won't work with non-in-memory string features

		SG_DEBUG("sorting string of length %i\n", len)

		//CMath::qsort(vec, len);
		CMath::radix_sort(vec, len);
	}
	return true;
}

/// apply preproc on single feature vector
uint64_t* CSortUlongString::apply_to_string(uint64_t* f, int32_t& len)
{
	uint64_t* vec=SG_MALLOC(uint64_t, len);
	int32_t i=0;

	for (i=0; i<len; i++)
		vec[i]=f[i];

	//CMath::qsort(vec, len);
	CMath::radix_sort(vec, len);

	return vec;
}
