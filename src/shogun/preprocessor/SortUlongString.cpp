/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg
 */

#include <shogun/base/range.h>
#include <shogun/features/Features.h>
#include <shogun/features/StringFeatures.h>
#include <shogun/mathematics/Math.h>
#include <shogun/preprocessor/SortUlongString.h>

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
	auto sf = f->as<CStringFeatures<uint64_t>>();
	auto num_vec = sf->get_num_vectors();

	for (auto i : range(num_vec))
	{
		int32_t len=0;
		bool free_vec;
		auto vec = sf->get_feature_vector(i, len, free_vec);
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

	for (auto i : range(len))
		vec[i] = f[i];

	//CMath::qsort(vec, len);
	CMath::radix_sort(vec, len);

	return vec;
}
