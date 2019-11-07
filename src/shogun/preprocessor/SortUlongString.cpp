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

SortUlongString::SortUlongString()
: StringPreprocessor<uint64_t>()
{
}

SortUlongString::~SortUlongString()
{
}

/// clean up allocated memory
void SortUlongString::cleanup()
{
}

/// initialize preprocessor from file
bool SortUlongString::load(FILE* f)
{
	SG_SET_LOCALE_C;
	SG_RESET_LOCALE;
	return false;
}

/// save preprocessor init-data to file
bool SortUlongString::save(FILE* f)
{
	SG_SET_LOCALE_C;
	SG_RESET_LOCALE;
	return false;
}

void SortUlongString::apply_to_string_list(std::vector<SGVector<uint64_t>>& string_list)
{
	for (auto& vec : string_list)
	{
		SG_DEBUG("sorting string of length {}", vec.vlen);

		//Math::qsort(vec, len);
		Math::radix_sort(vec.vector, vec.vlen);
	}
}

/// apply preproc on single feature vector
uint64_t* SortUlongString::apply_to_string(uint64_t* f, int32_t& len)
{
	uint64_t* vec=SG_MALLOC(uint64_t, len);

	std::copy(f, f + len, vec);

	//Math::qsort(vec, len);
	Math::radix_sort(vec, len);

	return vec;
}
