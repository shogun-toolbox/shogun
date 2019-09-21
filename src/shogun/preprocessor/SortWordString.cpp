/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Evan Shelhamer
 */

#include <shogun/preprocessor/SortWordString.h>
#include <shogun/features/Features.h>
#include <shogun/features/StringFeatures.h>
#include <shogun/mathematics/Math.h>

using namespace shogun;

SortWordString::SortWordString()
: StringPreprocessor<uint16_t>()
{
}

SortWordString::~SortWordString()
{
}

/// clean up allocated memory
void SortWordString::cleanup()
{
}

/// initialize preprocessor from file
bool SortWordString::load(FILE* f)
{
	SG_SET_LOCALE_C;
	SG_RESET_LOCALE;
	return false;
}

/// save preprocessor init-data to file
bool SortWordString::save(FILE* f)
{
	SG_SET_LOCALE_C;
	SG_RESET_LOCALE;
	return false;
}

void SortWordString::apply_to_string_list(std::vector<SGVector<uint16_t>>& string_list)
{
	for (auto& vec : string_list)
	{
		//Math::qsort(vec, len);
		Math::radix_sort(vec.vector, vec.vlen);
	}
}

/// apply preproc on single feature vector
uint16_t* SortWordString::apply_to_string(uint16_t* f, int32_t& len)
{
	uint16_t* vec=SG_MALLOC(uint16_t, len);
	int32_t i=0;

	for (i=0; i<len; i++)
		vec[i]=f[i];

	//Math::qsort(vec, len);
	Math::radix_sort(vec, len);

	return vec;
}
