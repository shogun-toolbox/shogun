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

CSortWordString::CSortWordString()
: CStringPreprocessor<uint16_t>()
{
}

CSortWordString::~CSortWordString()
{
}

/// clean up allocated memory
void CSortWordString::cleanup()
{
}

/// initialize preprocessor from file
bool CSortWordString::load(FILE* f)
{
	SG_SET_LOCALE_C;
	SG_RESET_LOCALE;
	return false;
}

/// save preprocessor init-data to file
bool CSortWordString::save(FILE* f)
{
	SG_SET_LOCALE_C;
	SG_RESET_LOCALE;
	return false;
}

void CSortWordString::apply_to_string_list(SGStringList<uint16_t> string_list)
{
	for (auto i : range(string_list.num_strings))
	{
		int32_t len = 0 ;
		auto& vec = string_list.strings[i];

		//CMath::qsort(vec, len);
		CMath::radix_sort(vec.string, vec.slen);
	}
}

/// apply preproc on single feature vector
uint16_t* CSortWordString::apply_to_string(uint16_t* f, int32_t& len)
{
	uint16_t* vec=SG_MALLOC(uint16_t, len);
	int32_t i=0;

	for (i=0; i<len; i++)
		vec[i]=f[i];

	//CMath::qsort(vec, len);
	CMath::radix_sort(vec, len);

	return vec;
}
