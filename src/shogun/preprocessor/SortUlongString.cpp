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

void CSortUlongString::apply_to_string_list(SGStringList<uint64_t> string_list)
{
	for (auto i : range(string_list.num_strings))
	{
		auto& vec = string_list.strings[i];

		SG_DEBUG("sorting string of length %i\n", vec.slen);

		//CMath::qsort(vec, len);
		CMath::radix_sort(vec.string, vec.slen);
	}
}

/// apply preproc on single feature vector
uint64_t* CSortUlongString::apply_to_string(uint64_t* f, int32_t& len)
{
	uint64_t* vec=SG_MALLOC(uint64_t, len);

	std::copy(f, f + len, vec);

	//CMath::qsort(vec, len);
	CMath::radix_sort(vec, len);

	return vec;
}
