/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2009 Soeren Sonnenburg
 * Written (W) 1999-2008 Gunnar Raetsch
 * Copyright (C) 1999-2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include <preprocessor/SortUlongString.h>
#include <features/Features.h>
#include <features/StringFeatures.h>
#include <mathematics/Math.h>

using namespace shogun;

CSortUlongString::CSortUlongString()
: CStringPreprocessor<uint64_t>()
{
}

CSortUlongString::~CSortUlongString()
{
}

/// initialize preprocessor from features
bool CSortUlongString::init(CFeatures* f)
{
	ASSERT(f->get_feature_class()==C_STRING)
	ASSERT(f->get_feature_type()==F_ULONG)

	return true;
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
	int32_t num_vec=((CStringFeatures<uint64_t>*)f)->get_num_vectors();

	for (i=0; i<num_vec; i++)
	{
		int32_t len=0;
		bool free_vec;
		uint64_t* vec=((CStringFeatures<uint64_t>*)f)->
			get_feature_vector(i, len, free_vec);
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
