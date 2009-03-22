/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2009 Soeren Sonnenburg
 * Copyright (C) 1999-2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include "preproc/SortWordString.h"
#include "features/Features.h"
#include "features/StringFeatures.h"
#include "lib/Mathematics.h"

CSortWordString::CSortWordString()
: CStringPreProc<uint16_t>("SortWordString", "STWS") 
{
}

CSortWordString::~CSortWordString()
{
}

/// initialize preprocessor from features
bool CSortWordString::init(CFeatures* f)
{
	ASSERT(f->get_feature_class()==C_STRING);
	ASSERT(f->get_feature_type()==F_WORD);

	return true;
}

/// clean up allocated memory
void CSortWordString::cleanup()
{
}

/// initialize preprocessor from file
bool CSortWordString::load(FILE* f)
{
	return false;
}

/// save preprocessor init-data to file
bool CSortWordString::save(FILE* f)
{
	return false;
}

/// apply preproc on feature matrix
/// result in feature matrix
/// return pointer to feature_matrix, i.e. f->get_feature_matrix();
bool CSortWordString::apply_to_string_features(CFeatures* f)
{
	int32_t i;
	int32_t num_vec=((CStringFeatures<uint16_t>*)f)->get_num_vectors() ;
	
	for (i=0; i<num_vec; i++)
	{
		int32_t len = 0 ;
		uint16_t* vec = ((CStringFeatures<uint16_t>*)f)->get_feature_vector(i, len) ;
		
		//CMath::qsort(vec, len);
		CMath::radix_sort(vec, len);
	}
	return true ;
}

/// apply preproc on single feature vector
uint16_t* CSortWordString::apply_to_string(uint16_t* f, int32_t& len)
{
	uint16_t* vec=new uint16_t[len];
	int32_t i=0;

	for (i=0; i<len; i++)
		vec[i]=f[i];

	//CMath::qsort(vec, len);
	CMath::radix_sort(vec, len);

	return vec;
}

/// initialize preprocessor from file
bool CSortWordString::load_init_data(FILE* src)
{
	return true;
}

/// save init-data (like transforamtion matrices etc) to file
bool CSortWordString::save_init_data(FILE* dst)
{
	return true;
}
