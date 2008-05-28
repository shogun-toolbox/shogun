/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2008 Soeren Sonnenburg
 * Written (W) 1999-2008 Gunnar Raetsch
 * Copyright (C) 1999-2008 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include "preproc/SortUlongString.h"
#include "features/Features.h"
#include "features/StringFeatures.h"
#include "lib/Mathematics.h"

CSortUlongString::CSortUlongString()
: CStringPreProc<ULONG>("SortUlongString", "STUS")
{
}

CSortUlongString::~CSortUlongString()
{
}

/// initialize preprocessor from features
bool CSortUlongString::init(CFeatures* f)
{
	ASSERT(f->get_feature_class()==C_STRING);
	ASSERT(f->get_feature_type()==F_ULONG);

	return true;
}

/// clean up allocated memory
void CSortUlongString::cleanup()
{
}

/// initialize preprocessor from file
bool CSortUlongString::load(FILE* f)
{
	return false;
}

/// save preprocessor init-data to file
bool CSortUlongString::save(FILE* f)
{
	return false;
}

/// apply preproc on feature matrix
/// result in feature matrix
/// return pointer to feature_matrix, i.e. f->get_feature_matrix();
bool CSortUlongString::apply_to_string_features(CFeatures* f)
{
	INT i;
	INT num_vec=((CStringFeatures<ULONG>*)f)->get_num_vectors() ;
	
	for (i=0; i<num_vec; i++)
	{
		INT len = 0 ;
		ULONG* vec = ((CStringFeatures<ULONG>*)f)->get_feature_vector(i, len) ;
		SG_DEBUG( "sorting string of length %i\n", len) ;
		
		//CMath::qsort(vec, len);
		CMath::radix_sort(vec, len);
	}
	return true ;
}

/// apply preproc on single feature vector
ULONG* CSortUlongString::apply_to_string(ULONG* f, INT& len)
{
	ULONG* vec=new ULONG[len];
	INT i=0;

	for (i=0; i<len; i++)
		vec[i]=f[i];

	//CMath::qsort(vec, len);
	CMath::radix_sort(vec, len);

	return vec;
}

/// initialize preprocessor from file
bool CSortUlongString::load_init_data(FILE* src)
{
	return true;
}

/// save init-data (like transforamtion matrices etc) to file
bool CSortUlongString::save_init_data(FILE* dst)
{
	return true;
}
