/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2008 Soeren Sonnenburg
 * Copyright (C) 2008 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include "features/ShortRealFeatures.h"
#include "lib/File.h"

bool CShortRealFeatures::load(char* fname)
{
	bool status=false;
	num_vectors=1;
    num_features=0;
	CFile f(fname, 'r', F_SHORTREAL);
	int64_t numf=0;
	free_feature_matrix();
	feature_matrix=f.load_shortreal_data(NULL, numf);
	num_features=numf;


    if (!f.is_ok()) {
      SG_ERROR( "loading file \"%s\" failed", fname);
    }
	else
		status=true;

	return status;
}

bool CShortRealFeatures::save(char* fname)
{
	int32_t len;
	bool free;
	SHORTREAL* fv;

	CFile f(fname, 'w', F_SHORTREAL);

    for (int32_t i=0; i< (int32_t) num_vectors && f.is_ok(); i++)
	{
		if (!(i % (num_vectors/10+1)))
			SG_PRINT( "%02d%%.", (int) (100.0*i/num_vectors));
		else if (!(i % (num_vectors/200+1)))
			SG_PRINT( ".");

		fv=get_feature_vector(i, len, free);
		f.save_shortreal_data(fv, len);
		free_feature_vector(fv, i, free) ;
	}

	if (f.is_ok())
		SG_INFO( "%d vectors with %d features each successfully written (filesize: %ld)\n", num_vectors, num_features, num_vectors*num_features*sizeof(SHORTREAL));

    return true;
}
