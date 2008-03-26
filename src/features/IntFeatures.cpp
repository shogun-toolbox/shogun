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

#include "features/IntFeatures.h"
#include "lib/File.h"

bool CIntFeatures::load(CHAR* fname)
{
	bool status=false;
	num_vectors=1;
    num_features=0;
	CFile f(fname, 'r', F_INT);
	LONG numf=0 ;
	feature_matrix=f.load_real_data(NULL, numf);
	num_features=numf;


    if (!f.is_ok()) {
      SG_ERROR( "loading file \"%s\" failed", fname);
    }
	else
		status=true;

	return status;
}

bool CIntFeatures::save(CHAR* fname)
{
	INT len;
	bool free;
	INT* fv;

	CFile f(fname, 'w', F_INT);

    for (INT i=0; i< (INT) num_vectors && f.is_ok(); i++)
	{
		if (!(i % (num_vectors/10+1)))
			SG_PRINT( "%02d%%.", (int) (100.0*i/num_vectors));
		else if (!(i % (num_vectors/200+1)))
			SG_PRINT( ".");

		fv=get_feature_vector(i, len, free);
		f.save_real_data(fv, len);
		free_feature_vector(fv, i, free) ;
	}

	if (f.is_ok())
		SG_INFO( "%d vectors with %d features each successfully written (filesize: %ld)\n", num_vectors, num_features, num_vectors*num_features*sizeof(INT));

    return true;
}
