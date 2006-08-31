/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2006 Soeren Sonnenburg
 * Copyright (C) 1999-2006 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include "features/ByteFeatures.h"
#include "lib/common.h"
#include "lib/File.h"

CByteFeatures::CByteFeatures(LONG size) : CSimpleFeatures<BYTE>(size)
{
}

CByteFeatures::CByteFeatures(const CByteFeatures & orig) : CSimpleFeatures<BYTE>(orig)
{
}

CByteFeatures::CByteFeatures(CHAR* fname) : CSimpleFeatures<BYTE>(fname)
{
	load(fname);
}

CFeatures* CByteFeatures::duplicate() const
{
	return new CByteFeatures(*this);
}


bool CByteFeatures::load(CHAR* fname)
{
	bool status=false;
	num_vectors=1;
	CFile f(fname, 'r', F_BYTE);
	LONG numf=0 ;
	feature_matrix=f.load_byte_data(NULL, numf);
	num_features=numf;

    if (!f.is_ok())
		CIO::message(M_ERROR, "loading file \"%s\" failed", fname);
	else
		status=true;

	return status;
}

bool CByteFeatures::save(CHAR* fname)
{
	return false;
}
