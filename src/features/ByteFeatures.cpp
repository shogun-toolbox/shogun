/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2008 Soeren Sonnenburg
 * Copyright (C) 1999-2008 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include "features/ByteFeatures.h"
#include "lib/common.h"
#include "lib/File.h"

CByteFeatures::CByteFeatures(EAlphabet a, INT size)
: CSimpleFeatures<BYTE>(size)
{
	alphabet=new CAlphabet(a);
}

CByteFeatures::CByteFeatures(CAlphabet* a, INT size)
: CSimpleFeatures<BYTE>(size)
{
	alphabet=a;
}

CByteFeatures::CByteFeatures(const CByteFeatures & orig)
: CSimpleFeatures<BYTE>(orig)
{
	alphabet=orig.alphabet;
}

CByteFeatures::CByteFeatures(EAlphabet a, BYTE* fm, INT num_feat, INT num_vec)
: CSimpleFeatures<BYTE>(fm, num_feat, num_vec)
{
	alphabet=new CAlphabet(a);
}

CByteFeatures::CByteFeatures(EAlphabet a, CHAR* fname)
: CSimpleFeatures<BYTE>(fname)
{
	alphabet=new CAlphabet(a);
	load(fname);
}

CByteFeatures::~CByteFeatures()
{
	delete alphabet;
	alphabet=NULL;
}

bool CByteFeatures::load(CHAR* fname)
{
	SG_INFO( "loading...\n");
    LONG length=0;
	LONG linelen=0;

	CFile f(fname, 'r', F_BYTE);
	free_feature_matrix();
	feature_matrix=f.load_byte_data(NULL, length);

    if (f.is_ok())
	{
		for (linelen=0; linelen<length; linelen++)
		{
			if (feature_matrix[linelen]=='\n')
			{
				num_features=linelen;
				linelen++;
				break;
			}
		}

		num_vectors=length/linelen;

		SG_INFO( "file contains %ldx%ld vectors x features\n", num_vectors, num_features);

		if (length && (num_vectors*linelen==length))
		{
			for (INT lines=0; lines<num_vectors; lines++)
			{
				for (INT columns=0; columns<num_features; columns++)
					feature_matrix[lines*num_features+columns]=feature_matrix[lines*linelen+columns];

				if (feature_matrix[lines*linelen+num_features]!='\n')
				{
               SG_ERROR( "line %d in file \"%s\" is corrupt\n", lines, fname);
					return false;
				}
			}

			return true;
		}
		else
         SG_ERROR( "file is of zero size or no rectangular featurematrix of type BYTE\n");
	}
	else
      SG_ERROR( "reading file failed\n");

	return false;
}

bool CByteFeatures::save(CHAR* fname)
{
	return false;
}
