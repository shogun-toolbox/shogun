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

CByteFeatures::CByteFeatures(EAlphabet a, int32_t size)
: CSimpleFeatures<uint8_t>(size)
{
	alphabet=new CAlphabet(a);
}

CByteFeatures::CByteFeatures(CAlphabet* a, int32_t size)
: CSimpleFeatures<uint8_t>(size)
{
	alphabet=a;
}

CByteFeatures::CByteFeatures(const CByteFeatures & orig)
: CSimpleFeatures<uint8_t>(orig)
{
	alphabet=orig.alphabet;
}

CByteFeatures::CByteFeatures(EAlphabet a, uint8_t* fm, int32_t num_feat, int32_t num_vec)
: CSimpleFeatures<uint8_t>(fm, num_feat, num_vec)
{
	alphabet=new CAlphabet(a);
}

CByteFeatures::CByteFeatures(EAlphabet a, char* fname)
: CSimpleFeatures<uint8_t>(fname)
{
	alphabet=new CAlphabet(a);
	load(fname);
}

CByteFeatures::~CByteFeatures()
{
	delete alphabet;
	alphabet=NULL;
}

bool CByteFeatures::load(char* fname)
{
	SG_INFO( "loading...\n");
	int64_t length=0;
	int64_t linelen=0;

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
			for (int32_t lines=0; lines<num_vectors; lines++)
			{
				for (int32_t columns=0; columns<num_features; columns++)
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

bool CByteFeatures::save(char* fname)
{
	return false;
}
