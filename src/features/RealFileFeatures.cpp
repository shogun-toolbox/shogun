/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2008 Soeren Sonnenburg
 * Copyright (C) 1999-2008 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include "features/RealFileFeatures.h"
#include "features/Features.h"
#include "lib/io.h"

#include <stdio.h>
#include <string.h>

CRealFileFeatures::CRealFileFeatures(INT size, CHAR* fname) : CRealFeatures(size)
{
	working_file=fopen(fname, "r");
	working_filename=strdup(fname);
	ASSERT(working_file);
	intlen=0;
	doublelen=0;
	endian=0;
	fourcc=0;
	preprocd=0;
	labels=NULL;
	status=load_base_data();
}

CRealFileFeatures::CRealFileFeatures(INT size, FILE* file) : CRealFeatures(size), working_file(file), working_filename(NULL)
{
	ASSERT(working_file);
	intlen=0;
	doublelen=0;
	endian=0;
	fourcc=0;
	preprocd=0;
	labels=NULL;
	status=load_base_data();
}

CRealFileFeatures::~CRealFileFeatures()
{
	delete[] feature_matrix;
	delete[] working_filename;
	delete[] labels;
}

CRealFileFeatures::CRealFileFeatures(const CRealFileFeatures & orig): CRealFeatures(orig), 
	working_file(orig.working_file), status(orig.status)
{
	if (orig.working_filename)
		working_filename=strdup(orig.working_filename);
	if (orig.labels && get_num_vectors())
	{
		labels=new int[get_num_vectors()];
		memcpy(labels, orig.labels, sizeof(int)*get_num_vectors()); 
	}
}

DREAL* CRealFileFeatures::compute_feature_vector(INT num, INT &len, DREAL* target)
{
	ASSERT(num<num_vectors);
	len=num_features;
	DREAL* featurevector=target;
	if (!featurevector)
		featurevector=new DREAL[num_features];
	ASSERT(working_file);
	fseek(working_file, filepos+num_features*doublelen*num, SEEK_SET);
	ASSERT(fread(featurevector, doublelen, num_features, working_file)==(size_t) num_features);
	return featurevector;
}

DREAL* CRealFileFeatures::load_feature_matrix()
{
	ASSERT(working_file);
	fseek(working_file, filepos, SEEK_SET);
	delete[] feature_matrix;

	SG_INFO( "allocating feature matrix of size %.2fM\n", sizeof(double)*num_features*num_vectors/1024.0/1024.0);
	free_feature_matrix();
	feature_matrix=new DREAL[num_features*num_vectors];

	SG_INFO( "loading... be patient.\n");

	for (INT i=0; i<(INT) num_vectors; i++)
	{
		if (!(i % (num_vectors/10+1)))
			SG_PRINT( "%02d%%.", (int) (100.0*i/num_vectors));
		else if (!(i % (num_vectors/200+1)))
			SG_PRINT( ".");

		ASSERT(fread(&feature_matrix[num_features*i], doublelen, num_features, working_file)==(size_t) num_features);
	}
	SG_INFO( "done.\n");

	return feature_matrix;
}

INT CRealFileFeatures::get_label(INT idx)
{
	ASSERT(idx<num_vectors);
	if (labels)
		return labels[idx];
	return 0;
}

bool CRealFileFeatures::load_base_data()
{
	ASSERT(working_file);
	UINT num_vec=0;
	UINT num_feat=0;

	ASSERT(fread(&intlen, sizeof(BYTE), 1, working_file)==1);
	ASSERT(fread(&doublelen, sizeof(BYTE), 1, working_file)==1);
	ASSERT(fread(&endian, (UINT) intlen, 1, working_file)== 1);
	ASSERT(fread(&fourcc, (UINT) intlen, 1, working_file)==1);
	ASSERT(fread(&num_vec, (UINT) intlen, 1, working_file)==1);
	ASSERT(fread(&num_feat, (UINT) intlen, 1, working_file)==1);
	ASSERT(fread(&preprocd, (UINT) intlen, 1, working_file)==1);
	SG_INFO( "detected: intsize=%d, doublesize=%d, num_vec=%d, num_feat=%d, preprocd=%d\n", intlen, doublelen, num_vec, num_feat, preprocd);
	filepos=ftell(working_file);
	set_num_vectors(num_vec);
	set_num_features(num_feat);
	fseek(working_file, filepos+num_features*num_vectors*doublelen, SEEK_SET);
	delete[] labels;
	labels=new int[num_vec];
	ASSERT(fread(labels, intlen, num_vec, working_file) == num_vec);
	return true;
}
