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

#include "features/RealFeatures.h"
#include "lib/File.h"

bool CRealFeatures::load(char* fname)
{
	bool status=false;
	num_vectors=1;
    num_features=0;
	CFile f(fname, 'r', F_DREAL);
	LONG numf=0 ;
	free_feature_matrix();
	feature_matrix=f.load_real_data(NULL, numf);
	num_features=numf;


    if (!f.is_ok()) {
      SG_ERROR( "loading file \"%s\" failed", fname);
    }
	else
		status=true;

	return status;
}

bool CRealFeatures::save(char* fname)
{
	int32_t len;
	bool free;
	DREAL* fv;

	CFile f(fname, 'w', F_DREAL);

    for (int32_t i=0; i< (int32_t) num_vectors && f.is_ok(); i++)
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
		SG_INFO( "%d vectors with %d features each successfully written (filesize: %ld)\n", num_vectors, num_features, num_vectors*num_features*sizeof(DREAL));

    return true;
}


bool CRealFeatures::Align_char_features(CCharFeatures* cf, CCharFeatures* Ref, DREAL gapCost)
{
	ASSERT(cf);

	num_vectors=cf->get_num_vectors();
	num_features=Ref->get_num_vectors();

	int32_t len=num_vectors*num_features;
	free_feature_matrix();
	feature_matrix=new DREAL[len];
	int32_t num_cf_feat=0;
	int32_t num_cf_vec=0;
	int32_t num_ref_feat=0;
	int32_t num_ref_vec=0;
	char* fm_cf=cf->get_feature_matrix(num_cf_feat, num_cf_vec);
	char* fm_ref=Ref->get_feature_matrix(num_ref_feat, num_ref_vec);

	ASSERT(num_cf_vec==num_vectors);
	ASSERT(num_ref_vec==num_features);

	SG_INFO( "computing aligments of %i vectors to %i reference vectors: ", num_cf_vec, num_ref_vec) ;
	for (int32_t i=0; i< num_ref_vec; i++)
	  {
	    if (i%10==0)
	      SG_PRINT( "%i..", i) ;
	    for (int32_t j=0; j<num_cf_vec; j++)
	      feature_matrix[i+j*num_features] = CMath::Align(&fm_cf[j*num_cf_feat], &fm_ref[i*num_ref_feat], num_cf_feat, num_ref_feat, gapCost);
	  } ;

	SG_INFO( "created %i x %i matrix (0x%p)\n", num_features, num_vectors, feature_matrix) ;
	return true;
}
