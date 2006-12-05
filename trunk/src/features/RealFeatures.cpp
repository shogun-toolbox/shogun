/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2006 Soeren Sonnenburg
 * Written (W) 1999-2006 Gunnar Raetsch
 * Copyright (C) 1999-2006 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include "features/RealFeatures.h"
#include "lib/File.h"

bool CRealFeatures::load(CHAR* fname)
{
	bool status=false;
	num_vectors=1;
    num_features=0;
	CFile f(fname, 'r', F_DREAL);
	LONG numf=0 ;
	feature_matrix=f.load_real_data(NULL, numf);
	num_features=numf;


    if (!f.is_ok()) {
#ifdef HAVE_PYTHON
      throw FeatureException("loading file \"%s\" failed", fname);
#else
		CIO::message(M_ERROR, "loading file \"%s\" failed", fname);
#endif
    }
	else
		status=true;

	return status;
}

bool CRealFeatures::save(CHAR* fname)
{
	INT len;
	bool free;
	DREAL* fv;

	CFile f(fname, 'w', F_DREAL);

    for (INT i=0; i< (INT) num_vectors && f.is_ok(); i++)
	{
		if (!(i % (num_vectors/10+1)))
			CIO::message(M_MESSAGEONLY, "%02d%%.", (int) (100.0*i/num_vectors));
		else if (!(i % (num_vectors/200+1)))
			CIO::message(M_MESSAGEONLY, ".");

		fv=get_feature_vector(i, len, free);
		f.save_real_data(fv, len);
		free_feature_vector(fv, i, free) ;
	}

	if (f.is_ok())
		CIO::message(M_INFO, "%d vectors with %d features each successfully written (filesize: %ld)\n", num_vectors, num_features, num_vectors*num_features*sizeof(DREAL));

    return true;
}


static inline DREAL min( DREAL a, DREAL b )
{
  return a < b ? a : b;
}


static inline void swap( DREAL*& a, DREAL*& b )
{
  DREAL* temp = a;
  a = b;
  b = temp;
}

DREAL CRealFeatures::Align(CHAR * seq1, CHAR* seq2, INT l1, INT l2, DREAL gapCost)
{
  DREAL actCost=0 ;
  INT i1, i2 ;
  DREAL* const gapCosts1 = new DREAL[ l1 ];
  DREAL* const gapCosts2 = new DREAL[ l2 ];
  DREAL* costs2_0 = new DREAL[ l2 + 1 ];
  DREAL* costs2_1 = new DREAL[ l2 + 1 ];

  // initialize borders
  for( i1 = 0; i1 < l1; ++i1 ) {
    gapCosts1[ i1 ] = gapCost * i1;
  }
  costs2_1[ 0 ] = 0;
  for( i2 = 0; i2 < l2; ++i2 ) {
    gapCosts2[ i2 ] = gapCost * i2;
    costs2_1[ i2+1 ] = costs2_1[ i2 ] + gapCosts2[ i2 ];
  }
  // compute alignment
  for( i1 = 0; i1 < l1; ++i1 ) {
    swap( costs2_0, costs2_1 );
    actCost = costs2_0[ 0 ] + gapCosts1[ i1 ];
    costs2_1[ 0 ] = actCost;
    for( i2 = 0; i2 < l2; ++i2 ) {
      const DREAL actMatch = costs2_0[ i2 ] + ( seq1[i1] == seq2[i2] );
      const DREAL actGap1 = costs2_0[ i2+1 ] + gapCosts1[ i1 ];
      const DREAL actGap2 = actCost + gapCosts2[ i2 ];
      const DREAL actGap = min( actGap1, actGap2 );
      actCost = min( actMatch, actGap );
      costs2_1[ i2+1 ] = actCost;
    }
  }

  delete [] gapCosts1;
  delete [] gapCosts2;
  delete [] costs2_0;
  delete [] costs2_1;
  
  // return the final cost
  return actCost;
} ;


bool CRealFeatures::Align_char_features(CCharFeatures* cf, CCharFeatures* Ref, DREAL gapCost)
{
	ASSERT(cf);

	num_vectors  = cf->get_num_vectors();
	num_features = Ref->get_num_vectors();

	INT len=num_vectors*num_features;
	delete[] feature_matrix;
	feature_matrix=new DREAL[len];
	ASSERT(feature_matrix);

	INT num_cf_feat;
	INT num_cf_vec;
	INT num_ref_feat;
	INT num_ref_vec;

	CHAR* fm_cf  = cf->get_feature_matrix(num_cf_feat, num_cf_vec);
	CHAR* fm_ref = Ref->get_feature_matrix(num_ref_feat, num_ref_vec);

	ASSERT(num_cf_vec==num_vectors);
	ASSERT(num_ref_vec==num_features);

	CIO::message(M_INFO, "computing aligments of %i vectors to %i reference vectors: ", num_cf_vec, num_ref_vec) ;
	for (INT i=0; i< num_ref_vec; i++)
	  {
	    if (i%10==0)
	      CIO::message(M_MESSAGEONLY, "%i..", i) ;
	    for (INT j=0; j<num_cf_vec; j++)
	      feature_matrix[i+j*num_features] = Align(&fm_cf[j*num_cf_feat], &fm_ref[i*num_ref_feat], num_cf_feat, num_ref_feat, gapCost);
	  } ;

	CIO::message(M_INFO, "created %i x %i matrix (%ld)\n", num_features, num_vectors, feature_matrix) ;
	return true;
}
