/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2006 Soeren Sonnenburg
 * Written (W) 1999-2006 Gunnar Raetsch
 * Written (W) 1999-2006 Fabio De Bona
 * Copyright (C) 1999-2006 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include "features/ShortFeatures.h"
#include "features/CharFeatures.h"

CShortFeatures::CShortFeatures(INT size) : CSimpleFeatures<SHORT>(size)
{
}

CShortFeatures::CShortFeatures(const CShortFeatures & orig) : CSimpleFeatures<SHORT>(orig)
{
}

CShortFeatures::CShortFeatures(CHAR* fname) : CSimpleFeatures<SHORT>(fname)
{
}

bool CShortFeatures::obtain_from_char_features(CCharFeatures* cf, INT start, INT order, INT gap)
{
	ASSERT(cf);

	num_vectors=cf->get_num_vectors();
	num_features=cf->get_num_features();

	INT len=num_vectors*num_features;
	delete[] feature_matrix;
	feature_matrix=new SHORT[len];
	ASSERT(feature_matrix);

	INT num_cf_feat;
	INT num_cf_vec;

	CHAR* fm=cf->get_feature_matrix(num_cf_feat, num_cf_vec);

	ASSERT(num_cf_vec==num_vectors);
	ASSERT(num_cf_feat==num_features);

	INT max_val=0;
	for (INT i=0; i<len; i++)
	{
		feature_matrix[i]=(SHORT) cf->remap(fm[i]);
		max_val=CMath::max((INT) feature_matrix[i],max_val);
	}

	for (INT line=0; line<num_vectors; line++)
		translate_from_single_order(&feature_matrix[line*num_features], num_features, start+gap, order+gap, max_val, gap);

	if (start+gap!=0)
	{
		// condensing feature matrix ... 
		ASSERT(start+gap>=0) ;
		for (INT line=0; line<num_vectors; line++)
			for (INT j=0; j<num_features-start-gap; j++)
				feature_matrix[line*(num_features-(start+gap))+j]=feature_matrix[line*num_features+j] ;
		num_features=num_features-(start+gap) ;
	}
	
	return true;
}


void CShortFeatures::translate_from_single_order(SHORT* obs, INT sequence_length, INT start, INT order, INT max_val, INT gap)
{
	ASSERT(gap>=0) ;

	const INT start_gap = (order - gap)/2 ;
	const INT end_gap = start_gap + gap ;

	INT i,j;
	SHORT value=0;

	// almost all positions
	for (i=sequence_length-1; i>= ((int) order)-1; i--)	//convert interval of size T
	{
		value=0;
		for (j=i; j>=i-((int) order)+1; j--)
		{
			if (i-j<start_gap)
				value= (value >> max_val) | (obs[j] << (max_val * (order-1-gap)));
			else if (i-j>=end_gap)
				value= (value >> max_val) | (obs[j] << (max_val * (order-1-gap)));
		}
		obs[i]=value;
	}
	
	// the remaining `order` positions
	for (i=order-2;i>=0;i--)
	{
		value=0;
		for (j=i; j>=i-order+1; j--)
		{
			if (i-j<start_gap)
			{
				value= (value >> max_val);
				if (j>=0)
					value|=obs[j] << (max_val * (order-1-gap));
			} 
			else if (i-j>=end_gap)
				{
					value= (value >> max_val);
					if (j>=0)
						value|=obs[j] << (max_val * (order-1-gap));
				}
		}
		obs[i]=value;
	}

	for (i=start; i<sequence_length; i++)	
		obs[i-start]=obs[i];
}

CFeatures* CShortFeatures::duplicate() const
{
	return new CShortFeatures(*this);
}

bool CShortFeatures::load(CHAR* fname)
{
	return false;
}

bool CShortFeatures::save(CHAR* fname)
{
	return false;
}
