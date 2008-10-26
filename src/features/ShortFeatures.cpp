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

#include "features/ShortFeatures.h"
#include "features/CharFeatures.h"

CShortFeatures::CShortFeatures(int32_t size)
: CSimpleFeatures<int16_t>(size)
{
}

CShortFeatures::CShortFeatures(const CShortFeatures & orig)
: CSimpleFeatures<int16_t>(orig)
{
}

CShortFeatures::CShortFeatures(char* fname)
: CSimpleFeatures<int16_t>(fname)
{
}

bool CShortFeatures::obtain_from_char_features(
	CCharFeatures* cf, int32_t start, int32_t order, int32_t gap)
{
	ASSERT(cf);

	num_vectors=cf->get_num_vectors();
	num_features=cf->get_num_features();

	CAlphabet* alpha=cf->get_alphabet();
	ASSERT(alpha);

	int32_t len=num_vectors*num_features;
	free_feature_matrix();
	feature_matrix=new int16_t[len];
	int32_t num_cf_feat=0;
	int32_t num_cf_vec=0;
	char* fm=cf->get_feature_matrix(num_cf_feat, num_cf_vec);

	ASSERT(num_cf_vec==num_vectors);
	ASSERT(num_cf_feat==num_features);

	int32_t max_val=0;
	for (int32_t i=0; i<len; i++)
	{
		feature_matrix[i]=(int16_t) alpha->remap_to_bin(fm[i]);
		max_val=CMath::max((int32_t) feature_matrix[i],max_val);
	}

	for (int32_t line=0; line<num_vectors; line++)
		translate_from_single_order(&feature_matrix[line*num_features], num_features, start+gap, order+gap, max_val, gap);

	if (start+gap!=0)
	{
		// condensing feature matrix ...
		ASSERT(start+gap>=0);
		for (int32_t line=0; line<num_vectors; line++)
			for (int32_t j=0; j<num_features-start-gap; j++)
				feature_matrix[line*(num_features-(start+gap))+j]=feature_matrix[line*num_features+j] ;
		num_features=num_features-(start+gap) ;
	}
	
	return true;
}


void CShortFeatures::translate_from_single_order(
	int16_t* obs, int32_t sequence_length, int32_t start, int32_t order,
	int32_t max_val, int32_t gap)
{
	ASSERT(gap>=0);

	const int32_t start_gap = (order - gap)/2;
	const int32_t end_gap = start_gap + gap;
	int32_t i,j;
	int16_t value=0;

	// almost all positions
	for (i=sequence_length-1; i>=order-1; i--) //convert interval of size T
	{
		value=0;
		for (j=i; j>=i-order+1; j--)
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

bool CShortFeatures::load(char* fname)
{
	return false;
}

bool CShortFeatures::save(char* fname)
{
	return false;
}
