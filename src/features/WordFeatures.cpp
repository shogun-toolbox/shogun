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

#include "features/WordFeatures.h"
#include "features/CharFeatures.h"
#include "lib/File.h"

CWordFeatures::CWordFeatures(int32_t size, int32_t num_sym)
: CSimpleFeatures<uint16_t>(size), num_symbols(num_sym),
	original_num_symbols(num_sym), order(0), symbol_mask_table(NULL)
{
}

CWordFeatures::CWordFeatures(const CWordFeatures & orig)
: CSimpleFeatures<uint16_t>(orig)
{
}

CWordFeatures::CWordFeatures(char* fname, int32_t num_sym)
: CSimpleFeatures<uint16_t>(fname), num_symbols(num_sym),
	original_num_symbols(num_sym), order(0), symbol_mask_table(NULL)
{
}

CWordFeatures::~CWordFeatures()
{
	delete[] symbol_mask_table;
}

bool CWordFeatures::obtain_from_char_features(
	CCharFeatures* cf, int32_t start, int32_t p_order, int32_t gap)
{
	ASSERT(cf);

	this->order=p_order;
	delete[] symbol_mask_table;
	symbol_mask_table=new uint16_t[256];

	num_vectors=cf->get_num_vectors();
	num_features=cf->get_num_features();

	CAlphabet* alpha=cf->get_alphabet();
	ASSERT(alpha);

	int32_t len=num_vectors*num_features;
	delete[] feature_matrix;
	feature_matrix=new uint16_t[len];
	int32_t num_cf_feat=0;
	int32_t num_cf_vec=0;
	char* fm=cf->get_feature_matrix(num_cf_feat, num_cf_vec);

	ASSERT(num_cf_vec==num_vectors);
	ASSERT(num_cf_feat==num_features);

	int32_t max_val=0;
	for (int32_t i=0; i<len; i++)
	{
		feature_matrix[i]=(uint16_t) alpha->remap_to_bin(fm[i]);
		max_val=CMath::max((int32_t) feature_matrix[i],max_val);
	}

	original_num_symbols=max_val+1;
	
	int32_t* hist = new int[max_val+1] ;
	for (int32_t i=0; i<=max_val; i++)
	  hist[i]=0 ;

	for (int32_t i=0; i<len; i++)
	{
		feature_matrix[i]=(uint16_t) alpha->remap_to_bin(fm[i]);
		hist[feature_matrix[i]]++ ;
	}
	for (int32_t i=0; i<=max_val; i++)
	  if (hist[i]>0)
	    SG_DEBUG( "symbol: %i  number of occurence: %i\n", i, hist[i]) ;

	delete[] hist;

	//number of bits the maximum value in feature matrix requires to get stored
	max_val= (int32_t) ceil(log((float64_t) max_val+1)/log((float64_t) 2));
	num_symbols=1<<(max_val*p_order);

	SG_INFO( "max_val (bit): %d order: %d -> results in num_symbols: %d\n", max_val, p_order, num_symbols);

	if (num_symbols>(1<<(sizeof(uint16_t)*8)))
	{
      SG_ERROR( "symbol does not fit into datatype \"%c\" (%d)\n", (char) max_val, (int) max_val);
		return false;
	}

	for (int32_t line=0; line<num_vectors; line++)
		translate_from_single_order(&feature_matrix[line*num_features], num_features, start+gap, p_order+gap, max_val, gap);

	if (start+gap!=0)
	{
		// condensing feature matrix ... 
		ASSERT(start+gap>=0);
		for (int32_t line=0; line<num_vectors; line++)
			for (int32_t j=0; j<num_features-start-gap; j++)
				feature_matrix[line*(num_features-(start+gap))+j]=feature_matrix[line*num_features+j] ;
		num_features=num_features-(start+gap) ;
	}
	
	for (int32_t i=0; i<256; i++)
		symbol_mask_table[i]=0;

	uint16_t mask=0;
	for (int32_t i=0; i<max_val; i++)
		mask=(mask<<1) | 1;

	for (int32_t i=0; i<256; i++)
	{
		uint8_t bits=(uint8_t) i;
		symbol_mask_table[i]=0;

		for (int32_t j=0; j<8; j++)
		{
			if (bits & 1)
				symbol_mask_table[i]|=mask<<(max_val*j);

			bits>>=1;
		}
	}

	return true;
}

void CWordFeatures::translate_from_single_order(
	uint16_t* obs, int32_t sequence_length, int32_t start, int32_t p_order,
	int32_t max_val, int32_t gap)
{
	ASSERT(gap>=0);
	
	const int32_t start_gap = (p_order - gap)/2;
	const int32_t end_gap = start_gap + gap;
	int32_t i,j;
	uint16_t value=0;

	// almost all positions
	for (i=sequence_length-1; i>=p_order-1; i--) //convert interval of size T
	{
		value=0;
		for (j=i; j>=i-p_order+1; j--)
		{
			if (i-j<start_gap)
			{
				value= (value >> max_val) | (obs[j] << (max_val * (p_order-1-gap)));
			}
			else if (i-j>=end_gap)
			{
				value= (value >> max_val) | (obs[j] << (max_val * (p_order-1-gap)));
			}
		}
		obs[i]=value;
	}

	// the remaining `order` positions
	for (i=p_order-2;i>=0;i--)
	{
		value=0;
		for (j=i; j>=i-p_order+1; j--)
		{
			if (i-j<start_gap)
			{
				value= (value >> max_val);
				if (j>=0)
					value|=obs[j] << (max_val * (p_order-1-gap));
			}
			else if (i-j>=end_gap)
			{
				value= (value >> max_val);
				if (j>=0)
					value|=obs[j] << (max_val * (p_order-1-gap));
			}			
		}
		obs[i]=value;
	}

	// shifting
	for (i=start; i<sequence_length; i++)	
		obs[i-start]=obs[i];
}

bool CWordFeatures::load(char* fname)
{
	return false;
}

bool CWordFeatures::save(char* fname)
{
	int32_t len;
	bool free;
	uint16_t* fv;

	CFile f(fname, 'w', F_WORD);

    for (int32_t i=0; i< (int32_t) num_vectors && f.is_ok(); i++)
	{
		if (!(i % (num_vectors/10+1)))
			SG_PRINT( "%02d%%.", (int) (100.0*i/num_vectors));
		else if (!(i % (num_vectors/200+1)))
			SG_PRINT( ".");

		fv=get_feature_vector(i, len, free);
		f.save_word_data(fv, len);
		free_feature_vector(fv, i, free) ;
	}

	if (f.is_ok())
		SG_INFO( "%d vectors with %d features each successfully written (filesize: %ld)\n", num_vectors, num_features, num_vectors*num_features*sizeof(uint16_t));

    return true;
}

/* 
XT=['ATTTTTTAA';'ATTTTTTAA']' ;
sg('send_command', 'loglevel ALL') ;
sg('set_features', 'TRAIN', XT)
sg('send_command', 'convert TRAIN SIMPLE CHAR SIMPLE WORD DNA 3 2 0') ;
*/
