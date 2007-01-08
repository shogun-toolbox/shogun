/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2007 Soeren Sonnenburg
 * Written (W) 1999-2007 Gunnar Raetsch
 * Copyright (C) 1999-2007 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include "features/WordFeatures.h"
#include "features/CharFeatures.h"
#include "lib/File.h"

CWordFeatures::CWordFeatures(INT size, INT num_sym) : CSimpleFeatures<WORD>(size), num_symbols(num_sym),original_num_symbols(num_sym),order(0),symbol_mask_table(NULL)
{
}

CWordFeatures::CWordFeatures(const CWordFeatures & orig) : CSimpleFeatures<WORD>(orig)
{
}

CWordFeatures::CWordFeatures(CHAR* fname, INT num_sym) : CSimpleFeatures<WORD>(fname), num_symbols(num_sym),original_num_symbols(num_sym),order(0),symbol_mask_table(NULL)
{
}

CWordFeatures::~CWordFeatures()
{
	delete[] symbol_mask_table;
}

bool CWordFeatures::obtain_from_char_features(CCharFeatures* cf, INT start, INT p_order, INT gap)
{
	ASSERT(cf);

	this->order=p_order;
	delete[] symbol_mask_table;
	symbol_mask_table=new WORD[256];

	num_vectors=cf->get_num_vectors();
	num_features=cf->get_num_features();

	CAlphabet* alpha=cf->get_alphabet();
	ASSERT(alpha);

	INT len=num_vectors*num_features;
	delete[] feature_matrix;
	feature_matrix=new WORD[len];
	ASSERT(feature_matrix);

	INT num_cf_feat;
	INT num_cf_vec;

	CHAR* fm=cf->get_feature_matrix(num_cf_feat, num_cf_vec);

	ASSERT(num_cf_vec==num_vectors);
	ASSERT(num_cf_feat==num_features);

	INT max_val=0;
	for (INT i=0; i<len; i++)
	{
		feature_matrix[i]=(WORD) alpha->remap_to_bin(fm[i]);
		max_val=CMath::max((INT) feature_matrix[i],max_val);
	}

	original_num_symbols=max_val+1;
	
	INT* hist = new int[max_val+1] ;
	for (INT i=0; i<=max_val; i++)
	  hist[i]=0 ;

	for (INT i=0; i<len; i++)
	{
		feature_matrix[i]=(WORD) alpha->remap_to_bin(fm[i]);
		hist[feature_matrix[i]]++ ;
	}
	for (INT i=0; i<=max_val; i++)
	  if (hist[i]>0)
	    CIO::message(M_DEBUG, "symbol: %i  number of occurence: %i\n", i, hist[i]) ;

	delete[] hist;

	//number of bits the maximum value in feature matrix requires to get stored
	max_val= (int) ceil(log((double) max_val+1)/log((double) 2));
	num_symbols=1<<(max_val*p_order);

	CIO::message(M_INFO, "max_val (bit): %d order: %d -> results in num_symbols: %d\n", max_val, p_order, num_symbols);

	if (num_symbols>(1<<(sizeof(WORD)*8)))
	{
      sg_error(sg_err_fun,"symbol does not fit into datatype \"%c\" (%d)\n", (char) max_val, (int) max_val);
		return false;
	}

	for (INT line=0; line<num_vectors; line++)
		translate_from_single_order(&feature_matrix[line*num_features], num_features, start+gap, p_order+gap, max_val, gap);

	if (start+gap!=0)
	{
		// condensing feature matrix ... 
		ASSERT(start+gap>=0) ;
		for (INT line=0; line<num_vectors; line++)
			for (INT j=0; j<num_features-start-gap; j++)
				feature_matrix[line*(num_features-(start+gap))+j]=feature_matrix[line*num_features+j] ;
		num_features=num_features-(start+gap) ;
	}
	
	for (INT i=0; i<256; i++)
		symbol_mask_table[i]=0;

	WORD mask=0;
	for (INT i=0; i<max_val; i++)
		mask=(mask<<1) | 1;

	for (INT i=0; i<256; i++)
	{
		BYTE bits=(BYTE) i;

		for (INT j=0; j<8; j++)
		{
			if (bits & 1)
				symbol_mask_table[i]|=mask<<(max_val*j);

			bits>>=1;
		}
	}

	return true;
}

void CWordFeatures::translate_from_single_order(WORD* obs, INT sequence_length, INT start, INT p_order, INT max_val, INT gap)
{
	ASSERT(gap>=0) ;
	
	const INT start_gap = (p_order - gap)/2 ;
	const INT end_gap = start_gap + gap ;
	
	INT i,j;
	WORD value=0;

	// almost all positions
	for (i=sequence_length-1; i>= ((int) p_order)-1; i--)	//convert interval of size T
	{
		value=0;
		for (j=i; j>=i-((int) p_order)+1; j--)
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
		obs[i]= (WORD) value;
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

bool CWordFeatures::load(CHAR* fname)
{
	return false;
}

bool CWordFeatures::save(CHAR* fname)
{
	INT len;
	bool free;
	WORD* fv;

	CFile f(fname, 'w', F_WORD);

    for (INT i=0; i< (INT) num_vectors && f.is_ok(); i++)
	{
		if (!(i % (num_vectors/10+1)))
			CIO::message(M_MESSAGEONLY, "%02d%%.", (int) (100.0*i/num_vectors));
		else if (!(i % (num_vectors/200+1)))
			CIO::message(M_MESSAGEONLY, ".");

		fv=get_feature_vector(i, len, free);
		f.save_word_data(fv, len);
		free_feature_vector(fv, i, free) ;
	}

	if (f.is_ok())
		CIO::message(M_INFO, "%d vectors with %d features each successfully written (filesize: %ld)\n", num_vectors, num_features, num_vectors*num_features*sizeof(WORD));

    return true;
}

/* 
XT=['ATTTTTTAA';'ATTTTTTAA']' ;
sg('send_command', 'loglevel ALL') ;
sg('set_features', 'TRAIN', XT)
sg('send_command', 'convert TRAIN SIMPLE CHAR SIMPLE WORD DNA 3 2 0') ;
*/
