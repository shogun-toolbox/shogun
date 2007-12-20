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

#include "distributions/hmm/LinearHMM.h"
#include "lib/common.h"
#include "features/StringFeatures.h"
#include "lib/io.h"

CLinearHMM::CLinearHMM(CStringFeatures<WORD>* f) : hist(NULL), log_hist(NULL)
{
	features=f;
	sequence_length = f->get_vector_length(0);
	num_symbols     = f->get_num_symbols();
	num_params      = sequence_length*num_symbols;
}

CLinearHMM::CLinearHMM(INT p_num_features, INT p_num_symbols) : hist(NULL), log_hist(NULL)
{
	sequence_length = p_num_features;
	num_symbols     = p_num_symbols;
	num_params      = sequence_length*num_symbols;
}

CLinearHMM::~CLinearHMM()
{
	delete[] hist;
	delete[] log_hist;
}

bool CLinearHMM::train()
{
	delete[] hist;
	delete[] log_hist;
	INT* int_hist = new INT[num_params];
	ASSERT(int_hist);

	INT vec;
	INT i;

	for (i=0; i< num_params; i++)
		int_hist[i]=0;

	for (vec=0; vec<features->get_num_vectors(); vec++)
	{
		INT len;

		WORD* vector=((CStringFeatures<WORD>*) features)->get_feature_vector(vec, len);

		//just count the symbols per position -> histogram
		for (INT feat=0; feat<len ; feat++)
			int_hist[feat*num_symbols+vector[feat]]++;
	}

	//trade memory for speed
	hist= new DREAL[num_params];
	log_hist= new DREAL[num_params];

	ASSERT(hist);
	ASSERT(log_hist);

	for (i=0;i<sequence_length;i++)
	{
		for (INT j=0; j<num_symbols; j++)
		{
			DREAL sum=0;
			INT offs=i*num_symbols+((CStringFeatures<WORD> *) features)->get_masked_symbols((WORD)j,(BYTE) 254);
			INT original_num_symbols=((CStringFeatures<WORD> *) features)->get_original_num_symbols();

			for (INT k=0; k<original_num_symbols; k++)
				sum+=int_hist[offs+k];

			hist[i*num_symbols+j]=(int_hist[i*num_symbols+j]+pseudo_count)/(sum+((CStringFeatures<WORD> *) features)->get_original_num_symbols()*pseudo_count);
			log_hist[i*num_symbols+j]=log(hist[i*num_symbols+j]);
		}
	}

	delete[] int_hist;
	return true;
}

bool CLinearHMM::train(const INT* indizes, INT num_indizes, DREAL pseudo)
{
	delete[] hist;
	delete[] log_hist;
	INT* int_hist = new INT[num_params];
	ASSERT(int_hist);

	INT vec;
	INT i;

	for (i=0; i< num_params; i++)
		int_hist[i]=0;

	for (vec=0; vec<num_indizes; vec++)
	{
		INT len;

		ASSERT(indizes[vec]>=0 && indizes[vec]<((CStringFeatures<WORD>*) features)->get_num_vectors());
		WORD* vector=((CStringFeatures<WORD>*) features)->get_feature_vector(indizes[vec], len);

		//just count the symbols per position -> histogram
		//
		for (INT feat=0; feat<len ; feat++)
			int_hist[feat*num_symbols+vector[feat]]++;
	}

	//trade memory for speed
	hist= new DREAL[num_params];
	log_hist= new DREAL[num_params];

	ASSERT(hist);
	ASSERT(log_hist);

	for (i=0;i<sequence_length;i++)
	{
		for (INT j=0; j<num_symbols; j++)
		{
			DREAL sum=0;
			INT original_num_symbols=((CStringFeatures<WORD> *) features)->get_original_num_symbols();
			for (INT k=0; k<original_num_symbols; k++)
			{
				sum+=int_hist[i*num_symbols+
					((CStringFeatures<WORD>*) features)->
						get_masked_symbols((WORD)j,(BYTE) 254)+k];
			}

			hist[i*num_symbols+j]=(int_hist[i*num_symbols+j]+pseudo)/(sum+((CStringFeatures<WORD>*) features)->get_original_num_symbols()*pseudo);
			log_hist[i*num_symbols+j]=log(hist[i*num_symbols+j]);
		}
	}

	delete[] int_hist;
	return true;
}

DREAL CLinearHMM::get_log_likelihood_example(WORD* vector, INT len)
{
	DREAL result=log_hist[vector[0]];

	for (INT i=1; i<len; i++)
		result+=log_hist[i*num_symbols+vector[i]];
	
	return result;
}

DREAL CLinearHMM::get_log_likelihood_example(INT num_example)
{
	INT len;
	WORD* vector=((CStringFeatures<WORD>*) features)->get_feature_vector(num_example, len);
	DREAL result=log_hist[vector[0]];

	for (INT i=1; i<len; i++)
		result+=log_hist[i*num_symbols+vector[i]];

	return result;
}

DREAL CLinearHMM::get_likelihood_example(WORD* vector, INT len)
{
	DREAL result=hist[vector[0]];

	for (INT i=1; i<len; i++)
		result*=hist[i*num_symbols+vector[i]];
	
	return result;
}

DREAL CLinearHMM::get_log_derivative(INT num_param, INT num_example)
{
	INT len;
	WORD* vector=((CStringFeatures<WORD>*) features)->get_feature_vector(num_example, len);
	DREAL result=0;
	WORD check=0;
	INT position=num_param/num_symbols;

	if (position!=0) {
		check=num_param/position;
	}

	if (vector[position]==check && hist[num_param]!=0)
	{
		result=1.0/hist[num_param];
	}

	return result;
}

void CLinearHMM::set_log_hist(const DREAL* new_log_hist)
{
	if (!log_hist)
		log_hist = new DREAL[num_params];

	if (!hist)
		hist = new DREAL[num_params];

	for (INT i=0; i< num_params; i++)
	{
		log_hist[i]=new_log_hist[i];
		hist[i]=exp(log_hist[i]);
	}
}

void CLinearHMM::set_hist(const DREAL* new_hist)
{
	if (!log_hist)
		log_hist = new DREAL[num_params];

	if (!hist)
		hist = new DREAL[num_params];

	for (INT i=0; i< num_params; i++)
	{
		hist[i]=new_hist[i];
		log_hist[i]=log(hist[i]);
	}
}
