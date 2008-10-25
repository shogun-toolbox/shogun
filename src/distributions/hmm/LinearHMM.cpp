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

#include "distributions/hmm/LinearHMM.h"
#include "lib/common.h"
#include "features/StringFeatures.h"
#include "lib/io.h"

CLinearHMM::CLinearHMM(CStringFeatures<uint16_t>* f)
: CDistribution(), transition_probs(NULL), log_transition_probs(NULL)
{
	features=f;
	sequence_length = f->get_vector_length(0);
	num_symbols     = (INT) f->get_num_symbols();
	num_params      = sequence_length*num_symbols;
}

CLinearHMM::CLinearHMM(INT p_num_features, INT p_num_symbols)
: CDistribution(), transition_probs(NULL), log_transition_probs(NULL)
{
	sequence_length = p_num_features;
	num_symbols     = p_num_symbols;
	num_params      = sequence_length*num_symbols;
}

CLinearHMM::~CLinearHMM()
{
	delete[] transition_probs;
	delete[] log_transition_probs;
}

bool CLinearHMM::train()
{
	delete[] transition_probs;
	delete[] log_transition_probs;
	INT* int_transition_probs=new INT[num_params];

	INT vec;
	INT i;

	for (i=0; i< num_params; i++)
		int_transition_probs[i]=0;

	for (vec=0; vec<features->get_num_vectors(); vec++)
	{
		INT len;

		uint16_t* vector=((CStringFeatures<uint16_t>*) features)->get_feature_vector(vec, len);

		//just count the symbols per position -> transition_probsogram
		for (INT feat=0; feat<len ; feat++)
			int_transition_probs[feat*num_symbols+vector[feat]]++;
	}

	//trade memory for speed
	transition_probs=new DREAL[num_params];
	log_transition_probs=new DREAL[num_params];

	for (i=0;i<sequence_length;i++)
	{
		for (INT j=0; j<num_symbols; j++)
		{
			DREAL sum=0;
			INT offs=i*num_symbols+((CStringFeatures<uint16_t> *) features)->get_masked_symbols((uint16_t)j,(uint8_t) 254);
			INT original_num_symbols=(INT) ((CStringFeatures<uint16_t> *) features)->get_original_num_symbols();

			for (INT k=0; k<original_num_symbols; k++)
				sum+=int_transition_probs[offs+k];

			transition_probs[i*num_symbols+j]=(int_transition_probs[i*num_symbols+j]+pseudo_count)/(sum+((CStringFeatures<uint16_t> *) features)->get_original_num_symbols()*pseudo_count);
			log_transition_probs[i*num_symbols+j]=log(transition_probs[i*num_symbols+j]);
		}
	}

	delete[] int_transition_probs;
	return true;
}

bool CLinearHMM::train(const INT* indizes, INT num_indizes, DREAL pseudo)
{
	delete[] transition_probs;
	delete[] log_transition_probs;
	INT* int_transition_probs=new INT[num_params];
	INT vec;
	INT i;

	for (i=0; i< num_params; i++)
		int_transition_probs[i]=0;

	for (vec=0; vec<num_indizes; vec++)
	{
		INT len;

		ASSERT(indizes[vec]>=0 && indizes[vec]<((CStringFeatures<uint16_t>*) features)->get_num_vectors());
		uint16_t* vector=((CStringFeatures<uint16_t>*) features)->get_feature_vector(indizes[vec], len);

		//just count the symbols per position -> transition_probsogram
		//
		for (INT feat=0; feat<len ; feat++)
			int_transition_probs[feat*num_symbols+vector[feat]]++;
	}

	//trade memory for speed
	transition_probs=new DREAL[num_params];
	log_transition_probs=new DREAL[num_params];

	for (i=0;i<sequence_length;i++)
	{
		for (INT j=0; j<num_symbols; j++)
		{
			DREAL sum=0;
			INT original_num_symbols= (INT) ((CStringFeatures<uint16_t> *) features)->get_original_num_symbols();
			for (INT k=0; k<original_num_symbols; k++)
			{
				sum+=int_transition_probs[i*num_symbols+
					((CStringFeatures<uint16_t>*) features)->
						get_masked_symbols((uint16_t)j,(uint8_t) 254)+k];
			}

			transition_probs[i*num_symbols+j]=(int_transition_probs[i*num_symbols+j]+pseudo)/(sum+((CStringFeatures<uint16_t>*) features)->get_original_num_symbols()*pseudo);
			log_transition_probs[i*num_symbols+j]=log(transition_probs[i*num_symbols+j]);
		}
	}

	delete[] int_transition_probs;
	return true;
}

DREAL CLinearHMM::get_log_likelihood_example(uint16_t* vector, INT len)
{
	DREAL result=log_transition_probs[vector[0]];

	for (INT i=1; i<len; i++)
		result+=log_transition_probs[i*num_symbols+vector[i]];
	
	return result;
}

DREAL CLinearHMM::get_log_likelihood_example(INT num_example)
{
	INT len;
	uint16_t* vector=((CStringFeatures<uint16_t>*) features)->get_feature_vector(num_example, len);
	DREAL result=log_transition_probs[vector[0]];

	for (INT i=1; i<len; i++)
		result+=log_transition_probs[i*num_symbols+vector[i]];

	return result;
}

DREAL CLinearHMM::get_likelihood_example(uint16_t* vector, INT len)
{
	DREAL result=transition_probs[vector[0]];

	for (INT i=1; i<len; i++)
		result*=transition_probs[i*num_symbols+vector[i]];
	
	return result;
}

DREAL CLinearHMM::get_log_derivative(INT num_param, INT num_example)
{
	INT len;
	uint16_t* vector=((CStringFeatures<uint16_t>*) features)->get_feature_vector(num_example, len);
	DREAL result=0;
	INT position=num_param/num_symbols;
	ASSERT(position>=0 && position<len);
	uint16_t sym=(uint16_t) (num_param-position*num_symbols);

	if (vector[position]==sym && transition_probs[num_param]!=0)
		result=1.0/transition_probs[num_param];

	return result;
}

void CLinearHMM::get_transition_probs(DREAL** dst, INT* num)
{
	*num=num_params;
	size_t sz=sizeof(*transition_probs)*(*num);
	*dst=(DREAL*) malloc(sz);
	ASSERT(dst);

	memcpy(*dst, transition_probs, sz);
}

bool CLinearHMM::set_transition_probs(const DREAL* src, INT num)
{
	if (num!=-1)
		ASSERT(num==num_params);

	if (!log_transition_probs)
		log_transition_probs=new DREAL[num_params];

	if (!transition_probs)
		transition_probs=new DREAL[num_params];

	for (INT i=0; i<num_params; i++)
	{
		transition_probs[i]=src[i];
		log_transition_probs[i]=log(transition_probs[i]);
	}

	return true;
}

void CLinearHMM::get_log_transition_probs(DREAL** dst, INT* num)
{
	*num=num_params;
	size_t sz=sizeof(*log_transition_probs)*(*num);
	*dst=(DREAL*) malloc(sz);
	ASSERT(dst);

	memcpy(*dst, log_transition_probs, sz);
}

bool CLinearHMM::set_log_transition_probs(const DREAL* src, INT num)
{
	if (num!=-1)
		ASSERT(num==num_params);

	if (!log_transition_probs)
		log_transition_probs=new DREAL[num_params];

	if (!transition_probs)
		transition_probs=new DREAL[num_params];

	for (INT i=0; i< num_params; i++)
	{
		log_transition_probs[i]=src[i];
		transition_probs[i]=exp(log_transition_probs[i]);
	}

	return true;
}




