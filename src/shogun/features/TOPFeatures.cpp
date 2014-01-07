/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2009 Soeren Sonnenburg
 * Written (W) 1999-2008 Gunnar Raetsch
 * Copyright (C) 1999-2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include <features/TOPFeatures.h>
#include <io/SGIO.h>
#include <mathematics/Math.h>

using namespace shogun;

CTOPFeatures::CTOPFeatures()
{
	init();
}

CTOPFeatures::CTOPFeatures(
	int32_t size, CHMM* p, CHMM* n, bool neglin, bool poslin)
: CDenseFeatures<float64_t>(size)
{
	init();
	neglinear=neglin;
	poslinear=poslin;

	set_models(p,n);
}

CTOPFeatures::CTOPFeatures(const CTOPFeatures &orig)
: CDenseFeatures<float64_t>(orig)
{
	init();
	pos=orig.pos;
	neg=orig.neg;
	neglinear=orig.neglinear;
	poslinear=orig.poslinear;
}

CTOPFeatures::~CTOPFeatures()
{
	SG_FREE(pos_relevant_indizes.idx_p);
	SG_FREE(pos_relevant_indizes.idx_q);
	SG_FREE(pos_relevant_indizes.idx_a_cols);
	SG_FREE(pos_relevant_indizes.idx_a_rows);
	SG_FREE(pos_relevant_indizes.idx_b_cols);
	SG_FREE(pos_relevant_indizes.idx_b_rows);

	SG_FREE(neg_relevant_indizes.idx_p);
	SG_FREE(neg_relevant_indizes.idx_q);
	SG_FREE(neg_relevant_indizes.idx_a_cols);
	SG_FREE(neg_relevant_indizes.idx_a_rows);
	SG_FREE(neg_relevant_indizes.idx_b_cols);
	SG_FREE(neg_relevant_indizes.idx_b_rows);

	SG_UNREF(pos);
	SG_UNREF(neg);
}

void CTOPFeatures::set_models(CHMM* p, CHMM* n)
{
	ASSERT(p && n)
	SG_REF(p);
	SG_REF(n);

	pos=p;
	neg=n;
	set_num_vectors(0);

	feature_matrix=SGMatrix<float64_t>();

	if (pos && pos->get_observations())
		set_num_vectors(pos->get_observations()->get_num_vectors());

	compute_relevant_indizes(p, &pos_relevant_indizes);
	compute_relevant_indizes(n, &neg_relevant_indizes);
	num_features=compute_num_features();

	SG_DEBUG("pos_feat=[%i,%i,%i,%i],neg_feat=[%i,%i,%i,%i] -> %i features\n", pos->get_N(), pos->get_N(), pos->get_N()*pos->get_N(), pos->get_N()*pos->get_M(), neg->get_N(), neg->get_N(), neg->get_N()*neg->get_N(), neg->get_N()*neg->get_M(),num_features)
}

float64_t* CTOPFeatures::compute_feature_vector(
	int32_t num, int32_t &len, float64_t* target)
{
	float64_t* featurevector=target;

	if (!featurevector)
		featurevector=SG_MALLOC(float64_t, get_num_features());

	if (!featurevector)
		return NULL;

	compute_feature_vector(featurevector, num, len);

	return featurevector;
}

void CTOPFeatures::compute_feature_vector(
	float64_t* featurevector, int32_t num, int32_t& len)
{
	int32_t i,j,p=0,x=num;
	int32_t idx=0;

	float64_t posx=(poslinear) ?
		(pos->linear_model_probability(x)) : (pos->model_probability(x));
	float64_t negx=(neglinear) ?
		(neg->linear_model_probability(x)) : (neg->model_probability(x));

	len=get_num_features();

	featurevector[p++]=(posx-negx);

	//first do positive model
	if (poslinear)
	{
		for (i=0; i<pos->get_N(); i++)
		{
			for (j=0; j<pos->get_M(); j++)
				featurevector[p++]=exp(pos->linear_model_derivative(i, j, x)-posx);
		}
	}
	else
	{
		for (idx=0; idx< pos_relevant_indizes.num_p; idx++)
			featurevector[p++]=exp(pos->model_derivative_p(pos_relevant_indizes.idx_p[idx], x)-posx);

		for (idx=0; idx< pos_relevant_indizes.num_q; idx++)
			featurevector[p++]=exp(pos->model_derivative_q(pos_relevant_indizes.idx_q[idx], x)-posx);

		for (idx=0; idx< pos_relevant_indizes.num_a; idx++)
				featurevector[p++]=exp(pos->model_derivative_a(pos_relevant_indizes.idx_a_rows[idx], pos_relevant_indizes.idx_a_cols[idx], x)-posx);

		for (idx=0; idx< pos_relevant_indizes.num_b; idx++)
				featurevector[p++]=exp(pos->model_derivative_b(pos_relevant_indizes.idx_b_rows[idx], pos_relevant_indizes.idx_b_cols[idx], x)-posx);


		//for (i=0; i<pos->get_N(); i++)
		//{
		//	featurevector[p++]=exp(pos->model_derivative_p(i, x)-posx);
		//	featurevector[p++]=exp(pos->model_derivative_q(i, x)-posx);

		//	for (j=0; j<pos->get_N(); j++)
		//		featurevector[p++]=exp(pos->model_derivative_a(i, j, x)-posx);

		//	for (j=0; j<pos->get_M(); j++)
		//		featurevector[p++]=exp(pos->model_derivative_b(i, j, x)-posx);
		//}
	}

	//then do negative
	if (neglinear)
	{
		for (i=0; i<neg->get_N(); i++)
		{
			for (j=0; j<neg->get_M(); j++)
				featurevector[p++]= - exp(neg->linear_model_derivative(i, j, x)-negx);
		}
	}
	else
	{
		for (idx=0; idx< neg_relevant_indizes.num_p; idx++)
			featurevector[p++]= - exp(neg->model_derivative_p(neg_relevant_indizes.idx_p[idx], x)-negx);

		for (idx=0; idx< neg_relevant_indizes.num_q; idx++)
			featurevector[p++]= - exp(neg->model_derivative_q(neg_relevant_indizes.idx_q[idx], x)-negx);

		for (idx=0; idx< neg_relevant_indizes.num_a; idx++)
				featurevector[p++]= - exp(neg->model_derivative_a(neg_relevant_indizes.idx_a_rows[idx], neg_relevant_indizes.idx_a_cols[idx], x)-negx);

		for (idx=0; idx< neg_relevant_indizes.num_b; idx++)
				featurevector[p++]= - exp(neg->model_derivative_b(neg_relevant_indizes.idx_b_rows[idx], neg_relevant_indizes.idx_b_cols[idx], x)-negx);

		//for (i=0; i<neg->get_N(); i++)
		//{
		//	featurevector[p++]= - exp(neg->model_derivative_p(i, x)-negx);
		//	featurevector[p++]= - exp(neg->model_derivative_q(i, x)-negx);

		//	for (j=0; j<neg->get_N(); j++)
		//		featurevector[p++]= - exp(neg->model_derivative_a(i, j, x)-negx);

		//	for (j=0; j<neg->get_M(); j++)
		//		featurevector[p++]= - exp(neg->model_derivative_b(i, j, x)-negx);
		//}
	}
}

float64_t* CTOPFeatures::set_feature_matrix()
{
	int32_t len=0;

	num_features=get_num_features();
	ASSERT(num_features)
	ASSERT(pos)
	ASSERT(pos->get_observations())

	num_vectors=pos->get_observations()->get_num_vectors();
	SG_INFO("allocating top feature cache of size %.2fM\n", sizeof(float64_t)*num_features*num_vectors/1024.0/1024.0)
	feature_matrix = SGMatrix<float64_t>(num_features,num_vectors);
	if (!feature_matrix.matrix)
	{
      SG_ERROR("allocation not successful!")
		return NULL ;
	} ;

	SG_INFO("calculating top feature matrix\n")

	for (int32_t x=0; x<num_vectors; x++)
	{
		if (!(x % (num_vectors/10+1)))
			SG_DEBUG("%02d%%.", (int) (100.0*x/num_vectors))
		else if (!(x % (num_vectors/200+1)))
			SG_DEBUG(".")

		compute_feature_vector(&feature_matrix[x*num_features], x, len);
	}

	SG_DONE()

	num_vectors=get_num_vectors() ;
	num_features=get_num_features() ;

	return feature_matrix.matrix;
}

bool CTOPFeatures::compute_relevant_indizes(CHMM* hmm, T_HMM_INDIZES* hmm_idx)
{
	int32_t i=0;
	int32_t j=0;

	hmm_idx->num_p=0;
	hmm_idx->num_q=0;
	hmm_idx->num_a=0;
	hmm_idx->num_b=0;

	for (i=0; i<hmm->get_N(); i++)
	{
		if (hmm->get_p(i)>CMath::ALMOST_NEG_INFTY)
			hmm_idx->num_p++;

		if (hmm->get_q(i)>CMath::ALMOST_NEG_INFTY)
			hmm_idx->num_q++;

		for (j=0; j<hmm->get_N(); j++)
		{
			if (hmm->get_a(i,j)>CMath::ALMOST_NEG_INFTY)
				hmm_idx->num_a++;
		}

		for (j=0; j<pos->get_M(); j++)
		{
			if (hmm->get_b(i,j)>CMath::ALMOST_NEG_INFTY)
				hmm_idx->num_b++;
		}
	}

	if (hmm_idx->num_p > 0)
	{
		hmm_idx->idx_p=SG_MALLOC(int32_t, hmm_idx->num_p);
		ASSERT(hmm_idx->idx_p)
	}

	if (hmm_idx->num_q > 0)
	{
		hmm_idx->idx_q=SG_MALLOC(int32_t, hmm_idx->num_q);
		ASSERT(hmm_idx->idx_q)
	}

	if (hmm_idx->num_a > 0)
	{
		hmm_idx->idx_a_rows=SG_MALLOC(int32_t, hmm_idx->num_a);
		hmm_idx->idx_a_cols=SG_MALLOC(int32_t, hmm_idx->num_a);
		ASSERT(hmm_idx->idx_a_rows)
		ASSERT(hmm_idx->idx_a_cols)
	}

	if (hmm_idx->num_b > 0)
	{
		hmm_idx->idx_b_rows=SG_MALLOC(int32_t, hmm_idx->num_b);
		hmm_idx->idx_b_cols=SG_MALLOC(int32_t, hmm_idx->num_b);
		ASSERT(hmm_idx->idx_b_rows)
		ASSERT(hmm_idx->idx_b_cols)
	}


	int32_t idx_p=0;
	int32_t idx_q=0;
	int32_t idx_a=0;
	int32_t idx_b=0;

	for (i=0; i<hmm->get_N(); i++)
	{
		if (hmm->get_p(i)>CMath::ALMOST_NEG_INFTY)
		{
			ASSERT(idx_p < hmm_idx->num_p)
			hmm_idx->idx_p[idx_p++]=i;
		}

		if (hmm->get_q(i)>CMath::ALMOST_NEG_INFTY)
		{
			ASSERT(idx_q < hmm_idx->num_q)
			hmm_idx->idx_q[idx_q++]=i;
		}

		for (j=0; j<hmm->get_N(); j++)
		{
			if (hmm->get_a(i,j)>CMath::ALMOST_NEG_INFTY)
			{
				ASSERT(idx_a < hmm_idx->num_a)
				hmm_idx->idx_a_rows[idx_a]=i;
				hmm_idx->idx_a_cols[idx_a++]=j;
			}
		}

		for (j=0; j<pos->get_M(); j++)
		{
			if (hmm->get_b(i,j)>CMath::ALMOST_NEG_INFTY)
			{
				ASSERT(idx_b < hmm_idx->num_b)
				hmm_idx->idx_b_rows[idx_b]=i;
				hmm_idx->idx_b_cols[idx_b++]=j;
			}
		}
	}

	return true;
}

int32_t CTOPFeatures::compute_num_features()
{
	int32_t num=0;

	if (pos && neg)
	{
		num+=1; //zeroth- component

		if (poslinear)
			num+=pos->get_N()*pos->get_M();
		else
		{
			num+= pos_relevant_indizes.num_p + pos_relevant_indizes.num_q + pos_relevant_indizes.num_a + pos_relevant_indizes.num_b;
		}

		if (neglinear)
			num+=neg->get_N()*neg->get_M();
		else
		{
			num+= neg_relevant_indizes.num_p + neg_relevant_indizes.num_q + neg_relevant_indizes.num_a + neg_relevant_indizes.num_b;
		}

		//num+=1; //zeroth- component
		//num+= (poslinear) ? (pos->get_N()*pos->get_M()) : (pos->get_N()*(1+pos->get_N()+1+pos->get_M()));
		//num+= (neglinear) ? (neg->get_N()*neg->get_M()) : (neg->get_N()*(1+neg->get_N()+1+neg->get_M()));
	}
	return num;
}

void CTOPFeatures::init()
{
	pos = NULL;
	neg = NULL;
	neglinear = false;
	poslinear = false;

	memset(&pos_relevant_indizes, 0, sizeof(pos_relevant_indizes));
	memset(&neg_relevant_indizes, 0, sizeof(neg_relevant_indizes));

	unset_generic();
	//TODO serialize HMMs
	//m_parameters->add((CSGObject**) &pos, "pos", "HMM for positive class.");
	//m_parameters->add((CSGObject**) &neg, "neg", "HMM for negative class.");
	m_parameters->add(&neglinear, "neglinear", "If negative HMM is a LinearHMM");
	m_parameters->add(&poslinear, "poslinear", "If positive HMM is a LinearHMM");
}
