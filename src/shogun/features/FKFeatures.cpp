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

#include <shogun/features/FKFeatures.h>
#include <shogun/features/StringFeatures.h>
#include <shogun/io/SGIO.h>
#include <shogun/base/Parameter.h>

using namespace shogun;

CFKFeatures::CFKFeatures() : CDenseFeatures<float64_t>()
{
	init();
}

CFKFeatures::CFKFeatures(int32_t size, CHMM* p, CHMM* n)
: CDenseFeatures<float64_t>(size)
{
	init();
	weight_a=-1;
	set_models(p,n);
}

CFKFeatures::CFKFeatures(const CFKFeatures &orig)
: CDenseFeatures<float64_t>(orig), pos(orig.pos), neg(orig.neg), weight_a(orig.weight_a)
{
}

CFKFeatures::~CFKFeatures()
{
	SG_UNREF(pos);
	SG_UNREF(neg);
}

float64_t CFKFeatures::deriv_a(float64_t a, int32_t dimension)
{
	CStringFeatures<uint16_t> *Obs=pos->get_observations() ;
	float64_t deriv=0.0 ;
	int32_t i=dimension ;

	if (dimension==-1)
	{
		for (i=0; i<Obs->get_num_vectors(); i++)
		{
			//float64_t pp=pos->model_probability(i) ;
			//float64_t pn=neg->model_probability(i) ;
			float64_t pp=(pos_prob) ? pos_prob[i] : pos->model_probability(i);
			float64_t pn=(neg_prob) ? neg_prob[i] : neg->model_probability(i);
			float64_t sub=pp ;
			if (pn>pp) sub=pn ;
			pp-=sub ;
			pn-=sub ;
			pp=exp(pp) ;
			pn=exp(pn) ;
			float64_t p=a*pp+(1-a)*pn ;
			deriv+=(pp-pn)/p ;

			/*float64_t d1=(pp-pn)/p ;
			  pp=exp(pos->model_probability(i)) ;
			  pn=exp(neg->model_probability(i)) ;
			  p=a*pp+(1-a)*pn ;
			  float64_t d2=(pp-pn)/p ;
			  fprintf(stderr, "d1=%e  d2=%e,  d1-d2=%e\n",d1,d2) ;*/
		} ;
	} else
	{
		float64_t pp=pos->model_probability(i) ;
		float64_t pn=neg->model_probability(i) ;
		float64_t sub=pp ;
		if (pn>pp) sub=pn ;
		pp-=sub ;
		pn-=sub ;
		pp=exp(pp) ;
		pn=exp(pn) ;
		float64_t p=a*pp+(1-a)*pn ;
		deriv+=(pp-pn)/p ;
	} ;

	return deriv ;
}


float64_t CFKFeatures::set_opt_a(float64_t a)
{
	if (a==-1)
	{
		SG_INFO("estimating a.\n")
		pos_prob=SG_MALLOC(float64_t, pos->get_observations()->get_num_vectors());
		neg_prob=SG_MALLOC(float64_t, pos->get_observations()->get_num_vectors());
		for (int32_t i=0; i<pos->get_observations()->get_num_vectors(); i++)
		{
			pos_prob[i]=pos->model_probability(i) ;
			neg_prob[i]=neg->model_probability(i) ;
		}

		float64_t la=0;
		float64_t ua=1;
		a=(la+ua)/2;
		while (CMath::abs(ua-la)>1e-6)
		{
			float64_t da=deriv_a(a);
			if (da>0)
				la=a;
			if (da<=0)
				ua=a;
			a=(la+ua)/2;
			SG_INFO("opt_a: a=%1.3e  deriv=%1.3e  la=%1.3e  ua=%1.3e\n", a, da, la ,ua)
		}
		SG_FREE(pos_prob);
		SG_FREE(neg_prob);
		pos_prob=NULL;
		neg_prob=NULL;
	}

	weight_a=a;
	SG_INFO("setting opt_a: %g\n", a)
	return a;
}

void CFKFeatures::set_models(CHMM* p, CHMM* n)
{
	ASSERT(p && n)
	SG_REF(p);
	SG_REF(n);

	pos=p;
	neg=n;
	set_num_vectors(0);

	free_feature_matrix();

	SG_INFO("pos_feat=[%i,%i,%i,%i],neg_feat=[%i,%i,%i,%i]\n", pos->get_N(), pos->get_N(), pos->get_N()*pos->get_N(), pos->get_N()*pos->get_M(), neg->get_N(), neg->get_N(), neg->get_N()*neg->get_N(), neg->get_N()*neg->get_M())

	if (pos && pos->get_observations())
		set_num_vectors(pos->get_observations()->get_num_vectors());
	if (pos && neg)
		num_features=1+pos->get_N()*(1+pos->get_N()+1+pos->get_M()) + neg->get_N()*(1+neg->get_N()+1+neg->get_M()) ;
}

float64_t* CFKFeatures::compute_feature_vector(
	int32_t num, int32_t &len, float64_t* target)
{
	float64_t* featurevector=target;

	if (!featurevector)
		featurevector=SG_MALLOC(float64_t,
			1+
			pos->get_N()*(1+pos->get_N()+1+pos->get_M())+
			neg->get_N()*(1+neg->get_N()+1+neg->get_M())
		);

	if (!featurevector)
		return NULL;

	compute_feature_vector(featurevector, num, len);

	return featurevector;
}

void CFKFeatures::compute_feature_vector(
	float64_t* featurevector, int32_t num, int32_t& len)
{
	int32_t i,j,p=0,x=num;

	float64_t posx=pos->model_probability(x);
	float64_t negx=neg->model_probability(x);

	len=1+pos->get_N()*(1+pos->get_N()+1+pos->get_M()) + neg->get_N()*(1+neg->get_N()+1+neg->get_M());

	featurevector[p++] = deriv_a(weight_a, x);
	float64_t px=CMath::logarithmic_sum(
		posx+log(weight_a),negx+log(1-weight_a));

	//first do positive model
	for (i=0; i<pos->get_N(); i++)
	{
		featurevector[p++]=weight_a*exp(pos->model_derivative_p(i, x)-px);
		featurevector[p++]=weight_a*exp(pos->model_derivative_q(i, x)-px);

		for (j=0; j<pos->get_N(); j++) {
			featurevector[p++]=weight_a*exp(pos->model_derivative_a(i, j, x)-px);
		}

		for (j=0; j<pos->get_M(); j++) {
			featurevector[p++]=weight_a*exp(pos->model_derivative_b(i, j, x)-px);
		}

	}

	//then do negative
	for (i=0; i<neg->get_N(); i++)
	{
		featurevector[p++]= (1-weight_a)*exp(neg->model_derivative_p(i, x)-px);
		featurevector[p++]= (1-weight_a)* exp(neg->model_derivative_q(i, x)-px);

		for (j=0; j<neg->get_N(); j++) {
			featurevector[p++]= (1-weight_a)*exp(neg->model_derivative_a(i, j, x)-px);
		}

		for (j=0; j<neg->get_M(); j++) {
			featurevector[p++]= (1-weight_a)*exp(neg->model_derivative_b(i, j, x)-px);
		}
	}
}

float64_t* CFKFeatures::set_feature_matrix()
{
	ASSERT(pos)
	ASSERT(pos->get_observations())
	ASSERT(neg)
	ASSERT(neg->get_observations())

	int32_t len=0;
	num_features=1+ pos->get_N()*(1+pos->get_N()+1+pos->get_M()) + neg->get_N()*(1+neg->get_N()+1+neg->get_M());

	num_vectors=pos->get_observations()->get_num_vectors();
	ASSERT(num_vectors)

	SG_INFO("allocating FK feature cache of size %.2fM\n", sizeof(float64_t)*num_features*num_vectors/1024.0/1024.0)
	free_feature_matrix();
	feature_matrix=SGMatrix<float64_t>(num_features,num_vectors);

	SG_INFO("calculating FK feature matrix\n")

	for (int32_t x=0; x<num_vectors; x++)
	{
		if (!(x % (num_vectors/10+1)))
			SG_DEBUG("%02d%%.", (int) (100.0*x/num_vectors))
		else if (!(x % (num_vectors/200+1)))
			SG_DEBUG(".")

		compute_feature_vector(&feature_matrix.matrix[x*num_features], x, len);
	}

	SG_DONE()

	num_vectors=get_num_vectors();
	num_features=get_num_features();

	return feature_matrix.matrix;
}

void CFKFeatures::init()
{
	pos = NULL;
	neg = NULL;
	pos_prob = NULL;
	neg_prob = NULL;
	weight_a = 0.0;

	unset_generic();
	//TODO serialize HMMs
	//m_parameters->add((CSGObject**) &pos, "pos", "HMM for positive class.");
	//m_parameters->add((CSGObject**) &neg, "neg", "HMM for negative class.");
	m_parameters->add(&weight_a, "weight_a", "Class prior.");
}
