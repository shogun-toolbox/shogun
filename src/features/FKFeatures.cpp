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

#include "features/FKFeatures.h"
#include "features/StringFeatures.h"
#include "lib/io.h"

CFKFeatures::CFKFeatures(INT size, CHMM* p, CHMM* n) : CRealFeatures(size)
{
	pos_prob=NULL;
	neg_prob=NULL;
	weight_a=-1;
	set_models(p,n);
}

 CFKFeatures::CFKFeatures(const CFKFeatures &orig): 
	CRealFeatures(orig), pos(orig.pos), neg(orig.neg), weight_a(orig.weight_a)
{ 
}

CFKFeatures::~CFKFeatures()
{
	SG_UNREF(pos);
	SG_UNREF(neg);
}

double CFKFeatures::deriv_a(double a, INT dimension)
{
	CStringFeatures<WORD> *Obs=pos->get_observations() ;
	double deriv=0.0 ;
	INT i=dimension ;

	if (dimension==-1)
	{
		for (i=0; i<Obs->get_num_vectors(); i++)
		{
			//double pp=pos->model_probability(i) ;
			//double pn=neg->model_probability(i) ;
			double pp=(pos_prob) ? pos_prob[i] : pos->model_probability(i);
			double pn=(neg_prob) ? neg_prob[i] : neg->model_probability(i);
			double sub=pp ;
			if (pn>pp) sub=pn ;
			pp-=sub ;
			pn-=sub ;
			pp=exp(pp) ;
			pn=exp(pn) ;
			double p=a*pp+(1-a)*pn ;
			deriv+=(pp-pn)/p ;

			/*double d1=(pp-pn)/p ;
			  pp=exp(pos->model_probability(i)) ;
			  pn=exp(neg->model_probability(i)) ;
			  p=a*pp+(1-a)*pn ;
			  double d2=(pp-pn)/p ;
			  fprintf(stderr, "d1=%e  d2=%e,  d1-d2=%e\n",d1,d2) ;*/
		} ;
	} else
	{
		double pp=pos->model_probability(i) ;
		double pn=neg->model_probability(i) ;
		double sub=pp ;
		if (pn>pp) sub=pn ;
		pp-=sub ;
		pn-=sub ;
		pp=exp(pp) ;
		pn=exp(pn) ;
		double p=a*pp+(1-a)*pn ;
		deriv+=(pp-pn)/p ;
	} ;

	return deriv ;
}


double CFKFeatures::set_opt_a(double a)
{
	if (a==-1)
	{
		SG_INFO( "estimating a.\n");
		pos_prob=new double[pos->get_observations()->get_num_vectors()];
		neg_prob=new double[pos->get_observations()->get_num_vectors()];
		ASSERT(pos_prob!=NULL);
		ASSERT(neg_prob!=NULL);
		for (INT i=0; i<pos->get_observations()->get_num_vectors(); i++)
		{
			pos_prob[i]=pos->model_probability(i) ;
			neg_prob[i]=neg->model_probability(i) ;
		}

		double la=0;
		double ua=1;
		a=(la+ua)/2;
		while (CMath::abs(ua-la)>1e-6)
		{
			double da=deriv_a(a);
			if (da>0)
				la=a;
			if (da<=0)
				ua=a;
			a=(la+ua)/2;
			SG_INFO( "opt_a: a=%1.3e  deriv=%1.3e  la=%1.3e  ua=%1.3e\n", a, da, la ,ua);
		}
		delete[] pos_prob;
		delete[] neg_prob;
		pos_prob=NULL;
		neg_prob=NULL;
	}

	weight_a=a;
	SG_INFO( "setting opt_a: %g\n", a);
	return a;
}

void CFKFeatures::set_models(CHMM* p, CHMM* n)
{
	ASSERT(p!=NULL && n!=NULL);
	SG_REF(p);
	SG_REF(n);

	pos=p; 
	neg=n;
	set_num_vectors(0);

	free_feature_matrix();

	SG_INFO( "pos_feat=[%i,%i,%i,%i],neg_feat=[%i,%i,%i,%i]\n", pos->get_N(), pos->get_N(), pos->get_N()*pos->get_N(), pos->get_N()*pos->get_M(), neg->get_N(), neg->get_N(), neg->get_N()*neg->get_N(), neg->get_N()*neg->get_M()) ;

	if (pos && pos->get_observations())
		set_num_vectors(pos->get_observations()->get_num_vectors());
	if (pos && neg)
		num_features=1+pos->get_N()*(1+pos->get_N()+1+pos->get_M()) + neg->get_N()*(1+neg->get_N()+1+neg->get_M()) ;
}

DREAL* CFKFeatures::compute_feature_vector(INT num, INT &len, DREAL* target)
{
  DREAL* featurevector=target;
  
  if (!featurevector)
	featurevector=new DREAL[ 1+pos->get_N()*(1+pos->get_N()+1+pos->get_M()) + neg->get_N()*(1+neg->get_N()+1+neg->get_M()) ];
  
  if (!featurevector)
    return NULL;
  
  compute_feature_vector(featurevector, num, len);

  return featurevector;
}

void CFKFeatures::compute_feature_vector(DREAL* featurevector, INT num, INT& len)
{
	INT i,j,p=0,x=num;

	double posx=pos->model_probability(x);
	double negx=neg->model_probability(x);

	len=1+pos->get_N()*(1+pos->get_N()+1+pos->get_M()) + neg->get_N()*(1+neg->get_N()+1+neg->get_M());

	featurevector[p++] = deriv_a(weight_a, x);
	double px=CMath::logarithmic_sum(posx+log(weight_a),negx+log(1-weight_a)) ;

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

DREAL* CFKFeatures::set_feature_matrix()
{
	ASSERT(pos);
	ASSERT(pos->get_observations());
	ASSERT(neg);
	ASSERT(neg->get_observations());

	INT len=0;
	num_features=1+ pos->get_N()*(1+pos->get_N()+1+pos->get_M()) + neg->get_N()*(1+neg->get_N()+1+neg->get_M());

	num_vectors=pos->get_observations()->get_num_vectors();
	ASSERT(num_vectors);

	SG_INFO( "allocating FK feature cache of size %.2fM\n", sizeof(double)*num_features*num_vectors/1024.0/1024.0);
	free_feature_matrix();
	feature_matrix=new DREAL[num_features*num_vectors];

	SG_INFO( "calculating FK feature matrix\n");

	for (INT x=0; x<num_vectors; x++)
	{
		if (!(x % (num_vectors/10+1)))
			SG_DEBUG("%02d%%.", (int) (100.0*x/num_vectors));
		else if (!(x % (num_vectors/200+1)))
			SG_DEBUG(".");

		compute_feature_vector(&feature_matrix[x*num_features], x, len);
	}

	SG_INFO("done.\n");
	
	num_vectors=get_num_vectors() ;
	num_features=get_num_features() ;

	return feature_matrix;
}
