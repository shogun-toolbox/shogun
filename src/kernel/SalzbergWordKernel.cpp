/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2008 Gunnar Raetsch
 * Copyright (C) 1999-2008 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include "lib/common.h"
#include "lib/io.h"
#include "kernel/SalzbergWordKernel.h"
#include "features/Features.h"
#include "features/StringFeatures.h"
#include "features/Labels.h"
#include "classifier/PluginEstimate.h"

CSalzbergWordKernel::CSalzbergWordKernel(INT size, CPluginEstimate* pie)
: CStringKernel<WORD>(size), estimate(pie), mean(NULL), variance(NULL),
	sqrtdiag_lhs(NULL), sqrtdiag_rhs(NULL),
	ld_mean_lhs(NULL), ld_mean_rhs(NULL),
	num_params(0), num_symbols(0), sum_m2_s2(0), pos_prior(0.5),
	neg_prior(0.5), initialized(false)
{
}

CSalzbergWordKernel::CSalzbergWordKernel(
	CStringFeatures<WORD>* l, CStringFeatures<WORD>* r, CPluginEstimate* pie)
: CStringKernel<WORD>(10),estimate(pie), mean(NULL), variance(NULL),
	sqrtdiag_lhs(NULL), sqrtdiag_rhs(NULL),
	ld_mean_lhs(NULL), ld_mean_rhs(NULL),
	num_params(0), num_symbols(0), sum_m2_s2(0), pos_prior(0.5),
	neg_prior(0.5), initialized(false)
{
	init(l, r);
}

CSalzbergWordKernel::~CSalzbergWordKernel()
{
	cleanup();
}

bool CSalzbergWordKernel::init(CFeatures* p_l, CFeatures* p_r)
{
	bool status=CStringKernel<WORD>::init(p_l,p_r);
	CStringFeatures<WORD>* l=(CStringFeatures<WORD>*) p_l;
	ASSERT(l);
	CStringFeatures<WORD>* r=(CStringFeatures<WORD>*) p_r;
	ASSERT(r);

	INT i;
	initialized=false;

	if (sqrtdiag_lhs!=sqrtdiag_rhs)
		delete[] sqrtdiag_rhs;
	sqrtdiag_rhs=NULL;
	delete[] sqrtdiag_lhs;
	sqrtdiag_lhs=NULL;
	if (ld_mean_lhs!=ld_mean_rhs)
		delete[] ld_mean_rhs;
	ld_mean_rhs=NULL;
	delete[] ld_mean_lhs;
	ld_mean_lhs=NULL;

	sqrtdiag_lhs=new DREAL[l->get_num_vectors()];
	ld_mean_lhs=new DREAL[l->get_num_vectors()];

	for (i=0; i<l->get_num_vectors(); i++)
		sqrtdiag_lhs[i]=1;

	if (l==r)
	{
		sqrtdiag_rhs=sqrtdiag_lhs;
		ld_mean_rhs=ld_mean_lhs;
	}
	else
	{
		sqrtdiag_rhs=new DREAL[r->get_num_vectors()];
		for (i=0; i<r->get_num_vectors(); i++)
			sqrtdiag_rhs[i]=1;

		ld_mean_rhs=new DREAL[r->get_num_vectors()];
	}

	DREAL* l_ld_mean_lhs=ld_mean_lhs;
	DREAL* l_ld_mean_rhs=ld_mean_rhs;

	//from our knowledge first normalize variance to 1 and then norm=1 does the job
	if (!initialized)
	{
		INT num_vectors=l->get_num_vectors();
		num_symbols=(INT) l->get_num_symbols();
		INT llen=l->get_vector_length(0);
		INT rlen=r->get_vector_length(0);
		num_params=(INT) llen*l->get_num_symbols();
		INT num_params2=(INT) llen*l->get_num_symbols()+rlen*r->get_num_symbols();
		if ((!estimate) || (!estimate->check_models()))
		{
			SG_ERROR( "no estimate available\n");
			return false ;
		} ;
		if (num_params2!=estimate->get_num_params())
		{
			SG_ERROR( "number of parameters of estimate and feature representation do not match\n");
			return false ;
		} ;

		delete[] variance;
		delete[] mean;
		mean=new DREAL[num_params];
		ASSERT(mean);
		variance=new DREAL[num_params];
		ASSERT(variance);

		for (i=0; i<num_params; i++)
		{
			mean[i]=0;
			variance[i]=0;
		}


		// compute mean
		for (i=0; i<num_vectors; i++)
		{
			INT len;
			WORD* vec=l->get_feature_vector(i, len);

			for (INT j=0; j<len; j++)
			{
				INT idx=compute_index(j, vec[j]);
				DREAL theta_p = 1/estimate->log_derivative_pos_obsolete(vec[j], j) ;
				DREAL theta_n = 1/estimate->log_derivative_neg_obsolete(vec[j], j) ;
				DREAL value   = (theta_p/(pos_prior*theta_p+neg_prior*theta_n)) ;

				mean[idx]   += value/num_vectors ;
			}
		}

		// compute variance
		for (i=0; i<num_vectors; i++)
		{
			INT len;
			WORD* vec=l->get_feature_vector(i, len);

			for (INT j=0; j<len; j++)
			{
				for (INT k=0; k<4; k++)
				{
					INT idx=compute_index(j, k);
					if (k!=vec[j])
						variance[idx]+=mean[idx]*mean[idx]/num_vectors;
					else
					{
						DREAL theta_p = 1/estimate->log_derivative_pos_obsolete(vec[j], j) ;
						DREAL theta_n = 1/estimate->log_derivative_neg_obsolete(vec[j], j) ;
						DREAL value = (theta_p/(pos_prior*theta_p+neg_prior*theta_n)) ;

						variance[idx] += CMath::sq(value-mean[idx])/num_vectors;
					}
				}
			}
		}


		// compute sum_i m_i^2/s_i^2
		sum_m2_s2=0 ;
		for (i=0; i<num_params; i++)
		{
			if (variance[i]<1e-14) // then it is likely to be numerical inaccuracy
				variance[i]=1 ;

			//fprintf(stderr, "%i: mean=%1.2e  std=%1.2e\n", i, mean[i], std[i]) ;
			sum_m2_s2 += mean[i]*mean[i]/(variance[i]) ;
		} ;
	} 

	// compute sum of 
	//result -= feature*mean[a_idx]/variance[a_idx] ;

	for (i=0; i<l->get_num_vectors(); i++)
	{
		INT alen ;
		WORD* avec=l->get_feature_vector(i, alen);
		DREAL  result=0 ;
		for (INT j=0; j<alen; j++)
		{
			INT a_idx = compute_index(j, avec[j]) ;
			DREAL theta_p = 1/estimate->log_derivative_pos_obsolete(avec[j], j) ;
			DREAL theta_n = 1/estimate->log_derivative_neg_obsolete(avec[j], j) ;
			DREAL value = (theta_p/(pos_prior*theta_p+neg_prior*theta_n)) ;

			if (variance[a_idx]!=0)
				result-=value*mean[a_idx]/variance[a_idx];
		}
		ld_mean_lhs[i]=result ;
	}

	if (ld_mean_lhs!=ld_mean_rhs)
	{
		// compute sum of 
		//result -= feature*mean[b_idx]/variance[b_idx] ;
		for (i=0; i<r->get_num_vectors(); i++)
		{
			INT alen ;
			WORD* avec=r->get_feature_vector(i, alen);
			DREAL  result=0 ;
			for (INT j=0; j<alen; j++)
			{
				INT a_idx = compute_index(j, avec[j]) ;
				DREAL theta_p = 1/estimate->log_derivative_pos_obsolete(avec[j], j) ;
				DREAL theta_n = 1/estimate->log_derivative_neg_obsolete(avec[j], j) ;
				DREAL value = (theta_p/(pos_prior*theta_p+neg_prior*theta_n)) ;

				result -= value*mean[a_idx]/variance[a_idx] ;
			}
			ld_mean_rhs[i]=result ;
		} ;
	} ;

	//warning hacky
	//
	this->lhs=l;
	this->rhs=l;
	ld_mean_lhs = l_ld_mean_lhs ;
	ld_mean_rhs = l_ld_mean_lhs ;

	//compute normalize to 1 values
	for (i=0; i<lhs->get_num_vectors(); i++)
	{
		sqrtdiag_lhs[i]=sqrt(compute(i,i));

		//trap divide by zero exception
		if (sqrtdiag_lhs[i]==0)
			sqrtdiag_lhs[i]=1e-16;
	}

	// if lhs is different from rhs (train/test data)
	// compute also the normalization for rhs
	if (sqrtdiag_lhs!=sqrtdiag_rhs)
	{
		this->lhs=r;
		this->rhs=r;
		ld_mean_lhs = l_ld_mean_rhs ;
		ld_mean_rhs = l_ld_mean_rhs ;

		//compute normalize to 1 values
		for (i=0; i<rhs->get_num_vectors(); i++)
		{
			sqrtdiag_rhs[i]=sqrt(compute(i,i));

			//trap divide by zero exception
			if (sqrtdiag_rhs[i]==0)
				sqrtdiag_rhs[i]=1e-16;
		}
	}

	this->lhs=l;
	this->rhs=r;
	ld_mean_lhs = l_ld_mean_lhs ;
	ld_mean_rhs = l_ld_mean_rhs ;

	initialized = true ;
	return status;
}

void CSalzbergWordKernel::cleanup()
{
	delete[] variance;
	variance=NULL;

	delete[] mean;
	mean=NULL;

	if (sqrtdiag_lhs != sqrtdiag_rhs)
		delete[] sqrtdiag_rhs;
	sqrtdiag_rhs=NULL;

	delete[] sqrtdiag_lhs;
	sqrtdiag_lhs=NULL;

	if (ld_mean_lhs!=ld_mean_rhs)
		delete[] ld_mean_rhs ;
	ld_mean_rhs=NULL;

	delete[] ld_mean_lhs ;
	ld_mean_lhs=NULL;

	CKernel::cleanup();
}

bool CSalzbergWordKernel::load_init(FILE* src)
{
	return false;
}

bool CSalzbergWordKernel::save_init(FILE* dest)
{
	return false;
}



DREAL CSalzbergWordKernel::compute(INT idx_a, INT idx_b)
{
	INT alen, blen;
	WORD* avec=((CStringFeatures<WORD>*) lhs)->get_feature_vector(idx_a, alen);
	WORD* bvec=((CStringFeatures<WORD>*) rhs)->get_feature_vector(idx_b, blen);
	// can only deal with strings of same length
	ASSERT(alen==blen);

	DREAL result = sum_m2_s2 ; // does not contain 0-th element

	for (INT i=0; i<alen; i++)
	{
		if (avec[i]==bvec[i])
		{
			INT a_idx = compute_index(i, avec[i]) ;

			DREAL theta_p = 1/estimate->log_derivative_pos_obsolete(avec[i], i) ;
			DREAL theta_n = 1/estimate->log_derivative_neg_obsolete(avec[i], i) ;
			DREAL value = (theta_p/(pos_prior*theta_p+neg_prior*theta_n)) ;

			result   += value*value/variance[a_idx] ;
		}
	}
	result += ld_mean_lhs[idx_a] + ld_mean_rhs[idx_b] ;


	if (initialized)
		result /=  (sqrtdiag_lhs[idx_a]*sqrtdiag_rhs[idx_b]) ;

	//fprintf(stderr, "%ld : %ld -> %f\n",idx_a, idx_b, result) ;
	return result;
}
