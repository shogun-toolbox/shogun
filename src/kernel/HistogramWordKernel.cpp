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

#include "lib/common.h"
#include "kernel/HistogramWordKernel.h"
#include "features/Features.h"
#include "features/WordFeatures.h"
#include "classifier/PluginEstimate.h"
#include "lib/io.h"

CHistogramWordKernel::CHistogramWordKernel(LONG size, CPluginEstimate* pie)
  : CSimpleKernel<WORD>(size),estimate(pie), mean(NULL), variance(NULL), 
    sqrtdiag_lhs(NULL), sqrtdiag_rhs(NULL), 
    ld_mean_lhs(NULL), ld_mean_rhs(NULL),
    plo_lhs(NULL), plo_rhs(NULL),
    num_params(0), num_params1(0), num_symbols(0), sum_m2_s2(0), initialized(false)
{
}

CHistogramWordKernel::~CHistogramWordKernel() 
{
  delete[] variance;
  delete[] mean;
  if (sqrtdiag_lhs != sqrtdiag_rhs)
    delete[] sqrtdiag_rhs;
  delete[] sqrtdiag_lhs;
  if (ld_mean_lhs!=ld_mean_rhs)
    delete[] ld_mean_rhs ;
  delete[] ld_mean_lhs ;
  if (plo_lhs!=plo_rhs)
    delete[] plo_rhs ;
  delete[] plo_lhs ;
}

bool CHistogramWordKernel::init(CFeatures* p_l, CFeatures* p_r)
{
	bool status=CSimpleKernel<WORD>::init(p_l,p_r);
	CWordFeatures* l=(CWordFeatures*) p_l;
	CWordFeatures* r=(CWordFeatures*) p_r;
	ASSERT(l) ;
	ASSERT(r) ;
	
	SG_DEBUG( "init: lhs: %ld   rhs: %ld\n", l, r) ;
	INT i;

	if (sqrtdiag_lhs != sqrtdiag_rhs)
	  delete[] sqrtdiag_rhs;
	sqrtdiag_rhs=NULL ;
	delete[] sqrtdiag_lhs;
	sqrtdiag_lhs=NULL ;
	if (ld_mean_lhs!=ld_mean_rhs)
		delete[] ld_mean_rhs ;
	ld_mean_rhs=NULL ;
	delete[] ld_mean_lhs ;
	ld_mean_lhs=NULL ;
	if (plo_lhs!=plo_rhs)
	  delete[] plo_rhs ;
	plo_rhs=NULL ;
	delete[] plo_lhs ;
	plo_lhs=NULL ;
	
	sqrtdiag_lhs= new DREAL[l->get_num_vectors()];
	ld_mean_lhs = new DREAL[l->get_num_vectors()];
	plo_lhs     = new DREAL[l->get_num_vectors()];
	
	for (i=0; i<l->get_num_vectors(); i++)
		sqrtdiag_lhs[i]=1;
	
	if (l==r)
	{
		sqrtdiag_rhs = sqrtdiag_lhs;
		ld_mean_rhs  = ld_mean_lhs ;
		plo_rhs      = plo_lhs ;
	}
	else
	{
		sqrtdiag_rhs= new DREAL[r->get_num_vectors()];
		for (i=0; i<r->get_num_vectors(); i++)
			sqrtdiag_rhs[i]=1;
		
		ld_mean_rhs = new DREAL[r->get_num_vectors()];
		plo_rhs = new DREAL[r->get_num_vectors()];
	}
	
	DREAL *l_plo_lhs = plo_lhs ;
	DREAL *l_plo_rhs = plo_rhs ;
	DREAL *l_ld_mean_lhs = ld_mean_lhs ;
	DREAL *l_ld_mean_rhs = ld_mean_rhs ;
	
	ASSERT(sqrtdiag_lhs);
	ASSERT(sqrtdiag_rhs);
	
	//from our knowledge first normalize variance to 1 and then norm=1 does the job
	if (!initialized)
	{
	    INT num_vectors=l->get_num_vectors();
	    num_symbols=l->get_num_symbols();
	    num_params1 = l->get_num_features() * l->get_num_symbols() ;
	    num_params=l->get_num_features() * l->get_num_symbols() +
			r->get_num_features() * r->get_num_symbols();
	    if ((!estimate) || (!estimate->check_models()))
		{
         SG_ERROR( "no estimate available\n");
			return false ;
		} ;
	    if (num_params!=estimate->get_num_params())
		{
         SG_ERROR( "number of parameters of estimate and feature representation do not match\n");
			return false ;
		} ;
	    
	    //add 1 as we have the 'bias' also in this vector
	    num_params++;
	    
	    delete[] variance;
	    variance=NULL ;
	    delete[] mean;
	    mean=NULL ;
	    mean= new DREAL[num_params];
	    variance= new DREAL[num_params];
	    
	    ASSERT(mean);
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
			bool freevec;
			
			WORD* vec=l->get_feature_vector(i, len, freevec);
			
			mean[0]+=estimate->posterior_log_odds_obsolete(vec, len)/num_vectors;
			
			ASSERT(len==l->get_num_features());
			
			for (INT j=0; j<len; j++)
			{
				INT idx=compute_index(j, vec[j]);
				mean[idx]             += estimate->log_derivative_pos_obsolete(vec[j], j)/num_vectors;
				mean[idx+num_params1] += estimate->log_derivative_neg_obsolete(vec[j], j)/num_vectors;
			}
			
			l->free_feature_vector(vec, i, freevec);
		}
	    
	    // compute variance
	    for (i=0; i<num_vectors; i++)
		{
			INT len;
			bool freevec;
			
			WORD* vec=l->get_feature_vector(i, len, freevec);
			
			variance[0] += CMath::sq(estimate->posterior_log_odds_obsolete(vec, len)-mean[0])/num_vectors;
			
			ASSERT(len==l->get_num_features());
			
			for (INT j=0; j<len; j++)
			{
				for (INT k=0; k<4; k++)
				{
					INT idx=compute_index(j, k);
					if (k!=vec[j])
					{
						variance[idx]+=mean[idx]*mean[idx]/num_vectors;
						variance[idx+num_params1]+=mean[idx+num_params1]*mean[idx+num_params1]/num_vectors;
					}
					else
					{
						variance[idx]             += CMath::sq(estimate->log_derivative_pos_obsolete(vec[j], j)
															 -mean[idx])/num_vectors;
						variance[idx+num_params1] += CMath::sq(estimate->log_derivative_neg_obsolete(vec[j], j)
															 -mean[idx+num_params1])/num_vectors;
					}
				}
				
				l->free_feature_vector(vec, i, freevec);
			}
		}
		
		
		// compute sum_i m_i^2/s_i^2
		sum_m2_s2=0 ;
	    for (i=1; i<num_params; i++)
		{
			if (variance[i]<1e-14) // then it is likely to be numerical inaccuracy
				variance[i]=1 ;
			
			//fprintf(stderr, "%i: mean=%1.2e  std=%1.2e\n", i, mean[i], std[i]) ;
			sum_m2_s2 += mean[i]*mean[i]/(variance[i]) ;
		} ;
	} 
	
	// compute sum of 
	//result -= estimate->log_derivative_pos(avec[i], i)*mean[a_idx]/variance[a_idx] ;
	//result -= estimate->log_derivative_neg(avec[i], i)*mean[a_idx+num_params1]/variance[a_idx+num_params1] ;
	for (i=0; i<l->get_num_vectors(); i++)
	{
	    INT alen ;
	    bool afree ;
	    WORD* avec = l->get_feature_vector(i, alen, afree);
	    DREAL  result=0 ;
	    for (INT j=0; j<alen; j++)
	      {
		INT a_idx = compute_index(j, avec[j]) ;
		result -= estimate->log_derivative_pos_obsolete(avec[j], j)*mean[a_idx]/variance[a_idx] ;
		result -= estimate->log_derivative_neg_obsolete(avec[j], j)*mean[a_idx+num_params1]/variance[a_idx+num_params1] ;
	      }
	    ld_mean_lhs[i]=result ;
	    
	    // precompute posterior-log-odds
	    plo_lhs[i] = estimate->posterior_log_odds_obsolete(avec, alen)-mean[0] ;
	    
	    l->free_feature_vector(avec, i, afree);
	  } ;
	
	if (ld_mean_lhs!=ld_mean_rhs)
	  {
	    // compute sum of 
	    //result -= estimate->log_derivative_pos(bvec[i], i)*mean[b_idx]/variance[b_idx] ;
	    //result -= estimate->log_derivative_neg(bvec[i], i)*mean[b_idx+num_params1]/variance[b_idx+num_params1] ;	
	    for (i=0; i < r->get_num_vectors(); i++)
	      {
		INT alen ;
		bool afree ;
		WORD* avec = r -> get_feature_vector(i, alen, afree);
		DREAL  result=0 ;
		for (INT j=0; j<alen; j++)
		  {
		    INT a_idx = compute_index(j, avec[j]) ;
		    result -= estimate->log_derivative_pos_obsolete(avec[j], j)*mean[a_idx]/variance[a_idx] ;
		    result -= estimate->log_derivative_neg_obsolete(avec[j], j)*mean[a_idx+num_params1]/variance[a_idx+num_params1] ;
		  }
		ld_mean_rhs[i]=result ;
		
		// precompute posterior-log-odds
		plo_rhs[i] = estimate->posterior_log_odds_obsolete(avec, alen)-mean[0] ;
		
		r->free_feature_vector(avec, i, afree);
	      } ;
	  } ;
	
	//warning hacky
	//
	this->lhs=l;
	this->rhs=r;
	plo_lhs = l_plo_lhs ;
	plo_rhs = l_plo_lhs ;
	ld_mean_lhs = l_ld_mean_lhs ;
	ld_mean_rhs = l_ld_mean_lhs ;
	
	//compute normalize to 1 values
	for (i=0; i<l->get_num_vectors(); i++)
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
		plo_lhs = l_plo_rhs ;
		plo_rhs = l_plo_rhs ;
		ld_mean_lhs = l_ld_mean_rhs ;
		ld_mean_rhs = l_ld_mean_rhs ;

		//compute normalize to 1 values
		for (i=0; i<r->get_num_vectors(); i++)
		{
		  sqrtdiag_rhs[i]=sqrt(compute(i,i));

			//trap divide by zero exception
			if (sqrtdiag_rhs[i]==0)
				sqrtdiag_rhs[i]=1e-16;
		}
	}

	this->lhs=l;
	this->rhs=r;
	plo_lhs = l_plo_lhs ;
	plo_rhs = l_plo_rhs ;
	ld_mean_lhs = l_ld_mean_lhs ;
	ld_mean_rhs = l_ld_mean_rhs ;

	initialized = true ;
	return status;
}
  
void CHistogramWordKernel::cleanup()
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

	if (plo_lhs!=plo_rhs)
		delete[] plo_rhs ;
	plo_rhs=NULL;

	delete[] plo_lhs ;
	plo_lhs=NULL;

	num_params=0;
	num_params1=0;
	num_symbols=0;
	sum_m2_s2=0;
	initialized = false;
}

bool CHistogramWordKernel::load_init(FILE* src)
{
	return false;
}

bool CHistogramWordKernel::save_init(FILE* dest)
{
	return false;
}
  


DREAL CHistogramWordKernel::compute(INT idx_a, INT idx_b)
{
  INT alen, blen;
  bool afree, bfree;

  WORD* avec=((CWordFeatures*) lhs)->get_feature_vector(idx_a, alen, afree);
  WORD* bvec=((CWordFeatures*) rhs)->get_feature_vector(idx_b, blen, bfree);

  // can only deal with strings of same length
  ASSERT(alen==blen);

  double result = plo_lhs[idx_a]*plo_rhs[idx_b]/variance[0];
  result+= sum_m2_s2 ; // does not contain 0-th element

  for (INT i=0; i<alen; i++)
  {
    if (avec[i]==bvec[i])
      {
	INT a_idx = compute_index(i, avec[i]) ;
	double dd = estimate->log_derivative_pos_obsolete(avec[i], i) ;
	result   += dd*dd/variance[a_idx] ;
	dd        = estimate->log_derivative_neg_obsolete(avec[i], i) ;
	result   += dd*dd/variance[a_idx+num_params1] ;
      } ;
  }
  result += ld_mean_lhs[idx_a] + ld_mean_rhs[idx_b] ;

  ((CWordFeatures*) lhs)->free_feature_vector(avec, idx_a, afree);
  ((CWordFeatures*) rhs)->free_feature_vector(bvec, idx_b, bfree);

  if (initialized)
    result /=  (sqrtdiag_lhs[idx_a]*sqrtdiag_rhs[idx_b]) ;

  //fprintf(stderr, "%ld : %ld -> %f\n",idx_a, idx_b, result) ;
#ifdef BLABLA
  DREAL result2 = compute_slow(idx_a, idx_b) ;
  if (fabs(result - result2)>1e-10)
    {
      fprintf(stderr, "new=%e  old = %e  diff = %e\n", result, result2, result - result2) ;
      ASSERT(0) ;
    } ;
#endif
  return result;
}

#ifdef BLABLA

DREAL CHistogramWordKernel::compute_slow(INT idx_a, INT idx_b)
{
  INT alen, blen;
  bool afree, bfree;

  //  fprintf(stderr, "start\n") ;

  WORD* avec=((CWordFeatures*) lhs)->get_feature_vector(idx_a, alen, afree);
  WORD* bvec=((CWordFeatures*) rhs)->get_feature_vector(idx_b, blen, bfree);

  // can only deal with strings of same length
  // ASSERT(alen==blen);

  double result=(estimate->posterior_log_odds_obsolete(avec, alen)-mean[0])*
    (estimate->posterior_log_odds_obsolete(bvec, blen)-mean[0])/(variance[0]);
  result+= sum_m2_s2 ; // does not contain 0-th element

  for (INT i=0; i<alen; i++)
  {
    INT a_idx = compute_index(i, avec[i]) ;
    INT b_idx = compute_index(i, bvec[i]) ;

    if (avec[i]==bvec[i])
      {
	double dd = estimate->log_derivative_pos_obsolete(avec[i], i) ;
	result   += dd*dd/variance[a_idx] ;
	dd        = estimate->log_derivative_neg_obsolete(avec[i], i) ;
	result   += dd*dd/variance[a_idx+num_params1] ;
      } ;
    
    result -= estimate->log_derivative_pos_obsolete(avec[i], i)*mean[a_idx]/variance[a_idx] ;
    result -= estimate->log_derivative_pos_obsolete(bvec[i], i)*mean[b_idx]/variance[b_idx] ;
    result -= estimate->log_derivative_neg_obsolete(avec[i], i)*mean[a_idx+num_params1]/variance[a_idx+num_params1] ;
    result -= estimate->log_derivative_neg_obsolete(bvec[i], i)*mean[b_idx+num_params1]/variance[b_idx+num_params1] ;
  }

  ((CWordFeatures*) lhs)->free_feature_vector(avec, idx_a, afree);
  ((CWordFeatures*) rhs)->free_feature_vector(bvec, idx_b, bfree);

  if (initialized)
    result /=  (sqrtdiag_lhs[idx_a]*sqrtdiag_rhs[idx_b]) ;

  return result;
}

#endif
