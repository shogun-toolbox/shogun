#include "lib/common.h"
#include "kernel/SalzbergWordKernel.h"
#include "features/Features.h"
#include "features/WordFeatures.h"
#include "classifier/PluginEstimate.h"
#include "lib/io.h"
#include "features/Labels.h"

#include <assert.h>

CSalzbergWordKernel::CSalzbergWordKernel(LONG size, CPluginEstimate* pie)
  : CWordKernel(size),estimate(pie), mean(NULL), variance(NULL), 
    sqrtdiag_lhs(NULL), sqrtdiag_rhs(NULL), 
    ld_mean_lhs(NULL), ld_mean_rhs(NULL),
    num_params(0), num_symbols(0), sum_m2_s2(0), pos_prior(0.5),
	neg_prior(0.5)
{
}

CSalzbergWordKernel::~CSalzbergWordKernel() 
{
  delete[] variance;
  delete[] mean;
  if (sqrtdiag_lhs != sqrtdiag_rhs)
    delete[] sqrtdiag_rhs;
  delete[] sqrtdiag_lhs;
  if (ld_mean_lhs!=ld_mean_rhs)
    delete[] ld_mean_rhs ;
  delete[] ld_mean_lhs ;
}

bool CSalzbergWordKernel::init(CFeatures* l, CFeatures* r, bool do_init)
{
	bool result=CWordKernel::init(l,r,do_init);
	initialized = false ;
	assert(l!=NULL) ;
	assert(r!=NULL) ;
	
	//  fprintf(stderr, "start\n") ;

	CWordFeatures* lhs=(CWordFeatures*) l;
	CWordFeatures* rhs=(CWordFeatures*) r;
	assert(lhs) ;
	assert(rhs) ;
	
	CIO::message(M_INFO, "init: lhs: %ld   rhs: %ld\n", lhs, rhs) ;
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
	
	sqrtdiag_lhs= new REAL[lhs->get_num_vectors()];
	ld_mean_lhs = new REAL[lhs->get_num_vectors()];
	
	for (i=0; i<lhs->get_num_vectors(); i++)
		sqrtdiag_lhs[i]=1;
	
	if (l==r)
	{
		sqrtdiag_rhs = sqrtdiag_lhs;
		ld_mean_rhs  = ld_mean_lhs ;
	}
	else
	{
		sqrtdiag_rhs= new REAL[rhs->get_num_vectors()];
		for (i=0; i<rhs->get_num_vectors(); i++)
			sqrtdiag_rhs[i]=1;
		
		ld_mean_rhs = new REAL[rhs->get_num_vectors()];
	}
	
	REAL *l_ld_mean_lhs = ld_mean_lhs ;
	REAL *l_ld_mean_rhs = ld_mean_rhs ;
	
	assert(sqrtdiag_lhs);
	assert(sqrtdiag_rhs);
	
	//from our knowledge first normalize variance to 1 and then norm=1 does the job
	if (do_init)
	{
	    INT num_vectors=lhs->get_num_vectors();
	    num_symbols=lhs->get_num_symbols();
	    num_params = lhs->get_num_features() * lhs->get_num_symbols() ;
	    int num_params2=lhs->get_num_features() * lhs->get_num_symbols() +
			rhs->get_num_features() * rhs->get_num_symbols();
	    if ((!estimate) || (!estimate->check_models()))
		{
			CIO::message(M_ERROR, "no estimate available\n") ;
			return false ;
		} ;
	    if (num_params2!=estimate->get_num_params())
		{
			CIO::message(M_ERROR, "number of parameters of estimate and feature representation do not match\n") ;
			return false ;
		} ;
	    
	    delete[] variance;
	    variance=NULL ;
	    delete[] mean;
	    mean=NULL ;
	    mean= new REAL[num_params];
	    variance= new REAL[num_params];
	    
	    assert(mean);
	    assert(variance);
	    
	    
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
			
			WORD* vec=lhs->get_feature_vector(i, len, freevec);
			
			assert(len==lhs->get_num_features());
			
			for (INT j=0; j<len; j++)
			{
				INT idx=compute_index(j, vec[j]);
				REAL theta_p = 1/estimate->log_derivative_pos_obsolete(vec[j], j) ;
				REAL theta_n = 1/estimate->log_derivative_neg_obsolete(vec[j], j) ;
				REAL value   = (theta_p/(pos_prior*theta_p+neg_prior*theta_n)) ;
				
				mean[idx]   += value/num_vectors ;
			}
			
			((CWordFeatures*) lhs)->free_feature_vector(vec, i, freevec);
		}
	    
	    // compute variance
	    for (i=0; i<num_vectors; i++)
		{
			INT len;
			bool freevec;
			
			WORD* vec=lhs->get_feature_vector(i, len, freevec);
			
			assert(len==lhs->get_num_features());
			
			for (INT j=0; j<len; j++)
			{
				for (INT k=0; k<4; k++)
				{
					INT idx=compute_index(j, k);
					if (k!=vec[j])
						variance[idx]+=mean[idx]*mean[idx]/num_vectors;
					else
					{
						REAL theta_p = 1/estimate->log_derivative_pos_obsolete(vec[j], j) ;
						REAL theta_n = 1/estimate->log_derivative_neg_obsolete(vec[j], j) ;
						REAL value = (theta_p/(pos_prior*theta_p+neg_prior*theta_n)) ;
						
						variance[idx] += math.sq(value-mean[idx])/num_vectors;
					}
				}
				
				((CWordFeatures*) lhs)->free_feature_vector(vec, i, freevec);
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

	for (i=0; i<lhs->get_num_vectors(); i++)
	{
	    INT alen ;
	    bool afree ;
	    WORD* avec = ((CWordFeatures*) lhs) -> get_feature_vector(i, alen, afree);
	    REAL  result=0 ;
	    for (INT j=0; j<alen; j++)
		{
			INT a_idx = compute_index(j, avec[j]) ;
			
			REAL theta_p = 1/estimate->log_derivative_pos_obsolete(avec[j], j) ;
			REAL theta_n = 1/estimate->log_derivative_neg_obsolete(avec[j], j) ;
			REAL value = (theta_p/(pos_prior*theta_p+neg_prior*theta_n)) ;
			
			result -= value*mean[a_idx]/variance[a_idx] ;
		}
	    ld_mean_lhs[i]=result ;
		
	    ((CWordFeatures*) lhs)->free_feature_vector(avec, i, afree);
	} ;
	
	if (ld_mean_lhs!=ld_mean_rhs)
	  {
	    // compute sum of 
	    //result -= feature*mean[b_idx]/variance[b_idx] ;
	    for (i=0; i<rhs->get_num_vectors(); i++)
	      {
			  INT alen ;
			  bool afree ;
			  WORD* avec = ((CWordFeatures*) rhs) -> get_feature_vector(i, alen, afree);
			  REAL  result=0 ;
			  for (INT j=0; j<alen; j++)
			  {
				  INT a_idx = compute_index(j, avec[j]) ;

				  REAL theta_p = 1/estimate->log_derivative_pos_obsolete(avec[j], j) ;
				  REAL theta_n = 1/estimate->log_derivative_neg_obsolete(avec[j], j) ;
				  REAL value = (theta_p/(pos_prior*theta_p+neg_prior*theta_n)) ;
				  
				  result -= value*mean[a_idx]/variance[a_idx] ;
			  }
			  ld_mean_rhs[i]=result ;
			  
			  // precompute posterior-log-odds
			  ((CWordFeatures*) rhs)->free_feature_vector(avec, i, afree);
	      } ;
	  } ;
	
	//warning hacky
	//
	this->lhs=lhs;
	this->rhs=lhs;
	ld_mean_lhs = l_ld_mean_lhs ;
	ld_mean_rhs = l_ld_mean_lhs ;
	
	//compute normalize to 1 values
	for (i=0; i<lhs->get_num_vectors(); i++)
		sqrtdiag_lhs[i]=sqrt(compute(i,i));

	// if lhs is different from rhs (train/test data)
	// compute also the normalization for rhs
	if (sqrtdiag_lhs!=sqrtdiag_rhs)
	{
		this->lhs=rhs;
		this->rhs=rhs;
		ld_mean_lhs = l_ld_mean_rhs ;
		ld_mean_rhs = l_ld_mean_rhs ;

		//compute normalize to 1 values
		for (i=0; i<rhs->get_num_vectors(); i++)
		  sqrtdiag_rhs[i]=sqrt(compute(i,i));
	}

	this->lhs=lhs;
	this->rhs=rhs;
	ld_mean_lhs = l_ld_mean_lhs ;
	ld_mean_rhs = l_ld_mean_rhs ;

	initialized = true ;
	return result;
}
  
void CSalzbergWordKernel::cleanup()
{
}

bool CSalzbergWordKernel::load_init(FILE* src)
{
	return false;
}

bool CSalzbergWordKernel::save_init(FILE* dest)
{
	return false;
}
  


REAL CSalzbergWordKernel::compute(INT idx_a, INT idx_b)
{
  INT alen, blen;
  bool afree, bfree;

  WORD* avec=((CWordFeatures*) lhs)->get_feature_vector(idx_a, alen, afree);
  WORD* bvec=((CWordFeatures*) rhs)->get_feature_vector(idx_b, blen, bfree);

  // can only deal with strings of same length
  assert(alen==blen);

  double result = sum_m2_s2 ; // does not contain 0-th element

  for (INT i=0; i<alen; i++)
  {
    if (avec[i]==bvec[i])
      {
		  INT a_idx = compute_index(i, avec[i]) ;

		  REAL theta_p = 1/estimate->log_derivative_pos_obsolete(avec[i], i) ;
		  REAL theta_n = 1/estimate->log_derivative_neg_obsolete(avec[i], i) ;
		  REAL value = (theta_p/(pos_prior*theta_p+neg_prior*theta_n)) ;

		  result   += value*value/variance[a_idx] ;
      } ;
  }
  result += ld_mean_lhs[idx_a] + ld_mean_rhs[idx_b] ;
  
  ((CWordFeatures*) lhs)->free_feature_vector(avec, idx_a, afree);
  ((CWordFeatures*) rhs)->free_feature_vector(bvec, idx_b, bfree);
  
  if (initialized)
	  result /=  (sqrtdiag_lhs[idx_a]*sqrtdiag_rhs[idx_b]) ;
  
  //fprintf(stderr, "%ld : %ld -> %f\n",idx_a, idx_b, result) ;
  return result;
}
