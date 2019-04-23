/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Bjoern Esser, Sergey Lisitsyn
 */

#include <shogun/lib/common.h>
#include <shogun/kernel/string/HistogramWordStringKernel.h>
#include <shogun/features/Features.h>
#include <shogun/features/StringFeatures.h>
#include <shogun/classifier/PluginEstimate.h>
#include <shogun/io/SGIO.h>

using namespace shogun;

HistogramWordStringKernel::HistogramWordStringKernel()
: StringKernel<uint16_t>()
{
	init();
}

HistogramWordStringKernel::HistogramWordStringKernel(int32_t size, std::shared_ptr<PluginEstimate> pie)
: StringKernel<uint16_t>(size)
{
	init();

	estimate=pie;

}

HistogramWordStringKernel::HistogramWordStringKernel(
	std::shared_ptr<StringFeatures<uint16_t>> l, std::shared_ptr<StringFeatures<uint16_t>> r, std::shared_ptr<PluginEstimate> pie)
: StringKernel<uint16_t>()
{
	init();

	estimate=pie;
	init(l, r);
}

HistogramWordStringKernel::~HistogramWordStringKernel()
{


	SG_FREE(variance);
	SG_FREE(mean);
	if (sqrtdiag_lhs != sqrtdiag_rhs)
		SG_FREE(sqrtdiag_rhs);
	SG_FREE(sqrtdiag_lhs);
	if (ld_mean_lhs!=ld_mean_rhs)
		SG_FREE(ld_mean_rhs);
	SG_FREE(ld_mean_lhs);
	if (plo_lhs!=plo_rhs)
		SG_FREE(plo_rhs);
	SG_FREE(plo_lhs);
}

bool HistogramWordStringKernel::init(std::shared_ptr<Features> p_l, std::shared_ptr<Features> p_r)
{
	StringKernel<uint16_t>::init(p_l,p_r);
	auto l=std::static_pointer_cast<StringFeatures<uint16_t>>(p_l);
	auto r=std::static_pointer_cast<StringFeatures<uint16_t>>(p_r);
	ASSERT(l)
	ASSERT(r)

	SG_DEBUG("init: lhs: %ld   rhs: %ld\n", l.get(), r.get())
	int32_t i;
	initialized=false;

	if (sqrtdiag_lhs != sqrtdiag_rhs)
		SG_FREE(sqrtdiag_rhs);
	sqrtdiag_rhs=NULL ;
	SG_FREE(sqrtdiag_lhs);
	sqrtdiag_lhs=NULL ;
	if (ld_mean_lhs!=ld_mean_rhs)
		SG_FREE(ld_mean_rhs);
	ld_mean_rhs=NULL ;
	SG_FREE(ld_mean_lhs);
	ld_mean_lhs=NULL ;
	if (plo_lhs!=plo_rhs)
		SG_FREE(plo_rhs);
	plo_rhs=NULL ;
	SG_FREE(plo_lhs);
	plo_lhs=NULL ;

	sqrtdiag_lhs= SG_MALLOC(float64_t, l->get_num_vectors());
	ld_mean_lhs = SG_MALLOC(float64_t, l->get_num_vectors());
	plo_lhs     = SG_MALLOC(float64_t, l->get_num_vectors());

	for (i=0; i<l->get_num_vectors(); i++)
		sqrtdiag_lhs[i]=1;

	if (l==r)
	{
		sqrtdiag_rhs=sqrtdiag_lhs;
		ld_mean_rhs=ld_mean_lhs;
		plo_rhs=plo_lhs;
	}
	else
	{
		sqrtdiag_rhs=SG_MALLOC(float64_t, r->get_num_vectors());
		for (i=0; i<r->get_num_vectors(); i++)
			sqrtdiag_rhs[i]=1;

		ld_mean_rhs=SG_MALLOC(float64_t, r->get_num_vectors());
		plo_rhs=SG_MALLOC(float64_t, r->get_num_vectors());
	}

	float64_t* l_plo_lhs=plo_lhs;
	float64_t* l_plo_rhs=plo_rhs;
	float64_t* l_ld_mean_lhs=ld_mean_lhs;
	float64_t* l_ld_mean_rhs=ld_mean_rhs;

	//from our knowledge first normalize variance to 1 and then norm=1 does the job
	if (!initialized)
	{
		int32_t num_vectors=l->get_num_vectors();
		num_symbols=(int32_t) l->get_num_symbols();
		int32_t llen=l->get_vector_length(0);
		int32_t rlen=r->get_vector_length(0);
		num_params=llen*((int32_t) l->get_num_symbols());
		num_params2=llen*((int32_t) l->get_num_symbols())+rlen*((int32_t) r->get_num_symbols());

		if ((!estimate) || (!estimate->check_models()))
		{
			SG_ERROR("no estimate available\n")
			return false ;
		} ;
		if (num_params2!=estimate->get_num_params())
		{
			SG_ERROR("number of parameters of estimate and feature representation do not match\n")
			return false ;
		} ;

		//add 1 as we have the 'bias' also in this vector
		num_params2++;

		SG_FREE(mean);
		mean=SG_MALLOC(float64_t, num_params2);
		SG_FREE(variance);
		variance=SG_MALLOC(float64_t, num_params2);

		for (i=0; i<num_params2; i++)
		{
			mean[i]=0;
			variance[i]=0;
		}

		// compute mean
		for (i=0; i<num_vectors; i++)
		{
			int32_t len;
			bool free_vec;
			uint16_t* vec=l->get_feature_vector(i, len, free_vec);

			mean[0]+=estimate->posterior_log_odds_obsolete(vec, len)/num_vectors;

			for (int32_t j=0; j<len; j++)
			{
				int32_t idx=compute_index(j, vec[j]);
				mean[idx]             += estimate->log_derivative_pos_obsolete(vec[j], j)/num_vectors;
				mean[idx+num_params] += estimate->log_derivative_neg_obsolete(vec[j], j)/num_vectors;
			}

			l->free_feature_vector(vec, i, free_vec);
		}

		// compute variance
		for (i=0; i<num_vectors; i++)
		{
			int32_t len;
			bool free_vec;
			uint16_t* vec=l->get_feature_vector(i, len, free_vec);

			variance[0] += Math::sq(estimate->posterior_log_odds_obsolete(vec, len)-mean[0])/num_vectors;

			for (int32_t j=0; j<len; j++)
			{
				for (int32_t k=0; k<4; k++)
				{
					int32_t idx=compute_index(j, k);
					if (k!=vec[j])
					{
						variance[idx]+=mean[idx]*mean[idx]/num_vectors;
						variance[idx+num_params]+=mean[idx+num_params]*mean[idx+num_params]/num_vectors;
					}
					else
					{
						variance[idx] += Math::sq(estimate->log_derivative_pos_obsolete(vec[j], j)
								-mean[idx])/num_vectors;
						variance[idx+num_params] += Math::sq(estimate->log_derivative_neg_obsolete(vec[j], j)
								-mean[idx+num_params])/num_vectors;
					}
				}
			}

			l->free_feature_vector(vec, i, free_vec);
		}


		// compute sum_i m_i^2/s_i^2
		sum_m2_s2=0 ;
		for (i=1; i<num_params2; i++)
		{
			if (variance[i]<1e-14) // then it is likely to be numerical inaccuracy
				variance[i]=1 ;

			//fprintf(stderr, "%i: mean=%1.2e  std=%1.2e\n", i, mean[i], std[i]) ;
			sum_m2_s2 += mean[i]*mean[i]/(variance[i]) ;
		} ;
	}

	// compute sum of
	//result -= estimate->log_derivative_pos(avec[i], i)*mean[a_idx]/variance[a_idx] ;
	//result -= estimate->log_derivative_neg(avec[i], i)*mean[a_idx+num_params]/variance[a_idx+num_params] ;
	for (i=0; i<l->get_num_vectors(); i++)
	{
		int32_t alen;
		bool free_avec;
		uint16_t* avec = l->get_feature_vector(i, alen, free_avec);

		float64_t  result=0 ;
		for (int32_t j=0; j<alen; j++)
		{
			int32_t a_idx = compute_index(j, avec[j]);
			result -= estimate->log_derivative_pos_obsolete(avec[j], j)*mean[a_idx]/variance[a_idx] ;
			result -= estimate->log_derivative_neg_obsolete(avec[j], j)*mean[a_idx+num_params]/variance[a_idx+num_params] ;
		}
		ld_mean_lhs[i]=result ;

		// precompute posterior-log-odds
		plo_lhs[i] = estimate->posterior_log_odds_obsolete(avec, alen)-mean[0] ;
		l->free_feature_vector(avec, i, free_avec);
	} ;

	if (ld_mean_lhs!=ld_mean_rhs)
	{
		// compute sum of
		//result -= estimate->log_derivative_pos(bvec[i], i)*mean[b_idx]/variance[b_idx] ;
		//result -= estimate->log_derivative_neg(bvec[i], i)*mean[b_idx+num_params]/variance[b_idx+num_params] ;
		for (i=0; i < r->get_num_vectors(); i++)
		{
			int32_t alen;
			bool free_avec;
			uint16_t* avec=r->get_feature_vector(i, alen, free_avec);

			float64_t  result=0 ;
			for (int32_t j=0; j<alen; j++)
			{
				int32_t a_idx = compute_index(j, avec[j]) ;
				result -= estimate->log_derivative_pos_obsolete(avec[j], j)*mean[a_idx]/variance[a_idx] ;
				result -= estimate->log_derivative_neg_obsolete(avec[j], j)*mean[a_idx+num_params]/variance[a_idx+num_params] ;
			}
			ld_mean_rhs[i]=result ;

			// precompute posterior-log-odds
			plo_rhs[i] = estimate->posterior_log_odds_obsolete(avec, alen)-mean[0] ;
			r->free_feature_vector(avec, i, free_avec);
		} ;
	} ;

	//warning hacky
	//
	this->lhs=l;
	this->rhs=l;
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
	return init_normalizer();
}

void HistogramWordStringKernel::cleanup()
{
	SG_FREE(variance);
	variance=NULL;

	SG_FREE(mean);
	mean=NULL;

	if (sqrtdiag_lhs != sqrtdiag_rhs)
		SG_FREE(sqrtdiag_rhs);
	sqrtdiag_rhs=NULL;

	SG_FREE(sqrtdiag_lhs);
	sqrtdiag_lhs=NULL;

	if (ld_mean_lhs!=ld_mean_rhs)
		SG_FREE(ld_mean_rhs);
	ld_mean_rhs=NULL;

	SG_FREE(ld_mean_lhs);
	ld_mean_lhs=NULL;

	if (plo_lhs!=plo_rhs)
		SG_FREE(plo_rhs);
	plo_rhs=NULL;

	SG_FREE(plo_lhs);
	plo_lhs=NULL;

	num_params2=0;
	num_params=0;
	num_symbols=0;
	sum_m2_s2=0;
	initialized = false;

	Kernel::cleanup();
}

float64_t HistogramWordStringKernel::compute(int32_t idx_a, int32_t idx_b)
{
	int32_t alen, blen;
	bool free_avec, free_bvec;
	uint16_t* avec=std::static_pointer_cast<StringFeatures<uint16_t>>(lhs)->get_feature_vector(idx_a, alen, free_avec);
	uint16_t* bvec=std::static_pointer_cast<StringFeatures<uint16_t>>(rhs)->get_feature_vector(idx_b, blen, free_bvec);
	// can only deal with strings of same length
	ASSERT(alen==blen)

	float64_t result = plo_lhs[idx_a]*plo_rhs[idx_b]/variance[0];
	result+= sum_m2_s2 ; // does not contain 0-th element

	for (int32_t i=0; i<alen; i++)
	{
		if (avec[i]==bvec[i])
		{
			int32_t a_idx = compute_index(i, avec[i]) ;
			float64_t dd = estimate->log_derivative_pos_obsolete(avec[i], i) ;
			result   += dd*dd/variance[a_idx] ;
			dd        = estimate->log_derivative_neg_obsolete(avec[i], i) ;
			result   += dd*dd/variance[a_idx+num_params] ;
		} ;
	}
	result += ld_mean_lhs[idx_a] + ld_mean_rhs[idx_b] ;

	if (initialized)
		result /=  (sqrtdiag_lhs[idx_a]*sqrtdiag_rhs[idx_b]) ;

#ifdef DEBUG_HWSK_COMPUTATION
	float64_t result2 = compute_slow(idx_a, idx_b) ;
	if (fabs(result - result2)>1e-10)
		SG_ERROR("new=%e  old = %e  diff = %e\n", result, result2, result - result2)
#endif
	std::static_pointer_cast<StringFeatures<uint16_t>>(lhs)->free_feature_vector(avec, idx_a, free_avec);
	std::static_pointer_cast<StringFeatures<uint16_t>>(rhs)->free_feature_vector(bvec, idx_b, free_bvec);
	return result;
}

void HistogramWordStringKernel::init()
{
	estimate=NULL;
	mean=NULL;
	variance=NULL;

	sqrtdiag_lhs=NULL;
	sqrtdiag_rhs=NULL;

	ld_mean_lhs=NULL;
	ld_mean_rhs=NULL;

	plo_lhs=NULL;
	plo_rhs=NULL;
	num_params=0;
	num_params2=0;

	num_symbols=0;
	sum_m2_s2=0;
	initialized=false;

	SG_ADD(&initialized, "initialized", "If kernel is initalized.");

	/*m_parameters->add_vector(&plo_lhs, &num_lhs, "plo_lhs");*/
	watch_param("plo_lhs", &plo_lhs, &num_lhs);

	/*m_parameters->add_vector(&plo_rhs, &num_rhs, "plo_rhs");*/
	watch_param("plo_rhs", &plo_rhs, &num_rhs);

	/*m_parameters->add_vector(&ld_mean_lhs, &num_lhs, "ld_mean_lhs");*/
	watch_param("ld_mean_lhs", &ld_mean_lhs, &num_lhs);

	/*m_parameters->add_vector(&ld_mean_rhs, &num_rhs, "ld_mean_rhs");*/
	watch_param("ld_mean_rhs", &ld_mean_rhs, &num_rhs);

	/*m_parameters->add_vector(&sqrtdiag_lhs, &num_lhs, "sqrtdiag_lhs");*/
	watch_param("sqrtdiag_lhs", &sqrtdiag_lhs, &num_lhs);

	/*m_parameters->add_vector(&sqrtdiag_rhs, &num_rhs, "sqrtdiag_rhs");*/
	watch_param("sqrtdiag_rhs", &sqrtdiag_rhs, &num_rhs);

	/*m_parameters->add_vector(&mean, &num_params2, "mean");*/
	watch_param("mean", &mean, &num_params2);

	/*m_parameters->add_vector(&variance, &num_params2, "variance");*/
	watch_param("variance", &variance, &num_params2);

	SG_ADD((std::shared_ptr<SGObject>*) &estimate, "estimate", "Plugin Estimate.");
}

#ifdef DEBUG_HWSK_COMPUTATION
float64_t CHistogramWordStringKernel::compute_slow(int32_t idx_a, int32_t idx_b)
{
	int32_t alen, blen;
	bool free_avec, free_bvec;
	uint16_t* avec=std::static_pointer_cast<CStringFeatures<uint16_t>>(lhs)->get_feature_vector(idx_a, alen, free_avec);
	uint16_t* bvec=std::static_pointer_cast<CStringFeatures<uint16_t>>(rhs)->get_feature_vector(idx_b, blen, free_bvec);
	// can only deal with strings of same length
	ASSERT(alen==blen)

	float64_t result=(estimate->posterior_log_odds_obsolete(avec, alen)-mean[0])*
		(estimate->posterior_log_odds_obsolete(bvec, blen)-mean[0])/(variance[0]);
	result+= sum_m2_s2 ; // does not contain 0-th element

	for (int32_t i=0; i<alen; i++)
	{
		int32_t a_idx = compute_index(i, avec[i]) ;
		int32_t b_idx = compute_index(i, bvec[i]) ;

		if (avec[i]==bvec[i])
		{
			float64_t dd = estimate->log_derivative_pos_obsolete(avec[i], i) ;
			result   += dd*dd/variance[a_idx] ;
			dd        = estimate->log_derivative_neg_obsolete(avec[i], i) ;
			result   += dd*dd/variance[a_idx+num_params] ;
		} ;

		result -= estimate->log_derivative_pos_obsolete(avec[i], i)*mean[a_idx]/variance[a_idx] ;
		result -= estimate->log_derivative_pos_obsolete(bvec[i], i)*mean[b_idx]/variance[b_idx] ;
		result -= estimate->log_derivative_neg_obsolete(avec[i], i)*mean[a_idx+num_params]/variance[a_idx+num_params] ;
		result -= estimate->log_derivative_neg_obsolete(bvec[i], i)*mean[b_idx+num_params]/variance[b_idx+num_params] ;
	}

	if (initialized)
		result /=  (sqrtdiag_lhs[idx_a]*sqrtdiag_rhs[idx_b]) ;

	std::static_pointer_cast<CStringFeatures<uint16_t>>(lhs)->free_feature_vector(avec, idx_a, free_avec);
	std::static_pointer_cast<CStringFeatures<uint16_t>>(rhs)->free_feature_vector(bvec, idx_b, free_bvec);
	return result;
}

#endif
