/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Sergey Lisitsyn
 */

#include <shogun/lib/common.h>
#include <shogun/io/SGIO.h>
#include <shogun/kernel/string/SalzbergWordStringKernel.h>
#include <shogun/features/Features.h>
#include <shogun/features/StringFeatures.h>
#include <shogun/labels/Labels.h>
#include <shogun/labels/BinaryLabels.h>
#include <shogun/classifier/PluginEstimate.h>

using namespace shogun;

SalzbergWordStringKernel::SalzbergWordStringKernel()
: StringKernel<uint16_t>(0)
{
	init();
}

SalzbergWordStringKernel::SalzbergWordStringKernel(int32_t size, std::shared_ptr<PluginEstimate> pie, std::shared_ptr<Labels> labels)
: StringKernel<uint16_t>(size)
{
	init();
	estimate=pie;

	if (labels)
		set_prior_probs_from_labels(labels);
}

SalzbergWordStringKernel::SalzbergWordStringKernel(
	std::shared_ptr<StringFeatures<uint16_t>> l, std::shared_ptr<StringFeatures<uint16_t>> r,
	std::shared_ptr<PluginEstimate> pie, std::shared_ptr<Labels> labels)
: StringKernel<uint16_t>(10),estimate(pie)
{
	init();
	estimate=pie;

	if (labels)
		set_prior_probs_from_labels(labels);

	init(l, r);
}

SalzbergWordStringKernel::~SalzbergWordStringKernel()
{
	cleanup();
}

bool SalzbergWordStringKernel::init(std::shared_ptr<Features> p_l, std::shared_ptr<Features> p_r)
{
	StringKernel<uint16_t>::init(p_l,p_r);
	auto l=std::static_pointer_cast<StringFeatures<uint16_t>>(p_l);
	ASSERT(l)
	auto r=std::static_pointer_cast<StringFeatures<uint16_t>>(p_r);
	ASSERT(r)

	int32_t i;
	initialized=false;

	if (sqrtdiag_lhs!=sqrtdiag_rhs)
		SG_FREE(sqrtdiag_rhs);
	sqrtdiag_rhs=NULL;
	SG_FREE(sqrtdiag_lhs);
	sqrtdiag_lhs=NULL;
	if (ld_mean_lhs!=ld_mean_rhs)
		SG_FREE(ld_mean_rhs);
	ld_mean_rhs=NULL;
	SG_FREE(ld_mean_lhs);
	ld_mean_lhs=NULL;

	sqrtdiag_lhs=SG_MALLOC(float64_t, l->get_num_vectors());
	ld_mean_lhs=SG_MALLOC(float64_t, l->get_num_vectors());

	for (i=0; i<l->get_num_vectors(); i++)
		sqrtdiag_lhs[i]=1;

	if (l==r)
	{
		sqrtdiag_rhs=sqrtdiag_lhs;
		ld_mean_rhs=ld_mean_lhs;
	}
	else
	{
		sqrtdiag_rhs=SG_MALLOC(float64_t, r->get_num_vectors());
		for (i=0; i<r->get_num_vectors(); i++)
			sqrtdiag_rhs[i]=1;

		ld_mean_rhs=SG_MALLOC(float64_t, r->get_num_vectors());
	}

	float64_t* l_ld_mean_lhs=ld_mean_lhs;
	float64_t* l_ld_mean_rhs=ld_mean_rhs;

	//from our knowledge first normalize variance to 1 and then norm=1 does the job
	if (!initialized)
	{
		int32_t num_vectors=l->get_num_vectors();
		num_symbols=(int32_t) l->get_num_symbols();
		int32_t llen=l->get_vector_length(0);
		int32_t rlen=r->get_vector_length(0);
		num_params=(int32_t) llen*l->get_num_symbols();
		int32_t num_params2=(int32_t) llen*l->get_num_symbols()+rlen*r->get_num_symbols();
		if ((!estimate) || (!estimate->check_models()))
		{
			error("no estimate available");
			return false ;
		} ;
		if (num_params2!=estimate->get_num_params())
		{
			error("number of parameters of estimate and feature representation do not match");
			return false ;
		} ;

		SG_FREE(variance);
		SG_FREE(mean);
		mean=SG_MALLOC(float64_t, num_params);
		ASSERT(mean)
		variance=SG_MALLOC(float64_t, num_params);
		ASSERT(variance)

		for (i=0; i<num_params; i++)
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

			for (int32_t j=0; j<len; j++)
			{
				int32_t idx=compute_index(j, vec[j]);
				float64_t theta_p = 1/estimate->log_derivative_pos_obsolete(vec[j], j) ;
				float64_t theta_n = 1/estimate->log_derivative_neg_obsolete(vec[j], j) ;
				float64_t value   = (theta_p/(pos_prior*theta_p+neg_prior*theta_n)) ;

				mean[idx]   += value/num_vectors ;
			}
			l->free_feature_vector(vec, i, free_vec);
		}

		// compute variance
		for (i=0; i<num_vectors; i++)
		{
			int32_t len;
			bool free_vec;
			uint16_t* vec=l->get_feature_vector(i, len, free_vec);

			for (int32_t j=0; j<len; j++)
			{
				for (int32_t k=0; k<4; k++)
				{
					int32_t idx=compute_index(j, k);
					if (k!=vec[j])
						variance[idx]+=mean[idx]*mean[idx]/num_vectors;
					else
					{
						float64_t theta_p = 1/estimate->log_derivative_pos_obsolete(vec[j], j) ;
						float64_t theta_n = 1/estimate->log_derivative_neg_obsolete(vec[j], j) ;
						float64_t value = (theta_p/(pos_prior*theta_p+neg_prior*theta_n)) ;

						variance[idx] += Math::sq(value-mean[idx])/num_vectors;
					}
				}
			}
			l->free_feature_vector(vec, i, free_vec);
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
		int32_t alen ;
		bool free_avec;
		uint16_t* avec=l->get_feature_vector(i, alen, free_avec);
		float64_t  result=0 ;
		for (int32_t j=0; j<alen; j++)
		{
			int32_t a_idx = compute_index(j, avec[j]) ;
			float64_t theta_p = 1/estimate->log_derivative_pos_obsolete(avec[j], j) ;
			float64_t theta_n = 1/estimate->log_derivative_neg_obsolete(avec[j], j) ;
			float64_t value = (theta_p/(pos_prior*theta_p+neg_prior*theta_n)) ;

			if (variance[a_idx]!=0)
				result-=value*mean[a_idx]/variance[a_idx];
		}
		ld_mean_lhs[i]=result ;

		l->free_feature_vector(avec, i, free_avec);
	}

	if (ld_mean_lhs!=ld_mean_rhs)
	{
		// compute sum of
		//result -= feature*mean[b_idx]/variance[b_idx] ;
		for (i=0; i<r->get_num_vectors(); i++)
		{
			int32_t alen;
			bool free_avec;
			uint16_t* avec=r->get_feature_vector(i, alen, free_avec);
			float64_t  result=0;

			for (int32_t j=0; j<alen; j++)
			{
				int32_t a_idx = compute_index(j, avec[j]) ;
				float64_t theta_p=1/estimate->log_derivative_pos_obsolete(
					avec[j], j) ;
				float64_t theta_n=1/estimate->log_derivative_neg_obsolete(
					avec[j], j) ;
				float64_t value=(theta_p/(pos_prior*theta_p+neg_prior*theta_n));

				result -= value*mean[a_idx]/variance[a_idx] ;
			}

			ld_mean_rhs[i]=result;
			r->free_feature_vector(avec, i, free_avec);
		}
	}

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
	return init_normalizer();
}

void SalzbergWordStringKernel::cleanup()
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

	Kernel::cleanup();
}

float64_t SalzbergWordStringKernel::compute(int32_t idx_a, int32_t idx_b)
{
	int32_t alen, blen;
	bool free_avec, free_bvec;
	uint16_t* avec=std::static_pointer_cast<StringFeatures<uint16_t>>(lhs)->get_feature_vector(idx_a, alen, free_avec);
	uint16_t* bvec=std::static_pointer_cast<StringFeatures<uint16_t>>(rhs)->get_feature_vector(idx_b, blen, free_bvec);
	// can only deal with strings of same length
	ASSERT(alen==blen)

	float64_t result = sum_m2_s2 ; // does not contain 0-th element

	for (int32_t i=0; i<alen; i++)
	{
		if (avec[i]==bvec[i])
		{
			int32_t a_idx = compute_index(i, avec[i]) ;

			float64_t theta_p = 1/estimate->log_derivative_pos_obsolete(avec[i], i) ;
			float64_t theta_n = 1/estimate->log_derivative_neg_obsolete(avec[i], i) ;
			float64_t value = (theta_p/(pos_prior*theta_p+neg_prior*theta_n)) ;

			result   += value*value/variance[a_idx] ;
		}
	}
	result += ld_mean_lhs[idx_a] + ld_mean_rhs[idx_b] ;

	std::static_pointer_cast<StringFeatures<uint16_t>>(lhs)->free_feature_vector(avec, idx_a, free_avec);
	std::static_pointer_cast<StringFeatures<uint16_t>>(rhs)->free_feature_vector(bvec, idx_b, free_bvec);

	if (initialized)
		result /=  (sqrtdiag_lhs[idx_a]*sqrtdiag_rhs[idx_b]) ;

	return result;
}

void SalzbergWordStringKernel::set_prior_probs_from_labels(std::shared_ptr<Labels> labels)
{
	ASSERT(labels)
	ASSERT(labels->get_label_type() == LT_BINARY)
	labels->ensure_valid();

	int32_t num_pos=0, num_neg=0;
	auto bl = binary_labels(labels);
	for (int32_t i=0; i<labels->get_num_labels(); i++)
	{
		if (bl->get_int_label(i)==1)
			num_pos++;
		if (bl->get_int_label(i)==-1)
			num_neg++;
	}

	io::info("priors: pos={:1.3f} ({})  neg={:1.3f} ({})",
		(float64_t) num_pos/(num_pos+num_neg), num_pos,
		(float64_t) num_neg/(num_pos+num_neg), num_neg);

	set_prior_probs(
		(float64_t)num_pos/(num_pos+num_neg),
		(float64_t)num_neg/(num_pos+num_neg));
}

void SalzbergWordStringKernel::init()
{
	estimate=NULL;
	mean=NULL;
	variance=NULL;

	sqrtdiag_lhs=NULL;
	sqrtdiag_rhs=NULL;

	ld_mean_lhs=NULL;
	ld_mean_rhs=NULL;

	num_params=0;
	num_symbols=0;
	sum_m2_s2=0;
	pos_prior=0.5;

	neg_prior=0.5;
	initialized=false;
}
