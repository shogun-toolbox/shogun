/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Heiko Strathmann
 */

#include <shogun/statistics/QuadraticTimeMMD.h>
#include <shogun/features/Features.h>
#include <shogun/mathematics/Statistics.h>
#include <shogun/kernel/Kernel.h>

using namespace shogun;

CQuadraticTimeMMD::CQuadraticTimeMMD() : CKernelTwoSampleTestStatistic()
{
	init();
}

CQuadraticTimeMMD::CQuadraticTimeMMD(CKernel* kernel, CFeatures* p_and_q,
		index_t m) :
		CKernelTwoSampleTestStatistic(kernel, p_and_q, m)
{
	init();

	if (p_and_q && m!=p_and_q->get_num_vectors()/2)
	{
		SG_ERROR("CQuadraticTimeMMD: Only features with equal number of vectors "
				"are currently possible\n");
	}
}

CQuadraticTimeMMD::CQuadraticTimeMMD(CKernel* kernel, CFeatures* p,
		CFeatures* q) : CKernelTwoSampleTestStatistic(kernel, p, q)
{
	init();

	if (p && q && p->get_num_vectors()!=q->get_num_vectors())
	{
		SG_ERROR("CQuadraticTimeMMD: Only features with equal number of vectors "
				"are currently possible\n");
	}
}

CQuadraticTimeMMD::~CQuadraticTimeMMD()
{

}

void CQuadraticTimeMMD::init()
{
	SG_ADD(&m_num_samples_spectrum, "num_samples_spectrum", "Number of samples"
			" for spectrum method null-distribution approximation",
			MS_NOT_AVAILABLE);
	SG_ADD(&m_num_eigenvalues_spectrum, "num_eigenvalues_spectrum", "Number of "
			" Eigenvalues for spectrum method null-distribution approximation",
			MS_NOT_AVAILABLE);
	SG_ADD((machine_int_t*)&m_statistic_type, "statistic_type",
			"Biased or unbiased MMD", MS_NOT_AVAILABLE);

	m_num_samples_spectrum=0;
	m_num_eigenvalues_spectrum=0;
	m_statistic_type=UNBIASED;
}

float64_t CQuadraticTimeMMD::compute_unbiased_statistic()
{
	/* split computations into three terms from JLMR paper (see documentation )*/
	index_t m=m_m;

	/* init kernel with features */
	m_kernel->init(m_p_and_q, m_p_and_q);

	/* first term */
	float64_t first=0;
	for (index_t i=0; i<m; ++i)
	{
		/* ensure i!=j while adding up */
		for (index_t j=0; j<m; ++j)
			first+=i==j ? 0 : m_kernel->kernel(i,j);
	}
	first/=(m-1);

	/* second term */
	float64_t second=0;
	for (index_t i=m_m; i<m_m+m; ++i)
	{
		/* ensure i!=j while adding up */
		for (index_t j=m_m; j<m_m+m; ++j)
			second+=i==j ? 0 : m_kernel->kernel(i,j);
	}
	second/=(m-1);

	/* third term */
	float64_t third=0;
	for (index_t i=0; i<m; ++i)
	{
		for (index_t j=m_m; j<m_m+m; ++j)
			third+=m_kernel->kernel(i,j);
	}
	third*=2.0/m;

	return first+second-third;
}

float64_t CQuadraticTimeMMD::compute_biased_statistic()
{
	/* split computations into three terms from JLMR paper (see documentation )*/
	index_t m=m_m;

	/* init kernel with features */
	m_kernel->init(m_p_and_q, m_p_and_q);

	/* first term */
	float64_t first=0;
	for (index_t i=0; i<m; ++i)
	{
		for (index_t j=0; j<m; ++j)
			first+=m_kernel->kernel(i,j);
	}
	first/=m;

	/* second term */
	float64_t second=0;
	for (index_t i=m_m; i<m_m+m; ++i)
	{
		for (index_t j=m_m; j<m_m+m; ++j)
			second+=m_kernel->kernel(i,j);
	}
	second/=m;

	/* third term */
	float64_t third=0;
	for (index_t i=0; i<m; ++i)
	{
		for (index_t j=m_m; j<m_m+m; ++j)
			third+=m_kernel->kernel(i,j);
	}
	third*=2.0/m;

	return first+second-third;
}

float64_t CQuadraticTimeMMD::compute_statistic()
{
	if (!m_kernel)
		SG_ERROR("%s::compute_statistic(): No kernel specified!\n", get_name());

	float64_t result=0;
	switch (m_statistic_type)
	{
	case UNBIASED:
		result=compute_unbiased_statistic();
		break;
	case BIASED:
		result=compute_biased_statistic();
		break;
	default:
		SG_ERROR("CQuadraticTimeMMD::compute_statistic(): Unknown statistic "
				"type!\n");
		break;
	}

	return result;
}

float64_t CQuadraticTimeMMD::compute_p_value(float64_t statistic)
{
	float64_t result=0;

	switch (m_null_approximation_method)
	{
	case MMD2_SPECTRUM:
	{
#ifdef HAVE_LAPACK
		/* get samples from null-distribution and compute p-value of statistic */
		SGVector<float64_t> null_samples=sample_null_spectrum(
				m_num_samples_spectrum, m_num_eigenvalues_spectrum);
		CMath::qsort(null_samples);
		index_t pos=CMath::find_position_to_insert(null_samples, statistic);
		result=1.0-((float64_t)pos)/null_samples.vlen;
#else // HAVE_LAPACK
		SG_ERROR("CQuadraticTimeMMD::compute_p_value(): Only possible if "
				"shogun is compiled with LAPACK enabled\n");
#endif // HAVE_LAPACK
		break;
	}

	case MMD2_GAMMA:
	{
		/* fit gamma and return cdf at statistic */
		SGVector<float64_t> params=fit_null_gamma();
		result=CStatistics::gamma_cdf(statistic, params[0], params[1]);
		break;
	}

	default:
		result=CKernelTwoSampleTestStatistic::compute_p_value(statistic);
		break;
	}

	return result;
}

float64_t CQuadraticTimeMMD::compute_threshold(float64_t alpha)
{
	float64_t result=0;

	switch (m_null_approximation_method)
	{
	case MMD2_SPECTRUM:
	{
#ifdef HAVE_LAPACK
		/* get samples from null-distribution and compute threshold */
		SGVector<float64_t> null_samples=sample_null_spectrum(
				m_num_samples_spectrum, m_num_eigenvalues_spectrum);
		CMath::qsort(null_samples);
		result=null_samples[CMath::floor(null_samples.vlen*(1-alpha))];
#else // HAVE_LAPACK
		SG_ERROR("CQuadraticTimeMMD::compute_threshold(): Only possible if "
				"shogun is compiled with LAPACK enabled\n");
#endif // HAVE_LAPACK
		break;
	}

	case MMD2_GAMMA:
	{
		/* fit gamma and return inverse cdf at alpha */
		SGVector<float64_t> params=fit_null_gamma();
		result=CStatistics::inverse_gamma_cdf(alpha, params[0], params[1]);
		break;
	}

	default:
		/* bootstrapping is handled here */
		result=CKernelTwoSampleTestStatistic::compute_threshold(alpha);
		break;
	}

	return result;
}


#ifdef HAVE_LAPACK
SGVector<float64_t> CQuadraticTimeMMD::sample_null_spectrum(index_t num_samples,
		index_t num_eigenvalues)
{
	if (m_m!=m_p_and_q->get_num_vectors()/2)
	{
		SG_ERROR("%s::sample_null_spectrum(): Currently, only equal "
				"sample sizes are supported\n", get_name());
	}

	if (num_samples<=2)
	{
		SG_ERROR("%s::sample_null_spectrum(): Number of samples has to be at"
				" least 2, better in the hundrets", get_name());
	}

	if (num_eigenvalues>2*m_m-1)
	{
		SG_ERROR("%s::sample_null_spectrum(): Number of Eigenvalues too large\n",
				get_name());
	}

	if (num_eigenvalues<1)
	{
		SG_ERROR("%s::sample_null_spectrum(): Number of Eigenvalues too small\n",
				get_name());
	}

	/* evtl. warn user not to use wrong statistic type */
	if (m_statistic_type!=BIASED)
	{
		SG_WARNING("%s::sample_null_spectrum(): Note: provided statistic has "
				"to be BIASED. Please ensure that! To get rid of warning,"
				"call %s::set_statistic_type(BIASED)\n", get_name(),
				get_name());
	}

	/* imaginary matrix K=[K KL; KL' L] (MATLAB notation)
	 * K is matrix for XX, L is matrix for YY, KL is XY, LK is YX
	 * works since X and Y are concatenated here */
	m_kernel->init(m_p_and_q, m_p_and_q);
	SGMatrix<float64_t> K=m_kernel->get_kernel_matrix();

	/* center matrix K=H*K*H */
	K.center();

	/* compute eigenvalues and select num_eigenvalues largest ones */
	SGVector<float64_t> eigenvalues=
			SGMatrix<float64_t>::compute_eigenvectors(K);
	SGVector<float64_t> largest_ev(num_eigenvalues);

	/* take largest EV, scale by 1/2/m on the fly and take abs value*/
	for (index_t i=0; i<num_eigenvalues; ++i)
		largest_ev[i]=CMath::abs(
				1.0/2/m_m*eigenvalues[eigenvalues.vlen-1-i]);

	/* finally, sample from null distribution */
	SGVector<float64_t> null_samples(num_samples);
	for (index_t i=0; i<num_samples; ++i)
	{
		/* 2*sum(kEigs.*(randn(length(kEigs),1)).^2); */
		null_samples[i]=0;
		for (index_t j=0; j<largest_ev.vlen; ++j)
			null_samples[i]+=largest_ev[j]*CMath::pow(CMath::randn_double(), 2);

		null_samples[i]*=2;
	}

	return null_samples;
}
#endif // HAVE_LAPACK

SGVector<float64_t> CQuadraticTimeMMD::fit_null_gamma()
{
	if (m_m!=m_p_and_q->get_num_vectors()/2)
	{
		SG_ERROR("%s::compute_p_value_gamma(): Currently, only equal "
				"sample sizes are supported\n", get_name());
	}

	/* evtl. warn user not to use wrong statistic type */
	if (m_statistic_type!=BIASED)
	{
		SG_WARNING("%s::compute_p_value(): Note: provided statistic has "
				"to be BIASED. Please ensure that! To get rid of warning,"
				"call %s::set_statistic_type(BIASED)\n", get_name(),
				get_name());
	}

	/* imaginary matrix K=[K KL; KL' L] (MATLAB notation)
	 * K is matrix for XX, L is matrix for YY, KL is XY, LK is YX
	 * works since X and Y are concatenated here */
	m_kernel->init(m_p_and_q, m_p_and_q);

	/* compute mean under H0 of MMD, which is
	 * meanMMD  = 2/m * ( 1  - 1/m*sum(diag(KL))  );
	 * in MATLAB.
	 * Remove diagonals on the fly */
	float64_t mean_mmd=0;
	for (index_t i=0; i<m_m; ++i)
	{
		/* virtual KL matrix is in upper right corner of SHOGUN K matrix
		 * so this sums the diagonal of the matrix between X and Y*/
		mean_mmd+=m_kernel->kernel(i, m_m+i);
	}
	mean_mmd=2.0/m_m*(1.0-1.0/m_m*mean_mmd);

	/* compute variance under H0 of MMD, which is
	 * varMMD = 2/m/(m-1) * 1/m/(m-1) * sum(sum( (K + L - KL - KL').^2 ));
	 * in MATLAB, so sum up all elements */
	float64_t var_mmd=0;
	for (index_t i=0; i<m_m; ++i)
	{
		for (index_t j=0; j<m_m; ++j)
		{
			/* dont add diagonal of all pairs of imaginary kernel matrices */
			if (i==j || m_m+i==j || m_m+j==i)
				continue;

			float64_t to_add=m_kernel->kernel(i, j);
			to_add+=m_kernel->kernel(m_m+i, m_m+j);
			to_add-=m_kernel->kernel(i, m_m+j);
			to_add-=m_kernel->kernel(m_m+i, j);
			var_mmd+=CMath::pow(to_add, 2);
		}
	}
	var_mmd*=2.0/m_m/(m_m-1)*1.0/m_m/(m_m-1);

	/* parameters for gamma distribution */
	float64_t a=CMath::pow(mean_mmd, 2)/var_mmd;
	float64_t b=var_mmd*m_m / mean_mmd;

	SGVector<float64_t> result(2);
	result[0]=a;
	result[1]=b;

	return result;
}

void CQuadraticTimeMMD::set_num_samples_sepctrum(index_t
		num_samples_spectrum)
{
	m_num_samples_spectrum=num_samples_spectrum;
}

void CQuadraticTimeMMD::set_num_eigenvalues_spectrum(
		index_t num_eigenvalues_spectrum)
{
	m_num_eigenvalues_spectrum=num_eigenvalues_spectrum;
}

void CQuadraticTimeMMD::set_statistic_type(EQuadraticMMDType
		statistic_type)
{
	m_statistic_type=statistic_type;
}
