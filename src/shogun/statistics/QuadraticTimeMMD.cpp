/*
 * Copyright (c) The Shogun Machine Learning Toolbox
 * Written (w) 2012-2013 Heiko Strathmann
 * Written (w) 2014 Soumyajit De
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * The views and conclusions contained in the software and documentation are those
 * of the authors and should not be interpreted as representing official policies,
 * either expressed or implied, of the Shogun Development Team.
 */

#include <shogun/statistics/QuadraticTimeMMD.h>
#include <shogun/features/Features.h>
#include <shogun/mathematics/Math.h>
#include <shogun/mathematics/Statistics.h>
#include <shogun/kernel/Kernel.h>
#include <shogun/kernel/CombinedKernel.h>
#include <shogun/kernel/CustomKernel.h>
#include <shogun/mathematics/linalg/linalg.h>

using namespace shogun;
using namespace linalg;

#ifdef HAVE_EIGEN3
#include <shogun/mathematics/eigen3.h>

using namespace Eigen;
#endif

CQuadraticTimeMMD::CQuadraticTimeMMD() : CKernelTwoSampleTest()
{
	init();
}

CQuadraticTimeMMD::CQuadraticTimeMMD(CKernel* kernel, CFeatures* p_and_q,
		index_t m) :
		CKernelTwoSampleTest(kernel, p_and_q, m)
{
	init();
}

CQuadraticTimeMMD::CQuadraticTimeMMD(CKernel* kernel, CFeatures* p,
		CFeatures* q) : CKernelTwoSampleTest(kernel, p, q)
{
	init();
}

CQuadraticTimeMMD::CQuadraticTimeMMD(CCustomKernel* custom_kernel, index_t m) :
		CKernelTwoSampleTest(custom_kernel, NULL, m)
{
	init();
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

SGVector<float64_t> CQuadraticTimeMMD::compute_unbiased_statistic_variance(
		int m, int n)
{
	SG_DEBUG("Entering!\n");

	/* init kernel with features. NULL check is handled in compute_statistic */
	m_kernel->init(m_p_and_q, m_p_and_q);

	/* computing kernel values and their sums on the fly that are used both in
	   computing statistic and variance */

	/* the following matrix stores row-wise sum of kernel values k(X,X') in
	   the first column and row-wise squared sum of kernel values k^2(X,X')
	   in the second column. m entries in both column */
	SGMatrix<float64_t> xx_sum_sq_sum_rowwise=m_kernel->
		row_wise_sum_squared_sum_symmetric_block(0, m);

	/* row-wise sum of kernel values k(Y,Y'), n entries */
	SGVector<float64_t> yy_sum_rowwise=m_kernel->
		row_wise_sum_symmetric_block(m, n);

	/* row-wise and col-wise sum of kernel values k(X,Y), m+n entries
	   first m entries are row-wise sum, rest n entries are col-wise sum */
	SGVector<float64_t> xy_sum_rowcolwise=m_kernel->
		row_col_wise_sum_block(0, m, m, n);

	/* computing overall sum and squared sum from above for convenience */

	SGVector<float64_t> xx_sum_rowwise(m);
	std::copy(xx_sum_sq_sum_rowwise.matrix, xx_sum_sq_sum_rowwise.matrix+m,
			xx_sum_rowwise.vector);

	SGVector<float64_t> xy_sum_rowwise(m);
	std::copy(xy_sum_rowcolwise.vector, xy_sum_rowcolwise.vector+m,
			xy_sum_rowwise.vector);

	SGVector<float64_t> xy_sum_colwise(n);
	std::copy(xy_sum_rowcolwise.vector+m, xy_sum_rowcolwise.vector+m+n,
			xy_sum_colwise.vector);

	float64_t xx_sq_sum=0.0;
	for (index_t i=0; i<xx_sum_sq_sum_rowwise.num_rows; i++)
		xx_sq_sum+=xx_sum_sq_sum_rowwise(i, 1);

	float64_t xx_sum=0.0;
	for (index_t i=0; i<xx_sum_rowwise.vlen; i++)
		xx_sum+=xx_sum_rowwise[i];

	float64_t yy_sum=0.0;
	for (index_t i=0; i<yy_sum_rowwise.vlen; i++)
		yy_sum+=yy_sum_rowwise[i];

	float64_t xy_sum=0.0;
	for (index_t i=0; i<xy_sum_rowwise.vlen; i++)
		xy_sum+=xy_sum_rowwise[i];

	/* compute statistic */

	/* split computations into three terms from JLMR paper (see documentation )*/

	/* first term */
	float64_t first=xx_sum/m/(m-1);

	/* second term */
	float64_t second=yy_sum/n/(n-1);

	/* third term */
	float64_t third=2.0*xy_sum/m/n;

	float64_t statistic=first+second-third;

	SG_INFO("Computed statistic!\n");
	SG_DEBUG("first=%f, second=%f, third=%f\n", first, second, third);

	/* compute variance under null */

	/* split computations into three terms (see documentation) */

	/* first term */
	float64_t kappa_0=CMath::sq(xx_sum/m/(m-1));

	/* second term */
	float64_t kappa_1=0.0;
	for (index_t i=0; i<m; ++i)
		kappa_1+=CMath::sq(xx_sum_rowwise[i]/(m-1));
	kappa_1/=m;

	/* third term */
	float64_t kappa_2=xx_sq_sum/m/(m-1);

	float64_t var_null=2*(kappa_0-2*kappa_1+kappa_2);

	SG_INFO("Computed variance under null!\n");
	SG_DEBUG("kappa_0=%f, kappa_1=%f, kappa_2=%f\n", kappa_0, kappa_1, kappa_2);

	/* compute variance under alternative */

	/* split computations into four terms (see documentation) */

	/* first term */
	float64_t alt_var_first=0.0;
	for (index_t i=0; i<m; ++i)
	{
		// use row-wise sum from k(X,X') and k(X,Y) blocks
		float64_t term=xx_sum_rowwise[i]/(m-1)-xy_sum_rowwise[i]/n;
		alt_var_first+=CMath::sq(term);
	}
	alt_var_first/=m;

	/* second term */
	float64_t alt_var_second=CMath::sq(xx_sum/m/(m-1)-xy_sum/m/n);

	/* third term */
	float64_t alt_var_third=0.0;
	for (index_t i=0; i<n; ++i)
	{
		// use row-wise sum from k(Y,Y') and col-wise sum from k(X,Y)
		// blocks to simulate row-wise sum from k(Y,X) blocks
		float64_t term=yy_sum_rowwise[i]/(n-1)-xy_sum_colwise[i]/m;
		alt_var_third+=CMath::sq(term);
	}
	alt_var_third/=n;

	/* fourth term */
	float64_t alt_var_fourth=CMath::sq(yy_sum/n/(n-1)-xy_sum/m/n);

	/* finally computing variance */
	float64_t rho_x=float64_t(m)/(m+n);
	float64_t rho_y=float64_t(n)/(m+n);

	float64_t var_alt=4*rho_x*(alt_var_first-alt_var_second)+
		4*rho_y*(alt_var_third-alt_var_fourth);

	SG_INFO("Computed variance under alternative!\n");
	SG_DEBUG("first=%f, second=%f, third=%f, fourth=%f\n", alt_var_first,
			alt_var_second, alt_var_third, alt_var_fourth);

	SGVector<float64_t> results(3);
	results[0]=statistic;
	results[1]=var_null;
	results[2]=var_alt;

	SG_DEBUG("Leaving!\n");

	return results;
}

SGVector<float64_t> CQuadraticTimeMMD::compute_biased_statistic_variance(int m, int n)
{
	SG_DEBUG("Entering!\n");

	/* init kernel with features. NULL check is handled in compute_statistic */
	m_kernel->init(m_p_and_q, m_p_and_q);

	/* computing kernel values and their sums on the fly that are used both in
	   computing statistic and variance */

	/* the following matrix stores row-wise sum of kernel values k(X,X') in
	   the first column and row-wise squared sum of kernel values k^2(X,X')
	   in the second column. m entries in both column */
	SGMatrix<float64_t> xx_sum_sq_sum_rowwise=m_kernel->
		row_wise_sum_squared_sum_symmetric_block(0, m, false);

	/* row-wise sum of kernel values k(Y,Y'), n entries */
	SGVector<float64_t> yy_sum_rowwise=m_kernel->
		row_wise_sum_symmetric_block(m, n, false);

	/* row-wise and col-wise sum of kernel values k(X,Y), m+n entries
	   first m entries are row-wise sum, rest n entries are col-wise sum */
	SGVector<float64_t> xy_sum_rowcolwise=m_kernel->
		row_col_wise_sum_block(0, m, m, n);

	/* computing overall sum and squared sum from above for convenience */

	SGVector<float64_t> xx_sum_rowwise(m);
	std::copy(xx_sum_sq_sum_rowwise.matrix, xx_sum_sq_sum_rowwise.matrix+m,
			xx_sum_rowwise.vector);

	SGVector<float64_t> xy_sum_rowwise(m);
	std::copy(xy_sum_rowcolwise.vector, xy_sum_rowcolwise.vector+m,
			xy_sum_rowwise.vector);

	SGVector<float64_t> xy_sum_colwise(n);
	std::copy(xy_sum_rowcolwise.vector+m, xy_sum_rowcolwise.vector+m+n,
			xy_sum_colwise.vector);

	float64_t xx_sq_sum=0.0;
	for (index_t i=0; i<xx_sum_sq_sum_rowwise.num_rows; i++)
		xx_sq_sum+=xx_sum_sq_sum_rowwise(i, 1);

	float64_t xx_sum=0.0;
	for (index_t i=0; i<xx_sum_rowwise.vlen; i++)
		xx_sum+=xx_sum_rowwise[i];

	float64_t yy_sum=0.0;
	for (index_t i=0; i<yy_sum_rowwise.vlen; i++)
		yy_sum+=yy_sum_rowwise[i];

	float64_t xy_sum=0.0;
	for (index_t i=0; i<xy_sum_rowwise.vlen; i++)
		xy_sum+=xy_sum_rowwise[i];

	/* compute statistic */

	/* split computations into three terms from JLMR paper (see documentation )*/

	/* first term */
	float64_t first=xx_sum/m/m;

	/* second term */
	float64_t second=yy_sum/n/n;

	/* third term */
	float64_t third=2.0*xy_sum/m/n;

	float64_t statistic=first+second-third;

	SG_INFO("Computed statistic!\n");
	SG_DEBUG("first=%f, second=%f, third=%f\n", first, second, third);

	/* compute variance under null */

	/* split computations into three terms (see documentation) */

	/* first term */
	float64_t kappa_0=CMath::sq(xx_sum/m/m);

	/* second term */
	float64_t kappa_1=0.0;
	for (index_t i=0; i<m; ++i)
		kappa_1+=CMath::sq(xx_sum_rowwise[i]/m);
	kappa_1/=m;

	/* third term */
	float64_t kappa_2=xx_sq_sum/m/m;

	float64_t var_null=2*(kappa_0-2*kappa_1+kappa_2);

	SG_INFO("Computed variance under null!\n");
	SG_DEBUG("kappa_0=%f, kappa_1=%f, kappa_2=%f\n", kappa_0, kappa_1, kappa_2);

	/* compute variance under alternative */

	/* split computations into four terms (see documentation) */

	/* first term */
	float64_t alt_var_first=0.0;
	for (index_t i=0; i<m; ++i)
	{
		// use row-wise sum from k(X,X') and k(X,Y) blocks
		float64_t term=xx_sum_rowwise[i]/m-xy_sum_rowwise[i]/n;
		alt_var_first+=CMath::sq(term);
	}
	alt_var_first/=m;

	/* second term */
	float64_t alt_var_second=CMath::sq(xx_sum/m/m-xy_sum/m/n);

	/* third term */
	float64_t alt_var_third=0.0;
	for (index_t i=0; i<n; ++i)
	{
		// use row-wise sum from k(Y,Y') and col-wise sum from k(X,Y)
		// blocks to simulate row-wise sum from k(Y,X) blocks
		float64_t term=yy_sum_rowwise[i]/n-xy_sum_colwise[i]/m;
		alt_var_third+=CMath::sq(term);
	}
	alt_var_third/=n;

	/* fourth term */
	float64_t alt_var_fourth=CMath::sq(yy_sum/n/n-xy_sum/m/n);

	/* finally computing variance */
	float64_t rho_x=float64_t(m)/(m+n);
	float64_t rho_y=float64_t(n)/(m+n);

	float64_t var_alt=4*rho_x*(alt_var_first-alt_var_second)+
		4*rho_y*(alt_var_third-alt_var_fourth);

	SG_INFO("Computed variance under alternative!\n");
	SG_DEBUG("first=%f, second=%f, third=%f, fourth=%f\n", alt_var_first,
			alt_var_second, alt_var_third, alt_var_fourth);

	SGVector<float64_t> results(3);
	results[0]=statistic;
	results[1]=var_null;
	results[2]=var_alt;

	SG_DEBUG("Leaving!\n");

	return results;
}

SGVector<float64_t> CQuadraticTimeMMD::compute_incomplete_statistic_variance(int n)
{
	SG_DEBUG("Entering!\n");

	/* init kernel with features. NULL check is handled in compute_statistic */
	m_kernel->init(m_p_and_q, m_p_and_q);

	/* computing kernel values and their sums on the fly that are used both in
	   computing statistic and variance */

	/* the following matrix stores row-wise sum of kernel values k(X,X') in
	   the first column and row-wise squared sum of kernel values k^2(X,X')
	   in the second column. n entries in both column */
	SGMatrix<float64_t> xx_sum_sq_sum_rowwise=m_kernel->
		row_wise_sum_squared_sum_symmetric_block(0, n);

	/* row-wise sum of kernel values k(Y,Y'), n entries */
	SGVector<float64_t> yy_sum_rowwise=m_kernel->
		row_wise_sum_symmetric_block(n, n);

	/* row-wise and col-wise sum of kernel values k(X,Y), 2n entries
	   first n entries are row-wise sum, rest n entries are col-wise sum */
	SGVector<float64_t> xy_sum_rowcolwise=m_kernel->
		row_col_wise_sum_block(0, n, n, n, true);

	/* computing overall sum and squared sum from above for convenience */

	SGVector<float64_t> xx_sum_rowwise(n);
	std::copy(xx_sum_sq_sum_rowwise.matrix, xx_sum_sq_sum_rowwise.matrix+n,
			xx_sum_rowwise.vector);

	SGVector<float64_t> xy_sum_rowwise(n);
	std::copy(xy_sum_rowcolwise.vector, xy_sum_rowcolwise.vector+n,
			xy_sum_rowwise.vector);

	SGVector<float64_t> xy_sum_colwise(n);
	std::copy(xy_sum_rowcolwise.vector+n, xy_sum_rowcolwise.vector+2*n,
			xy_sum_colwise.vector);

	float64_t xx_sq_sum=0.0;
	for (index_t i=0; i<xx_sum_sq_sum_rowwise.num_rows; i++)
		xx_sq_sum+=xx_sum_sq_sum_rowwise(i, 1);

	float64_t xx_sum=0.0;
	for (index_t i=0; i<xx_sum_rowwise.vlen; i++)
		xx_sum+=xx_sum_rowwise[i];

	float64_t yy_sum=0.0;
	for (index_t i=0; i<yy_sum_rowwise.vlen; i++)
		yy_sum+=yy_sum_rowwise[i];

	float64_t xy_sum=0.0;
	for (index_t i=0; i<xy_sum_rowwise.vlen; i++)
		xy_sum+=xy_sum_rowwise[i];

	/* compute statistic */

	/* split computations into three terms from JLMR paper (see documentation )*/

	/* first term */
	float64_t first=xx_sum/n/(n-1);

	/* second term */
	float64_t second=yy_sum/n/(n-1);

	/* third term */
	float64_t third=2.0*xy_sum/n/(n-1);

	float64_t statistic=first+second-third;

	SG_INFO("Computed statistic!\n");
	SG_DEBUG("first=%f, second=%f, third=%f\n", first, second, third);

	/* compute variance under null */

	/* split computations into three terms (see documentation) */

	/* first term */
	float64_t kappa_0=CMath::sq(xx_sum/n/(n-1));

	/* second term */
	float64_t kappa_1=0.0;
	for (index_t i=0; i<n; ++i)
		kappa_1+=CMath::sq(xx_sum_rowwise[i]/(n-1));
	kappa_1/=n;

	/* third term */
	float64_t kappa_2=xx_sq_sum/n/(n-1);

	float64_t var_null=2*(kappa_0-2*kappa_1+kappa_2);

	SG_INFO("Computed variance under null!\n");
	SG_DEBUG("kappa_0=%f, kappa_1=%f, kappa_2=%f\n", kappa_0, kappa_1, kappa_2);

	/* compute variance under alternative */

	/* split computations into four terms (see documentation) */

	/* first term */
	float64_t alt_var_first=0.0;
	for (index_t i=0; i<n; ++i)
	{
		// use row-wise sum from k(X,X') and k(X,Y) blocks
		float64_t term=(xx_sum_rowwise[i]-xy_sum_rowwise[i])/(n-1);
		alt_var_first+=CMath::sq(term);
	}
	alt_var_first/=n;

	/* second term */
	float64_t alt_var_second=CMath::sq(xx_sum/n/(n-1)-xy_sum/n/(n-1));

	/* third term */
	float64_t alt_var_third=0.0;
	for (index_t i=0; i<n; ++i)
	{
		// use row-wise sum from k(Y,Y') and col-wise sum from k(X,Y)
		// blocks to simulate row-wise sum from k(Y,X) blocks
		float64_t term=(yy_sum_rowwise[i]-xy_sum_colwise[i])/(n-1);
		alt_var_third+=CMath::sq(term);
	}
	alt_var_third/=n;

	/* fourth term */
	float64_t alt_var_fourth=CMath::sq(yy_sum/n/(n-1)-xy_sum/n/(n-1));

	/* finally computing variance */
	float64_t rho_x=0.5;
	float64_t rho_y=0.5;

	float64_t var_alt=4*rho_x*(alt_var_first-alt_var_second)+
		4*rho_y*(alt_var_third-alt_var_fourth);

	SG_INFO("Computed variance under alternative!\n");
	SG_DEBUG("first=%f, second=%f, third=%f, fourth=%f\n", alt_var_first,
			alt_var_second, alt_var_third, alt_var_fourth);

	SGVector<float64_t> results(3);
	results[0]=statistic;
	results[1]=var_null;
	results[2]=var_alt;

	SG_DEBUG("Leaving!\n");

	return results;
}

float64_t CQuadraticTimeMMD::compute_unbiased_statistic(int m, int n)
{
	return compute_unbiased_statistic_variance(m, n)[0];
}

float64_t CQuadraticTimeMMD::compute_biased_statistic(int m, int n)
{
	return compute_biased_statistic_variance(m, n)[0];
}

float64_t CQuadraticTimeMMD::compute_incomplete_statistic(int n)
{
	return compute_incomplete_statistic_variance(n)[0];
}

float64_t CQuadraticTimeMMD::compute_statistic()
{
	REQUIRE(m_kernel, "No kernel specified!\n")

	index_t m=m_m;
	index_t n=0;

	/* check if kernel is precomputed (custom kernel) */
	if (m_kernel->get_kernel_type()==K_CUSTOM)
		n=m_kernel->get_num_vec_lhs()-m;
	else
	{
		REQUIRE(m_p_and_q, "The samples are not initialized!\n");
		n=m_p_and_q->get_num_vectors()-m;
	}

	SG_DEBUG("Computing MMD with %d samples from p and %d samples from q!\n",
			m, n);

	float64_t result=0;
	switch (m_statistic_type)
	{
	case UNBIASED:
		result=compute_unbiased_statistic(m, n);
		result*=m*n/float64_t(m+n);
		break;
	case UNBIASED_DEPRECATED:
		result=compute_unbiased_statistic(m, n);
		result*=m==n ? m : (m+n);
		break;
	case BIASED:
		result=compute_biased_statistic(m, n);
		result*=m*n/float64_t(m+n);
		break;
	case BIASED_DEPRECATED:
		result=compute_biased_statistic(m, n);
		result*=m==n? m : (m+n);
		break;
	case INCOMPLETE:
		REQUIRE(m==n, "Only possible with equal number of samples from both"
				"distribution!\n")
		result=compute_incomplete_statistic(n);
		result*=n/2;
		break;
	default:
		SG_ERROR("Unknown statistic type!\n");
		break;
	}

	return result;
}

SGVector<float64_t> CQuadraticTimeMMD::compute_variance()
{
	REQUIRE(m_kernel, "No kernel specified!\n")

	index_t m=m_m;
	index_t n=0;

	/* check if kernel is precomputed (custom kernel) */
	if (m_kernel->get_kernel_type()==K_CUSTOM)
		n=m_kernel->get_num_vec_lhs()-m;
	else
	{
		REQUIRE(m_p_and_q, "The samples are not initialized!\n");
		n=m_p_and_q->get_num_vectors()-m;
	}

	SG_DEBUG("Computing MMD with %d samples from p and %d samples from q!\n",
			m, n);

	SGVector<float64_t> result(2);
	switch (m_statistic_type)
	{
	case UNBIASED:
	case UNBIASED_DEPRECATED:
	{
		SGVector<float64_t> res=compute_unbiased_statistic_variance(m, n);
		result[0]=res[1];
		result[1]=res[2];
		break;
	}
	case BIASED:
	case BIASED_DEPRECATED:
	{
		SGVector<float64_t> res=compute_biased_statistic_variance(m, n);
		result[0]=res[1];
		result[1]=res[2];
		break;
	}
	case INCOMPLETE:
	{
		REQUIRE(m==n, "Only possible with equal number of samples from both"
				"distribution!\n")
		SGVector<float64_t> res=compute_incomplete_statistic_variance(n);
		result[0]=res[1];
		result[1]=res[2];
		break;
	}
	default:
		SG_ERROR("Unknown statistic type!\n");
		break;
	}

	return result;
}

float64_t CQuadraticTimeMMD::compute_variance_under_null()
{
	return compute_variance()[0];
}

float64_t CQuadraticTimeMMD::compute_variance_under_alternative()
{
	return compute_variance()[1];
}

SGVector<float64_t> CQuadraticTimeMMD::compute_statistic(bool multiple_kernels)
{
	SGVector<float64_t> mmds;
	if (!multiple_kernels)
	{
		/* just one mmd result */
		mmds=SGVector<float64_t>(1);
		mmds[0]=compute_statistic();
	}
	else
	{
		REQUIRE(m_kernel, "No kernel specified!\n")
		REQUIRE(m_kernel->get_kernel_type()==K_COMBINED,
			"multiple kernels specified, but underlying kernel is not of type "
			"K_COMBINED\n");

		/* cast and allocate memory for results */
		CCombinedKernel* combined=(CCombinedKernel*)m_kernel;
		SG_REF(combined);
		mmds=SGVector<float64_t>(combined->get_num_subkernels());

		/* iterate through all kernels and compute statistic */
		/* TODO this might be done in parallel */
		for (index_t i=0; i<mmds.vlen; ++i)
		{
			CKernel* current=combined->get_kernel(i);
			/* temporarily replace underlying kernel and compute statistic */
			m_kernel=current;
			mmds[i]=compute_statistic();

			SG_UNREF(current);
		}

		/* restore combined kernel */
		m_kernel=combined;
		SG_UNREF(combined);
	}

	return mmds;
}

SGMatrix<float64_t> CQuadraticTimeMMD::compute_variance(bool multiple_kernels)
{
	SGMatrix<float64_t> vars;
	if (!multiple_kernels)
	{
		/* just one mmd result */
		vars=SGMatrix<float64_t>(1, 2);
		SGVector<float64_t> result=compute_variance();
		vars(0, 0)=result[0];
		vars(0, 1)=result[1];
	}
	else
	{
		REQUIRE(m_kernel, "No kernel specified!\n")
		REQUIRE(m_kernel->get_kernel_type()==K_COMBINED,
			"multiple kernels specified, but underlying kernel is not of type "
			"K_COMBINED\n");

		/* cast and allocate memory for results */
		CCombinedKernel* combined=(CCombinedKernel*)m_kernel;
		SG_REF(combined);
		vars=SGMatrix<float64_t>(combined->get_num_subkernels(), 2);

		/* iterate through all kernels and compute variance */
		/* TODO this might be done in parallel */
		for (index_t i=0; i<vars.num_rows; ++i)
		{
			CKernel* current=combined->get_kernel(i);
			/* temporarily replace underlying kernel and compute variance */
			m_kernel=current;
			SGVector<float64_t> result=compute_variance();
			vars(i, 0)=result[0];
			vars(i, 1)=result[1];

			SG_UNREF(current);
		}

		/* restore combined kernel */
		m_kernel=combined;
		SG_UNREF(combined);
	}

	return vars;
}

float64_t CQuadraticTimeMMD::compute_p_value(float64_t statistic)
{
	float64_t result=0;

	switch (m_null_approximation_method)
	{
	case MMD2_SPECTRUM:
	{
#ifdef HAVE_EIGEN3
		/* get samples from null-distribution and compute p-value of statistic */
		SGVector<float64_t> null_samples=sample_null_spectrum(
				m_num_samples_spectrum, m_num_eigenvalues_spectrum);
		CMath::qsort(null_samples);
		index_t pos=null_samples.find_position_to_insert(statistic);
		result=1.0-((float64_t)pos)/null_samples.vlen;
#else // HAVE_EIGEN3
		SG_ERROR("Only possible if shogun is compiled with EIGEN3 enabled\n");
#endif // HAVE_EIGEN3
		break;
	}

	case MMD2_SPECTRUM_DEPRECATED:
	{
#ifdef HAVE_EIGEN3
		/* get samples from null-distribution and compute p-value of statistic */
		SGVector<float64_t> null_samples=sample_null_spectrum_DEPRECATED(
				m_num_samples_spectrum, m_num_eigenvalues_spectrum);
		CMath::qsort(null_samples);
		index_t pos=null_samples.find_position_to_insert(statistic);
		result=1.0-((float64_t)pos)/null_samples.vlen;
#else // HAVE_EIGEN3
		SG_ERROR("Only possible if shogun is compiled with EIGEN3 enabled\n");
#endif // HAVE_EIGEN3
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
		result=CKernelTwoSampleTest::compute_p_value(statistic);
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
#ifdef HAVE_EIGEN3
		/* get samples from null-distribution and compute threshold */
		SGVector<float64_t> null_samples=sample_null_spectrum(
				m_num_samples_spectrum, m_num_eigenvalues_spectrum);
		CMath::qsort(null_samples);
		result=null_samples[index_t(CMath::floor(null_samples.vlen*(1-alpha)))];
#else // HAVE_EIGEN3
		SG_ERROR("Only possible if shogun is compiled with EIGEN3 enabled\n");
#endif // HAVE_EIGEN3
		break;
	}

	case MMD2_SPECTRUM_DEPRECATED:
	{
#ifdef HAVE_EIGEN3
		/* get samples from null-distribution and compute threshold */
		SGVector<float64_t> null_samples=sample_null_spectrum_DEPRECATED(
				m_num_samples_spectrum, m_num_eigenvalues_spectrum);
		CMath::qsort(null_samples);
		result=null_samples[index_t(CMath::floor(null_samples.vlen*(1-alpha)))];
#else // HAVE_EIGEN3
		SG_ERROR("Only possible if shogun is compiled with EIGEN3 enabled\n");
#endif // HAVE_EIGEN3
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
		/* sampling null is handled here */
		result=CKernelTwoSampleTest::compute_threshold(alpha);
		break;
	}

	return result;
}


#ifdef HAVE_EIGEN3
SGVector<float64_t> CQuadraticTimeMMD::sample_null_spectrum(index_t num_samples,
		index_t num_eigenvalues)
{
	REQUIRE(m_kernel, "(%d, %d): No kernel set!\n", num_samples,
			num_eigenvalues);
	REQUIRE(m_kernel->get_kernel_type()==K_CUSTOM || m_p_and_q,
			"(%d, %d): No features set and no custom kernel in use!\n",
			num_samples, num_eigenvalues);

	index_t m=m_m;
	index_t n=0;

	/* check if kernel is precomputed (custom kernel) */
	if (m_kernel && m_kernel->get_kernel_type()==K_CUSTOM)
		n=m_kernel->get_num_vec_lhs()-m;
	else
	{
		REQUIRE(m_p_and_q, "The samples are not initialized!\n");
		n=m_p_and_q->get_num_vectors()-m;
	}

	if (num_samples<=2)
	{
		SG_ERROR("Number of samples has to be at least 2, "
				"better in the hundreds");
	}

	if (num_eigenvalues>m+n-1)
		SG_ERROR("Number of Eigenvalues too large\n");

	if (num_eigenvalues<1)
		SG_ERROR("Number of Eigenvalues too small\n");

	/* imaginary matrix K=[K KL; KL' L] (MATLAB notation)
	 * K is matrix for XX, L is matrix for YY, KL is XY, LK is YX
	 * works since X and Y are concatenated here */
	m_kernel->init(m_p_and_q, m_p_and_q);
	SGMatrix<float64_t> K=m_kernel->get_kernel_matrix();

	/* center matrix K=H*K*H */
	K.center();

	/* compute eigenvalues and select num_eigenvalues largest ones */
	Map<MatrixXd> c_kernel_matrix(K.matrix, K.num_rows, K.num_cols);
	SelfAdjointEigenSolver<MatrixXd> eigen_solver(c_kernel_matrix);
	REQUIRE(eigen_solver.info()==Eigen::Success,
			"Eigendecomposition failed!\n");
	index_t max_num_eigenvalues=eigen_solver.eigenvalues().rows();

	/* finally, sample from null distribution */
	SGVector<float64_t> null_samples(num_samples);
	for (index_t i=0; i<num_samples; ++i)
	{
		null_samples[i]=0;
		for (index_t j=0; j<num_eigenvalues; ++j)
		{
			float64_t z_j=CMath::randn_double();

			SG_DEBUG("z_j=%f\n", z_j);

			float64_t multiple=CMath::sq(z_j);

			/* take largest EV, scale by 1/(m+n) on the fly and take abs value*/
			float64_t eigenvalue_estimate=CMath::abs(1.0/(m+n)
				*eigen_solver.eigenvalues()[max_num_eigenvalues-1-j]);

			if (m_statistic_type==UNBIASED)
				multiple-=1;

			SG_DEBUG("multiple=%f, eigenvalue=%f\n", multiple,
					eigenvalue_estimate);

			null_samples[i]+=eigenvalue_estimate*multiple;
		}
	}

	return null_samples;
}

SGVector<float64_t> CQuadraticTimeMMD::sample_null_spectrum_DEPRECATED(
		index_t num_samples, index_t num_eigenvalues)
{
	REQUIRE(m_kernel, "(%d, %d): No kernel set!\n", num_samples,
			num_eigenvalues);
	REQUIRE(m_kernel->get_kernel_type()==K_CUSTOM || m_p_and_q,
			"(%d, %d): No features set and no custom kernel in use!\n",
			num_samples, num_eigenvalues);

	index_t m=m_m;
	index_t n=0;

	/* check if kernel is precomputed (custom kernel) */
	if (m_kernel && m_kernel->get_kernel_type()==K_CUSTOM)
		n=m_kernel->get_num_vec_lhs()-m;
	else
	{
		REQUIRE(m_p_and_q, "The samples are not initialized!\n");
		n=m_p_and_q->get_num_vectors()-m;
	}

	if (num_samples<=2)
	{
		SG_ERROR("Number of samples has to be at least 2, "
				"better in the hundreds");
	}

	if (num_eigenvalues>m+n-1)
		SG_ERROR("Number of Eigenvalues too large\n");

	if (num_eigenvalues<1)
		SG_ERROR("Number of Eigenvalues too small\n");

	/* imaginary matrix K=[K KL; KL' L] (MATLAB notation)
	 * K is matrix for XX, L is matrix for YY, KL is XY, LK is YX
	 * works since X and Y are concatenated here */
	m_kernel->init(m_p_and_q, m_p_and_q);
	SGMatrix<float64_t> K=m_kernel->get_kernel_matrix();

	/* center matrix K=H*K*H */
	K.center();

	/* compute eigenvalues and select num_eigenvalues largest ones */
	Map<MatrixXd> c_kernel_matrix(K.matrix, K.num_rows, K.num_cols);
	SelfAdjointEigenSolver<MatrixXd> eigen_solver(c_kernel_matrix);
	REQUIRE(eigen_solver.info()==Eigen::Success,
			"Eigendecomposition failed!\n");
	index_t max_num_eigenvalues=eigen_solver.eigenvalues().rows();

	/* precomputing terms with rho_x and rho_y of equation 10 in [1]
	 * (see documentation) */
	float64_t rho_x=float64_t(m)/(m+n);
	float64_t rho_y=1-rho_x;

	/* instead of using two Gaussian rv's ~ N(0,1), we'll use just one rv
	 * ~ N(0, 1/rho_x+1/rho_y) (derived from eq 10 in [1]) */
	float64_t std_dev=CMath::sqrt(1/rho_x+1/rho_y);
	float64_t inv_rho_x_y=1/(rho_x*rho_y);

	SG_DEBUG("Using Gaussian samples ~ N(0,%f)\n", std_dev*std_dev);

	/* finally, sample from null distribution */
	SGVector<float64_t> null_samples(num_samples);
	for (index_t i=0; i<num_samples; ++i)
	{
		null_samples[i]=0;
		for (index_t j=0; j<num_eigenvalues; ++j)
		{
			/* compute the right hand multiple of eq. 10 in [1] using one RV.
			 * randn_double() gives a sample from N(0,1), we need samples
			 * from N(0,1/rho_x+1/rho_y) */
			float64_t z_j=std_dev*CMath::randn_double();

			SG_DEBUG("z_j=%f\n", z_j);

			float64_t multiple=CMath::pow(z_j, 2);

			/* take largest EV, scale by 1/(m+n) on the fly and take abs value*/
			float64_t eigenvalue_estimate=CMath::abs(1.0/(m+n)
				*eigen_solver.eigenvalues()[max_num_eigenvalues-1-j]);

			if (m_statistic_type==UNBIASED_DEPRECATED)
				multiple-=inv_rho_x_y;

			SG_DEBUG("multiple=%f, eigenvalue=%f\n", multiple,
					eigenvalue_estimate);

			null_samples[i]+=eigenvalue_estimate*multiple;
		}
	}

	/* when m=n, return m*MMD^2 instead */
	if (m==n) {
		linalg::scale<linalg::Backend::NATIVE>(null_samples, 0.5);
	}

	return null_samples;
}
#endif // HAVE_EIGEN3

SGVector<float64_t> CQuadraticTimeMMD::fit_null_gamma()
{
	REQUIRE(m_kernel, "No kernel set!\n");
	REQUIRE(m_kernel->get_kernel_type()==K_CUSTOM || m_p_and_q,
			"No features set and no custom kernel in use!\n");

	index_t n=0;

	/* check if kernel is precomputed (custom kernel) */
	if (m_kernel && m_kernel->get_kernel_type()==K_CUSTOM)
		n=m_kernel->get_num_vec_lhs()-m_m;
	else
	{
		REQUIRE(m_p_and_q, "The samples are not initialized!\n");
		n=m_p_and_q->get_num_vectors()-m_m;
	}
	REQUIRE(m_m==n, "Only possible with equal number of samples "
			"from both distribution!\n")

	index_t num_data;
	if (m_kernel->get_kernel_type()==K_CUSTOM)
		num_data=m_kernel->get_num_vec_rhs();
	else
		num_data=m_p_and_q->get_num_vectors();

	if (m_m!=num_data/2)
		SG_ERROR("Currently, only equal sample sizes are supported\n");

	/* evtl. warn user not to use wrong statistic type */
	if (m_statistic_type!=BIASED_DEPRECATED)
	{
		SG_WARNING("Note: provided statistic has "
				"to be BIASED. Please ensure that! To get rid of warning,"
				"call %s::set_statistic_type(BIASED_DEPRECATED)\n", get_name());
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

void CQuadraticTimeMMD::set_num_samples_spectrum(index_t
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

