/*
 * Copyright (c) The Shogun Machine Learning Toolbox
 * Written (w) 2014 Wu Lin
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
 *
 */


#ifdef HAVE_EIGEN3

#include <shogun/machine/gp/LogitPiecewiseBoundLikelihood.h>


using namespace shogun;
using namespace Eigen;

CLogitPiecewiseBoundLikelihood::CLogitPiecewiseBoundLikelihood()
	: CLogitLikelihood()
{
	init();
}

CLogitPiecewiseBoundLikelihood::~CLogitPiecewiseBoundLikelihood()
{
}


void CLogitPiecewiseBoundLikelihood::set_bound(SGMatrix<float64_t> bound)
{
	m_bound = bound;
}

void CLogitPiecewiseBoundLikelihood::set_distribution(SGVector<float64_t> mu,
	SGVector<float64_t> s2, const CLabels* lab)
{
	REQUIRE(lab, "Labels are required (lab should not be NULL)\n");

	REQUIRE((mu.vlen==s2.vlen) && (mu.vlen==lab->get_num_labels()),
		"Length of the vector of means (%d), length of the vector of "
		"variances (%d) and number of labels (%d) should be the same\n",
		mu.vlen, s2.vlen, lab->get_num_labels());

	REQUIRE(lab->get_label_type()==LT_BINARY,
		"Labels must be type of CBinaryLabels\n");

	for(index_t i = 0; i < s2.vlen; ++i)
		ASSERT(s2[i] > 0.0);

	m_lab = ((CBinaryLabels*)lab)->get_labels();
	/** convert the input label to standard label used in the class
	 *
	 *  Note that Shogun uses  -1 and 1 as labels and this class uses 
	 *  0 and 1 repectively.
	 *
	 */
	for(index_t i = 0; i < m_lab.size(); ++i)
		m_lab[i] = CMath::max(m_lab[i], 0.0);

	m_mu = mu;

	m_s2 = s2;

	precompute();
}

void CLogitPiecewiseBoundLikelihood::init()
{
	SG_ADD(&m_bound, "bound", 
		"Variational piecewise bound for logit likelihood",
		MS_NOT_AVAILABLE);

	SG_ADD(&m_mu, "mu", 
		"The mean of variational normal distribution",
		MS_AVAILABLE, GRADIENT_AVAILABLE);

	SG_ADD(&m_s2, "sigma2", 
		"The variance of variational normal distribution",
		MS_AVAILABLE,GRADIENT_AVAILABLE);

	SG_ADD(&m_lab, "y", 
		"The data/labels (must be 0 or 1) drawn from the distribution",
		MS_NOT_AVAILABLE);

	SG_ADD(&m_pl, "pdf_l", 
		"The pdf given the lower range and parameters(mu and variance)",
		MS_NOT_AVAILABLE);
	SG_ADD(&m_ph, "pdf_h", 
		"The pdf given the higher range and parameters(mu and variance)",
		MS_NOT_AVAILABLE);
	SG_ADD(&m_cdf_diff, "cdf_h_minus_cdf_l", 
		"The CDF difference between the lower and higher range given the parameters(mu and variance)",
		MS_NOT_AVAILABLE);
	SG_ADD(&m_l2_plus_s2, "l2_plus_sigma2", 
		"The result of l^2 + sigma^2",
		MS_NOT_AVAILABLE);
	SG_ADD(&m_h2_plus_s2, "h2_plus_sigma2", 
		"The result of h^2 + sigma^2",
		MS_NOT_AVAILABLE);
	SG_ADD(&m_weighted_pdf_diff, "weighted_pdf_diff", 
		"The result of l*pdf(l_norm)-h*pdf(h_norm)",
		MS_NOT_AVAILABLE);
}
#endif /* HAVE_EIGEN3 */
