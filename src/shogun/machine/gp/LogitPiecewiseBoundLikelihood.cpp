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

SGVector<float64_t> CLogitPiecewiseBoundLikelihood::get_variational_expection()
{
	const Map<VectorXd> eigen_c(m_bound.get_column_vector(0), m_bound.num_rows);
	const Map<VectorXd> eigen_b(m_bound.get_column_vector(1), m_bound.num_rows);
	const Map<VectorXd> eigen_a(m_bound.get_column_vector(2), m_bound.num_rows);
	const Map<VectorXd> eigen_mu(m_mu.vector, m_mu.vlen);
	const Map<VectorXd> eigen_s2(m_s2.vector, m_s2.vlen);
	Map<VectorXd> eigen_l(m_bound.get_column_vector(3), m_bound.num_rows);
	Map<VectorXd> eigen_h(m_bound.get_column_vector(4), m_bound.num_rows);

	index_t num_rows = m_bound.num_rows; 
	index_t num_cols = m_mu.vlen;

	const Map<MatrixXd> eigen_cdf_diff(m_cdf_diff.matrix, num_rows, num_cols);
	const Map<MatrixXd> eigen_pl(m_pl.matrix, num_rows, num_cols);
	const Map<MatrixXd> eigen_ph(m_ph.matrix, num_rows, num_cols);

	float64_t l_bak = eigen_l(0);
	//l(1) = 0; 
	eigen_l(0) = 0;

	float64_t h_bak = eigen_h(eigen_h.size()-1);
	//h(end) = 0; 
	eigen_h(eigen_h.size()-1) = 0;

	//ex0 = ch-cl;
	const Map<MatrixXd> & eigen_ex0 = eigen_cdf_diff;

	//%ex1= v.*(pl-ph) + m.*(ch-cl);
	MatrixXd eigen_ex1 = ((eigen_pl - eigen_ph).array().rowwise()*eigen_s2.array().transpose() 
		+ eigen_cdf_diff.array().rowwise()*eigen_mu.array().transpose()).matrix();

	//ex2 = bsxfun(@times, v, (bsxfun(@plus, l, m)).*pl - (bsxfun(@plus, h, m)).*ph) + bsxfun(@times, (v+m.^2), ex0);
	
	//bsxfun(@plus, l, m)).*pl - (bsxfun(@plus, h, m)).*ph
	MatrixXd eigen_ex2 = ((eigen_mu.replicate(1,eigen_l.rows()).array().transpose().colwise() + eigen_l.array())*eigen_pl.array()
		- (eigen_mu.replicate(1,eigen_h.rows()).array().transpose().colwise() + eigen_h.array())*eigen_ph.array()).matrix();

	//bsxfun(@times, v, (bsxfun(@plus, l, m).*pl - bsxfun(@plus, h, m).*ph
	eigen_ex2 = (eigen_ex2.array().rowwise()*eigen_s2.array().transpose()).matrix();

	//ex2 = bsxfun(@times, v, (bsxfun(@plus, l, m)).*pl - (bsxfun(@plus, h, m)).*ph) + bsxfun(@times, (v+m.^2), ex0);
	eigen_ex2 += (eigen_cdf_diff.array().rowwise()*(eigen_mu.array().pow(2)
			+ eigen_s2.array()).transpose()).matrix();

	SGVector<float64_t> f(m_mu.vlen);
	Map<VectorXd> eigen_f(f.vector, f.vlen);

	//%f = sum((a.*ex2 + b.*ex1 + c.*ex0),1);
	eigen_f = (eigen_ex2.array().colwise()*eigen_a.array() 
		+ eigen_ex1.array().colwise()*eigen_b.array() 
		+ eigen_ex0.array().colwise()*eigen_c.array()).colwise().sum().matrix();


	eigen_l(0) = l_bak;
	eigen_h(eigen_h.size()-1) = h_bak;

	Map<VectorXd>eigen_lab(m_lab.vector, m_lab.vlen);
	eigen_f = eigen_lab.cwiseProduct(eigen_mu) - eigen_f; 
	return f;
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

void CLogitPiecewiseBoundLikelihood::precompute()
{
	const Map<VectorXd> eigen_c(m_bound.get_column_vector(0), m_bound.num_rows);
	const Map<VectorXd> eigen_b(m_bound.get_column_vector(1), m_bound.num_rows);
	const Map<VectorXd> eigen_a(m_bound.get_column_vector(2), m_bound.num_rows);
	const Map<VectorXd> eigen_mu(m_mu.vector, m_mu.vlen);
	const Map<VectorXd> eigen_s2(m_s2.vector, m_s2.vlen);
	Map<VectorXd> eigen_l(m_bound.get_column_vector(3), m_bound.num_rows);
	Map<VectorXd> eigen_h(m_bound.get_column_vector(4), m_bound.num_rows);

	index_t num_rows = m_bound.num_rows; 
	index_t num_cols = m_mu.vlen;

	m_pl = SGMatrix<float64_t>(num_rows,num_cols);
	m_ph = SGMatrix<float64_t>(num_rows,num_cols);
	m_cdf_diff = SGMatrix<float64_t>(num_rows,num_cols);
	m_l2_plus_s2 = SGMatrix<float64_t>(num_rows,num_cols);
	m_h2_plus_s2 = SGMatrix<float64_t>(num_rows,num_cols);
	m_weighted_pdf_diff = SGMatrix<float64_t>(num_rows,num_cols);

	Map<MatrixXd> eigen_pl(m_pl.matrix, num_rows, num_cols);

	Map<MatrixXd> eigen_ph(m_ph.matrix, num_rows, num_cols);

	Map<MatrixXd> eigen_cdf_diff(m_cdf_diff.matrix, num_rows, num_cols);

	Map<MatrixXd> eigen_l2_plus_s2(m_l2_plus_s2.matrix, num_rows, num_cols);

	Map<MatrixXd> eigen_h2_plus_s2(m_h2_plus_s2.matrix, num_rows, num_cols);

	Map<MatrixXd> eigen_weighted_pdf_diff(m_weighted_pdf_diff.matrix, num_rows, num_cols);


	SGMatrix<float64_t> zl(num_rows, num_cols);
	Map<MatrixXd> eigen_zl(zl.matrix, num_rows, num_cols);

	SGMatrix<float64_t> zh(num_rows, num_cols);
	Map<MatrixXd> eigen_zh(zh.matrix, num_rows, num_cols);

	//bsxfun(@minus,l,m)
	eigen_zl = ((-eigen_mu).replicate(1,eigen_l.rows()).array().transpose().colwise() + eigen_l.array()).matrix();
	//bsxfun(@minus,h,m)
	eigen_zh = ((-eigen_mu).replicate(1,eigen_h.rows()).array().transpose().colwise() + eigen_h.array()).matrix();

	VectorXd eigen_s_inv = eigen_s2.array().sqrt().inverse().matrix(); 

	//zl = bsxfun(@times, bsxfun(@minus,l,m), 1./sqrt(v))
	eigen_zl = (eigen_zl.array().rowwise()*eigen_s_inv.array().transpose()).matrix();
	//zh = bsxfun(@times, bsxfun(@minus,h,m), 1./sqrt(v))
	eigen_zh = (eigen_zh.array().rowwise()*eigen_s_inv.array().transpose()).matrix();


	for (index_t r = 0; r < zl.num_rows; r++)
		for (index_t c = 0; c < zl.num_cols; c++)
		{
			if (zl(r, c) == CMath::INFTY || zl(r, c) == -CMath::INFTY)
				m_pl(r, c) = 0;
			else
				m_pl(r, c) = CMath::exp(CGaussianDistribution::univariate_log_pdf(zl(r, c)));
			if (zh(r, c) == CMath::INFTY || zh(r, c) == -CMath::INFTY)
				m_ph(r, c) = 0;
			else
				m_ph(r, c) = CMath::exp(CGaussianDistribution::univariate_log_pdf(zh(r, c)));
		}

	//pl = bsxfun(@times, normpdf(zl), 1./sqrt(v));
	eigen_pl = (eigen_pl.array().rowwise()*eigen_s_inv.array().transpose()).matrix();
	//ph = bsxfun(@times, normpdf(zh), 1./sqrt(v));
	eigen_ph = (eigen_ph.array().rowwise()*eigen_s_inv.array().transpose()).matrix();

	SGMatrix<float64_t> & cl = zl; 
	SGMatrix<float64_t> & ch = zh;

	for (index_t r = 0; r < zl.num_rows; r++)
		for (index_t c = 0; c < zl.num_cols; c++)
		{
			//cl = 0.5*erf(zl/sqrt(2)); %normal cdf -const
			cl(r, c) = CStatistics::normal_cdf(zl(r, c)) - 0.5;
			//ch = 0.5*erf(zl/sqrt(2)); %normal cdf -const
			ch(r, c) = CStatistics::normal_cdf(zh(r, c)) - 0.5;
		}

	Map<MatrixXd> eigen_cl(cl.matrix, num_rows, num_cols);
	Map<MatrixXd> eigen_ch(ch.matrix, num_rows, num_cols);

	eigen_cdf_diff = eigen_ch - eigen_cl;

	float64_t l_bak = eigen_l(0);
	eigen_l(0) = 0;

	float64_t h_bak = eigen_h(eigen_h.size()-1);
	eigen_h(eigen_h.size()-1) = 0;

	//bsxfun(@plus, l.^2, v)
	eigen_l2_plus_s2 = (eigen_s2.replicate(1,eigen_l.rows()).array().transpose().colwise() + eigen_l.array().pow(2)).matrix();
	//bsxfun(@plus, h.^2, v)
	eigen_h2_plus_s2 = (eigen_s2.replicate(1,eigen_h.rows()).array().transpose().colwise() + eigen_h.array().pow(2)).matrix();
	//pl.*l - ph.*h
	eigen_weighted_pdf_diff = (eigen_pl.array().colwise()*eigen_l.array() - eigen_ph.array().colwise()*eigen_h.array()).matrix();

	eigen_l(0) = l_bak;
	eigen_h(eigen_h.size()-1) = h_bak;
}

#endif /* HAVE_EIGEN3 */
