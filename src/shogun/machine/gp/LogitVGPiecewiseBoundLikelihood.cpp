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
 * Code adapted from 
 * https://github.com/emtiyaz/VariationalApproxExample
 * and the reference paper is
 * Marlin, Benjamin M., Mohammad Emtiyaz Khan, and Kevin P. Murphy.
 * "Piecewise Bounds for Estimating Bernoulli-Logistic Latent Gaussian Models." ICML. 2011.
 *
 * This code specifically adapted from ElogLik.m
 * and from the formula of the appendix
 * http://www.cs.ubc.ca/~emtiyaz/papers/truncatedGaussianMoments.pdf
 */

#ifdef HAVE_EIGEN3

#include <shogun/machine/gp/LogitVGPiecewiseBoundLikelihood.h>
#include <shogun/mathematics/Statistics.h>
#include <shogun/distributions/classical/GaussianDistribution.h>
#include <shogun/mathematics/Math.h>
#include <shogun/labels/BinaryLabels.h>

using namespace Eigen;

namespace shogun
{

CLogitVGPiecewiseBoundLikelihood::CLogitVGPiecewiseBoundLikelihood()
	: CVariationalGaussianLikelihood()
{
	init();
}

CLogitVGPiecewiseBoundLikelihood::~CLogitVGPiecewiseBoundLikelihood()
{
}

void CLogitVGPiecewiseBoundLikelihood::set_variational_bound(SGMatrix<float64_t> bound)
{
	m_bound = bound;
}

void CLogitVGPiecewiseBoundLikelihood::set_default_variational_bound()
{
	index_t row = 20;
	index_t col = 5;
	m_bound = SGMatrix<float64_t>(row, col);

	m_bound(0,0)=0.000188712193000;
	m_bound(1,0)=0.028090310300000;
	m_bound(2,0)=0.110211757000000;
	m_bound(3,0)=0.232736440000000;
	m_bound(4,0)=0.372524706000000;
	m_bound(5,0)=0.504567936000000;
	m_bound(6,0)=0.606280283000000;
	m_bound(7,0)=0.666125432000000;
	m_bound(8,0)=0.689334264000000;
	m_bound(9,0)=0.693147181000000;
	m_bound(10,0)=0.693147181000000;
	m_bound(11,0)=0.689334264000000;
	m_bound(12,0)=0.666125432000000;
	m_bound(13,0)=0.606280283000000;
	m_bound(14,0)=0.504567936000000;
	m_bound(15,0)=0.372524706000000;
	m_bound(16,0)=0.232736440000000;
	m_bound(17,0)=0.110211757000000;
	m_bound(18,0)=0.028090310400000;
	m_bound(19,0)=0.000188712000000;

	m_bound(0,1)=0;
	m_bound(1,1)=0.006648614600000;
	m_bound(2,1)=0.034432684600000;
	m_bound(3,1)=0.088701969900000;
	m_bound(4,1)=0.168024214000000;
	m_bound(5,1)=0.264032863000000;
	m_bound(6,1)=0.360755794000000;
	m_bound(7,1)=0.439094482000000;
	m_bound(8,1)=0.485091758000000;
	m_bound(9,1)=0.499419205000000;
	m_bound(10,1)=0.500580795000000;
	m_bound(11,1)=0.514908242000000;
	m_bound(12,1)=0.560905518000000;
	m_bound(13,1)=0.639244206000000;
	m_bound(14,1)=0.735967137000000;
	m_bound(15,1)=0.831975786000000;
	m_bound(16,1)=0.911298030000000;
	m_bound(17,1)=0.965567315000000;
	m_bound(18,1)=0.993351385000000;
	m_bound(19,1)=1.000000000000000;

	m_bound(0,2)=0;
	m_bound(1,2)=0.000397791059000;
	m_bound(2,2)=0.002753100850000;
	m_bound(3,2)=0.008770186980000;
	m_bound(4,2)=0.020034759300000;
	m_bound(5,2)=0.037511596000000;
	m_bound(6,2)=0.060543032900000;
	m_bound(7,2)=0.086256780600000;
	m_bound(8,2)=0.109213531000000;
	m_bound(9,2)=0.123026104000000;
	m_bound(10,2)=0.123026104000000;
	m_bound(11,2)=0.109213531000000;
	m_bound(12,2)=0.086256780600000;
	m_bound(13,2)=0.060543032900000;
	m_bound(14,2)=0.037511596000000;
	m_bound(15,2)=0.020034759300000;
	m_bound(16,2)=0.008770186980000;
	m_bound(17,2)=0.002753100850000;
	m_bound(18,2)=0.000397791059000;
	m_bound(19,2)=0;

	m_bound(0,3)=-CMath::INFTY;
	m_bound(1,3)=-8.575194939999999;
	m_bound(2,3)=-5.933689180000000;
	m_bound(3,3)=-4.525933600000000;
	m_bound(4,3)=-3.528107790000000;
	m_bound(5,3)=-2.751548540000000;
	m_bound(6,3)=-2.097898790000000;
	m_bound(7,3)=-1.519690830000000;
	m_bound(8,3)=-0.989533382000000;
	m_bound(9,3)=-0.491473077000000;
	m_bound(10,3)=0;
	m_bound(11,3)=0.491473077000000;
	m_bound(12,3)=0.989533382000000;
	m_bound(13,3)=1.519690830000000;
	m_bound(14,3)=2.097898790000000;
	m_bound(15,3)=2.751548540000000;
	m_bound(16,3)=3.528107790000000;
	m_bound(17,3)=4.525933600000000;
	m_bound(18,3)=5.933689180000000;
	m_bound(19,3)=8.575194939999999;

	m_bound(0,4)=-8.575194939999999;
	m_bound(1,4)=-5.933689180000000;
	m_bound(2,4)=-4.525933600000000;
	m_bound(3,4)=-3.528107790000000;
	m_bound(4,4)=-2.751548540000000;
	m_bound(5,4)=-2.097898790000000;
	m_bound(6,4)=-1.519690830000000;
	m_bound(7,4)=-0.989533382000000;
	m_bound(8,4)=-0.491473077000000;
	m_bound(9,4)=0;
	m_bound(10,4)=0.491473077000000;
	m_bound(11,4)=0.989533382000000;
	m_bound(12,4)=1.519690830000000;
	m_bound(13,4)=2.097898790000000;
	m_bound(14,4)=2.751548540000000;
	m_bound(15,4)=3.528107790000000;
	m_bound(16,4)=4.525933600000000;
	m_bound(17,4)=5.933689180000000;
	m_bound(18,4)=8.575194939999999;
	m_bound(19,4)= CMath::INFTY;
}

SGVector<float64_t> CLogitVGPiecewiseBoundLikelihood::get_variational_expection()
{
	//This function is based on the Matlab code,
	//function [f, gm, gv] = Ellp(m, v, bound, ind), to compute f
	//and the formula of the appendix
	
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


SGVector<float64_t> CLogitVGPiecewiseBoundLikelihood::get_variational_first_derivative(
		const TParameter* param) const
{
	//This function is based on the Matlab code
	//function [f, gm, gv] = Ellp(m, v, bound, ind), to compute gm and gv
	//and the formula of the appendix
	REQUIRE(param, "Param is required (param should not be NULL)\n");
	REQUIRE(param->m_name, "Param name is required (param->m_name should not be NULL)\n");
	//We take the derivative wrt to param. Only mu or sigma2 can be the param 
	REQUIRE(!(strcmp(param->m_name, "mu") && strcmp(param->m_name, "sigma2")),
		"Can't compute derivative of the variational expection ", 
		"of log LogitLikelihood using the piecewise bound ", 
		"wrt %s.%s parameter. The function only accepts mu and sigma2 as parameter",
		get_name(), param->m_name);

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
	const Map<MatrixXd> eigen_weighted_pdf_diff(m_weighted_pdf_diff.matrix, num_rows, num_cols);
	const Map<MatrixXd> eigen_h2_plus_s2(m_h2_plus_s2.matrix, num_rows, num_cols);
	const Map<MatrixXd> eigen_l2_plus_s2(m_l2_plus_s2.matrix, num_rows, num_cols);

	float64_t l_bak = eigen_l(0);
	//l(1) = 0; 
	eigen_l(0) = 0;
	float64_t h_bak = eigen_h(eigen_h.size()-1);
	//h(end) = 0; 
	eigen_h(eigen_h.size()-1) = 0;

	SGVector<float64_t> result(m_mu.vlen);

	if (strcmp(param->m_name, "mu") == 0)
	{
		//Compute the derivative wrt mu

		//bsxfun(@plus, bsxfun(@plus, l.^2, v), v).*pl - bsxfun(@plus, bsxfun(@plus, h.^2, v), v).*ph;
		MatrixXd eigen_dmu2 = ((eigen_l2_plus_s2.array().rowwise()+eigen_s2.array().transpose())*eigen_pl.array()
			- (eigen_h2_plus_s2.array().rowwise()+eigen_s2.array().transpose())*eigen_ph.array()).matrix();
		//bsxfun(@times, ch-cl, (2.0*mu))
		eigen_dmu2 += (eigen_cdf_diff.array().rowwise()*(2.0*eigen_mu).array().transpose()).matrix();

		SGVector<float64_t> & gmu = result;
		Map<VectorXd> eigen_gmu(gmu.vector, gmu.vlen);

		//gmu = bsxfun(@times, dmu2, _a) + bsxfun(@times, pl.*l - ph.*h + ch - chl, b) + bsxfun(@times, pl - ph, c)
		//gmu = sum(gmu,1)
		eigen_gmu = ((eigen_dmu2.array().colwise()*eigen_a.array()) 
			+ ((eigen_weighted_pdf_diff + eigen_cdf_diff).array().colwise()*eigen_b.array())
			+ ( (eigen_pl - eigen_ph).array().colwise()*eigen_c.array())).colwise().sum().matrix();

		Map<VectorXd>eigen_lab(m_lab.vector, m_lab.vlen);
		eigen_gmu = eigen_lab - eigen_gmu;
	}
	else
	{
		//Compute the derivative wrt sigma2

		//gv_0 = bsxfun(@plus, l, -mu).*pl - bsxfun(@plus, h, -mu).*ph;
		MatrixXd eigen_gs2_0 = (((-eigen_mu).replicate(1,eigen_l.rows()).array().transpose().colwise() + eigen_l.array())*eigen_pl.array()
			- ((-eigen_mu).replicate(1,eigen_h.rows()).array().transpose().colwise() + eigen_h.array())*eigen_ph.array()).matrix();
		//gv_0 = bsxfun(times, gv_0, c);
		eigen_gs2_0 = (eigen_gs2_0.array().colwise()*eigen_c.array()).matrix();

		//tmpl = l2_plus_v - bsxfun(@times, l, mu)
		MatrixXd tmpl = (eigen_l2_plus_s2 - (eigen_mu.replicate(1,eigen_l.rows()).array().transpose().colwise()*eigen_l.array()).matrix()
			).cwiseProduct(eigen_pl);

		//tmph = h2_plus_v - bsxfun(@times, h, mu)
		MatrixXd tmph = (eigen_h2_plus_s2 - (eigen_mu.replicate(1,eigen_h.rows()).array().transpose().colwise()*eigen_h.array()).matrix()
			).cwiseProduct(eigen_ph);

		//gv_1 = bsxfun(@times, tmpl - tmph, b);
		MatrixXd eigen_gs2_1 = ((tmpl - tmph).array().colwise()*eigen_b.array()).matrix();

		//gv_2 = bsxfun(@times, tmpl, l) - bsxfun(@times, tmph, h);
		MatrixXd eigen_gs2_2 = (tmpl.array().colwise()*eigen_l.array() - tmph.array().colwise()*eigen_h.array()).matrix();

		//gv_2 = bsxfun(@times, gv_2, a);
		eigen_gs2_2 = (eigen_gs2_2.array().colwise()*eigen_a.array()).matrix();

		SGVector<float64_t> & gs2 = result;
		Map<VectorXd> eigen_gs2(gs2.vector, gs2.vlen);

		//gv = (bsxfun(@times, ch - cl + 0.5*pl.*l - ph.*h, a) + bsxfun(@times, gv_0 + gv_1 + gv_2, 1.0/(2.0*v))
		//gv = sum(gv,1);
		eigen_gs2 = ((eigen_cdf_diff + 0.5*eigen_weighted_pdf_diff).array().colwise()*eigen_a.array()
			+ (eigen_gs2_0 + eigen_gs2_1 + eigen_gs2_2).array().rowwise()/(2.0*eigen_s2).array().transpose()
			).colwise().sum().matrix();

		//gv = -gv
		eigen_gs2 = -eigen_gs2;
	}
	eigen_l(0) = l_bak;
	eigen_h(eigen_h.size()-1) = h_bak;

	return result;
}

void CLogitVGPiecewiseBoundLikelihood::set_variational_distribution(SGVector<float64_t> mu,
	SGVector<float64_t> s2, const CLabels* lab)
{
	CVariationalGaussianLikelihood::set_variational_distribution(mu, s2, lab);

	REQUIRE(lab->get_label_type()==LT_BINARY,
		"Labels must be type of CBinaryLabels\n");

	m_lab = (((CBinaryLabels*)lab)->get_labels()).clone();

	//Convert the input label to standard label used in the class
	//Note that Shogun uses  -1 and 1 as labels and this class internally uses 
	//0 and 1 repectively.
	for(index_t i = 0; i < m_lab.size(); ++i)
		m_lab[i] = CMath::max(m_lab[i], 0.0);

	precompute();
}

void CLogitVGPiecewiseBoundLikelihood::init_likelihood()
{
	set_likelihood(new CLogitLikelihood());
}

void CLogitVGPiecewiseBoundLikelihood::init()
{
	SG_ADD(&m_bound, "bound", 
		"Variational piecewise bound for logit likelihood",
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

	init_likelihood();
}

void CLogitVGPiecewiseBoundLikelihood::precompute()
{
	//This function is based on the Matlab code
	//function [f, gm, gv] = Ellp(m, v, bound, ind), to compute common variables later
	//used in get_variational_expection and get_variational_first_derivative
	
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

	//Usually we use pdf in log-domain and the log_sum_exp trick
	//to avoid numerical underflow in particular for IID samples
	for (index_t r = 0; r < zl.num_rows; r++)
	{
		for (index_t c = 0; c < zl.num_cols; c++)
		{
			if (CMath::abs(zl(r, c)) == CMath::INFTY)
				m_pl(r, c) = 0;
			else
				m_pl(r, c) = CMath::exp(CGaussianDistribution::univariate_log_pdf(zl(r, c)));

			if (CMath::abs(zh(r, c)) == CMath::INFTY)
				m_ph(r, c) = 0;
			else
				m_ph(r, c) = CMath::exp(CGaussianDistribution::univariate_log_pdf(zh(r, c)));
		}
	}

	//pl = bsxfun(@times, normpdf(zl), 1./sqrt(v));
	eigen_pl = (eigen_pl.array().rowwise()*eigen_s_inv.array().transpose()).matrix();
	//ph = bsxfun(@times, normpdf(zh), 1./sqrt(v));
	eigen_ph = (eigen_ph.array().rowwise()*eigen_s_inv.array().transpose()).matrix();

	SGMatrix<float64_t> & cl = zl; 
	SGMatrix<float64_t> & ch = zh;

	for (index_t r = 0; r < zl.num_rows; r++)
	{
		for (index_t c = 0; c < zl.num_cols; c++)
		{
			//cl = 0.5*erf(zl/sqrt(2)); %normal cdf -const
			cl(r, c) = CStatistics::normal_cdf(zl(r, c)) - 0.5;
			//ch = 0.5*erf(zl/sqrt(2)); %normal cdf -const
			ch(r, c) = CStatistics::normal_cdf(zh(r, c)) - 0.5;
		}
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
	eigen_weighted_pdf_diff = (eigen_pl.array().colwise() * eigen_l.array() - eigen_ph.array().colwise() * eigen_h.array()).matrix();

	eigen_l(0) = l_bak;
	eigen_h(eigen_h.size()-1) = h_bak;
}

SGVector<float64_t> CLogitVGPiecewiseBoundLikelihood::get_first_derivative_wrt_hyperparameter(
	const TParameter* param) const
{
	return SGVector<float64_t>();
}

} /* namespace shogun */
#endif /* HAVE_EIGEN3 */
