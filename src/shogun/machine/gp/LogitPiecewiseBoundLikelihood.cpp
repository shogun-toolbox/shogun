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

MatrixXd CLogitPiecewiseBoundLikelihood::my_bsxfun_vec(MyBsxfunOp op, const MatrixXd & x,
	const VectorXd & y, bool is_col_vec)
{

	ASSERT(op == plus || op == times);
	switch(op)
	{
	case plus:
		if (is_col_vec)
			return (x.array().colwise()+y.array()).matrix();
		return (x.array().rowwise()+y.array().transpose()).matrix();
	case times:
		if (is_col_vec)
			return (x.array().colwise()*y.array()).matrix();
		return (x.array().rowwise()*y.array().transpose()).matrix();
	}
	return x;

}

template<typename M1, typename M2>
MatrixXd CLogitPiecewiseBoundLikelihood::my_bsxfun(MyBsxfunOp op, const MatrixBase<M1> & x,
	const MatrixBase<M2> & y)
{

	ASSERT(op == plus || op == times);
	ASSERT((x.rows()==y.rows() || x.rows()==1 || y.rows()==1) && (x.cols()==y.cols() || x.cols()==1 || y.cols()==1));
	if ((x.rows() > 1 && x.cols() >1) && (y.rows() == 1 || y.cols() ==1))
	{
		if (y.rows() >1)
			// matrix_op_col_vec
			return my_bsxfun_vec(op, x, y, true);
		// matrix_op_row_vec
		return my_bsxfun_vec(op, x, y, false);
	}
	else if ((y.rows() > 1 && y.cols() > 1) && (x.rows() ==1 || x.cols() ==1))
	{
		if (y.rows() >1)
			// col_vec_op_matrix
			return my_bsxfun_vec(op, y, x, true);
		// row_vec_op_matrix
		return my_bsxfun_vec(op, y, x, false);
	}
	else if (x.rows() != y.rows() || x.cols() != y.cols())
	{
		if(x.rows()==1)
			// row_vec_op_col_vec
			return my_bsxfun_vec(op, x.replicate(y.rows(), 1), y, true);
		else if(x.cols()==1)
			// col_vec_op_row_vec
			return my_bsxfun_vec(op, y.replicate(x.rows(), 1), x, true);
	}
	switch(op)
	{
	case plus:
		return x+y;
	case times:
		return (x.array() * y.array()).matrix();
	}
	return x;
}

float64_t CLogitPiecewiseBoundLikelihood::_standard_norm_pdf(float64_t x)
{
	if(x == CMath::INFTY || x == -CMath::INFTY)
		return 0.0;
	else
		return CMath::exp(-0.5*(x*x + CMath::log(2.0*CMath::PI)));
}

template<typename M1>
Eigen::MatrixXd CLogitPiecewiseBoundLikelihood::standard_norm_pdf(const Eigen::MatrixBase<M1> &x)
{
	return x.unaryExpr(std::ptr_fun(CLogitPiecewiseBoundLikelihood::_standard_norm_pdf));
}

float64_t CLogitPiecewiseBoundLikelihood::_norm_cdf_minus_const(float64_t x)
{
	return CStatistics::normal_cdf(x) - 0.5;
}

template<typename M1>
Eigen::MatrixXd CLogitPiecewiseBoundLikelihood::normal_cdf_minus_const(const Eigen::MatrixBase<M1> &x)
{
	return x.unaryExpr(std::ptr_fun(CLogitPiecewiseBoundLikelihood::_norm_cdf_minus_const));
}

CLogitPiecewiseBoundLikelihood::CLogitPiecewiseBoundLikelihood()
	: CLogitLikelihood()
{
	init();
}

CLogitPiecewiseBoundLikelihood::~CLogitPiecewiseBoundLikelihood()
{
	clean();
}

void CLogitPiecewiseBoundLikelihood::clean()
{
	SG_UNREF((SGRefObject *&)m_bound);
	SG_UNREF((SGRefObject *&)m_mu);
	SG_UNREF((SGRefObject *&)m_s2);
	SG_UNREF((SGRefObject *&)m_lab);
	SG_UNREF((SGRefObject *&)m_pl);
	SG_UNREF((SGRefObject *&)m_ph);
	SG_UNREF((SGRefObject *&)m_cdf_diff);
	SG_UNREF((SGRefObject *&)m_l2_plus_s2);
	SG_UNREF((SGRefObject *&)m_h2_plus_s2);
	SG_UNREF((SGRefObject *&)m_weighted_pdf_diff);
}

void CLogitPiecewiseBoundLikelihood::set_bound(SGMatrix<float64_t> bound)
{
	*m_bound = bound;
}

SGVector<float64_t> CLogitPiecewiseBoundLikelihood::get_variational_expection()
{
	Map<VectorXd> eigen_c(m_bound->get_column_vector(0), m_bound->num_rows);
	Map<VectorXd> eigen_b(m_bound->get_column_vector(1), m_bound->num_rows);
	Map<VectorXd> eigen_a(m_bound->get_column_vector(2), m_bound->num_rows);
	Map<VectorXd> eigen_l(m_bound->get_column_vector(3), m_bound->num_rows);
	Map<VectorXd> eigen_h(m_bound->get_column_vector(4), m_bound->num_rows);
	Map<VectorXd> eigen_mu(m_mu->vector, m_mu->vlen);
	Map<VectorXd> eigen_s2(m_s2->vector, m_s2->vlen);

	index_t num_rows = m_bound->num_rows; 
	index_t num_cols = m_mu->vlen;

	Map<Eigen::MatrixXd> eigen_cdf_diff(m_cdf_diff->matrix, num_rows, num_cols);
	Map<Eigen::MatrixXd> eigen_pl(m_pl->matrix, num_rows, num_cols);
	Map<Eigen::MatrixXd> eigen_ph(m_ph->matrix, num_rows, num_cols);

	float64_t l_bak = eigen_l(0);
	eigen_l(0) = 0;

	float64_t h_bak = eigen_h(eigen_h.size()-1);
	eigen_h(eigen_h.size()-1) = 0;

	//ex0 = ch-cl;
	Map<MatrixXd> & eigen_ex0 = eigen_cdf_diff;

	//%ex1= v.*(pl-ph) + m.*(ch-cl);
	MatrixXd eigen_ex1 = my_bsxfun(times, eigen_pl - eigen_ph, eigen_s2.transpose()) 
		+ my_bsxfun(times, eigen_cdf_diff, eigen_mu.transpose());

	//ex2 = bsxfun(@times, v, (bsxfun(@plus, l, m)).*pl - (bsxfun(@plus, h, m)).*ph) + bsxfun(@times, (v+m.^2), ex0);
	MatrixXd eigen_ex2 = my_bsxfun(plus, eigen_l, eigen_mu.transpose()).cwiseProduct(eigen_pl)
		- my_bsxfun(plus, eigen_h, eigen_mu.transpose()).cwiseProduct(eigen_ph);
	eigen_ex2 = my_bsxfun(times, eigen_ex2, eigen_s2.transpose());
	eigen_ex2 += my_bsxfun(times, eigen_cdf_diff, (eigen_mu.array().pow(2)
			+ eigen_s2.array()).matrix().transpose());


	SGVector<float64_t> f(m_mu->vlen);
	Map<VectorXd> eigen_f(f.vector, f.vlen);

	//%f = sum((a.*ex2 + b.*ex1 + c.*ex0),1);
	eigen_f = (my_bsxfun(times, eigen_ex2, eigen_a) 
		+ my_bsxfun(times, eigen_ex1, eigen_b) 
		+ my_bsxfun(times, eigen_ex0, eigen_c)).colwise().sum();


	eigen_l(0) = l_bak;
	eigen_h(eigen_h.size()-1) = h_bak;

	Map<VectorXd>eigen_lab(m_lab->vector, m_lab->vlen);
	eigen_f = eigen_lab.cwiseProduct(eigen_mu) - eigen_f; 
	return f;
}


SGVector<float64_t> CLogitPiecewiseBoundLikelihood::get_variational_first_derivative(
		const TParameter* param) const
{
	REQUIRE(param, "Param name is required (param should not be NULL)\n");
	REQUIRE(!(strcmp(param->m_name, "mu") && strcmp(param->m_name, "sigma2")),
		"Can't compute derivative of the variational expection", 
		"of log LogitLikelihood using the piecewise bound", 
		"wrt %s.%s parameter",
		get_name(), param->m_name);

	Map<VectorXd> eigen_c(m_bound->get_column_vector(0), m_bound->num_rows);
	Map<VectorXd> eigen_b(m_bound->get_column_vector(1), m_bound->num_rows);
	Map<VectorXd> eigen_a(m_bound->get_column_vector(2), m_bound->num_rows);
	Map<VectorXd> eigen_l(m_bound->get_column_vector(3), m_bound->num_rows);
	Map<VectorXd> eigen_h(m_bound->get_column_vector(4), m_bound->num_rows);
	Map<VectorXd> eigen_mu(m_mu->vector, m_mu->vlen);
	Map<VectorXd> eigen_s2(m_s2->vector, m_s2->vlen);
	index_t num_rows = m_bound->num_rows; 
	index_t num_cols = m_mu->vlen;
	Map<Eigen::MatrixXd> eigen_cdf_diff(m_cdf_diff->matrix, num_rows, num_cols);
	Map<Eigen::MatrixXd> eigen_pl(m_pl->matrix, num_rows, num_cols);
	Map<Eigen::MatrixXd> eigen_ph(m_ph->matrix, num_rows, num_cols);
	Map<Eigen::MatrixXd> eigen_weighted_pdf_diff(m_weighted_pdf_diff->matrix, num_rows, num_cols);
	Map<Eigen::MatrixXd> eigen_h2_plus_s2(m_h2_plus_s2->matrix, num_rows, num_cols);
	Map<Eigen::MatrixXd> eigen_l2_plus_s2(m_l2_plus_s2->matrix, num_rows, num_cols);
	float64_t l_bak = eigen_l(0);
	eigen_l(0) = 0;
	float64_t h_bak = eigen_h(eigen_h.size()-1);
	eigen_h(eigen_h.size()-1) = 0;

	SGVector<float64_t> result(m_mu->vlen);
	if (strcmp(param->m_name, "mu") ==0)
	{
		MatrixXd eigen_dmu2 = my_bsxfun(plus, eigen_l2_plus_s2, eigen_s2.transpose()).cwiseProduct(eigen_pl)
			- my_bsxfun(plus, eigen_h2_plus_s2, eigen_s2.transpose()).cwiseProduct(eigen_ph);
		eigen_dmu2 += my_bsxfun(times, eigen_cdf_diff, (2.0*eigen_mu).transpose());

		SGVector<float64_t> & gmu = result;
		Map<VectorXd> eigen_gmu(gmu.vector, gmu.vlen);
		eigen_gmu = (my_bsxfun(times, eigen_dmu2, eigen_a) 
			+ my_bsxfun(times, eigen_weighted_pdf_diff + eigen_cdf_diff, eigen_b)
			+ my_bsxfun(times, eigen_pl - eigen_ph, eigen_c)).colwise().sum();

		Map<VectorXd>eigen_lab(m_lab->vector, m_lab->vlen);
		eigen_gmu = eigen_lab - eigen_gmu;
	}
	else
	{
		MatrixXd eigen_gs2_0 = my_bsxfun(plus, eigen_l, (-eigen_mu).transpose()).cwiseProduct(eigen_pl) 
			- (my_bsxfun(plus, eigen_h, (-eigen_mu).transpose())).cwiseProduct(eigen_ph);
		eigen_gs2_0 = my_bsxfun(times, eigen_gs2_0, eigen_c);


		MatrixXd tmpl = (eigen_l2_plus_s2 - my_bsxfun(times, eigen_l, eigen_mu.transpose())
			).cwiseProduct(eigen_pl);
		MatrixXd tmph = (eigen_h2_plus_s2 - my_bsxfun(times, eigen_h, eigen_mu.transpose())
			).cwiseProduct(eigen_ph);

		MatrixXd eigen_gs2_1 = my_bsxfun(times, tmpl - tmph, eigen_b);

		MatrixXd eigen_gs2_2 = my_bsxfun(times, tmpl, eigen_l) - my_bsxfun(times, tmph, eigen_h);

		eigen_gs2_2 = my_bsxfun(times, eigen_gs2_2, eigen_a);

		SGVector<float64_t> & gs2 = result;
		Map<VectorXd> eigen_gs2(gs2.vector, gs2.vlen);

		eigen_gs2 = (my_bsxfun(times, eigen_cdf_diff + 0.5*eigen_weighted_pdf_diff, eigen_a)
			+ my_bsxfun(times, eigen_gs2_0 + eigen_gs2_1
				+ eigen_gs2_2, (2.0*eigen_s2).array().inverse().matrix().transpose())
			).colwise().sum();

		eigen_gs2 = - eigen_gs2;

	}
	eigen_l(0) = l_bak;
	eigen_h(eigen_h.size()-1) = h_bak;


	return result;
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

	
	Map<VectorXd> eigen_s2(s2.vector, s2.vlen);
	eigen_s2.unaryExpr(std::ptr_fun(CLogitPiecewiseBoundLikelihood::_check_variance));

	*m_lab = ((CBinaryLabels*)lab)->get_labels();
	Map<VectorXd>eigen_lab(m_lab->vector, m_lab->vlen);
	eigen_lab = eigen_lab.unaryExpr(std::ptr_fun(CLogitPiecewiseBoundLikelihood::_convert_label));


	*m_mu = mu;

	*m_s2 = s2;

	precompute();
}

void CLogitPiecewiseBoundLikelihood::init()
{

	m_bound = new SGMatrix<float64_t>();
	SG_REF((SGRefObject *)m_bound);
	SG_ADD((CSGObject**)&m_bound, "bound", 
		"Variational piecewise bound for logit likelihood",
		MS_NOT_AVAILABLE);
	m_mu = new SGVector<float64_t>();
	SG_REF((SGRefObject *)m_mu);
	SG_ADD((CSGObject**)&m_mu, "mu", 
		"The mean of variational normal distribution",
		MS_AVAILABLE, GRADIENT_AVAILABLE);
	m_s2 =  new SGVector<float64_t>();
	SG_REF((SGRefObject *)m_s2);
	SG_ADD((CSGObject**)&m_s2, "sigma2", 
		"The variance of variational normal distribution",
		MS_AVAILABLE,GRADIENT_AVAILABLE);

	m_lab =  new SGVector<float64_t>();
	SG_REF((SGRefObject *)m_lab);
	SG_ADD((CSGObject**)&m_lab, "y", 
		"The data/labels (must be 0 or 1) drawn from the distribution",
		MS_NOT_AVAILABLE);

	m_pl = NULL;
	SG_REF((SGRefObject *)m_pl);
	SG_ADD((CSGObject**)&m_pl, "pdf_l", 
		"The pdf given the lower range and parameters(mu and variance)",
		MS_NOT_AVAILABLE);

	m_ph = NULL;
	SG_REF((SGRefObject *)m_ph);
	SG_ADD((CSGObject**)&m_ph, "pdf_h", 
		"The pdf given the higher range and parameters(mu and variance)",
		MS_NOT_AVAILABLE);

	m_cdf_diff = NULL;
	SG_REF((SGRefObject *)m_cdf_diff);
	SG_ADD((CSGObject**)&m_cdf_diff, "cdf_h_minus_cdf_l", 
		"The CDF difference between the lower and higher range given the parameters(mu and variance)",
		MS_NOT_AVAILABLE);

	m_l2_plus_s2 = NULL;
	SG_REF((SGRefObject *)m_l2_plus_s2);
	SG_ADD((CSGObject**)&m_l2_plus_s2, "l2_plus_sigma2", 
		"The result of l^2 + sigma^2",
		MS_NOT_AVAILABLE);

	m_h2_plus_s2 = NULL;
	SG_REF((SGRefObject *)m_h2_plus_s2);
	SG_ADD((CSGObject**)&m_h2_plus_s2, "h2_plus_sigma2", 
		"The result of h^2 + sigma^2",
		MS_NOT_AVAILABLE);

	m_weighted_pdf_diff = NULL;
	SG_REF((SGRefObject *)m_weighted_pdf_diff);
	SG_ADD((CSGObject**)&m_weighted_pdf_diff, "weighted_pdf_diff", 
		"The result of l*pdf(l_norm)-h*pdf(h_norm)",
		MS_NOT_AVAILABLE);
}
void CLogitPiecewiseBoundLikelihood::init_matrix(SGMatrix<float64_t> ** ptr, index_t num_rows, index_t num_cols)
{
	SG_UNREF((SGRefObject *&)(*ptr));
	(*ptr) = new SGMatrix<float64_t>(num_rows, num_cols);
	SG_REF((SGRefObject *)(*ptr));
}

void CLogitPiecewiseBoundLikelihood::precompute()
{
	Map<VectorXd> eigen_c(m_bound->get_column_vector(0), m_bound->num_rows);
	Map<VectorXd> eigen_b(m_bound->get_column_vector(1), m_bound->num_rows);
	Map<VectorXd> eigen_a(m_bound->get_column_vector(2), m_bound->num_rows);
	Map<VectorXd> eigen_l(m_bound->get_column_vector(3), m_bound->num_rows);
	Map<VectorXd> eigen_h(m_bound->get_column_vector(4), m_bound->num_rows);

	index_t num_rows = m_bound->num_rows; 
	index_t num_cols = m_mu->vlen;

	init_matrix(&m_pl, num_rows, num_cols);
	Map<Eigen::MatrixXd> eigen_pl(m_pl->matrix, num_rows, num_cols);

	init_matrix(&m_ph, num_rows, num_cols);
	Map<Eigen::MatrixXd> eigen_ph(m_ph->matrix, num_rows, num_cols);

	init_matrix(&m_cdf_diff, num_rows, num_cols);
	Map<Eigen::MatrixXd> eigen_cdf_diff(m_cdf_diff->matrix, num_rows, num_cols);

	init_matrix(&m_l2_plus_s2, num_rows, num_cols);
	Map<Eigen::MatrixXd> eigen_l2_plus_s2(m_l2_plus_s2->matrix, num_rows, num_cols);

	init_matrix(&m_h2_plus_s2, num_rows, num_cols);
	Map<Eigen::MatrixXd> eigen_h2_plus_s2(m_h2_plus_s2->matrix, num_rows, num_cols);

	init_matrix(&m_weighted_pdf_diff, num_rows, num_cols);
	Map<Eigen::MatrixXd> eigen_weighted_pdf_diff(m_weighted_pdf_diff->matrix, num_rows, num_cols);


	Map<VectorXd> eigen_mu(m_mu->vector, m_mu->vlen);
	Map<VectorXd> eigen_s2(m_s2->vector, m_s2->vlen);


	MatrixXd eigen_zl = my_bsxfun(plus, eigen_l, (-eigen_mu).transpose());
	MatrixXd eigen_zh = my_bsxfun(plus, eigen_h, (-eigen_mu).transpose());

	VectorXd eigen_s_inv = eigen_s2.array().sqrt().inverse().matrix(); 

	//zl = bsxfun(@times, bsxfun(@minus,l,m), 1./sqrt(v))
	eigen_zl = my_bsxfun(times, eigen_zl, eigen_s_inv.transpose());
	//zh = bsxfun(@times, bsxfun(@minus,h,m), 1./sqrt(v))
	eigen_zh = my_bsxfun(times, eigen_zh, eigen_s_inv.transpose());

	//pl = bsxfun(@times, normpdf(zl), 1./sqrt(v));
	eigen_pl = my_bsxfun(times, standard_norm_pdf(eigen_zl), eigen_s_inv.transpose());
	//ph = bsxfun(@times, normpdf(zh), 1./sqrt(v));
	eigen_ph = my_bsxfun(times, standard_norm_pdf(eigen_zh), eigen_s_inv.transpose());

	MatrixXd & eigen_cl = eigen_zl;
	MatrixXd & eigen_ch = eigen_zh;

	//cl = 0.5*erf(zl/sqrt(2)); %normal cdf -const
	eigen_cl = normal_cdf_minus_const(eigen_zl);
	//ch = 0.5*erf(zl/sqrt(2)); %normal cdf -const
	eigen_ch = normal_cdf_minus_const(eigen_zh);

	eigen_cdf_diff = eigen_ch - eigen_cl;

	float64_t l_bak = eigen_l(0);
	eigen_l(0) = 0;

	float64_t h_bak = eigen_h(eigen_h.size()-1);
	eigen_h(eigen_h.size()-1) = 0;

	eigen_l2_plus_s2 = my_bsxfun(plus, eigen_l.array().pow(2).matrix(), eigen_s2.transpose());
	eigen_h2_plus_s2 = my_bsxfun(plus, eigen_h.array().pow(2).matrix(), eigen_s2.transpose());
	eigen_weighted_pdf_diff = my_bsxfun(times, eigen_pl, eigen_l) - my_bsxfun(times, eigen_ph, eigen_h);


	eigen_l(0) = l_bak;
	eigen_h(eigen_h.size()-1) = h_bak;

}

float64_t CLogitPiecewiseBoundLikelihood::_check_variance(float64_t x)
{
	REQUIRE(x > 0.0, "Variance should always be positive\n");
	return x;
}
float64_t CLogitPiecewiseBoundLikelihood::_convert_label(float64_t x)
{
	if (x <= 0.0)
		return 0.0;
	return 1.0;
}

#endif /* HAVE_EIGEN3 */
