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
 *
 * Code adapted from
 * https://gist.github.com/yorkerlin/14ace49f2278f3859614
 * Gaussian Process Machine Learning Toolbox
 * http://www.gaussianprocess.org/gpml/code/matlab/doc/
 * and
 * GPstuff - Gaussian process models for Bayesian analysis
 * http://becs.aalto.fi/en/research/bayes/gpstuff/
 *
 * The reference pseudo code is the algorithm 3.3 of the GPML textbook
 */

#include <shogun/machine/gp/MultiLaplacianInferenceMethod.h>

#ifdef HAVE_EIGEN3
#include <shogun/mathematics/eigen3.h>
#include <shogun/labels/MulticlassLabels.h>
#include <shogun/mathematics/Math.h>
#include <shogun/lib/external/brent.h>

using namespace shogun;
using namespace Eigen;

namespace shogun
{

#ifndef DOXYGEN_SHOULD_SKIP_THIS

/** Wrapper class used for the Brent minimizer */
class CMultiPsiLine : public func_base
{
public:
	float64_t scale;
	MatrixXd K;
	VectorXd dalpha;
	VectorXd start_alpha;
	Map<VectorXd>* alpha;
	SGVector<float64_t>* dlp;
	SGVector<float64_t>* f;
	SGVector<float64_t>* m;
	CLikelihoodModel* lik;
	CLabels* lab;

	virtual double operator() (double x)
	{
		const index_t C=((CMulticlassLabels*)lab)->get_num_classes();
		const index_t n=((CMulticlassLabels*)lab)->get_num_labels();
		Map<VectorXd> eigen_f(f->vector, f->vlen);
		Map<VectorXd> eigen_m(m->vector, m->vlen);

		// compute alpha=alpha+x*dalpha and f=K*alpha+m
		(*alpha)=start_alpha+x*dalpha;

		float64_t result=0;
		for(index_t bl=0; bl<C; bl++)
		{
			eigen_f.block(bl*n,0,n,1)=K*alpha->block(bl*n,0,n,1)*CMath::sq(scale);
			result+=alpha->block(bl*n,0,n,1).dot(eigen_f.block(bl*n,0,n,1))/2.0;
			eigen_f.block(bl*n,0,n,1)+=eigen_m;
		}

		// get first and second derivatives of log likelihood
		(*dlp)=lik->get_log_probability_derivative_f(lab, (*f), 1);

		result -=SGVector<float64_t>::sum(lik->get_log_probability_f(lab, *f));

		return result;
	}
};

#endif /* DOXYGEN_SHOULD_SKIP_THIS */

CMultiLaplacianInferenceMethod::CMultiLaplacianInferenceMethod() : CLaplacianInferenceBase()
{
	init();
}

CMultiLaplacianInferenceMethod::CMultiLaplacianInferenceMethod(CKernel* kern,
		CFeatures* feat, CMeanFunction* m, CLabels* lab, CLikelihoodModel* mod)
		: CLaplacianInferenceBase(kern, feat, m, lab, mod)
{
	init();
}

void CMultiLaplacianInferenceMethod::init()
{
	m_nlz=0;
	SG_ADD(&m_nlz, "nlz", "negative log marginal likelihood ", MS_NOT_AVAILABLE);
	SG_ADD(&m_U, "U", "the matrix used to compute gradient wrt hyperparameters", MS_NOT_AVAILABLE);
}

CMultiLaplacianInferenceMethod::~CMultiLaplacianInferenceMethod()
{
}

void CMultiLaplacianInferenceMethod::check_members() const
{
	CInferenceMethod::check_members();

	REQUIRE(m_labels->get_label_type()==LT_MULTICLASS,
		"Labels must be type of CMulticlassLabels\n");
	REQUIRE(m_model->supports_multiclass(),
		"likelihood model should support multi-classification\n");
}

SGVector<float64_t> CMultiLaplacianInferenceMethod::get_diagonal_vector()
{
	if (parameter_hash_changed())
		update();

	get_dpi_helper();

	return SGVector<float64_t>(m_W);
}

float64_t CMultiLaplacianInferenceMethod::get_negative_log_marginal_likelihood()
{
	if (parameter_hash_changed())
		update();

	return m_nlz;
}

SGVector<float64_t> CMultiLaplacianInferenceMethod::get_derivative_wrt_likelihood_model(
		const TParameter* param)
{
	//SoftMax likelihood does not have this kind of derivative
	SG_ERROR("Not Implemented!\n");
	return SGVector<float64_t> ();
}

CMultiLaplacianInferenceMethod* CMultiLaplacianInferenceMethod::obtain_from_generic(
		CInferenceMethod* inference)
{
	REQUIRE(inference, "Inference pointer not set.\n");

	if (inference->get_inference_type()!=INF_LAPLACIAN_MULTIPLE)
		SG_SERROR("Provided inference is not of type CMultiLaplacianInferenceMethod!\n")

	SG_REF(inference);
	return (CMultiLaplacianInferenceMethod*)inference;
}


void CMultiLaplacianInferenceMethod::update_approx_cov()
{
	//Sigma=K-K*(E-E*R(M*M')^{-1}*R'*E)*K
	const index_t C=((CMulticlassLabels*)m_labels)->get_num_classes();
	const index_t n=m_labels->get_num_labels();
	Map<MatrixXd> eigen_M(m_L.matrix, m_L.num_rows, m_L.num_cols);
	Map<MatrixXd> eigen_E(m_E.matrix, m_E.num_rows, m_E.num_cols);
	Map<MatrixXd> eigen_K(m_ktrtr.matrix, m_ktrtr.num_rows, m_ktrtr.num_cols);

	m_Sigma=SGMatrix<float64_t>(C*n, C*n);
	Map<MatrixXd> eigen_Sigma(m_Sigma.matrix, m_Sigma.num_rows,	m_Sigma.num_cols);
	eigen_Sigma.fill(0);

	MatrixXd eigen_U(C*n,n);
	for(index_t bl=0; bl<C; bl++)
	{
		eigen_U.block(bl*n,0,n,n)=eigen_K*CMath::sq(m_scale)*eigen_E.block(0,bl*n,n,n);
		eigen_Sigma.block(bl*n,bl*n,n,n)=(MatrixXd::Identity(n,n)-eigen_U.block(bl*n,0,n,n))*(eigen_K*CMath::sq(m_scale));
	}
	MatrixXd eigen_V=eigen_M.triangularView<Upper>().adjoint().solve(eigen_U.transpose());
	eigen_Sigma+=eigen_V.transpose()*eigen_V;
}

void CMultiLaplacianInferenceMethod::update_chol()
{
}

void CMultiLaplacianInferenceMethod::get_dpi_helper()
{
	const index_t C=((CMulticlassLabels*)m_labels)->get_num_classes();
	const index_t n=m_labels->get_num_labels();
	Map<VectorXd> eigen_dpi(m_W.vector, m_W.vlen);
	Map<MatrixXd> eigen_dpi_matrix(eigen_dpi.data(),n,C);

	Map<VectorXd> eigen_mu(m_mu, m_mu.vlen);
	Map<MatrixXd> eigen_mu_matrix(eigen_mu.data(),n,C);
	// with log_sum_exp trick
	VectorXd max_coeff=eigen_mu_matrix.rowwise().maxCoeff();
	eigen_dpi_matrix=eigen_mu_matrix.array().colwise()-max_coeff.array();
	VectorXd log_sum_exp=((eigen_dpi_matrix.array().exp()).rowwise().sum()).array().log();
	eigen_dpi_matrix=(eigen_dpi_matrix.array().colwise()-log_sum_exp.array()).exp();

	// without log_sum_exp trick
	//eigen_dpi_matrix=eigen_mu_matrix.array().exp();
	//VectorXd tmp_for_dpi=eigen_dpi_matrix.rowwise().sum();
	//eigen_dpi_matrix=eigen_dpi_matrix.array().colwise()/tmp_for_dpi.array();
}

void CMultiLaplacianInferenceMethod::update_alpha()
{
	float64_t Psi_Old = CMath::INFTY;
	float64_t Psi_New;
	float64_t Psi_Def;
	const index_t C=((CMulticlassLabels*)m_labels)->get_num_classes();
	const index_t n=m_labels->get_num_labels();

	// get mean vector and create eigen representation of it
	SGVector<float64_t> mean=m_mean->get_mean_vector(m_features);
	Map<VectorXd> eigen_mean_bl(mean.vector, mean.vlen);
	VectorXd eigen_mean=eigen_mean_bl.replicate(C,1);

	// create eigen representation of kernel matrix
	Map<MatrixXd> eigen_ktrtr(m_ktrtr.matrix, m_ktrtr.num_rows, m_ktrtr.num_cols);

	// create shogun and eigen representation of function vector
	m_mu=SGVector<float64_t>(mean.vlen*C);
	Map<VectorXd> eigen_mu(m_mu, m_mu.vlen);

	// f = mean as default value
	eigen_mu=eigen_mean;

	Psi_Def=-SGVector<float64_t>::sum(m_model->get_log_probability_f(
			m_labels, m_mu));

	if (m_alpha.vlen!=C*n)
	{
		// set alpha a zero vector
		m_alpha=SGVector<float64_t>(C*n);
		m_alpha.zero();
		Psi_New=Psi_Def;
		m_E=SGMatrix<float64_t>(n,C*n);
		m_L=SGMatrix<float64_t>(n,n);
		m_W=SGVector<float64_t>(C*n);
	}
	else
	{
		Map<VectorXd> alpha(m_alpha.vector, m_alpha.vlen);
		for(index_t bl=0; bl<C; bl++)
			eigen_mu.block(bl*n,0,n,1)=eigen_ktrtr*CMath::sq(m_scale)*alpha.block(bl*n,0,n,1);

		//alpha'*(f-m)/2.0
		Psi_New=alpha.dot(eigen_mu)/2.0;
		// compute f = K * alpha + m
		eigen_mu+=eigen_mean;

		Psi_New-=SGVector<float64_t>::sum(m_model->get_log_probability_f(m_labels, m_mu));

		// if default is better, then use it
		if (Psi_Def < Psi_New)
		{
			m_alpha.zero();
			eigen_mu=eigen_mean;
			Psi_New=Psi_Def;
		}
	}

	Map<VectorXd> eigen_alpha(m_alpha.vector, m_alpha.vlen);
	Map<VectorXd> eigen_W(m_W.vector, m_W.vlen);
	Map<MatrixXd> eigen_L(m_L.matrix, m_L.num_rows, m_L.num_cols);
	Map<MatrixXd> eigen_E(m_E.matrix, m_E.num_rows, m_E.num_cols);

	// get first derivative of log probability function
	m_dlp=m_model->get_log_probability_derivative_f(m_labels, m_mu, 1);

	index_t iter=0;
	Map<MatrixXd> & eigen_M=eigen_L;

	while (Psi_Old-Psi_New>m_tolerance && iter<m_iter)
	{
		Map<VectorXd> eigen_dlp(m_dlp.vector, m_dlp.vlen);

		get_dpi_helper();
		Map<VectorXd> eigen_dpi(m_W.vector, m_W.vlen);


		Psi_Old = Psi_New;
		iter++;

		m_nlz=0;

		for(index_t bl=0; bl<C; bl++)
		{
			VectorXd eigen_sD=eigen_dpi.block(bl*n,0,n,1).cwiseSqrt();
			LLT<MatrixXd> chol_tmp((eigen_sD*eigen_sD.transpose()).cwiseProduct(eigen_ktrtr*CMath::sq(m_scale))+
				MatrixXd::Identity(m_ktrtr.num_rows, m_ktrtr.num_cols));
			MatrixXd eigen_L_tmp=chol_tmp.matrixU();
			MatrixXd eigen_E_bl=eigen_L_tmp.triangularView<Upper>().adjoint().solve(MatrixXd(eigen_sD.asDiagonal()));
			eigen_E_bl=eigen_E_bl.transpose()*eigen_E_bl;
			eigen_E.block(0,bl*n,n,n)=eigen_E_bl;
			if (bl==0)
				eigen_M=eigen_E_bl;
			else
				eigen_M+=eigen_E_bl;
			m_nlz+=eigen_L_tmp.diagonal().array().log().sum();
		}

		LLT<MatrixXd> chol_tmp(eigen_M);
		eigen_M = chol_tmp.matrixU();
		m_nlz+=eigen_M.diagonal().array().log().sum();

		VectorXd eigen_b=eigen_dlp;
		Map<VectorXd> & tmp1=eigen_dlp;
		tmp1=eigen_dpi.array()*(eigen_mu-eigen_mean).array();
		Map<MatrixXd> m_tmp(tmp1.data(),n,C);
		VectorXd tmp2=m_tmp.array().rowwise().sum();

		for(index_t bl=0; bl<C; bl++)
			eigen_b.block(bl*n,0,n,1)+=eigen_dpi.block(bl*n,0,n,1).cwiseProduct(eigen_mu.block(bl*n,0,n,1)-eigen_mean_bl-tmp2);

		Map<VectorXd> &eigen_c=eigen_W;
		for(index_t bl=0; bl<C; bl++)
			eigen_c.block(bl*n,0,n,1)=eigen_E.block(0,bl*n,n,n)*(eigen_ktrtr*CMath::sq(m_scale)*eigen_b.block(bl*n,0,n,1));

		Map<MatrixXd> c_tmp(eigen_c.data(),n,C);

		VectorXd tmp3=c_tmp.array().rowwise().sum();
		VectorXd tmp4=eigen_M.triangularView<Upper>().adjoint().solve(tmp3);

		VectorXd &eigen_dalpha=eigen_b;
		eigen_dalpha+=eigen_E.transpose()*(eigen_M.triangularView<Upper>().solve(tmp4))-eigen_c-eigen_alpha;

		// perform Brent's optimization
		CMultiPsiLine func;

		func.scale=m_scale;
		func.K=eigen_ktrtr;
		func.dalpha=eigen_dalpha;
		func.start_alpha=eigen_alpha;
		func.alpha=&eigen_alpha;
		func.dlp=&m_dlp;
		func.f=&m_mu;
		func.m=&mean;
		func.lik=m_model;
		func.lab=m_labels;

		float64_t x;
		Psi_New=local_min(0, m_opt_max, m_opt_tolerance, func, x);
		m_nlz+=Psi_New;
	}

}

void CMultiLaplacianInferenceMethod::update_deriv()
{
	const index_t C=((CMulticlassLabels*)m_labels)->get_num_classes();
	const index_t n=m_labels->get_num_labels();
	m_U=SGMatrix<float64_t>(n, n*C);
	Map<MatrixXd> eigen_U(m_U.matrix, m_U.num_rows, m_U.num_cols);
	Map<MatrixXd> eigen_M(m_L.matrix, m_L.num_rows, m_L.num_cols);
	Map<MatrixXd> eigen_E(m_E.matrix, m_E.num_rows, m_E.num_cols);
	eigen_U=eigen_M.triangularView<Upper>().adjoint().solve(eigen_E);
}

float64_t CMultiLaplacianInferenceMethod::get_derivative_helper(SGMatrix<float64_t> dK)
{
	Map<MatrixXd> eigen_dK(dK.matrix, dK.num_rows, dK.num_cols);
	//currently only explicit term is computed
	const index_t C=((CMulticlassLabels*)m_labels)->get_num_classes();
	const index_t n=m_labels->get_num_labels();
	Map<MatrixXd> eigen_U(m_U.matrix, m_U.num_rows, m_U.num_cols);
	Map<MatrixXd> eigen_E(m_E.matrix, m_E.num_rows, m_E.num_cols);
	Map<VectorXd> eigen_alpha(m_alpha.vector, m_alpha.vlen);
	float64_t result=0;
	//currently only explicit term is computed
	for(index_t bl=0; bl<C; bl++)
	{
		result+=((eigen_E.block(0,bl*n,n,n)-eigen_U.block(0,bl*n,n,n).transpose()*eigen_U.block(0,bl*n,n,n)).array()
			*eigen_dK.array()).sum();
		result-=(eigen_dK*eigen_alpha.block(bl*n,0,n,1)).dot(eigen_alpha.block(bl*n,0,n,1));
	}

	return result/2.0;
}

SGVector<float64_t> CMultiLaplacianInferenceMethod::get_derivative_wrt_inference_method(
		const TParameter* param)
{
	REQUIRE(!strcmp(param->m_name, "scale"), "Can't compute derivative of "
			"the nagative log marginal likelihood wrt %s.%s parameter\n",
			get_name(), param->m_name)

	Map<MatrixXd> eigen_K(m_ktrtr.matrix, m_ktrtr.num_rows, m_ktrtr.num_cols);

	SGVector<float64_t> result(1);

	SGMatrix<float64_t> dK(m_ktrtr.num_rows, m_ktrtr.num_cols);
	Map<MatrixXd> eigen_dK(dK.matrix, dK.num_rows, dK.num_cols);

	// compute derivative K wrt scale
	eigen_dK=eigen_K*m_scale*2.0;

	result[0]=get_derivative_helper(dK);

	return result;
}

SGVector<float64_t> CMultiLaplacianInferenceMethod::get_derivative_wrt_kernel(
		const TParameter* param)
{
	// create eigen representation of K, Z, dfhat, dlp and alpha
	Map<MatrixXd> eigen_K(m_ktrtr.matrix, m_ktrtr.num_rows, m_ktrtr.num_cols);

	SGVector<float64_t> result;

	if (param->m_datatype.m_ctype==CT_VECTOR ||
		param->m_datatype.m_ctype==CT_SGVECTOR ||
		param->m_datatype.m_ctype==CT_MATRIX ||
		param->m_datatype.m_ctype==CT_SGMATRIX)
	{
		REQUIRE(param->m_datatype.m_length_y,
			"Length of the parameter %s should not be NULL\n", param->m_name)
		result=SGVector<float64_t>(*(param->m_datatype.m_length_y));
	}
	else
	{
		result=SGVector<float64_t>(1);
	}

	for (index_t i=0; i<result.vlen; i++)
	{
		SGMatrix<float64_t> dK;

		if (result.vlen==1)
			dK=m_kernel->get_parameter_gradient(param);
		else
			dK=m_kernel->get_parameter_gradient(param, i);

		Map<MatrixXd> eigen_dK(dK.matrix, dK.num_rows, dK.num_cols);
		eigen_dK=eigen_dK*CMath::sq(m_scale);

		result[i]=get_derivative_helper(dK);
	}

	return result;
}

SGVector<float64_t> CMultiLaplacianInferenceMethod::get_derivative_wrt_mean(
		const TParameter* param)
{
	// create eigen representation of K, Z, dfhat and alpha
	Map<MatrixXd> eigen_K(m_ktrtr.matrix, m_ktrtr.num_rows, m_ktrtr.num_cols);
	Map<VectorXd> eigen_alpha(m_alpha.vector, m_alpha.vlen);
	const index_t C=((CMulticlassLabels*)m_labels)->get_num_classes();
	const index_t n=m_labels->get_num_labels();

	SGVector<float64_t> result;

	if (param->m_datatype.m_ctype==CT_VECTOR ||
			param->m_datatype.m_ctype==CT_SGVECTOR)
	{
		REQUIRE(param->m_datatype.m_length_y,
				"Length of the parameter %s should not be NULL\n", param->m_name)
		result=SGVector<float64_t>(*(param->m_datatype.m_length_y));
	}
	else
	{
		result=SGVector<float64_t>(1);
	}

	for (index_t i=0; i<result.vlen; i++)
	{
		SGVector<float64_t> dmu;

		if (result.vlen==1)
			dmu=m_mean->get_parameter_derivative(m_features, param);
		else
			dmu=m_mean->get_parameter_derivative(m_features, param, i);

		Map<VectorXd> eigen_dmu(dmu.vector, dmu.vlen);

		result[i]=0;
		//currently only compute the explicit term
		for(index_t bl=0; bl<C; bl++)
			result[i]-=eigen_alpha.block(bl*n,0,n,1).dot(eigen_dmu);
	}

	return result;
}

}

#endif /* HAVE_EIGEN3 */
