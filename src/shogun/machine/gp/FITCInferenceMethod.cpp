/*
 * Copyright (c) The Shogun Machine Learning Toolbox
 * Written (W) 2015 Wu Lin
 * Written (W) 2013 Roman Votyakov
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

#include <shogun/machine/gp/FITCInferenceMethod.h>
#include <shogun/machine/gp/GaussianLikelihood.h>
#include <shogun/mathematics/Math.h>
#include <shogun/labels/RegressionLabels.h>
#include <shogun/mathematics/eigen3.h>

#include <utility>

using namespace shogun;
using namespace Eigen;

FITCInferenceMethod::FITCInferenceMethod() : SingleFITCInference()
{
	init();
}

FITCInferenceMethod::FITCInferenceMethod(std::shared_ptr<Kernel> kern, std::shared_ptr<Features> feat,
		std::shared_ptr<MeanFunction> m, std::shared_ptr<Labels> lab, std::shared_ptr<LikelihoodModel> mod, std::shared_ptr<Features> lat)
		: SingleFITCInference(std::move(kern), std::move(feat), std::move(m), std::move(lab), std::move(mod), std::move(lat))
{
	init();
}

void FITCInferenceMethod::init()
{
}

FITCInferenceMethod::~FITCInferenceMethod()
{
}
void FITCInferenceMethod::compute_gradient()
{
	Inference::compute_gradient();

	if (!m_gradient_update)
	{
		update_deriv();
		m_gradient_update=true;
		update_parameter_hash();
	}
}

void FITCInferenceMethod::update()
{
	SG_TRACE("entering");

	Inference::update();
	update_chol();
	update_alpha();
	m_gradient_update=false;
	update_parameter_hash();

	SG_TRACE("leaving");
}

std::shared_ptr<FITCInferenceMethod> FITCInferenceMethod::obtain_from_generic(
		const std::shared_ptr<Inference>& inference)
{
	if (inference==NULL)
		return NULL;

	if (inference->get_inference_type()!=INF_FITC_REGRESSION)
		error("Provided inference is not of type FITCInferenceMethod!");

	return inference->as<FITCInferenceMethod>();
}

void FITCInferenceMethod::check_members() const
{
	SingleFITCInference::check_members();

	require(m_model->get_model_type()==LT_GAUSSIAN,
			"FITC inference method can only use Gaussian likelihood function");
	require(m_labels->get_label_type()==LT_REGRESSION, "Labels must be type "
			"of RegressionLabels");
}

SGVector<float64_t> FITCInferenceMethod::get_diagonal_vector()
{
	if (parameter_hash_changed())
		update();

	// get the sigma variable from the Gaussian likelihood model
	auto lik = m_model->as<GaussianLikelihood>();
	float64_t sigma=lik->get_sigma();

	// compute diagonal vector: sW=1/sigma
	SGVector<float64_t> result(m_features->get_num_vectors());
	result.fill_vector(result.vector, m_features->get_num_vectors(), 1.0/sigma);

	return result;
}

float64_t FITCInferenceMethod::get_negative_log_marginal_likelihood()
{
	if (parameter_hash_changed())
		update();

	//time complexity of the following operations is O(m*n)

	// create eigen representations of chol_utr, dg, r, be
	SGMatrix<double>::EigenMatrixXtMap eigen_chol_utr = m_chol_utr;
	SGVector<double>::EigenVectorXtMap eigen_t = m_t;
	SGVector<double>::EigenVectorXtMap eigen_r = m_r;
	SGVector<double>::EigenVectorXtMap eigen_be = m_be;

	// compute negative log marginal likelihood:
	// nlZ=sum(log(diag(utr)))+(sum(log(dg))+r'*r-be'*be+n*log(2*pi))/2
	float64_t result =
	    eigen_chol_utr.diagonal().array().log().sum() +
	    (-eigen_t.array().log().sum() + eigen_r.dot(eigen_r) -
	     eigen_be.dot(eigen_be) + m_ktrtr_diag.vlen * std::log(2 * Math::PI)) /
	        2.0;

	return result;
}

void FITCInferenceMethod::update_chol()
{
	//time complexits O(m^2*n)

	// get the sigma variable from the Gaussian likelihood model
	auto lik = m_model->as<GaussianLikelihood>();
	float64_t sigma=lik->get_sigma();

	// eigen3 representation of covariance matrix of inducing features (m_kuu)
	// and training features (m_ktru)
	SGMatrix<double>::EigenMatrixXtMap eigen_kuu = m_kuu;
	SGMatrix<double>::EigenMatrixXtMap eigen_Ktru = m_ktru;

	SGVector<double>::EigenVectorXtMap eigen_ktrtr_diag = m_ktrtr_diag;

	// solve Luu' * Luu = Kuu + m_ind_noise * I
	//Luu  = chol(Kuu+snu2*eye(nu));                         % Kuu + snu2*I = Luu'*Luu
	LLT<MatrixXd> Luu(
	    eigen_kuu * std::exp(m_log_scale * 2.0) +
	    std::exp(m_log_ind_noise) *
	        MatrixXd::Identity(m_kuu.num_rows, m_kuu.num_cols));

	// create shogun and eigen3 representation of cholesky of covariance of
	// inducing features Luu (m_chol_uu and eigen_chol_uu)
	m_chol_uu=SGMatrix<float64_t>(Luu.rows(), Luu.cols());
	SGMatrix<double>::EigenMatrixXtMap eigen_chol_uu = m_chol_uu;
	eigen_chol_uu=Luu.matrixU();

	// solve Luu' * V = Ktru
	//V  = Luu'\Ku;                                     % V = inv(Luu')*Ku => V'*V = Q

	m_V=SGMatrix<float64_t>(m_chol_uu.num_cols, m_ktru.num_cols);
	SGMatrix<double>::EigenMatrixXtMap V = m_V;
	V = eigen_chol_uu.triangularView<Upper>().adjoint().solve(
	    eigen_Ktru * std::exp(m_log_scale * 2.0));

	// create shogun and eigen3 representation of
	// dg = diag(K) + sn2 - diag(Q)
	// t = 1/dg
	m_t=SGVector<float64_t>(m_ktrtr_diag.vlen);
	SGVector<double>::EigenVectorXtMap eigen_t = m_t;

	//g_sn2 = diagK + sn2 - sum(V.*V,1)';          % g + sn2 = diag(K) + sn2 - diag(Q)
	eigen_t = eigen_ktrtr_diag * std::exp(m_log_scale * 2.0) +
	          Math::sq(sigma) * VectorXd::Ones(m_t.vlen) -
	          (V.cwiseProduct(V)).colwise().sum().adjoint();
	eigen_t=MatrixXd::Ones(eigen_t.rows(),1).cwiseQuotient(eigen_t);

	// solve Lu' * Lu = V * diag(1/dg) * V' + I
	//Lu = chol(eye(nu) + (V./repmat(g_sn2',nu,1))*V');  % Lu'*Lu=I+V*diag(1/g_sn2)*V'
	LLT<MatrixXd> Lu(V*((VectorXd::Ones(m_t.vlen)).cwiseProduct(eigen_t)).asDiagonal()*
			V.adjoint()+MatrixXd::Identity(m_kuu.num_rows, m_kuu.num_cols));

	// create shogun and eigen3 representation of cholesky of covariance of
	// training features Luu (m_chol_utr and eigen_chol_utr)
	m_chol_utr=SGMatrix<float64_t>(Lu.rows(), Lu.cols());
	SGMatrix<double>::EigenMatrixXtMap eigen_chol_utr = m_chol_utr;
	eigen_chol_utr=Lu.matrixU();

	// create eigen representation of labels and mean vectors
	SGVector<float64_t> y=regression_labels(m_labels)->get_labels();
	SGVector<double>::EigenVectorXtMap eigen_y = y;
	SGVector<float64_t> m=m_mean->get_mean_vector(m_features);
	SGVector<double>::EigenVectorXtMap eigen_m = m;

	// compute sgrt_dg = sqrt(dg)
	VectorXd sqrt_t=eigen_t.array().sqrt();

	// create shogun and eigen3 representation of labels adjusted for
	// noise and means (m_r)
	//r  = (y-m)./sqrt(g_sn2);
	m_r=SGVector<float64_t>(y.vlen);
	SGVector<double>::EigenVectorXtMap eigen_r = m_r;
	eigen_r=(eigen_y-eigen_m).cwiseProduct(sqrt_t);

	// compute be
	//be = Lu'\(V*(r./sqrt(g_sn2)));
	m_be=SGVector<float64_t>(m_chol_utr.num_cols);
	SGVector<double>::EigenVectorXtMap eigen_be = m_be;
	eigen_be=eigen_chol_utr.triangularView<Upper>().adjoint().solve(
		V*eigen_r.cwiseProduct(sqrt_t));

	// compute iKuu
	//iKuu = solve_chol(Luu,eye(nu));
	MatrixXd iKuu=Luu.solve(MatrixXd::Identity(m_kuu.num_rows, m_kuu.num_cols));

	// create shogun and eigen3 representation of posterior cholesky
	MatrixXd eigen_prod=eigen_chol_utr*eigen_chol_uu;
	m_L=SGMatrix<float64_t>(m_kuu.num_rows, m_kuu.num_cols);
	SGMatrix<double>::EigenMatrixXtMap eigen_chol = m_L;

	//post.L  = solve_chol(Lu*Luu,eye(nu)) - iKuu;
	eigen_chol=eigen_prod.triangularView<Upper>().adjoint().solve(
		MatrixXd::Identity(m_kuu.num_rows, m_kuu.num_cols));
	eigen_chol=eigen_prod.triangularView<Upper>().solve(eigen_chol)-iKuu;
}

void FITCInferenceMethod::update_alpha()
{
	//time complexity O(m^2) since triangular.solve is O(m^2)
	SGMatrix<double>::EigenMatrixXtMap eigen_chol_uu = m_chol_uu;
	SGMatrix<double>::EigenMatrixXtMap eigen_chol_utr = m_chol_utr;
	SGVector<double>::EigenVectorXtMap eigen_be = m_be;

	// create shogun and eigen representations of alpha
	// and solve Luu * Lu * alpha = be
	m_alpha=SGVector<float64_t>(m_chol_uu.num_rows);
	SGVector<double>::EigenVectorXtMap eigen_alpha = m_alpha;

	//post.alpha = Luu\(Lu\be);
	eigen_alpha=eigen_chol_utr.triangularView<Upper>().solve(eigen_be);
	eigen_alpha=eigen_chol_uu.triangularView<Upper>().solve(eigen_alpha);
}

void FITCInferenceMethod::update_deriv()
{
	//time complexits O(m^2*n)

	// create eigen representation of Ktru, Lu, Luu, dg, be
	SGMatrix<double>::EigenMatrixXtMap eigen_Ktru = m_ktru;
	SGMatrix<double>::EigenMatrixXtMap eigen_Lu = m_chol_utr;
	SGMatrix<double>::EigenMatrixXtMap eigen_Luu = m_chol_uu;
	SGVector<double>::EigenVectorXtMap eigen_t = m_t;
	SGVector<double>::EigenVectorXtMap eigen_be = m_be;

	// get and create eigen representation of labels
	SGVector<float64_t> y=regression_labels(m_labels)->get_labels();
	SGVector<double>::EigenVectorXtMap eigen_y = y;

	// get and create eigen representation of mean vector
	SGVector<float64_t> m=m_mean->get_mean_vector(m_features);
	SGVector<double>::EigenVectorXtMap eigen_m = m;

	// compute V=inv(Luu')*Ku
	SGMatrix<double>::EigenMatrixXtMap V = m_V;

	// create shogun and eigen representation of al
	m_al=SGVector<float64_t>(m.vlen);
	SGVector<double>::EigenVectorXtMap eigen_al = m_al;

	// compute al=(Kt+sn2*eye(n))\y
	//r  = (y-m)./sqrt(g_sn2);
	//al = r./sqrt(g_sn2) - (V'*(Lu\be))./g_sn2;      % al = (Kt+sn2*eye(n))\(y-m)
	eigen_al=((eigen_y-eigen_m)-(V.adjoint()*
		eigen_Lu.triangularView<Upper>().solve(eigen_be))).cwiseProduct(eigen_t);

	// compute inv(Kuu+snu2*I)=iKuu
	MatrixXd iKuu=eigen_Luu.triangularView<Upper>().adjoint().solve(
			MatrixXd::Identity(m_kuu.num_rows, m_kuu.num_cols));
	iKuu=eigen_Luu.triangularView<Upper>().solve(iKuu);

	// create shogun and eigen representation of B
	m_B=SGMatrix<float64_t>(iKuu.rows(), eigen_Ktru.cols());
	SGMatrix<double>::EigenMatrixXtMap eigen_B = m_B;

	//B = iKuu*Ku; w = B*al;
	eigen_B = iKuu * eigen_Ktru * std::exp(m_log_scale * 2.0);

	// create shogun and eigen representation of w
	m_w=SGVector<float64_t>(m_B.num_rows);
	SGVector<double>::EigenVectorXtMap eigen_w = m_w;

	eigen_w=eigen_B*eigen_al;

	// create shogun and eigen representation of W
	m_Rvdd=SGMatrix<float64_t>(m_chol_utr.num_cols, m_t.vlen);
	SGMatrix<double>::EigenMatrixXtMap eigen_W = m_Rvdd;

	// compute W=Lu'\(V./repmat(g_sn2',nu,1))
	//W = Lu'\(V./repmat(g_sn2',nu,1));
	eigen_W=eigen_Lu.triangularView<Upper>().adjoint().solve(V*VectorXd::Ones(
		m_t.vlen).cwiseProduct(eigen_t).asDiagonal());
}


SGVector<float64_t> FITCInferenceMethod::get_posterior_mean()
{
	compute_gradient();

	m_mu=SGVector<float64_t>(m_al.vlen);
	SGVector<double>::EigenVectorXtMap eigen_mu = m_mu;

	/*
	//true posterior mean with equivalent FITC prior
	//time complexity of the following operations is O(n)
	SGVector<double>::EigenVectorXtMap eigen_al = m_al;
	SGVector<float64_t> y=((RegressionLabels*) m_labels)->get_labels();
	SGVector<double>::EigenVectorXtMap eigen_y = y;
	SGVector<float64_t> m=m_mean->get_mean_vector(m_features);
	SGVector<double>::EigenVectorXtMap eigen_m = m;
	auto lik=GaussianLikelihood::obtain_from_generic(m_model);
	float64_t sigma=lik->get_sigma();
	SG_UNREF(lik);
	eigen_mu=(eigen_y-eigen_m)-eigen_al*Math::sq(sigma);
	*/

	//FITC approximated posterior mean
	SGVector<double>::EigenVectorXtMap eigen_alpha = m_alpha;
	SGMatrix<double>::EigenMatrixXtMap eigen_Ktru = m_ktru;
	//time complexity of the following operation is O(m*n)
	eigen_mu = std::exp(m_log_scale * 2.0) * eigen_Ktru.adjoint() * eigen_alpha;

	return SGVector<float64_t>(m_mu);
}

SGMatrix<float64_t> FITCInferenceMethod::get_posterior_covariance()
{
	compute_gradient();

	//time complexity of the following operations is O(m*n^2)
	//Warning: the the time complexity increases from O(m^2*n) to O(n^2*m) if this method is called
	m_Sigma=SGMatrix<float64_t>(m_ktrtr_diag.vlen, m_ktrtr_diag.vlen);
	SGMatrix<double>::EigenMatrixXtMap eigen_Sigma = m_Sigma;
	SGMatrix<double>::EigenMatrixXtMap eigen_V = m_V;
	SGMatrix<double>::EigenMatrixXtMap eigen_Lu = m_chol_utr;

	/*
	//true posterior mean with equivalent FITC prior
	auto lik=GaussianLikelihood::obtain_from_generic(m_model);
	float64_t sigma=lik->get_sigma();
	SG_UNREF(lik);
	SGVector<double>::EigenVectorXtMap eigen_t = m_t;
	VectorXd diag_part=Math::sq(sigma)*eigen_t;
	// diag(sigma2/dg)*V'*(Lu\eye(n))
	MatrixXd part1=diag_part.asDiagonal()*eigen_V.adjoint()*
		(eigen_Lu.triangularView<Upper>().solve(MatrixXd::Identity(
		m_kuu.num_rows, m_kuu.num_cols)));
	eigen_Sigma=part1*part1.adjoint();
	VectorXd part2=(VectorXd::Ones(m_t.vlen)-diag_part)*Math::sq(sigma);
	eigen_Sigma+=part2.asDiagonal();
	*/

	//FITC approximated posterior covariance
	//TO DO we only need tha diagonal elements of m_ktrtr
	SGVector<double>::EigenVectorXtMap eigen_ktrtr_diag = m_ktrtr_diag;
	MatrixXd part1=eigen_V.adjoint()*(eigen_Lu.triangularView<Upper>().solve(MatrixXd::Identity(
		m_kuu.num_rows, m_kuu.num_cols)));
	eigen_Sigma=part1*part1.adjoint();
	VectorXd part2 = eigen_ktrtr_diag * std::exp(m_log_scale * 2.0) -
	                 (eigen_V.cwiseProduct(eigen_V)).colwise().sum().adjoint();
	eigen_Sigma+=part2.asDiagonal();

	return SGMatrix<float64_t>(m_Sigma);
}

SGVector<float64_t> FITCInferenceMethod::get_derivative_wrt_likelihood_model(
		Parameters::const_reference param)
{
	//time complexity O(m*n)
	require(param.first == "log_sigma", "Can't compute derivative of "
			"the nagative log marginal likelihood wrt {}.{} parameter",
			m_model->get_name(), param.first);

	// create eigen representation of dg, al, w, W and B
	SGVector<double>::EigenVectorXtMap eigen_t = m_t;
	SGVector<double>::EigenVectorXtMap eigen_al = m_al;
	SGVector<double>::EigenVectorXtMap eigen_w = m_w;
	SGMatrix<double>::EigenMatrixXtMap eigen_W = m_Rvdd;
	SGMatrix<double>::EigenMatrixXtMap eigen_B = m_B;

	// get the sigma variable from the Gaussian likelihood model
	auto lik = m_model->as<GaussianLikelihood>();
	float64_t sigma=lik->get_sigma();

	SGVector<float64_t> result(1);

	//diag_dK = 1./g_sn2 - sum(W.*W,1)' - al.*al;                  % diag(dnlZ/dK)
	//dnlZ.lik = sn2*sum(diag_dK) without noise term
	result[0]=Math::sq(sigma)*(VectorXd::Ones(m_t.vlen).cwiseProduct(
		eigen_t).sum()-eigen_W.cwiseProduct(eigen_W).sum()-eigen_al.dot(eigen_al));

	return result;
}

void FITCInferenceMethod::register_minimizer(std::shared_ptr<Minimizer> minimizer)
{
	io::warn("The method does not require a minimizer. The provided minimizer will not be used.");
}
