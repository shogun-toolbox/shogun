/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Roman Votyakov
 * Copyright (C) 2012 Jacob Walker
 * Copyright (C) 2013 Roman Votyakov
 *
 * Code adapted from Gaussian Process Machine Learning Toolbox
 * http://www.gaussianprocess.org/gpml/code/matlab/doc/
 */

#include <shogun/machine/gp/ExactInferenceMethod.h>

#ifdef HAVE_EIGEN3

#include <shogun/machine/gp/GaussianLikelihood.h>
#include <shogun/labels/RegressionLabels.h>
#include <shogun/mathematics/Math.h>
#include <shogun/mathematics/eigen3.h>

using namespace shogun;
using namespace Eigen;

CExactInferenceMethod::CExactInferenceMethod() : CInferenceMethod()
{
}

CExactInferenceMethod::CExactInferenceMethod(CKernel* kern, CFeatures* feat,
		CMeanFunction* m, CLabels* lab, CLikelihoodModel* mod) :
		CInferenceMethod(kern, feat, m, lab, mod)
{
}

CExactInferenceMethod::~CExactInferenceMethod()
{
}

void CExactInferenceMethod::update()
{
	CInferenceMethod::update();
	update_chol();
	update_alpha();
}

void CExactInferenceMethod::check_members() const
{
	CInferenceMethod::check_members();

	REQUIRE(m_model->get_model_type()==LT_GAUSSIAN,
		"Exact inference method can only use Gaussian likelihood function\n")
	REQUIRE(m_labels->get_label_type()==LT_REGRESSION,
		"Labels must be type of CRegressionLabels\n")
}

CMap<TParameter*, SGVector<float64_t> > CExactInferenceMethod::
get_marginal_likelihood_derivatives(CMap<TParameter*, CSGObject*>& para_dict)
{
	if (update_parameter_hash())
		update();

	// get the sigma variable from the Gaussian likelihood model
	CGaussianLikelihood* lik=CGaussianLikelihood::obtain_from_generic(m_model);
	float64_t sigma=lik->get_sigma();
	SG_UNREF(lik);

	// create eigen representation of derivative matrix and cholesky
	MatrixXd eigen_Q(m_L.num_rows, m_L.num_cols);
	Map<MatrixXd> eigen_L(m_L.matrix, m_L.num_rows, m_L.num_cols);

	// solve L * L' * Q = I
	eigen_Q=eigen_L.triangularView<Upper>().adjoint().solve(
		MatrixXd::Identity(m_L.num_rows, m_L.num_cols));
	eigen_Q=eigen_L.triangularView<Upper>().solve(eigen_Q);

	// divide Q by sigma^2
	eigen_Q/=CMath::sq(sigma);

	// create eigen representation of alpha and compute Q = Q - alpha * alpha'
	Map<VectorXd> eigen_alpha(m_alpha.vector, m_alpha.vlen);
	eigen_Q-=eigen_alpha*eigen_alpha.transpose();

	// build parameter dictionary for kernel and mean
	m_kernel->build_parameter_dictionary(para_dict);
	m_mean->build_parameter_dictionary(para_dict);

	// this will be the vector we return
	CMap<TParameter*, SGVector<float64_t> > gradient;

	for (index_t i=0; i<para_dict.get_num_elements(); i++)
	{
		TParameter* param=para_dict.get_node_ptr(i)->key;

		index_t length=1;

		if ((param->m_datatype.m_ctype==CT_VECTOR ||
				param->m_datatype.m_ctype==CT_SGVECTOR) &&
				param->m_datatype.m_length_y!=NULL)
			length=*(param->m_datatype.m_length_y);

		SGVector<float64_t> variables(length);

		bool deriv_found=false;

		for (index_t g=0; g<length; g++)
		{
			SGMatrix<float64_t> deriv;
			SGVector<float64_t> mean_derivatives;

			if (param->m_datatype.m_ctype==CT_VECTOR ||
					param->m_datatype.m_ctype==CT_SGVECTOR)
			{
				deriv=m_kernel->get_parameter_gradient(param, g);
				mean_derivatives=m_mean->get_parameter_derivative(param, m_feat, g);
			}
			else
			{
				deriv=m_kernel->get_parameter_gradient(param);
				mean_derivatives=m_mean->get_parameter_derivative(param, m_feat);
			}

			if (deriv.num_cols*deriv.num_rows>0)
			{
				Map<MatrixXd> eigen_deriv(deriv.matrix, deriv.num_rows, deriv.num_cols);
				MatrixXd eigen_S=eigen_Q.cwiseProduct(eigen_deriv)*CMath::sq(m_scale);
				variables[g]=eigen_S.sum()/2.0;
				deriv_found=true;
			}
			else if (mean_derivatives.vlen>0)
			{
				variables[g]=mean_derivatives.dot(mean_derivatives.vector,
						m_alpha.vector, m_alpha.vlen);
				deriv_found=true;
			}
		}

		if (deriv_found)
			gradient.add(param, variables);
	}

	TParameter* param=m_model_selection_parameters->get_parameter("scale");
	para_dict.add(param, this);

	Map<MatrixXd> eigen_K(m_ktrtr.matrix, m_ktrtr.num_rows, m_ktrtr.num_cols);
	MatrixXd eigen_S=eigen_Q.cwiseProduct(eigen_K)*m_scale*2.0;

	SGVector<float64_t> vscale(1);
	vscale[0]=eigen_S.sum()/2.0;

	gradient.add(param, vscale);

	param=m_model->m_model_selection_parameters->get_parameter("sigma");
	para_dict.add(param, m_model);

	SGVector<float64_t> vsigma(1);
	vsigma[0]=CMath::sq(sigma)*eigen_Q.trace();

	gradient.add(param, vsigma);

	return gradient;
}

SGVector<float64_t> CExactInferenceMethod::get_diagonal_vector()
{
	if (update_parameter_hash())
		update();

	// get the sigma variable from the Gaussian likelihood model
	CGaussianLikelihood* lik=CGaussianLikelihood::obtain_from_generic(m_model);
	float64_t sigma=lik->get_sigma();
	SG_UNREF(lik);

	// compute diagonal vector: sW=1/sigma
	SGVector<float64_t> result(m_features->get_num_vectors());
	result.fill_vector(result.vector, m_features->get_num_vectors(), 1.0/sigma);

	return result;
}

float64_t CExactInferenceMethod::get_negative_marginal_likelihood()
{
	if (update_parameter_hash())
		update();

	// get the sigma variable from the Gaussian likelihood model
	CGaussianLikelihood* lik=CGaussianLikelihood::obtain_from_generic(m_model);
	float64_t sigma=lik->get_sigma();
	SG_UNREF(lik);

	// create eigen representation of alpha and L
	Map<VectorXd> eigen_alpha(m_alpha.vector, m_alpha.vlen);
	Map<MatrixXd> eigen_L(m_L.matrix, m_L.num_rows, m_L.num_cols);

	// get labels and mean vectors and create eigen representation
	SGVector<float64_t> y=((CRegressionLabels*) m_labels)->get_labels();
	Map<VectorXd> eigen_y(y.vector, y.vlen);
	SGVector<float64_t> m=m_mean->get_mean_vector(m_feat);
	Map<VectorXd> eigen_m(m.vector, m.vlen);

	// compute negative log of the marginal likelihood:
	// nlZ=(y-m)'*alpha/2+sum(log(diag(L)))+n*log(2*pi*sigma^2)/2
	float64_t result=(eigen_y-eigen_m).dot(eigen_alpha)/2.0+
		eigen_L.diagonal().array().log().sum()+m_L.num_rows*
		CMath::log(2*CMath::PI*CMath::sq(sigma))/2.0;

	return result;
}

SGVector<float64_t> CExactInferenceMethod::get_alpha()
{
	if (update_parameter_hash())
		update();

	return SGVector<float64_t>(m_alpha);
}

SGMatrix<float64_t> CExactInferenceMethod::get_cholesky()
{
	if (update_parameter_hash())
		update();

	return SGMatrix<float64_t>(m_L);
}

void CExactInferenceMethod::update_chol()
{
	// get the sigma variable from the Gaussian likelihood model
	CGaussianLikelihood* lik=CGaussianLikelihood::obtain_from_generic(m_model);
	float64_t sigma=lik->get_sigma();
	SG_UNREF(lik);

	/* check whether to allocate cholesky memory */
	if (!m_L.matrix || m_L.num_rows!=m_ktrtr.num_rows)
		m_L=SGMatrix<float64_t>(m_ktrtr.num_rows, m_ktrtr.num_cols);

	/* creates views on kernel and cholesky matrix and perform cholesky */
	Map<MatrixXd> K(m_ktrtr.matrix, m_ktrtr.num_rows, m_ktrtr.num_cols);
	Map<MatrixXd> L(m_L.matrix, m_ktrtr.num_rows, m_ktrtr.num_cols);
	LLT<MatrixXd> llt(K*(CMath::sq(m_scale)/CMath::sq(sigma))+
		MatrixXd::Identity(m_ktrtr.num_rows, m_ktrtr.num_cols));
	L=llt.matrixU();
}

void CExactInferenceMethod::update_alpha()
{
	// get the sigma variable from the Gaussian likelihood model
	CGaussianLikelihood* lik=CGaussianLikelihood::obtain_from_generic(m_model);
	float64_t sigma=lik->get_sigma();
	SG_UNREF(lik);

	// get labels and mean vector and create eigen representation
	SGVector<float64_t> y=((CRegressionLabels*) m_labels)->get_labels();
	Map<VectorXd> eigen_y(y.vector, y.vlen);
	SGVector<float64_t> m=m_mean->get_mean_vector(m_feat);
	Map<VectorXd> eigen_m(m.vector, m.vlen);

	m_alpha=SGVector<float64_t>(y.vlen);

	/* creates views on cholesky matrix and alpha and solve system
	 * (L * L^T) * a = y for a */
	Map<VectorXd> a(m_alpha.vector, m_alpha.vlen);
	Map<MatrixXd> L(m_L.matrix, m_L.num_rows, m_L.num_cols);

	a=L.triangularView<Upper>().adjoint().solve(eigen_y-eigen_m);
	a=L.triangularView<Upper>().solve(a);

	a/=CMath::sq(sigma);
}

#endif // HAVE_EIGEN3
