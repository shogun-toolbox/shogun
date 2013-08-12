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
 * Adapted from the GPML toolbox, specifically likT.m
 * http://www.gaussianprocess.org/gpml/code/matlab/doc/
 */

#include <shogun/machine/gp/StudentsTLikelihood.h>

#ifdef HAVE_EIGEN3

#include <shogun/labels/RegressionLabels.h>
#include <shogun/mathematics/Statistics.h>
#include <shogun/mathematics/Math.h>
#include <shogun/mathematics/eigen3.h>

using namespace shogun;
using namespace Eigen;

CStudentsTLikelihood::CStudentsTLikelihood() : CLikelihoodModel()
{
	init();
}

CStudentsTLikelihood::CStudentsTLikelihood(float64_t sigma, float64_t df)
		: CLikelihoodModel()
{
	REQUIRE(sigma>0.0, "Scale parameter must be greater than zero\n")
	REQUIRE(df>1.0, "Number of degrees of freedom must be greater than one\n")

	init();
	m_sigma=sigma;
	m_df=df;
}

void CStudentsTLikelihood::init()
{
	m_sigma=1.0;
	m_df=3.0;
	SG_ADD(&m_df, "df", "Degrees of Freedom.", MS_AVAILABLE);
	SG_ADD(&m_sigma, "sigma", "Scale Parameter.", MS_AVAILABLE);
}

CStudentsTLikelihood::~CStudentsTLikelihood()
{
}

CStudentsTLikelihood* CStudentsTLikelihood::obtain_from_generic(
		CLikelihoodModel* lik)
{
	ASSERT(lik!=NULL);

	if (lik->get_model_type()!=LT_STUDENTST)
		SG_SERROR("Provided likelihood is not of type CStudentsTLikelihood!\n")

	SG_REF(lik);
	return (CStudentsTLikelihood*)lik;
}

SGVector<float64_t> CStudentsTLikelihood::evaluate_log_probabilities(
		SGVector<float64_t> mu,	SGVector<float64_t> s2, const CLabels* lab) const
{
	SG_NOTIMPLEMENTED // since we have not something for integral evaluation yet

	return SGVector<float64_t>();
}

SGVector<float64_t> CStudentsTLikelihood::evaluate_means(SGVector<float64_t> mu,
		SGVector<float64_t> s2, const CLabels* lab) const
{
	return SGVector<float64_t>(mu);
}

SGVector<float64_t> CStudentsTLikelihood::evaluate_variances(
		SGVector<float64_t> mu,	SGVector<float64_t> s2, const CLabels* lab) const
{
	SGVector<float64_t> result(s2);
	Map<VectorXd> eigen_result(result.vector, result.vlen);

	if (m_df<2.0)
		eigen_result=CMath::INFTY*VectorXd::Ones(result.vlen);
	else
	{
		eigen_result+=CMath::sq(m_sigma)*m_df/(m_df-2.0)*
			VectorXd::Ones(result.vlen);
	}

	return result;
}

SGVector<float64_t> CStudentsTLikelihood::get_log_probability_f(const CLabels* lab,
		SGVector<float64_t> func) const
{
	// check the parameters
	REQUIRE(lab, "Labels are required (lab should not be NULL)\n")
	REQUIRE(lab->get_label_type()==LT_REGRESSION,
			"Labels must be type of CRegressionLabels\n")
	REQUIRE(lab->get_num_labels()==func.vlen, "Number of labels must match "
			"length of the function vector\n")

	Map<VectorXd> eigen_f(func.vector, func.vlen);

	SGVector<float64_t> r(func.vlen);
	Map<VectorXd> eigen_r(r.vector, r.vlen);

	SGVector<float64_t> y=((CRegressionLabels*)lab)->get_labels();
	Map<VectorXd> eigen_y(y.vector, y.vlen);

	// compute lZ=log(gamma(df/2+1/2))-log(gamma(df/2))-log(df*pi*sigma^2)/2
	VectorXd eigen_lZ=(CStatistics::lgamma(m_df/2.0+0.5)-
		CStatistics::lgamma(m_df/2.0)-log(m_df*CMath::PI*CMath::sq(m_sigma))/2.0)*
		VectorXd::Ones(r.vlen);

	// compute log probability: lp=lZ-(df+1)*log(1+(y-f).^2./(df*sigma^2))/2
	eigen_r=eigen_y-eigen_f;
	eigen_r=eigen_r.cwiseProduct(eigen_r)/(m_df*CMath::sq(m_sigma));
	eigen_r=eigen_lZ-(m_df+1)*
		(eigen_r+VectorXd::Ones(r.vlen)).array().log().matrix()/2.0;

	return r;
}

SGVector<float64_t> CStudentsTLikelihood::get_log_probability_derivative_f(
		const CLabels* lab, SGVector<float64_t> func, index_t i) const
{
	// check the parameters
	REQUIRE(lab, "Labels are required (lab should not be NULL)\n")
	REQUIRE(lab->get_label_type()==LT_REGRESSION,
			"Labels must be type of CRegressionLabels\n")
	REQUIRE(lab->get_num_labels()==func.vlen, "Number of labels must match "
			"length of the function vector\n")
	REQUIRE(i>=1 && i<=3, "Index for derivative should be 1, 2 or 3\n")

	Map<VectorXd> eigen_f(func.vector, func.vlen);

	SGVector<float64_t> r(func.vlen);
	Map<VectorXd> eigen_r(r.vector, r.vlen);

	SGVector<float64_t> y=((CRegressionLabels*)lab)->get_labels();
	Map<VectorXd> eigen_y(y.vector, y.vlen);

	// compute r=y-f, r2=r.^2
	eigen_r=eigen_y-eigen_f;
	VectorXd eigen_r2=eigen_r.cwiseProduct(eigen_r);

	// compute a=(y-f).^2+df*sigma^2
	VectorXd a=eigen_r2+VectorXd::Ones(r.vlen)*m_df*CMath::sq(m_sigma);

	if (i==1)
	{
		// compute first derivative of log probability wrt f:
		// dlp=(df+1)*(y-f)./a
		eigen_r=(m_df+1)*eigen_r.cwiseQuotient(a);
	}
	else if (i==2)
	{
		// compute second derivative of log probability wrt f:
		// d2lp=(df+1)*((y-f)^2-df*sigma^2)./a.^2
		VectorXd b=eigen_r2-VectorXd::Ones(r.vlen)*m_df*CMath::sq(m_sigma);

		eigen_r=(m_df+1)*b.cwiseQuotient(a.cwiseProduct(a));
	}
	else if (i==3)
	{
		// compute third derivative of log probability wrt f:
		// d3lp=(f+1)*2*(y-f).*((y-f)^2-3*f*sigma^2)./a.^3
		VectorXd c=eigen_r2-VectorXd::Ones(r.vlen)*3*m_df*CMath::sq(m_sigma);
		VectorXd a2=a.cwiseProduct(a);

		eigen_r=(m_df+1)*2*eigen_r.cwiseProduct(c).cwiseQuotient(
			a2.cwiseProduct(a));
	}

	return r;
}

SGVector<float64_t> CStudentsTLikelihood::get_first_derivative(const CLabels* lab,
		const TParameter* param, SGVector<float64_t> func) const
{
	// check the parameters
	REQUIRE(lab, "Labels are required (lab should not be NULL)\n")
	REQUIRE(lab->get_label_type()==LT_REGRESSION,
			"Labels must be type of CRegressionLabels\n")
	REQUIRE(lab->get_num_labels()==func.vlen, "Number of labels must match "
			"length of the function vector\n")

	SGVector<float64_t> r(func.vlen);
	Map<VectorXd> eigen_r(r.vector, r.vlen);

	Map<VectorXd> eigen_f(func.vector, func.vlen);

	SGVector<float64_t> y=((CRegressionLabels*)lab)->get_labels();
	Map<VectorXd> eigen_y(y.vector, y.vlen);

	// compute r=y-f and r2=(y-f).^2
	eigen_r=eigen_y-eigen_f;
	VectorXd eigen_r2=eigen_r.cwiseProduct(eigen_r);

	if (!strcmp(param->m_name, "df"))
	{
		// compute derivative of log probability wrt df:
		// lp_ddf=df*(dloggamma(df/2+1/2)-dloggamma(df/2))/2-1/2-
		// df*log(1+r2/(df*sigma^2))/2 +(df/2+1/2)*r2./(df*sigma^2+r2)
		eigen_r=(m_df*(CStatistics::dlgamma(m_df*0.5+0.5)-
			CStatistics::dlgamma(m_df*0.5))*0.5-0.5)*VectorXd::Ones(r.vlen);

		eigen_r-=m_df*(VectorXd::Ones(r.vlen)+
			eigen_r2/(m_df*CMath::sq(m_sigma))).array().log().matrix()/2.0;

		eigen_r+=(m_df/2.0+0.5)*eigen_r2.cwiseQuotient(
			eigen_r2+VectorXd::Ones(r.vlen)*(m_df*CMath::sq(m_sigma)));

		eigen_r*=(1.0-1.0/m_df);

		return r;
	}
	else if (!strcmp(param->m_name, "sigma"))
	{
		// compute derivative of log probability wrt sigma:
		// lp_dsigma=(df+1)*r2./a-1
		eigen_r=(m_df+1)*eigen_r2.cwiseQuotient(eigen_r2+
			VectorXd::Ones(r.vlen)*(m_df*CMath::sq(m_sigma)));
		eigen_r-=VectorXd::Ones(r.vlen);

		return r;
	}

	return SGVector<float64_t>();
}

SGVector<float64_t> CStudentsTLikelihood::get_second_derivative(const CLabels* lab,
		const TParameter* param, SGVector<float64_t> func) const
{
	// check the parameters
	REQUIRE(lab, "Labels are required (lab should not be NULL)\n")
	REQUIRE(lab->get_label_type()==LT_REGRESSION,
			"Labels must be type of CRegressionLabels\n")
	REQUIRE(lab->get_num_labels()==func.vlen, "Number of labels must match "
			"length of the function vector\n")

	SGVector<float64_t> r(func.vlen);
	Map<VectorXd> eigen_r(r.vector, r.vlen);

	Map<VectorXd> eigen_f(func.vector, func.vlen);

	SGVector<float64_t> y=((CRegressionLabels*)lab)->get_labels();
	Map<VectorXd> eigen_y(y.vector, y.vlen);

	// compute r=y-f and r2=(y-f).^2
	eigen_r=eigen_y-eigen_f;
	VectorXd eigen_r2=eigen_r.cwiseProduct(eigen_r);

	// compute a=r+sigma^2*df and a2=a.^2
	VectorXd a=eigen_r2+CMath::sq(m_sigma)*m_df*VectorXd::Ones(r.vlen);
	VectorXd a2=a.cwiseProduct(a);

	if (!strcmp(param->m_name, "df"))
	{
		// compute derivative of first derivative of log probability wrt df:
		// dlp_ddf=df*r.*(a-sigma^2*(df+1))./a2
		eigen_r=m_df*eigen_r.cwiseProduct(a-CMath::sq(m_sigma)*(m_df+1.0)*
			VectorXd::Ones(r.vlen)).cwiseQuotient(a2);
		eigen_r*=(1.0-1.0/m_df);

		return r;
	}
	else if (!strcmp(param->m_name, "sigma"))
	{
		// compute derivative of first derivative of log probability wrt
		// sigma: dlp_dsigma=-(df+1)*2*df*sigma^2*r./a2
		eigen_r=-(m_df+1.0)*2*m_df*CMath::sq(m_sigma)*
			eigen_r.cwiseQuotient(a2);

		return r;
	}

	return SGVector<float64_t>();
}

SGVector<float64_t> CStudentsTLikelihood::get_third_derivative(const CLabels* lab,
		const TParameter* param, SGVector<float64_t> func) const
{
	// check the parameters
	REQUIRE(lab, "Labels are required (lab should not be NULL)\n")
	REQUIRE(lab->get_label_type()==LT_REGRESSION,
			"Labels must be type of CRegressionLabels\n")
	REQUIRE(lab->get_num_labels()==func.vlen, "Number of labels must match "
			"length of the function vector\n")

	SGVector<float64_t> r(func.vlen);
	Map<VectorXd> eigen_r(r.vector, r.vlen);

	Map<VectorXd> eigen_f(func.vector, func.vlen);

	SGVector<float64_t> y=((CRegressionLabels*)lab)->get_labels();
	Map<VectorXd> eigen_y(y.vector, y.vlen);

	// compute r=y-f and r2=(y-f).^2
	eigen_r=eigen_y-eigen_f;
	VectorXd eigen_r2=eigen_r.cwiseProduct(eigen_r);

	// compute a=r+sigma^2*df and a3=a.^3
	VectorXd a=eigen_r2+CMath::sq(m_sigma)*m_df*VectorXd::Ones(r.vlen);
	VectorXd a3=(a.cwiseProduct(a)).cwiseProduct(a);

	if (!strcmp(param->m_name, "df"))
	{
		// compute derivative of second derivative of log probability wrt
		// df: d2lp_ddf=df*(r2.*(r2-3*sigma^2*(1+df))+df*sigma^4)./a3
		float64_t sigma2=CMath::sq(m_sigma);

		eigen_r=m_df*(eigen_r2.cwiseProduct(eigen_r2-3*sigma2*(1.0+m_df)*
			VectorXd::Ones(r.vlen))+(m_df*CMath::sq(sigma2))*VectorXd::Ones(r.vlen));
		eigen_r=eigen_r.cwiseQuotient(a3);

		eigen_r*=(1.0-1.0/m_df);

		return r;
	}
	else if (!strcmp(param->m_name, "sigma"))
	{
		// compute derivative of second derivative of log probability wrt
		// sigma: d2lp_dsigma=(df+1)*2*df*sigma^2*(a-4*r2)./a3
		eigen_r=(m_df+1.0)*2*m_df*CMath::sq(m_sigma)*
			(a-4.0*eigen_r2).cwiseQuotient(a3);

		return r;
	}

	return SGVector<float64_t>();
}

#endif // HAVE_EIGEN3
