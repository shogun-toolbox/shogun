/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Roman Votyakov
 */

#include <shogun/machine/gp/ProbitLikelihood.h>

#ifdef HAVE_EIGEN3

#include <shogun/labels/BinaryLabels.h>
#include <shogun/mathematics/eigen3.h>
#include <shogun/mathematics/Statistics.h>

using namespace shogun;
using namespace Eigen;

CProbitLikelihood::CProbitLikelihood()
{
}

CProbitLikelihood::~CProbitLikelihood()
{
}

SGVector<float64_t> CProbitLikelihood::get_predictive_means(
		SGVector<float64_t> mu, SGVector<float64_t> s2, const CLabels* lab) const
{
	SGVector<float64_t> lp=get_log_zeroth_moments(mu, s2, lab);
	Map<VectorXd> eigen_lp(lp.vector, lp.vlen);

	SGVector<float64_t> r(lp.vlen);
	Map<VectorXd> eigen_r(r.vector, r.vlen);

	// evaluate predictive mean: ymu=2*p-1
	eigen_r=2.0*eigen_lp.array().exp()-1.0;

	return r;
}

SGVector<float64_t> CProbitLikelihood::get_predictive_variances(
		SGVector<float64_t> mu, SGVector<float64_t> s2, const CLabels* lab) const
{
	SGVector<float64_t> lp=get_log_zeroth_moments(mu, s2, lab);
	Map<VectorXd> eigen_lp(lp.vector, lp.vlen);

	SGVector<float64_t> r(lp.vlen);
	Map<VectorXd> eigen_r(r.vector, r.vlen);

	// evaluate predictive variance: ys2=1-(2*p-1).^2
	eigen_r=1-(2.0*eigen_lp.array().exp()-1.0).square();

	return r;
}

SGVector<float64_t> CProbitLikelihood::get_log_probability_f(const CLabels* lab,
		SGVector<float64_t> func) const
{
	// check the parameters
	REQUIRE(lab, "Labels are required (lab should not be NULL)\n")
	REQUIRE(lab->get_label_type()==LT_BINARY,
			"Labels must be type of CBinaryLabels\n")
	REQUIRE(lab->get_num_labels()==func.vlen, "Number of labels must match "
			"length of the function vector\n")

	SGVector<float64_t> y=((CBinaryLabels*)lab)->get_labels();
	Map<VectorXd> eigen_y(y.vector, y.vlen);

	Map<VectorXd> eigen_f(func.vector, func.vlen);

	SGVector<float64_t> r(func.vlen);
	Map<VectorXd> eigen_r(r.vector, r.vlen);

	// compute log pobability: log(normal_cdf(f.*y))
	eigen_r=eigen_y.cwiseProduct(eigen_f);

	for (index_t i=0; i<eigen_r.size(); i++)
		eigen_r[i]=CStatistics::lnormal_cdf(eigen_r[i]);

	return r;
}

SGVector<float64_t> CProbitLikelihood::get_log_probability_derivative_f(
		const CLabels* lab, SGVector<float64_t> func, index_t i) const
{
	// check the parameters
	REQUIRE(lab, "Labels are required (lab should not be NULL)\n")
	REQUIRE(lab->get_label_type()==LT_BINARY,
			"Labels must be type of CBinaryLabels\n")
	REQUIRE(lab->get_num_labels()==func.vlen, "Number of labels must match "
			"length of the function vector\n")
	REQUIRE(i>=1 && i<=3, "Index for derivative should be 1, 2 or 3\n")

	SGVector<float64_t> y=((CBinaryLabels*)lab)->get_labels();
	Map<VectorXd> eigen_y(y.vector, y.vlen);

	Map<VectorXd> eigen_f(func.vector, func.vlen);

	SGVector<float64_t> dlp(func.vlen);
	Map<VectorXd> eigen_dlp(dlp.vector, dlp.vlen);

	VectorXd eigen_yf=eigen_y.cwiseProduct(eigen_f);

	for (index_t j=0; j<eigen_yf.size(); j++)
	{
		float64_t v = eigen_yf[j];
		if (v<CStatistics::ERFC_CASE2)
		{
			//dlp( id2) = abs(den./num) * sqrt(2/pi); % strictly positive first derivative
			eigen_dlp[j]=CMath::sqrt(2.0/CMath::PI)
				/CMath::abs(CStatistics::erfc8_weighted_sum(v));
		}
		else
		{
			//dlp(~id2) = exp(-z(~id2).*z(~id2)/2-lp(~id2))/sqrt(2*pi); % safe computation
			eigen_dlp[j]=CMath::exp(-v*v/2.0-CStatistics::lnormal_cdf(v))
				/CMath::sqrt(2.0*CMath::PI);
		}
	}

	SGVector<float64_t> r(func.vlen);
	Map<VectorXd> eigen_r(r.vector, r.vlen);

	// compute derivatives of log probability wrt f

	if (i==1)
		eigen_r=eigen_dlp;
	else
		//d2lp = -dlp.*abs(z+dlp);             % strictly negative second derivative
		eigen_r=(-eigen_dlp.array()*((eigen_yf.array()+eigen_dlp.array()).abs().array())).matrix();

	if (i==3)
		//d3lp = -d2lp.*abs(z+2*dlp)-dlp;     % strictly positive third derivative
		eigen_r=(-eigen_r.array()*((eigen_yf.array()+2.0*eigen_dlp.array()).abs().array())
		 -eigen_dlp.array()).matrix();

	if (i==1 || i==3)
	{
		//varargout{2} = y.*varargout{2}
		//varargout{4} = y.*varargout{4}
		eigen_r=(eigen_r.array()*eigen_y.array()).matrix();
	}

	return r;
}

SGVector<float64_t> CProbitLikelihood::get_log_zeroth_moments(
		SGVector<float64_t> mu, SGVector<float64_t> s2, const CLabels* lab) const
{
	SGVector<float64_t> y;

	if (lab)
	{
		REQUIRE((mu.vlen==s2.vlen) && (mu.vlen==lab->get_num_labels()),
				"Length of the vector of means (%d), length of the vector of "
				"variances (%d) and number of labels (%d) should be the same\n",
				mu.vlen, s2.vlen, lab->get_num_labels())
		REQUIRE(lab->get_label_type()==LT_BINARY,
				"Labels must be type of CBinaryLabels\n")

		y=((CBinaryLabels*)lab)->get_labels();
	}
	else
	{
		REQUIRE(mu.vlen==s2.vlen, "Length of the vector of means (%d) and "
				"length of the vector of variances (%d) should be the same\n",
				mu.vlen, s2.vlen)

		y=SGVector<float64_t>(mu.vlen);
		y.set_const(1.0);
	}

	Map<VectorXd> eigen_y(y.vector, y.vlen);
	Map<VectorXd> eigen_mu(mu.vector, mu.vlen);
	Map<VectorXd> eigen_s2(s2.vector, s2.vlen);

	SGVector<float64_t> r(y.vlen);
	Map<VectorXd> eigen_r(r.vector, r.vlen);

	// compute: lp=log(normal_cdf((mu.*y)./sqrt(1+sigma^2)))
	eigen_r=eigen_mu.array()*eigen_y.array()/((1.0+eigen_s2.array()).sqrt());

	for (index_t i=0; i<eigen_r.size(); i++)
		eigen_r[i]=CStatistics::lnormal_cdf(eigen_r[i]);

	return r;
}

float64_t CProbitLikelihood::get_first_moment(SGVector<float64_t> mu,
		SGVector<float64_t> s2, const CLabels *lab, index_t i) const
{
	// check the parameters
	REQUIRE(lab, "Labels are required (lab should not be NULL)\n")
	REQUIRE((mu.vlen==s2.vlen) && (mu.vlen==lab->get_num_labels()),
			"Length of the vector of means (%d), length of the vector of "
			"variances (%d) and number of labels (%d) should be the same\n",
			mu.vlen, s2.vlen, lab->get_num_labels())
	REQUIRE(i>=0 && i<=mu.vlen, "Index (%d) out of bounds!\n", i)
	REQUIRE(lab->get_label_type()==LT_BINARY,
			"Labels must be type of CBinaryLabels\n")

	SGVector<float64_t> y=((CBinaryLabels*)lab)->get_labels();

	float64_t z=y[i]*mu[i]/CMath::sqrt(1.0+s2[i]);

	// compute ncdf=normal_cdf(z)
	float64_t ncdf=CStatistics::normal_cdf(z);

	// compute npdf=normal_pdf(z)=(1/sqrt(2*pi))*exp(-z.^2/2)
	float64_t npdf=(1.0/CMath::sqrt(2.0*CMath::PI))*CMath::exp(-0.5*CMath::sq(z));

	// compute the 1st moment: E[x] = mu + (y*s2*N(z))/(Phi(z)*sqrt(1+s2))
	float64_t Ex=mu[i]+(npdf/ncdf)*(y[i]*s2[i])/CMath::sqrt(1.0+s2[i]);

	return Ex;
}

float64_t CProbitLikelihood::get_second_moment(SGVector<float64_t> mu,
		SGVector<float64_t> s2, const CLabels *lab, index_t i) const
{
	// check the parameters
	REQUIRE(lab, "Labels are required (lab should not be NULL)\n")
	REQUIRE((mu.vlen==s2.vlen) && (mu.vlen==lab->get_num_labels()),
			"Length of the vector of means (%d), length of the vector of "
			"variances (%d) and number of labels (%d) should be the same\n",
			mu.vlen, s2.vlen, lab->get_num_labels())
	REQUIRE(i>=0 && i<=mu.vlen, "Index (%d) out of bounds!\n", i)
	REQUIRE(lab->get_label_type()==LT_BINARY,
			"Labels must be type of CBinaryLabels\n")

	SGVector<float64_t> y=((CBinaryLabels*)lab)->get_labels();

	float64_t z=y[i]*mu[i]/CMath::sqrt(1.0+s2[i]);

	// compute ncdf=normal_cdf(z)
	float64_t ncdf=CStatistics::normal_cdf(z);

	// compute npdf=normal_pdf(z)=(1/sqrt(2*pi))*exp(-z.^2/2)
	float64_t npdf=(1.0/CMath::sqrt(2.0*CMath::PI))*CMath::exp(-0.5*CMath::sq(z));

	SGVector<float64_t> r(y.vlen);
	Map<VectorXd> eigen_r(r.vector, r.vlen);

	// compute the 2nd moment:
	// Var[x] = s2 - (s2^2*N(z))/((1+s2)*Phi(z))*(z+N(z)/Phi(z))
	float64_t Var=s2[i]-(CMath::sq(s2[i])/(1.0+s2[i]))*(npdf/ncdf)*(z+(npdf/ncdf));

	return Var;
}

#endif /* HAVE_EIGEN3 */
