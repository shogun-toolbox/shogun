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

	SGVector<float64_t> r(func.vlen);
	Map<VectorXd> eigen_r(r.vector, r.vlen);

	// compute ncdf=normal_cdf(y.*f)
	VectorXd eigen_ncdf=eigen_y.cwiseProduct(eigen_f);

	for (index_t j=0; j<eigen_ncdf.size(); j++)
		eigen_ncdf[j]=CStatistics::normal_cdf(eigen_ncdf[j]);

	// compute npdf=normal_pdf(f)=(1/sqrt(2*pi))*exp(-f.^2/2)
	VectorXd eigen_npdf=(1.0/CMath::sqrt(2.0*CMath::PI))*
		(-0.5*eigen_f.array().square()).exp();

	// compute z=npdf/ncdf
	VectorXd eigen_z=eigen_npdf.cwiseQuotient(eigen_ncdf);

	// compute derivatives of log probability wrt f
	if (i == 1)
	{
		// compute the first derivative: dlp=y*z
		eigen_r=eigen_y.cwiseProduct(eigen_z);
	}
	else if (i == 2)
	{
		// compute the second derivative: d2lp=-z.^2-y.*f.*z
		eigen_r=-eigen_z.array().square()-eigen_y.array()*eigen_f.array()*
			eigen_z.array();
	}
	else if (i == 3)
	{
		VectorXd eigen_z2=eigen_z.cwiseProduct(eigen_z);
		VectorXd eigen_z3=eigen_z2.cwiseProduct(eigen_z);

		// compute the third derivative: d3lp=2*y.*z.^3+3*f.*z.^2+z.*y.*(f.^2-1)
		eigen_r=2.0*eigen_y.array()*eigen_z3.array()+3.0*eigen_f.array()*
			eigen_z2.array()+eigen_z.array()*eigen_y.array()*
			(eigen_f.array().square()-1.0);
	}

	return r;
}

SGVector<float64_t> CProbitLikelihood::get_first_derivative(const CLabels* lab,
		const TParameter* param, SGVector<float64_t> func) const
{
	return SGVector<float64_t>();
}

SGVector<float64_t> CProbitLikelihood::get_second_derivative(const CLabels* lab,
		const TParameter* param, SGVector<float64_t> func) const
{
	return SGVector<float64_t>();
}

SGVector<float64_t> CProbitLikelihood::get_third_derivative(const CLabels* lab,
		const TParameter* param, SGVector<float64_t> func) const
{
	return SGVector<float64_t>();
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
