/*
 * Copyright (c) The Shogun Machine Learning Toolbox
 * Written (W) 2013 Roman Votyakov
 * Written (W) 2012 Jacob Walker
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
 * Adapted from the GPML toolbox, specifically likT.m
 * http://www.gaussianprocess.org/gpml/code/matlab/doc/
 */
#include <shogun/machine/gp/StudentsTLikelihood.h>


#include <shogun/mathematics/Function.h>
#ifdef USE_GPL_SHOGUN
#include <shogun/mathematics/Integration.h>
#endif //USE_GPL_SHOGUN
#include <shogun/labels/RegressionLabels.h>
#include <shogun/mathematics/Statistics.h>
#include <shogun/mathematics/Math.h>
#include <shogun/mathematics/eigen3.h>

#include <utility>

using namespace shogun;
using namespace Eigen;

namespace shogun
{

#ifndef DOXYGEN_SHOULD_SKIP_THIS

/** Class of the probability density function of the normal distribution */
class NormalPDF : public Function
{
public:
	/** default constructor */
	NormalPDF()
	{
		m_mu=0.0;
		m_sigma=1.0;
	}

	/** constructor
	 *
	 * @param mu mean
	 * @param sigma standard deviation
	 */
	NormalPDF(float64_t mu, float64_t sigma)
	{
		m_mu=mu;
		m_sigma=sigma;
	}

	/** set mean
	 *
	 * @param mu mean to set
	 */
	void set_mu(float64_t mu) { m_mu=mu; }

	/** set standard deviation
	 *
	 * @param sigma standard deviation to set
	 */
	void set_sigma(float64_t sigma) { m_sigma=sigma; }

	/** returns value of the function at given point
	 *
	 * @param x argument
	 *
	 * @return f(x)=(1/sqrt(2*PI*sigma^2))*exp(-(x-mu)^2/(2*sigma^2))
	 */
	float64_t operator() (float64_t x) override
	{
		return (1.0 / (std::sqrt(2 * Math::PI) * m_sigma)) *
			   std::exp(-Math::sq(x - m_mu) / (2.0 * Math::sq(m_sigma)));
	}

private:
	/* mean */
	float64_t m_mu;

	/* standard deviation */
	float64_t m_sigma;
};

/** Class of the probability density function of the non-standardized Student's
 * t-distribution
 */
class StudentsTPDF : public Function
{
public:
	/** default constructor */
	StudentsTPDF()
	{
		m_sigma=1.0;
		m_mu=0.0;
		m_nu=3.0;
	}

	/** constructor
	 *
	 * @param sigma scale parameter
	 * @param nu degrees of freedom
	 * @param mu location parameter
	 */
	StudentsTPDF(float64_t sigma, float64_t nu, float64_t mu)
	{
		m_sigma=sigma;
		m_mu=mu;
		m_nu=nu;
	}

	/** set scale
	 *
	 * @param sigma scale to set
	 */
	void set_sigma(float64_t sigma) { m_sigma=sigma; }

	/** set location parameter
	 *
	 * @param mu location parameter to set
	 */
	void set_mu(float64_t mu) { m_mu=mu; }

	/** set degrees of freedom
	 *
	 * @param nu degrees of freedom to set
	 */
	void set_nu(float64_t nu) { m_nu=nu; }

	/** returns value of the function at given point
	 *
	 * @param x argument
	 *
	 * @return f(x)=Gamma((nu+1)/2)/(Gamma(nu/2)*sqrt(nu*pi*sigma^2))*
	 * (1+1/nu*(x-mu)^2/sigma^2)^(-(nu+1)/2)
	 */
	float64_t operator() (float64_t x) override
	{
		float64_t lZ = Statistics::lgamma((m_nu + 1.0) / 2.0) -
			           Statistics::lgamma(m_nu / 2.0) -
			           std::log(m_nu * Math::PI * Math::sq(m_sigma)) / 2.0;
		return std::exp(
			lZ -
			((m_nu + 1.0) / 2.0) *
			    std::log(
			        1.0 + Math::sq(x - m_mu) / (m_nu * Math::sq(m_sigma))));
	}

private:
	/** scale parameter */
	float64_t m_sigma;

	/** location parameter */
	float64_t m_mu;

	/** degrees of freedom */
	float64_t m_nu;
};

/** Class of the function, which is a product of two given functions */
class ProductFunction : public Function
{
public:
	/** constructor
	 *
	 * @param f f(x)
	 * @param g g(x)
	 */
	ProductFunction(std::shared_ptr<Function> f, std::shared_ptr<Function> g)
	{


		m_f=std::move(f);
		m_g=std::move(g);
	}

	~ProductFunction() override
	{


	}

	/** returns value of the function at given point
	 *
	 * @param x argument
	 *
	 * @return h(x)=f(x)*g(x)
	 */
	float64_t operator() (float64_t x) override
	{
		return (*m_f)(x)*(*m_g)(x);
	}

private:
	/** function f(x) */
	std::shared_ptr<Function> m_f;
	/**	function g(x) */
	std::shared_ptr<Function> m_g;
};

/** Class of the f(x)=x */
class LinearFunction : public Function
{
public:
	/** default constructor */
	LinearFunction() { }

	~LinearFunction() override { }

	/** returns value of the function at given point
	 *
	 * @param x argument
	 *
	 * @return f(x)=x
	 */
	float64_t operator() (float64_t x) override
	{
		return x;
	}
};

/** Class of the f(x)=x^2 */
class QuadraticFunction : public Function
{
public:
	/** default constructor */
	QuadraticFunction() { }

	~QuadraticFunction() override { }

	/** returns value of the function at given point
	 *
	 * @param x argument
	 *
	 * @return f(x)=x^2
	 */
	float64_t operator() (float64_t x) override
	{
		return Math::sq(x);
	}
};

#endif /* DOXYGEN_SHOULD_SKIP_THIS */

StudentsTLikelihood::StudentsTLikelihood() : LikelihoodModel()
{
	init();
}

StudentsTLikelihood::StudentsTLikelihood(float64_t sigma, float64_t df)
		: LikelihoodModel()
{
	init();
	set_sigma(sigma);
	set_degrees_freedom(df);
}

void StudentsTLikelihood::init()
{
	m_log_sigma=0.0;
	m_log_df = std::log(2.0);
	SG_ADD(&m_log_df, "log_df", "Degrees of freedom in log domain", ParameterProperties::HYPER | ParameterProperties::GRADIENT);
	SG_ADD(&m_log_sigma, "log_sigma", "Scale parameter in log domain", ParameterProperties::HYPER | ParameterProperties::GRADIENT);
}

StudentsTLikelihood::~StudentsTLikelihood()
{
}

std::shared_ptr<StudentsTLikelihood> StudentsTLikelihood::obtain_from_generic(
		const std::shared_ptr<LikelihoodModel>& lik)
{
	ASSERT(lik!=NULL);

	if (lik->get_model_type()!=LT_STUDENTST)
		error("Provided likelihood is not of type StudentsTLikelihood!");

	return lik->as<StudentsTLikelihood>();
}

SGVector<float64_t> StudentsTLikelihood::get_predictive_means(
		SGVector<float64_t> mu, SGVector<float64_t> s2, std::shared_ptr<const Labels> lab) const
{
	return SGVector<float64_t>(mu);
}

SGVector<float64_t> StudentsTLikelihood::get_predictive_variances(
		SGVector<float64_t> mu, SGVector<float64_t> s2, std::shared_ptr<const Labels> lab) const
{
	SGVector<float64_t> result(s2);
	Map<VectorXd> eigen_result(result.vector, result.vlen);
	float64_t df=get_degrees_freedom();
	if (df<2.0)
		eigen_result=Math::INFTY*VectorXd::Ones(result.vlen);
	else
	{
		eigen_result += std::exp(m_log_sigma * 2.0) * df / (df - 2.0) *
			            VectorXd::Ones(result.vlen);
	}

	return result;
}

SGVector<float64_t> StudentsTLikelihood::get_log_probability_f(std::shared_ptr<const Labels> lab,
		SGVector<float64_t> func) const
{
	// check the parameters
	require(lab, "Labels are required (lab should not be NULL)");
	require(lab->get_label_type()==LT_REGRESSION,
			"Labels must be type of RegressionLabels");
	require(lab->get_num_labels()==func.vlen, "Number of labels must match "
			"length of the function vector");

	Map<VectorXd> eigen_f(func.vector, func.vlen);

	SGVector<float64_t> r(func.vlen);
	Map<VectorXd> eigen_r(r.vector, r.vlen);

	SGVector<float64_t> y=lab->as<RegressionLabels>()->get_labels();
	Map<VectorXd> eigen_y(y.vector, y.vlen);

	float64_t df=get_degrees_freedom();
	// compute lZ=log(gamma(df/2+1/2))-log(gamma(df/2))-log(df*pi*sigma^2)/2
	VectorXd eigen_lZ =
		(Statistics::lgamma(df / 2.0 + 0.5) - Statistics::lgamma(df / 2.0) -
		 log(df * Math::PI * std::exp(m_log_sigma * 2.0)) / 2.0) *
		VectorXd::Ones(r.vlen);

	// compute log probability: lp=lZ-(df+1)*log(1+(y-f).^2./(df*sigma^2))/2
	eigen_r=eigen_y-eigen_f;
	eigen_r =
		eigen_r.cwiseProduct(eigen_r) / (df * std::exp(m_log_sigma * 2.0));
	eigen_r=eigen_lZ-(df+1)*
		(eigen_r+VectorXd::Ones(r.vlen)).array().log().matrix()/2.0;

	return r;
}

SGVector<float64_t> StudentsTLikelihood::get_log_probability_derivative_f(
		std::shared_ptr<const Labels> lab, SGVector<float64_t> func, index_t i) const
{
	// check the parameters
	require(lab, "Labels are required (lab should not be NULL)");
	require(lab->get_label_type()==LT_REGRESSION,
			"Labels must be type of RegressionLabels");
	require(lab->get_num_labels()==func.vlen, "Number of labels must match "
			"length of the function vector");
	require(i>=1 && i<=3, "Index for derivative should be 1, 2 or 3");

	Map<VectorXd> eigen_f(func.vector, func.vlen);

	SGVector<float64_t> r(func.vlen);
	Map<VectorXd> eigen_r(r.vector, r.vlen);

	SGVector<float64_t> y=lab->as<RegressionLabels>()->get_labels();
	Map<VectorXd> eigen_y(y.vector, y.vlen);

	// compute r=y-f, r2=r.^2
	eigen_r=eigen_y-eigen_f;
	VectorXd eigen_r2=eigen_r.cwiseProduct(eigen_r);
	float64_t df=get_degrees_freedom();

	// compute a=(y-f).^2+df*sigma^2
	VectorXd a =
		eigen_r2 + VectorXd::Ones(r.vlen) * df * std::exp(m_log_sigma * 2.0);

	if (i==1)
	{
		// compute first derivative of log probability wrt f:
		// dlp=(df+1)*(y-f)./a
		eigen_r=(df+1)*eigen_r.cwiseQuotient(a);
	}
	else if (i==2)
	{
		// compute second derivative of log probability wrt f:
		// d2lp=(df+1)*((y-f)^2-df*sigma^2)./a.^2
		VectorXd b = eigen_r2 -
			         VectorXd::Ones(r.vlen) * df * std::exp(m_log_sigma * 2.0);

		eigen_r=(df+1)*b.cwiseQuotient(a.cwiseProduct(a));
	}
	else if (i==3)
	{
		// compute third derivative of log probability wrt f:
		// d3lp=(f+1)*2*(y-f).*((y-f)^2-3*f*sigma^2)./a.^3
		VectorXd c =
			eigen_r2 -
			VectorXd::Ones(r.vlen) * 3 * df * std::exp(m_log_sigma * 2.0);
		VectorXd a2=a.cwiseProduct(a);

		eigen_r=(df+1)*2*eigen_r.cwiseProduct(c).cwiseQuotient(
			a2.cwiseProduct(a));
	}

	return r;
}

SGVector<float64_t> StudentsTLikelihood::get_first_derivative(std::shared_ptr<const Labels> lab,
		SGVector<float64_t> func, Parameters::const_reference param) const
{
	// check the parameters
	require(lab, "Labels are required (lab should not be NULL)");
	require(lab->get_label_type()==LT_REGRESSION,
			"Labels must be type of RegressionLabels");
	require(lab->get_num_labels()==func.vlen, "Number of labels must match "
			"length of the function vector");

	SGVector<float64_t> r(func.vlen);
	Map<VectorXd> eigen_r(r.vector, r.vlen);

	Map<VectorXd> eigen_f(func.vector, func.vlen);

	SGVector<float64_t> y=lab->as<RegressionLabels>()->get_labels();
	Map<VectorXd> eigen_y(y.vector, y.vlen);

	// compute r=y-f and r2=(y-f).^2
	eigen_r=eigen_y-eigen_f;
	VectorXd eigen_r2=eigen_r.cwiseProduct(eigen_r);
	float64_t df=get_degrees_freedom();

	if (param.first == "log_df")
	{
		// compute derivative of log probability wrt df:
		// lp_ddf=df*(dloggamma(df/2+1/2)-dloggamma(df/2))/2-1/2-
		// df*log(1+r2/(df*sigma^2))/2 +(df/2+1/2)*r2./(df*sigma^2+r2)
		eigen_r=(df*(Statistics::dlgamma(df*0.5+0.5)-
			Statistics::dlgamma(df*0.5))*0.5-0.5)*VectorXd::Ones(r.vlen);

		eigen_r -= df *
			       (VectorXd::Ones(r.vlen) +
			        eigen_r2 / (df * std::exp(m_log_sigma * 2.0)))
			           .array()
			           .log()
			           .matrix() /
			       2.0;

		eigen_r +=
			(df / 2.0 + 0.5) *
			eigen_r2.cwiseQuotient(
			    eigen_r2 +
			    VectorXd::Ones(r.vlen) * (df * std::exp(m_log_sigma * 2.0)));

		eigen_r*=(1.0-1.0/df);

		return r;
	}
	else if (param.first == "log_sigma")
	{
		// compute derivative of log probability wrt sigma:
		// lp_dsigma=(df+1)*r2./a-1
		eigen_r =
			(df + 1) *
			eigen_r2.cwiseQuotient(
			    eigen_r2 +
			    VectorXd::Ones(r.vlen) * (df * std::exp(m_log_sigma * 2.0)));
		eigen_r-=VectorXd::Ones(r.vlen);

		return r;
	}

	return SGVector<float64_t>();
}

SGVector<float64_t> StudentsTLikelihood::get_second_derivative(std::shared_ptr<const Labels> lab,
		SGVector<float64_t> func, Parameters::const_reference param) const
{
	// check the parameters
	require(lab, "Labels are required (lab should not be NULL)");
	require(lab->get_label_type()==LT_REGRESSION,
			"Labels must be type of RegressionLabels");
	require(lab->get_num_labels()==func.vlen, "Number of labels must match "
			"length of the function vector");

	SGVector<float64_t> r(func.vlen);
	Map<VectorXd> eigen_r(r.vector, r.vlen);

	Map<VectorXd> eigen_f(func.vector, func.vlen);

	SGVector<float64_t> y=lab->as<RegressionLabels>()->get_labels();
	Map<VectorXd> eigen_y(y.vector, y.vlen);

	// compute r=y-f and r2=(y-f).^2
	eigen_r=eigen_y-eigen_f;
	VectorXd eigen_r2=eigen_r.cwiseProduct(eigen_r);
	float64_t df=get_degrees_freedom();

	// compute a=r+sigma^2*df and a2=a.^2
	VectorXd a =
		eigen_r2 + std::exp(m_log_sigma * 2.0) * df * VectorXd::Ones(r.vlen);
	VectorXd a2=a.cwiseProduct(a);

	if (param.first == "log_df")
	{
		// compute derivative of first derivative of log probability wrt df:
		// dlp_ddf=df*r.*(a-sigma^2*(df+1))./a2
		eigen_r = df *
			      eigen_r
			          .cwiseProduct(
			              a -
			              std::exp(m_log_sigma * 2.0) * (df + 1.0) *
			                  VectorXd::Ones(r.vlen))
			          .cwiseQuotient(a2);
		eigen_r*=(1.0-1.0/df);

		return r;
	}
	else if (param.first == "log_sigma")
	{
		// compute derivative of first derivative of log probability wrt sigma:
		// dlp_dsigma=-(df+1)*2*df*sigma^2*r./a2
		eigen_r = -(df + 1.0) * 2 * df * std::exp(m_log_sigma * 2.0) *
			      eigen_r.cwiseQuotient(a2);

		return r;
	}

	return SGVector<float64_t>();
}

SGVector<float64_t> StudentsTLikelihood::get_third_derivative(std::shared_ptr<const Labels> lab,
		SGVector<float64_t> func, Parameters::const_reference param) const
{
	// check the parameters
	require(lab, "Labels are required (lab should not be NULL)");
	require(lab->get_label_type()==LT_REGRESSION,
			"Labels must be type of RegressionLabels");
	require(lab->get_num_labels()==func.vlen, "Number of labels must match "
			"length of the function vector");

	SGVector<float64_t> r(func.vlen);
	Map<VectorXd> eigen_r(r.vector, r.vlen);

	Map<VectorXd> eigen_f(func.vector, func.vlen);

	SGVector<float64_t> y=lab->as<RegressionLabels>()->get_labels();
	Map<VectorXd> eigen_y(y.vector, y.vlen);

	// compute r=y-f and r2=(y-f).^2
	eigen_r=eigen_y-eigen_f;
	VectorXd eigen_r2=eigen_r.cwiseProduct(eigen_r);
	float64_t df=get_degrees_freedom();

	// compute a=r+sigma^2*df and a3=a.^3
	VectorXd a =
		eigen_r2 + std::exp(m_log_sigma * 2.0) * df * VectorXd::Ones(r.vlen);
	VectorXd a3=(a.cwiseProduct(a)).cwiseProduct(a);

	if (param.first == "log_df")
	{
		// compute derivative of second derivative of log probability wrt df:
		// d2lp_ddf=df*(r2.*(r2-3*sigma^2*(1+df))+df*sigma^4)./a3
		float64_t sigma2 = std::exp(m_log_sigma * 2.0);

		eigen_r=df*(eigen_r2.cwiseProduct(eigen_r2-3*sigma2*(1.0+df)*
			VectorXd::Ones(r.vlen))+(df*Math::sq(sigma2))*VectorXd::Ones(r.vlen));
		eigen_r=eigen_r.cwiseQuotient(a3);

		eigen_r*=(1.0-1.0/df);

		return r;
	}
	else if (param.first == "log_sigma")
	{
		// compute derivative of second derivative of log probability wrt sigma:
		// d2lp_dsigma=(df+1)*2*df*sigma^2*(a-4*r2)./a3
		eigen_r = (df + 1.0) * 2 * df * std::exp(m_log_sigma * 2.0) *
			      (a - 4.0 * eigen_r2).cwiseQuotient(a3);

		return r;
	}

	return SGVector<float64_t>();
}

SGVector<float64_t> StudentsTLikelihood::get_log_zeroth_moments(
		SGVector<float64_t> mu, SGVector<float64_t> s2, std::shared_ptr<const Labels> lab) const
{
	SGVector<float64_t> y;

	if (lab)
	{
		require((mu.vlen==s2.vlen) && (mu.vlen==lab->get_num_labels()),
				"Length of the vector of means ({}), length of the vector of "
				"variances ({}) and number of labels ({}) should be the same",
				mu.vlen, s2.vlen, lab->get_num_labels());
		require(lab->get_label_type()==LT_REGRESSION,
				"Labels must be type of RegressionLabels");

		y=lab->as<RegressionLabels>()->get_labels();
	}
	else
	{
		require(mu.vlen==s2.vlen, "Length of the vector of means ({}) and "
				"length of the vector of variances ({}) should be the same",
				mu.vlen, s2.vlen);

		y=SGVector<float64_t>(mu.vlen);
		y.set_const(1.0);
	}

	// create an object of normal pdf
	auto f=std::make_shared<NormalPDF>();

	// create an object of Student's t pdf
	auto g=std::make_shared<StudentsTPDF>();

	g->set_nu(get_degrees_freedom());
	g->set_sigma(std::exp(m_log_sigma));

	// create an object of product of Student's-t pdf and normal pdf
	auto h=std::make_shared<ProductFunction>(f, g);


	// compute probabilities using numerical integration
	SGVector<float64_t> r(mu.vlen);

	for (index_t i=0; i<mu.vlen; i++)
	{
		// set normal pdf parameters
		f->set_mu(mu[i]);
		f->set_sigma(std::sqrt(s2[i]));

		// set Stundent's-t pdf parameters
		g->set_mu(y[i]);

#ifdef USE_GPL_SHOGUN
		// evaluate integral on (-inf, inf)
		r[i]=Integration::integrate_quadgk(h, -Math::INFTY, mu[i])+
			Integration::integrate_quadgk(h, mu[i], Math::INFTY);
#else
			error("StudentsT likelihood moments only supported under GPL.");
#endif //USE_GPL_SHOGUN
	}



	for (index_t i=0; i<r.vlen; i++)
		r[i] = std::log(r[i]);

	return r;
}

float64_t StudentsTLikelihood::get_first_moment(SGVector<float64_t> mu,
		SGVector<float64_t> s2, std::shared_ptr<const Labels >lab, index_t i) const
{
	// check the parameters
	require(lab, "Labels are required (lab should not be NULL)");
	require((mu.vlen==s2.vlen) && (mu.vlen==lab->get_num_labels()),
			"Length of the vector of means ({}), length of the vector of "
			"variances ({}) and number of labels ({}) should be the same",
			mu.vlen, s2.vlen, lab->get_num_labels());
	require(i>=0 && i<=mu.vlen, "Index ({}) out of bounds!", i);
	require(lab->get_label_type()==LT_REGRESSION,
			"Labels must be type of RegressionLabels");

	SGVector<float64_t> y=lab->as<RegressionLabels>()->get_labels();

	// create an object of normal pdf
	auto f = std::make_shared<NormalPDF>(mu[i], std::sqrt(s2[i]));

	// create an object of Student's t pdf
	auto g =
		std::make_shared<StudentsTPDF>(std::exp(m_log_sigma), get_degrees_freedom(), y[i]);

	// create an object of h(x)=N(x|mu,sigma)*t(x|mu,sigma,nu)
	auto h=std::make_shared<ProductFunction>(f, g);

	// create an object of k(x)=x*N(x|mu,sigma)*t(x|mu,sigma,nu)
	auto k=std::make_shared<ProductFunction>(std::make_shared<LinearFunction>(), h);


	float64_t Ex=0;
#ifdef USE_GPL_SHOGUN
	// compute Z = \int N(x|mu,sigma)*t(x|mu,sigma,nu) dx
	float64_t Z=Integration::integrate_quadgk(h, -Math::INFTY, mu[i])+
		Integration::integrate_quadgk(h, mu[i], Math::INFTY);

	// compute 1st moment:
	// E[x] = Z^-1 * \int x*N(x|mu,sigma)*t(x|mu,sigma,nu)dx
	Ex=(Integration::integrate_quadgk(k, -Math::INFTY, mu[i])+
			Integration::integrate_quadgk(k, mu[i], Math::INFTY))/Z;
#else
			error("StudentsT likelihood moments only supported under GPL.");
#endif //USE_GPL_SHOGUN


	return Ex;
}

float64_t StudentsTLikelihood::get_second_moment(SGVector<float64_t> mu,
		SGVector<float64_t> s2, std::shared_ptr<const Labels >lab, index_t i) const
{
	// check the parameters
	require(lab, "Labels are required (lab should not be NULL)");
	require((mu.vlen==s2.vlen) && (mu.vlen==lab->get_num_labels()),
			"Length of the vector of means ({}), length of the vector of "
			"variances ({}) and number of labels ({}) should be the same",
			mu.vlen, s2.vlen, lab->get_num_labels());
	require(i>=0 && i<=mu.vlen, "Index ({}) out of bounds!", i);
	require(lab->get_label_type()==LT_REGRESSION,
			"Labels must be type of RegressionLabels");

	SGVector<float64_t> y=lab->as<RegressionLabels>()->get_labels();

	// create an object of normal pdf
	auto f = std::make_shared<NormalPDF>(mu[i], std::sqrt(s2[i]));

	// create an object of Student's t pdf
	auto g =
		std::make_shared<StudentsTPDF>(std::exp(m_log_sigma), get_degrees_freedom(), y[i]);

	// create an object of h(x)=N(x|mu,sigma)*t(x|mu,sigma,nu)
	auto h=std::make_shared<ProductFunction>(f, g);

	// create an object of k(x)=x*N(x|mu,sigma)*t(x|mu,sigma,nu)
	auto k=std::make_shared<ProductFunction>(std::make_shared<LinearFunction>(), h);


	// create an object of p(x)=x^2*N(x|mu,sigma^2)*t(x|mu,sigma,nu)
	auto p=std::make_shared<ProductFunction>(std::make_shared<QuadraticFunction>(), h);


	float64_t Ex=0;
	float64_t Ex2=0;
#ifdef USE_GPL_SHOGUN
	// compute Z = \int N(x|mu,sigma)*t(x|mu,sigma,nu) dx
	float64_t Z=Integration::integrate_quadgk(h, -Math::INFTY, mu[i])+
		Integration::integrate_quadgk(h, mu[i], Math::INFTY);

	// compute 1st moment:
	// E[x] = Z^-1 * \int x*N(x|mu,sigma)*t(x|mu,sigma,nu)dx
	Ex=(Integration::integrate_quadgk(k, -Math::INFTY, mu[i])+
			Integration::integrate_quadgk(k, mu[i], Math::INFTY))/Z;

	// compute E[x^2] = Z^-1 * \int x^2*N(x|mu,sigma)*t(x|mu,sigma,nu)dx
	Ex2=(Integration::integrate_quadgk(p, -Math::INFTY, mu[i])+
			Integration::integrate_quadgk(p, mu[i], Math::INFTY))/Z;
#else
	gpl_only(SOURCE_LOCATION);
#endif //USE_GPL_SHOGUN



	// return 2nd moment: Var[x]=E[x^2]-E[x]^2
	return Ex2-Math::sq(Ex);
}
}

