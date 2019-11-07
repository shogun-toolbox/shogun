/*
 * Copyright (c) The Shogun Machine Learning Toolbox
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
#include <shogun/machine/gp/LogitLikelihood.h>


#include <shogun/mathematics/Function.h>
#ifdef USE_GPL_SHOGUN
#include <shogun/mathematics/Integration.h>
#endif //USE_GPL_SHOGUN
#include <shogun/labels/BinaryLabels.h>
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
	virtual float64_t operator() (float64_t x)
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

/** Class of the sigmoid function */
class SigmoidFunction : public Function
{
public:
	/** default constructor */
	SigmoidFunction()
	{
		m_a=0.0;
	}

	/** constructor
	 *
	 * @param a slope parameter
	 */
	SigmoidFunction(float64_t a)
	{
		m_a=a;
	}

	/** slope parameter
	 *
	 * @param a slope parameter to set
	 */
	void set_a(float64_t a) { m_a=a; }

	/** return value of the function at given point
	 *
	 * @param x argument
	 *
	 * @return f(x)=1/(1+exp(-a*x))
	 */
	virtual float64_t operator() (float64_t x)
	{
		return 1.0 / (1.0 + std::exp(-m_a * x));
	}

private:
	/** slope parameter */
	float64_t m_a;
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

	virtual ~ProductFunction()
	{


	}

	/** returns value of the function at given point
	 *
	 * @param x argument
	 *
	 * @return h(x)=f(x)*g(x)
	 */
	virtual float64_t operator() (float64_t x)
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

	virtual ~LinearFunction() { }

	/** returns value of the function at given point
	 *
	 * @param x argument
	 *
	 * @return f(x)=x
	 */
	virtual float64_t operator() (float64_t x)
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

	virtual ~QuadraticFunction() { }

	/** returns value of the function at given point
	 *
	 * @param x argument
	 *
	 * @return f(x)=x^2
	 */
	virtual float64_t operator() (float64_t x)
	{
		return Math::sq(x);
	}
};

#endif /* DOXYGEN_SHOULD_SKIP_THIS */

LogitLikelihood::LogitLikelihood() : LikelihoodModel()
{
}

LogitLikelihood::~LogitLikelihood()
{
}

SGVector<float64_t> LogitLikelihood::get_predictive_means(
		SGVector<float64_t> mu, SGVector<float64_t> s2, std::shared_ptr<const Labels> lab) const
{
	SGVector<float64_t> lp=get_log_zeroth_moments(mu, s2, lab);
	Map<VectorXd> eigen_lp(lp.vector, lp.vlen);

	SGVector<float64_t> r(lp.vlen);
	Map<VectorXd> eigen_r(r.vector, r.vlen);

	// evaluate predictive mean: ymu=2*p-1
	// Note that the distribution is Bernoulli distribution with p(x=1)=p and
	// p(x=-1)=(1-p)
	// the mean of the Bernoulli distribution is 2*p-1
	eigen_r=2.0*eigen_lp.array().exp()-1.0;

	return r;
}

SGVector<float64_t> LogitLikelihood::get_predictive_variances(
		SGVector<float64_t> mu, SGVector<float64_t> s2, std::shared_ptr<const Labels> lab) const
{
	SGVector<float64_t> lp=get_log_zeroth_moments(mu, s2, lab);
	Map<VectorXd> eigen_lp(lp.vector, lp.vlen);

	SGVector<float64_t> r(lp.vlen);
	Map<VectorXd> eigen_r(r.vector, r.vlen);

	// evaluate predictive variance: ys2=1-(2*p-1).^2
	// Note that the distribution is Bernoulli distribution with p(x=1)=p and
	// p(x=-1)=(1-p)
	// the variance of the Bernoulli distribution is 1-(2*p-1).^2
	eigen_r=1-(2.0*eigen_lp.array().exp()-1.0).square();

	return r;
}

SGVector<float64_t> LogitLikelihood::get_log_probability_f(std::shared_ptr<const Labels> lab,
		SGVector<float64_t> func) const
{
	// check the parameters
	require(lab, "Labels are required (lab should not be NULL)");
	require(lab->get_label_type()==LT_BINARY,
			"Labels must be type of BinaryLabels");
	require(lab->get_num_labels()==func.vlen, "Number of labels must match "
			"length of the function vector");

	SGVector<float64_t> y=lab->as<BinaryLabels>()->get_labels();
	Map<VectorXd> eigen_y(y.vector, y.vlen);

	Map<VectorXd> eigen_f(func.vector, func.vlen);

	SGVector<float64_t> r(func.vlen);
	Map<VectorXd> eigen_r(r.vector, r.vlen);

	// compute log probability: -log(1+exp(-f.*y))
	eigen_r=-(1.0+(-eigen_y.array()*eigen_f.array()).exp()).log();

	return r;
}

SGVector<float64_t> LogitLikelihood::get_log_probability_derivative_f(
		std::shared_ptr<const Labels> lab, SGVector<float64_t> func, index_t i) const
{
	// check the parameters
	require(lab, "Labels are required (lab should not be NULL)");
	require(lab->get_label_type()==LT_BINARY,
			"Labels must be type of BinaryLabels");
	require(lab->get_num_labels()==func.vlen, "Number of labels must match "
			"length of the function vector");
	require(i>=1 && i<=3, "Index for derivative should be 1, 2 or 3");

	SGVector<float64_t> y=lab->as<BinaryLabels>()->get_labels();
	Map<VectorXd> eigen_y(y.vector, y.vlen);

	Map<VectorXd> eigen_f(func.vector, func.vlen);

	SGVector<float64_t> r(func.vlen);
	Map<VectorXd> eigen_r(r.vector, r.vlen);

	// compute s(f)=1./(1+exp(-f))
	VectorXd eigen_s=(VectorXd::Ones(func.vlen)).cwiseQuotient((1.0+
		(-eigen_f).array().exp()).matrix());

	// compute derivatives of log probability wrt f
	if (i == 1)
	{
		// compute the first derivative: dlp=(y+1)/2-s(f)
		eigen_r=(eigen_y.array()+1.0)/2.0-eigen_s.array();
	}
	else if (i == 2)
	{
		// compute the second derivative: d2lp=-s(f).*(1-s(f))
		eigen_r=-eigen_s.array()*(1.0-eigen_s.array());
	}
	else if (i == 3)
	{
		// compute the third derivative: d2lp=-s(f).*(1-s(f)).*(1-2*s(f))
		eigen_r=-eigen_s.array()*(1.0-eigen_s.array())*(1.0-2.0*eigen_s.array());
	}
	else
	{
		error("Invalid index for derivative");
	}

	return r;
}

SGVector<float64_t> LogitLikelihood::get_log_zeroth_moments(
		SGVector<float64_t> mu, SGVector<float64_t> s2, std::shared_ptr<const Labels> lab) const
{
	SGVector<float64_t> y;

	if (lab)
	{
		require((mu.vlen==s2.vlen) && (mu.vlen==lab->get_num_labels()),
				"Length of the vector of means ({}), length of the vector of "
				"variances ({}) and number of labels ({}) should be the same",
				mu.vlen, s2.vlen, lab->get_num_labels());
		require(lab->get_label_type()==LT_BINARY,
				"Labels must be type of BinaryLabels");
		y=lab->as<BinaryLabels>()->get_labels();
	}
	else
	{
		require(mu.vlen==s2.vlen, "Length of the vector of means ({}) and "
				"length of the vector of variances ({}) should be the same",
				mu.vlen, s2.vlen);

		y=SGVector<float64_t>(mu.vlen);
		y.set_const(1.0);
	}

	// create an object of normal pdf function
	auto f=std::make_shared<NormalPDF>();

	// create an object of sigmoid function
	auto g=std::make_shared<SigmoidFunction>();

	// create an object of product of sigmoid and normal pdf functions
	auto h=std::make_shared<ProductFunction>(f, g);


	// compute probabilities using numerical integration
	SGVector<float64_t> r(mu.vlen);

	for (index_t i=0; i<mu.vlen; i++)
	{
		// set normal pdf parameters
		f->set_mu(mu[i]);
		f->set_sigma(std::sqrt(s2[i]));

		// set sigmoid parameters
		g->set_a(y[i]);

		// evaluate integral on (-inf, inf)
#ifdef USE_GPL_SHOGUN
		r[i]=Integration::integrate_quadgk(h, -Math::INFTY, mu[i])+
			Integration::integrate_quadgk(h, mu[i], Math::INFTY);
#else
		gpl_only(SOURCE_LOCATION);
#endif //USE_GPL_SHOGUN
	}



	for (index_t i=0; i<r.vlen; i++)
		r[i] = std::log(r[i]);

	return r;
}

float64_t LogitLikelihood::get_first_moment(SGVector<float64_t> mu,
		SGVector<float64_t> s2, std::shared_ptr<const Labels >lab, index_t i) const
{
	// check the parameters
	require(lab, "Labels are required (lab should not be NULL)");
	require((mu.vlen==s2.vlen) && (mu.vlen==lab->get_num_labels()),
			"Length of the vector of means ({}), length of the vector of "
			"variances ({}) and number of labels ({}) should be the same",
			mu.vlen, s2.vlen, lab->get_num_labels());
	require(i>=0 && i<=mu.vlen, "Index ({}) out of bounds!", i);
	require(lab->get_label_type()==LT_BINARY,
			"Labels must be type of BinaryLabels");

	SGVector<float64_t> y=lab->as<BinaryLabels>()->get_labels();

	// create an object of f(x)=N(x|mu,sigma^2)
	auto f = std::make_shared<NormalPDF>(mu[i], std::sqrt(s2[i]));

	// create an object of g(x)=sigmoid(x)
	auto g=std::make_shared<SigmoidFunction>(y[i]);

	// create an object of h(x)=N(x|mu,sigma^2)*sigmoid(x)
	auto h=std::make_shared<ProductFunction>(f, g);

	// create an object of l(x)=x
	auto l=std::make_shared<LinearFunction>();

	// create an object of k(x)=x*N(x|mu,sigma^2)*sigmoid(x)
	auto k=std::make_shared<ProductFunction>(l, h);

	float64_t Ex=0;
#ifdef USE_GPL_SHOGUN
	// compute Z = \int N(x|mu,sigma)*sigmoid(a*x) dx
	float64_t Z=Integration::integrate_quadgk(h, -Math::INFTY, mu[i])+
		Integration::integrate_quadgk(h, mu[i], Math::INFTY);

	// compute 1st moment: E[x] = Z^-1 * \int x*N(x|mu,sigma)*sigmoid(a*x)dx
	Ex=(Integration::integrate_quadgk(k, -Math::INFTY, mu[i])+
			Integration::integrate_quadgk(k, mu[i], Math::INFTY))/Z;
#else
	gpl_only(SOURCE_LOCATION);
#endif //USE_GPL_SHOGUN


	return Ex;
}

float64_t LogitLikelihood::get_second_moment(SGVector<float64_t> mu,
		SGVector<float64_t> s2, std::shared_ptr<const Labels >lab, index_t i) const
{
	// check the parameters
	require(lab, "Labels are required (lab should not be NULL)");
	require((mu.vlen==s2.vlen) && (mu.vlen==lab->get_num_labels()),
			"Length of the vector of means ({}), length of the vector of "
			"variances ({}) and number of labels ({}) should be the same",
			mu.vlen, s2.vlen, lab->get_num_labels());
	require(i>=0 && i<=mu.vlen, "Index ({}) out of bounds!", i);
	require(lab->get_label_type()==LT_BINARY,
			"Labels must be type of BinaryLabels");

	SGVector<float64_t> y=lab->as<BinaryLabels>()->get_labels();

	// create an object of f(x)=N(x|mu,sigma^2)
	auto f = std::make_shared<NormalPDF>(mu[i], std::sqrt(s2[i]));

	// create an object of g(x)=sigmoid(a*x)
	auto g=std::make_shared<SigmoidFunction>(y[i]);

	// create an object of h(x)=N(x|mu,sigma^2)*sigmoid(a*x)
	auto h=std::make_shared<ProductFunction>(f, g);

	// create an object of l(x)=x
	auto l=std::make_shared<LinearFunction>();

	// create an object of k(x)=x*N(x|mu,sigma^2)*sigmoid(a*x)
	auto k=std::make_shared<ProductFunction>(l, h);


	// create an object of q(x)=x^2
	auto q=std::make_shared<QuadraticFunction>();

	// create an object of p(x)=x^2*N(x|mu,sigma^2)*sigmoid(x)
	auto p=std::make_shared<ProductFunction>(q, h);


	float64_t Ex=0;
	float64_t Ex2=0;
#ifdef USE_GPL_SHOGUN
	// compute Z = \int N(x|mu,sigma)*sigmoid(a*x) dx
	float64_t Z=Integration::integrate_quadgk(h, -Math::INFTY, mu[i])+
		Integration::integrate_quadgk(h, mu[i], Math::INFTY);

	// compute 1st moment: E[x] = Z^-1 * \int x*N(x|mu,sigma)*sigmoid(a*x)dx
	Ex=(Integration::integrate_quadgk(k, -Math::INFTY, mu[i])+
			Integration::integrate_quadgk(k, mu[i], Math::INFTY))/Z;

	// compute E[x^2] = Z^-1 * \int x^2*N(x|mu,sigma)*sigmoid(a*x)dx
	Ex2=(Integration::integrate_quadgk(p, -Math::INFTY, mu[i])+
			Integration::integrate_quadgk(p, mu[i], Math::INFTY))/Z;
#else
	gpl_only(SOURCE_LOCATION);
#endif //USE_GPL_SHOGUN



	// return 2nd moment: Var[x]=E[x^2]-E[x]^2
	return Ex2-Math::sq(Ex);;
}
}

