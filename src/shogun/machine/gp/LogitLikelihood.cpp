/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Roman Votyakov
 */

#include <shogun/machine/gp/LogitLikelihood.h>

#ifdef HAVE_EIGEN3

#include <shogun/mathematics/Function.h>
#include <shogun/mathematics/Integration.h>
#include <shogun/labels/BinaryLabels.h>
#include <shogun/mathematics/eigen3.h>

using namespace shogun;
using namespace Eigen;

namespace shogun
{

#ifndef DOXYGEN_SHOULD_SKIP_THIS

/** Class of the probability density function of the normal distribution */
class CNormalPDF : public CFunction
{
public:
	/** default constructor */
	CNormalPDF()
	{
		m_mu=0.0;
		m_sigma=1.0;
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
		return (1.0/(CMath::sqrt(2*CMath::PI)*m_sigma))*
			CMath::exp(-CMath::sq(x-m_mu)/(2.0*CMath::sq(m_sigma)));
	}

private:
	/* mean */
	float64_t m_mu;

	/* standard deviation */
	float64_t m_sigma;
};

/** Class of the sigmoid function */
class CSigmoidFunction : public CFunction
{
public:
	/** default constructor */
	CSigmoidFunction()
	{
		m_a=0.0;
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
		return 1.0/(1.0+CMath::exp(-m_a*x));
	}

private:
	/** slope parameter */
	float64_t m_a;
};

/** Class of the function, which is a product of two given functions */
class CProductFunction : public CFunction
{
public:
	/** constructor
	 *
	 * @param f f(x)
	 * @param g g(x)
	 */
	CProductFunction(CFunction* f, CFunction* g)
	{
		SG_REF(f);
		SG_REF(g);
		m_f=f;
		m_g=g;
	}

	virtual ~CProductFunction()
	{
		SG_UNREF(m_f);
		SG_UNREF(m_g);
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
	CFunction* m_f;
	/**	function g(x) */
	CFunction* m_g;
};

#endif /* DOXYGEN_SHOULD_SKIP_THIS */

CLogitLikelihood::CLogitLikelihood() : CLikelihoodModel()
{
}

CLogitLikelihood::~CLogitLikelihood()
{
}

SGVector<float64_t> CLogitLikelihood::get_predictive_log_probabilities(
		SGVector<float64_t> mu,	SGVector<float64_t> s2, const CLabels* lab) const
{
	SGVector<float64_t> y;

	if (lab)
	{
		REQUIRE((mu.vlen==s2.vlen) && (mu.vlen==lab->get_num_labels()),
				"Length of the vector of means (mu), length of the vector of "
				"variances (s2) and number of labels (lab) should be the same\n")
		REQUIRE(lab->get_label_type()==LT_BINARY,
				"Labels must be type of CBinaryLabels\n")

		y=((CBinaryLabels*)lab)->get_labels();
	}
	else
	{
		REQUIRE(mu.vlen==s2.vlen, "Length of the vector of means (mu) and "
				"length of the vector of variances (s2) should be the same\n")

		y=SGVector<float64_t>(mu.vlen);
		y.set_const(1.0);
	}

	// create an object of normal pdf function
	CNormalPDF* f=new CNormalPDF();

	// create an object of sigmoid function
	CSigmoidFunction* g=new CSigmoidFunction();

	// create and object of product of sigmoid and normal pdf
	// functions
	CProductFunction* h=new CProductFunction(f, g);
	SG_REF(h);

	// compute probabilities using numerical integration
	SGVector<float64_t> r(mu.vlen);

	for (index_t i=0; i<mu.vlen; i++)
	{
		// set normal pdf parameters
		f->set_mu(mu[i]);
		f->set_sigma(CMath::sqrt(s2[i]));

		// set sigmoid parameters
		g->set_a(y[i]);

		// evaluate integral on (-inf, inf)
		r[i]=CIntegration::integrate_quadgk(h, -CMath::INFTY, mu[i])+
			CIntegration::integrate_quadgk(h, mu[i], CMath::INFTY);
	}

	SG_UNREF(h);

	r.log();

	return r;
}

SGVector<float64_t> CLogitLikelihood::get_predictive_means(
		SGVector<float64_t> mu,	SGVector<float64_t> s2, const CLabels* lab) const
{
	SGVector<float64_t> y;

	if (lab)
	{
		REQUIRE((mu.vlen==s2.vlen) && (mu.vlen==lab->get_num_labels()),
				"Length of the vector of means (mu), length of the vector of "
				"variances (s2) and number of labels (lab) should be the same\n")
		REQUIRE(lab->get_label_type()==LT_BINARY,
				"Labels must be type of CBinaryLabels\n")

		y=((CBinaryLabels*)lab)->get_labels();
	}
	else
	{
		REQUIRE(mu.vlen==s2.vlen, "Length of the vector of means (mu) and "
				"length of the vector of variances (s2) should be the same\n")

		y=SGVector<float64_t>(mu.vlen);
		y.set_const(1.0);
	}

	// create an object of normal pdf function
	CNormalPDF* f=new CNormalPDF();

	// create an object of sigmoid function
	CSigmoidFunction* g=new CSigmoidFunction();

	// create and object of product of sigmoid and normal pdf
	// functions
	CProductFunction* h=new CProductFunction(f, g);
	SG_REF(h);

	// compute probabilities using numerical integration
	SGVector<float64_t> p(mu.vlen);

	for (index_t i=0; i<mu.vlen; i++)
	{
		// set normal pdf parameters
		f->set_mu(mu[i]);
		f->set_sigma(CMath::sqrt(s2[i]));

		// set sigmoid parameters
		g->set_a(y[i]);

		// evaluate integral on (-inf, inf)
		p[i]=CIntegration::integrate_quadgk(h, -CMath::INFTY, mu[i])+
			CIntegration::integrate_quadgk(h, mu[i], CMath::INFTY);
	}

	SG_UNREF(h);

	Map<VectorXd> eigen_p(p.vector, p.vlen);

	SGVector<float64_t> r(p.vlen);
	Map<VectorXd> eigen_r(r.vector, r.vlen);

	// evaluate predictive mean: ymu=2*p-1
	eigen_r=2.0*eigen_p.array()-1.0;

	return r;
}

SGVector<float64_t> CLogitLikelihood::get_predictive_variances(
		SGVector<float64_t> mu,	SGVector<float64_t> s2, const CLabels* lab) const
{
	SGVector<float64_t> y;

	if (lab)
	{
		REQUIRE((mu.vlen==s2.vlen) && (mu.vlen==lab->get_num_labels()),
				"Length of the vector of means (mu), length of the vector of "
				"variances (s2) and number of labels (lab) should be the same\n")
		REQUIRE(lab->get_label_type()==LT_BINARY,
				"Labels must be type of CBinaryLabels\n")

		y=((CBinaryLabels*)lab)->get_labels();
	}
	else
	{
		REQUIRE(mu.vlen==s2.vlen, "Length of the vector of means (mu) and "
				"length of the vector of variances (s2) should be the same\n")

		y=SGVector<float64_t>(mu.vlen);
		y.set_const(1.0);
	}

	// create an object of normal pdf function
	CNormalPDF* f=new CNormalPDF();

	// create an object of sigmoid function
	CSigmoidFunction* g= new CSigmoidFunction();

	// create and object of product of sigmoid and normal pdf
	// functions
	CProductFunction* h=new CProductFunction(f, g);
	SG_REF(h);

	// compute probabilities using numerical integration
	SGVector<float64_t> p(mu.vlen);

	for (index_t i=0; i<mu.vlen; i++)
	{
		// set normal pdf parameters
		f->set_mu(mu[i]);
		f->set_sigma(CMath::sqrt(s2[i]));

		// set sigmoid parameters
		g->set_a(y[i]);

		// evaluate integral on (-inf, inf)
		p[i]=CIntegration::integrate_quadgk(h, -CMath::INFTY, mu[i])+
			CIntegration::integrate_quadgk(h, mu[i], CMath::INFTY);
	}

	SG_UNREF(h);

	Map<VectorXd> eigen_p(p.vector, p.vlen);

	SGVector<float64_t> r(p.vlen);
	Map<VectorXd> eigen_r(r.vector, r.vlen);

	// evaluate predictive variance: ys2=1-(2*p-1).^2
	eigen_r=1-(2.0*eigen_p.array()-1.0).square();

	return r;
}

SGVector<float64_t> CLogitLikelihood::get_log_probability_f(const CLabels* lab,
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

	// compute log probability: -log(1+exp(-f.*y))
	eigen_r=-(1.0+(-eigen_y.array()*eigen_f.array()).exp()).log();

	return r;
}

SGVector<float64_t> CLogitLikelihood::get_log_probability_derivative_f(
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
		eigen_r=-eigen_s.array()*(1.0-eigen_s.array())*(1.0-2*eigen_s.array());
	}
	else
	{
		SG_ERROR("Invalid index for derivative\n")
	}

	return r;
}

SGVector<float64_t> CLogitLikelihood::get_first_derivative(const CLabels* lab,
		const TParameter* param, SGVector<float64_t> func) const
{
	return SGVector<float64_t>();
}

SGVector<float64_t> CLogitLikelihood::get_second_derivative(const CLabels* lab,
		const TParameter* param, SGVector<float64_t> func) const
{
	return SGVector<float64_t>();
}

SGVector<float64_t> CLogitLikelihood::get_third_derivative(const CLabels* lab,
		const TParameter* param, SGVector<float64_t> func) const
{
	return SGVector<float64_t>();
}
}

#endif /* HAVE_EIGEN3 */
