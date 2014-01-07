/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Roman Votyakov
 */

#include <lib/config.h>

#ifdef HAVE_EIGEN3

#include <mathematics/Math.h>
#include <mathematics/Statistics.h>
#include <mathematics/Function.h>
#include <mathematics/Integration.h>
#include <gtest/gtest.h>

using namespace shogun;

/** @brief Class of the simple function
 */
class CSimpleFunction : public CFunction
{
public:
	/** returns value of the function at given point
	 *
	 * @param x argument
	 *
	 * @return f(x)=3*x^2
	 */
	virtual float64_t operator() (float64_t x)
	{
		return 3*x*x;
	}
};

/** @brief Class of the probability density function of the normal
 * distribution
 */
class CNormalPDF : public CFunction
{
public:
	/** constructor
	 *
	 * @param mu mean
	 * @param sigma standard deviation
	 */
	CNormalPDF(float64_t mu, float64_t sigma)
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
		return (1.0/(CMath::sqrt(2*CMath::PI)*m_sigma))*
			CMath::exp(-CMath::sq(x-m_mu)/(2.0*CMath::sq(m_sigma)));
	}

private:
	/* mean */
	float64_t m_mu;

	/* standard deviation */
	float64_t m_sigma;
};

/** @brief Class of the probability density function of the
 * non-standardized Student's t-distribution
 */
class CStudentsTPDF : public CFunction
{
public:
	/** constructor
	 *
	 * @param sigma scale parameter
	 * @param mu location parameter
	 * @param nu degrees of freedom
	 */
	CStudentsTPDF(float64_t sigma, float64_t mu, float64_t nu)
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
	virtual float64_t operator() (float64_t x)
	{
		float64_t lZ=CStatistics::lgamma(m_nu/2.0+0.5)-CStatistics::lgamma(m_nu/2.0)-
			CMath::log(m_nu*CMath::PI*CMath::sq(m_sigma))/2.0;
		return CMath::exp(lZ-(m_nu/2.0+0.5)*CMath::log(1.0+CMath::sq(x-m_mu)/
			(m_nu*CMath::sq(m_sigma))));
	}

private:
	/** scale parameter */
	float64_t m_sigma;

	/** location parameter */
	float64_t m_mu;

	/** degrees of freedom */
	float64_t m_nu;
};

/** @brief Class of the sigmoid function
 */
class CSigmoidFunction : public CFunction
{
public:
	/** return value of the function at given point
	 *
	 * @param x argument
	 *
	 * @return f(x)=1/(1+exp(-x))
	 */
	virtual float64_t operator() (float64_t x)
	{
		return 1.0/(1.0+CMath::exp(-x));
	}
};

/** @brief Class of the function, which is a product of two given
 * functions h(x)=f(x)*g(x)
 */
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

/** @brief Class of the transform function
 * g(x)=(1/sqrt(pi))*f(sqrt(2)*sigma*x+mu), which is used to compute
 * integral of N(x,mu,sigma^2)*f(x) on (-inf, inf) using Gauss-Hermite
 * quadrature formula
 */
class CTransformFunction : public CFunction
{
public:
	/** constructor
	 *
	 * @param f given function f(x)
	 * @param mu mean
	 * @param sigma standard deviation
	 */
	CTransformFunction(CFunction* f, float64_t mu, float64_t sigma)
	{
		SG_REF(f);
		m_f=f;
		m_mu=mu;
		m_sigma=sigma;
	}

	virtual ~CTransformFunction() { SG_UNREF(m_f); }

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
	 * @return f(x)=(1/sqrt(pi))*f(sqrt(2*sigma^2)*x+mu)
	 */
	virtual float64_t operator() (float64_t x)
	{
		return (1.0/CMath::sqrt(CMath::PI))*
			((*m_f)(CMath::sqrt(2.0)*m_sigma*x+m_mu));
	}

private:
	/* function f(x) */
	CFunction* m_f;

	/* mean */
	float64_t m_mu;

	/* standard deviation */
	float64_t m_sigma;
};

TEST(Integration,integrate_quadgk_simple_function)
{
	// create object of the simple function
	CSimpleFunction* f=new CSimpleFunction();
	SG_REF(f);

	// compare approximate value of definite integral with result form
	// Octave package (quadgk.m)
	float64_t q=CIntegration::integrate_quadgk(f, -5.1, 3.2);
	EXPECT_NEAR(q, 165.419000000000, 1E-12);

	q=CIntegration::integrate_quadgk(f, 0.0, 0.01);
	EXPECT_NEAR(q, 0.00000100000000, 1E-15);

	q=CIntegration::integrate_quadgk(f, -2.0, -0.2);
	EXPECT_NEAR(q, 7.99200000000000, 1E-14);

	q=CIntegration::integrate_quadgk(f, -21.23, 0.0);
	EXPECT_NEAR(q, 9568.63486700000, 1E-11);

	q=CIntegration::integrate_quadgk(f, 0.0, 21.23);
	EXPECT_NEAR(q, 9568.63486700000, 1E-11);

	q=CIntegration::integrate_quadgk(f, -21.11, 21.11);
	EXPECT_NEAR(q, 18814.5872620000, 1E-10);

	// clean up
	SG_UNREF(f);
}

TEST(Integration,integrate_quadgk_normal_pdf)
{
	// create object of the normal PDF with mean=0 and variance=0.01
	CNormalPDF* f=new CNormalPDF(0, 0.1);
	SG_REF(f);

	// compare approximate value of definite integral with result form
	// Octave package (quadgk.m)
	float64_t q=CIntegration::integrate_quadgk(f, -CMath::INFTY, CMath::INFTY);
	EXPECT_NEAR(q, 1.000000000000000, 1E-15);

	q=CIntegration::integrate_quadgk(f, -CMath::INFTY, -0.1);
	EXPECT_NEAR(q, 0.158655253931457, 1E-15);

	q=CIntegration::integrate_quadgk(f, -CMath::INFTY, 0.01);
	EXPECT_NEAR(q, 0.539827837277029, 1E-15);

	q=CIntegration::integrate_quadgk(f, 0.03, CMath::INFTY);
	EXPECT_NEAR(q, 0.382088577811047, 1E-15);

	q=CIntegration::integrate_quadgk(f, -0.4, CMath::INFTY);
	EXPECT_NEAR(q, 0.999968328759696, 1E-15);

	q=CIntegration::integrate_quadgk(f, -0.1959963984540054, 0.1959963984540054);
	EXPECT_NEAR(q, 0.950000000000000, 1E-15);

	q=CIntegration::integrate_quadgk(f, -0.3, 0.1);
	EXPECT_NEAR(q, 0.839994848036913, 1E-15);

	q=CIntegration::integrate_quadgk(f, 0.2, 0.4);
	EXPECT_NEAR(q, 0.0227184607063461, 1E-15);

	q=CIntegration::integrate_quadgk(f, -0.1, -0.01);
	EXPECT_NEAR(q, 0.301516908791514, 1E-15);

	// clean up
	SG_UNREF(f);
}

TEST(Integration,integrate_quadgk_students_t_pdf)
{
	// create object of the Student's-t PDF (sigma=0.5, mu=1.5,
	// nu=4.0)
	CStudentsTPDF* f=new CStudentsTPDF(0.5, 1.5, 4.0);
	SG_REF(f);

	// compare approximate value of definite integral with result form
	// Octave package (quadgk.m)
	float64_t q=CIntegration::integrate_quadgk(f, -CMath::INFTY, CMath::INFTY);
	EXPECT_NEAR(q, 1.000000000000000, 1E-10);

	q=CIntegration::integrate_quadgk(f, -CMath::INFTY, -0.7);
	EXPECT_NEAR(q, 0.00584559356909982, 1E-15);

	q=CIntegration::integrate_quadgk(f, -CMath::INFTY, 2.0);
	EXPECT_NEAR(q, 0.813049516849971, 1E-15);

	q=CIntegration::integrate_quadgk(f, 1.3, CMath::INFTY);
	EXPECT_NEAR(q, 0.645201369285002, 1E-15);

	q=CIntegration::integrate_quadgk(f, -0.4, CMath::INFTY);
	EXPECT_NEAR(q, 0.990448168804432, 1E-15);

	q=CIntegration::integrate_quadgk(f, 1.25, 1.75);
	EXPECT_NEAR(q, 0.356670036818137, 1E-15);

	q=CIntegration::integrate_quadgk(f, -0.3, 0.1);
	EXPECT_NEAR(q, 0.0130267055593582, 1E-15);

	q=CIntegration::integrate_quadgk(f, -0.2, 2.0);
	EXPECT_NEAR(q, 0.7994108541947792, 1E-15);

	// clean up
	SG_UNREF(f);
}

TEST(Integration,integrate_quadgk_sigmoid_function)
{
	// create object of the sigmoid function
	CSigmoidFunction* f=new CSigmoidFunction();
	SG_REF(f);

	// compare approximate value of definite integral with result form
	// Octave package (quadgk.m)
	float64_t q=CIntegration::integrate_quadgk(f, -CMath::INFTY, 0.0);
	EXPECT_NEAR(q, 0.693147180559945, 1E-15);

	q=CIntegration::integrate_quadgk(f, -CMath::INFTY, -2.7);
	EXPECT_NEAR(q, 0.0650435617765905, 1E-15);

	q=CIntegration::integrate_quadgk(f, -CMath::INFTY, 3.0);
	EXPECT_NEAR(q, 3.04858735157374, 1E-14);

	q=CIntegration::integrate_quadgk(f, -1.0, 5.0);
	EXPECT_NEAR(q, 4.69345366097090, 1E-14);

	q=CIntegration::integrate_quadgk(f, 10.0, 20.0);
	EXPECT_NEAR(q, 9.99995460316194, 1E-14);

	q=CIntegration::integrate_quadgk(f, -3.0, -2.0);
	EXPECT_NEAR(q, 0.0783406594692304, 1E-15);

	// clean up
	SG_UNREF(f);
}

TEST(Integration,integrate_quadgk_product_sigmoid_normal_pdf)
{
	// create object of the sigmoid function
	CSigmoidFunction* f=new CSigmoidFunction();

	// create object of the normal PDF function with mean=0.0 and
	// variance=0.01
	CNormalPDF* g=new CNormalPDF(0.0, 0.1);

	// create object of the product function
	CProductFunction* h=new CProductFunction(f, g);
	SG_REF(h);

	// compare approximate value of definite integral with result form
	// Octave package (quadgk.m)
	float64_t q=CIntegration::integrate_quadgk(h, -CMath::INFTY, CMath::INFTY);
	EXPECT_NEAR(q, 0.500000000000000, 1E-15);

	q=CIntegration::integrate_quadgk(h, -CMath::INFTY, 0.2);
	EXPECT_NEAR(q, 0.487281864084370, 1E-15);

	q=CIntegration::integrate_quadgk(h, -CMath::INFTY, -0.1);
	EXPECT_NEAR(q, 0.0732934168890405, 1E-15);

	q=CIntegration::integrate_quadgk(h, 0.03, CMath::INFTY);
	EXPECT_NEAR(q, 0.200562444119857, 1E-15);

	q=CIntegration::integrate_quadgk(h, -0.4, CMath::INFTY);
	EXPECT_NEAR(q, 0.499987460846813, 1E-15);

	q=CIntegration::integrate_quadgk(h, -2.0, 0.0);
	EXPECT_NEAR(q, 0.240042999495053, 1E-15);

	// clean up
	SG_UNREF(h);
}

TEST(Integration,integrate_quadgk_product_students_t_pdf_normal_pdf)
{
	// create object of the Student's-t PDF (sigma=1.5, mu=-1.5,
	// nu=4.0)
	CStudentsTPDF* f=new CStudentsTPDF(1.5, -1.5, 4.0);

	// create object of the normal PDF function with mean=0.0 and
	// variance=0.01
	CNormalPDF* g=new CNormalPDF(0.0, 0.5);

	// create object of the product function
	CProductFunction* h=new CProductFunction(f, g);
	SG_REF(h);

	// compare approximate value of definite integral with result form
	// Octave package (quadgk.m)
	float64_t q=CIntegration::integrate_quadgk(h, -CMath::INFTY, CMath::INFTY);
	EXPECT_NEAR(q, 0.145255619704035, 1E-15);

	q=CIntegration::integrate_quadgk(h, -5.0, CMath::INFTY);
	EXPECT_NEAR(q, 0.145255619704012, 1E-15);

	q=CIntegration::integrate_quadgk(h, 0.2, CMath::INFTY);
	EXPECT_NEAR(q, 0.0339899906690967, 1E-15);

	q=CIntegration::integrate_quadgk(h, -CMath::INFTY, -0.8);
	EXPECT_NEAR(q, 0.0127254813827678, 1E-15);

	q=CIntegration::integrate_quadgk(h, -CMath::INFTY, 2.17);
	EXPECT_NEAR(q, 0.145255453120542, 1E-15);

	q=CIntegration::integrate_quadgk(h, -20.0, 3.0);
	EXPECT_NEAR(q, 0.145255619691797, 1E-14);

	// clean up
	SG_UNREF(h);
}

TEST(Integration,integrate_quadgh_product_sigmoid_normal_pdf)
{
	// create object of the sigmoid function
	CSigmoidFunction* f=new CSigmoidFunction();

	// create object of transform function
	// g(x)=(1/sqrt(pi))*f(sqrt(2)*sigma*x+mu)
	CTransformFunction* g=new CTransformFunction(f, 0.0, 0.1);
	SG_REF(g);

	// compute integral of sigmoid(x)*N(x, 0.0, 0.01) on (-inf, inf)
	// using Gauss-Hermite quadrature
	float64_t q=CIntegration::integrate_quadgh(g);
	EXPECT_NEAR(q, 0.500000000000000, 1E-15);

	// compute integral of sigmoid(x)*N(x, 2.0, 0.04) on (-inf, inf)
	// using Gauss-Hermite quadrature formula
	g->set_mu(2.0);
	g->set_sigma(0.2);

	q=CIntegration::integrate_quadgh(g);
	EXPECT_NEAR(q, 0.879202123093179, 1E-15);

	// compute integral of sigmoid(x)*N(x, -1.0, 0.0009) on (-inf,
	// inf) using Gauss-Hermite quadrature formula
	g->set_mu(-1.0);
	g->set_sigma(0.03);

	q=CIntegration::integrate_quadgh(g);
	EXPECT_NEAR(q, 0.268982294855682, 1E-15);

	// compute integral of sigmoid(x)*N(x, -2.5, 1.0) on (-inf, inf)
	// using Gauss-Hermite quadrature formula
	g->set_mu(-2.5);
	g->set_sigma(1.0);

	q=CIntegration::integrate_quadgh(g);
	EXPECT_NEAR(q, 0.105362117215756, 1E-15);

	// clean up
	SG_UNREF(g);
}

TEST(Integration,integrate_quadgh_product_students_t_pdf_normal_pdf)
{
	// create object of the Student's-t PDF (sigma=0.1, mu=0.7,
	// nu=3.0)
	CStudentsTPDF* f=new CStudentsTPDF(0.1, 0.7, 3.0);

	// create object of transform function
	// g(x)=(1/sqrt(pi))*f(sqrt(2)*sigma*x+mu)
	CTransformFunction* g=new CTransformFunction(f, 0.0, 0.1);
	SG_REF(g);

	// compute integral of t(x, 0.1, 0.7, 0.3)*N(x, 0, 0.01) on (-inf,
	// inf) using Gauss-Hermite quadrature formula
	float64_t q=CIntegration::integrate_quadgh(g);
	EXPECT_NEAR(q, 0.0149093164709605, 1E-15);

	// compute integral of t(x, 1, 1.5, 5)*N(x, 0, 0.16) on (-inf,
	// inf) using Gauss-Hermite quadrature formula
	f->set_sigma(1.0);
	f->set_mu(1.5);
	f->set_nu(5.0);

	g->set_sigma(0.4);
	g->set_mu(1.0);

	q=CIntegration::integrate_quadgh(g);
	EXPECT_NEAR(q, 0.310385847323180, 1E-15);

	// compute integral of t(x, 1, 0.5, 10)*N(x, 1, 0.49) on (-inf,
	// inf) using Gauss-Hermite quadrature formula
	f->set_sigma(1.0);
	f->set_mu(0.5);
	f->set_nu(10.0);

	g->set_sigma(0.7);
	g->set_mu(1.0);

	q=CIntegration::integrate_quadgh(g);
	EXPECT_NEAR(q, 0.290698368717942, 1E-15);

	// clean up
	SG_UNREF(g);
}

#endif /* HAVE_EIGEN3 */
