/*
 * Copyright (c) The Shogun Machine Learning Toolbox
 * Written (w) 2014 Wu Lin
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
 */
#include <gtest/gtest.h>

#include <shogun/lib/config.h>

#include <shogun/mathematics/Math.h>
#include <shogun/mathematics/Statistics.h>
#include <shogun/mathematics/Function.h>
#ifdef USE_GPL_SHOGUN
#include <shogun/mathematics/Integration.h>
#endif //USE_GPL_SHOGUN
#include <shogun/lib/SGVector.h>

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
		float64_t lZ = CStatistics::lgamma(m_nu / 2.0 + 0.5) -
		               CStatistics::lgamma(m_nu / 2.0) -
		               std::log(m_nu * CMath::PI * CMath::sq(m_sigma)) / 2.0;
		return CMath::exp(
		    lZ -
		    (m_nu / 2.0 + 0.5) *
		        std::log(
		            1.0 + CMath::sq(x - m_mu) / (m_nu * CMath::sq(m_sigma))));
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

#ifdef USE_GPL_SHOGUN
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

TEST(Integration, generate_gauher)
{
	index_t n = 20;
	float64_t abs_tolerance, rel_tolerance = 1e-2;
	SGVector<float64_t> xgh(n);
	SGVector<float64_t> wgh(n);

	CIntegration::generate_gauher(xgh, wgh);

	SGVector<index_t> index = CMath::argsort(xgh);

	abs_tolerance = CMath::get_abs_tolerance(-7.619048541679757, rel_tolerance);
	EXPECT_NEAR(xgh[index[0]],  -7.619048541679757,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(-6.510590157013656, rel_tolerance);
	EXPECT_NEAR(xgh[index[1]],  -6.510590157013656,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(-5.578738805893203, rel_tolerance);
	EXPECT_NEAR(xgh[index[2]],  -5.578738805893203,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(-4.734581334046057, rel_tolerance);
	EXPECT_NEAR(xgh[index[3]],  -4.734581334046057,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(-3.943967350657318, rel_tolerance);
	EXPECT_NEAR(xgh[index[4]],  -3.943967350657318,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(-3.189014816553390, rel_tolerance);
	EXPECT_NEAR(xgh[index[5]],  -3.189014816553390,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(-2.458663611172367, rel_tolerance);
	EXPECT_NEAR(xgh[index[6]],  -2.458663611172367,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(-1.745247320814127, rel_tolerance);
	EXPECT_NEAR(xgh[index[7]],  -1.745247320814127,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(-1.042945348802751, rel_tolerance);
	EXPECT_NEAR(xgh[index[8]],  -1.042945348802751,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(-0.346964157081356, rel_tolerance);
	EXPECT_NEAR(xgh[index[9]],  -0.346964157081356,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.346964157081356, rel_tolerance);
	EXPECT_NEAR(xgh[index[10]],  0.346964157081356,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(1.042945348802751, rel_tolerance);
	EXPECT_NEAR(xgh[index[11]],  1.042945348802751,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(1.745247320814127, rel_tolerance);
	EXPECT_NEAR(xgh[index[12]],  1.745247320814127,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(2.458663611172367, rel_tolerance);
	EXPECT_NEAR(xgh[index[13]],  2.458663611172367,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(3.189014816553390, rel_tolerance);
	EXPECT_NEAR(xgh[index[14]],  3.189014816553390,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(3.943967350657316, rel_tolerance);
	EXPECT_NEAR(xgh[index[15]],  3.943967350657316,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(4.734581334046057, rel_tolerance);
	EXPECT_NEAR(xgh[index[16]],  4.734581334046057,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(5.578738805893201, rel_tolerance);
	EXPECT_NEAR(xgh[index[17]],  5.578738805893201,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(6.510590157013653, rel_tolerance);
	EXPECT_NEAR(xgh[index[18]],  6.510590157013653,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(7.619048541679757, rel_tolerance);
	EXPECT_NEAR(xgh[index[19]],  7.619048541679757,  abs_tolerance);

	abs_tolerance = CMath::get_abs_tolerance(0.000000000000126, rel_tolerance);
	EXPECT_NEAR(wgh[index[0]],  0.000000000000126,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.000000000248206, rel_tolerance);
	EXPECT_NEAR(wgh[index[1]],  0.000000000248206,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.000000061274903, rel_tolerance);
	EXPECT_NEAR(wgh[index[2]],  0.000000061274903,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.000004402121090, rel_tolerance);
	EXPECT_NEAR(wgh[index[3]],  0.000004402121090,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.000128826279962, rel_tolerance);
	EXPECT_NEAR(wgh[index[4]],  0.000128826279962,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.001830103131080, rel_tolerance);
	EXPECT_NEAR(wgh[index[5]],  0.001830103131080,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.013997837447101, rel_tolerance);
	EXPECT_NEAR(wgh[index[6]],  0.013997837447101,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.061506372063977, rel_tolerance);
	EXPECT_NEAR(wgh[index[7]],  0.061506372063977,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.161739333984000, rel_tolerance);
	EXPECT_NEAR(wgh[index[8]],  0.161739333984000,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.260793063449555, rel_tolerance);
	EXPECT_NEAR(wgh[index[9]],  0.260793063449555,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.260793063449555, rel_tolerance);
	EXPECT_NEAR(wgh[index[10]],  0.260793063449555,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.161739333984000, rel_tolerance);
	EXPECT_NEAR(wgh[index[11]],  0.161739333984000,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.061506372063977, rel_tolerance);
	EXPECT_NEAR(wgh[index[12]],  0.061506372063977,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.013997837447101, rel_tolerance);
	EXPECT_NEAR(wgh[index[13]],  0.013997837447101,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.001830103131080, rel_tolerance);
	EXPECT_NEAR(wgh[index[14]],  0.001830103131080,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.000128826279962, rel_tolerance);
	EXPECT_NEAR(wgh[index[15]],  0.000128826279962,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.000004402121090, rel_tolerance);
	EXPECT_NEAR(wgh[index[16]],  0.000004402121090,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.000000061274903, rel_tolerance);
	EXPECT_NEAR(wgh[index[17]],  0.000000061274903,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.000000000248206, rel_tolerance);
	EXPECT_NEAR(wgh[index[18]],  0.000000000248206,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.000000000000126, rel_tolerance);
	EXPECT_NEAR(wgh[index[19]],  0.000000000000126,  abs_tolerance);
}

TEST(Integration, generate_gauher20)
{
	index_t n = 20;
	float64_t abs_tolerance, rel_tolerance = 1e-2;
	SGVector<float64_t> xgh(n);
	SGVector<float64_t> wgh(n);

	CIntegration::generate_gauher20(xgh, wgh);

	abs_tolerance = CMath::get_abs_tolerance(-7.619048541679757, rel_tolerance);
	EXPECT_NEAR(xgh[0],  -7.619048541679757,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(-6.510590157013656, rel_tolerance);
	EXPECT_NEAR(xgh[1],  -6.510590157013656,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(-5.578738805893203, rel_tolerance);
	EXPECT_NEAR(xgh[2],  -5.578738805893203,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(-4.734581334046057, rel_tolerance);
	EXPECT_NEAR(xgh[3],  -4.734581334046057,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(-3.943967350657318, rel_tolerance);
	EXPECT_NEAR(xgh[4],  -3.943967350657318,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(-3.189014816553390, rel_tolerance);
	EXPECT_NEAR(xgh[5],  -3.189014816553390,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(-2.458663611172367, rel_tolerance);
	EXPECT_NEAR(xgh[6],  -2.458663611172367,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(-1.745247320814127, rel_tolerance);
	EXPECT_NEAR(xgh[7],  -1.745247320814127,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(-1.042945348802751, rel_tolerance);
	EXPECT_NEAR(xgh[8],  -1.042945348802751,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(-0.346964157081356, rel_tolerance);
	EXPECT_NEAR(xgh[9],  -0.346964157081356,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.346964157081356, rel_tolerance);
	EXPECT_NEAR(xgh[10],  0.346964157081356,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(1.042945348802751, rel_tolerance);
	EXPECT_NEAR(xgh[11],  1.042945348802751,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(1.745247320814127, rel_tolerance);
	EXPECT_NEAR(xgh[12],  1.745247320814127,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(2.458663611172367, rel_tolerance);
	EXPECT_NEAR(xgh[13],  2.458663611172367,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(3.189014816553390, rel_tolerance);
	EXPECT_NEAR(xgh[14],  3.189014816553390,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(3.943967350657316, rel_tolerance);
	EXPECT_NEAR(xgh[15],  3.943967350657316,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(4.734581334046057, rel_tolerance);
	EXPECT_NEAR(xgh[16],  4.734581334046057,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(5.578738805893201, rel_tolerance);
	EXPECT_NEAR(xgh[17],  5.578738805893201,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(6.510590157013653, rel_tolerance);
	EXPECT_NEAR(xgh[18],  6.510590157013653,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(7.619048541679757, rel_tolerance);
	EXPECT_NEAR(xgh[19],  7.619048541679757,  abs_tolerance);

	abs_tolerance = CMath::get_abs_tolerance(0.000000000000126, rel_tolerance);
	EXPECT_NEAR(wgh[0],  0.000000000000126,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.000000000248206, rel_tolerance);
	EXPECT_NEAR(wgh[1],  0.000000000248206,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.000000061274903, rel_tolerance);
	EXPECT_NEAR(wgh[2],  0.000000061274903,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.000004402121090, rel_tolerance);
	EXPECT_NEAR(wgh[3],  0.000004402121090,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.000128826279962, rel_tolerance);
	EXPECT_NEAR(wgh[4],  0.000128826279962,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.001830103131080, rel_tolerance);
	EXPECT_NEAR(wgh[5],  0.001830103131080,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.013997837447101, rel_tolerance);
	EXPECT_NEAR(wgh[6],  0.013997837447101,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.061506372063977, rel_tolerance);
	EXPECT_NEAR(wgh[7],  0.061506372063977,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.161739333984000, rel_tolerance);
	EXPECT_NEAR(wgh[8],  0.161739333984000,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.260793063449555, rel_tolerance);
	EXPECT_NEAR(wgh[9],  0.260793063449555,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.260793063449555, rel_tolerance);
	EXPECT_NEAR(wgh[10],  0.260793063449555,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.161739333984000, rel_tolerance);
	EXPECT_NEAR(wgh[11],  0.161739333984000,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.061506372063977, rel_tolerance);
	EXPECT_NEAR(wgh[12],  0.061506372063977,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.013997837447101, rel_tolerance);
	EXPECT_NEAR(wgh[13],  0.013997837447101,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.001830103131080, rel_tolerance);
	EXPECT_NEAR(wgh[14],  0.001830103131080,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.000128826279962, rel_tolerance);
	EXPECT_NEAR(wgh[15],  0.000128826279962,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.000004402121090, rel_tolerance);
	EXPECT_NEAR(wgh[16],  0.000004402121090,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.000000061274903, rel_tolerance);
	EXPECT_NEAR(wgh[17],  0.000000061274903,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.000000000248206, rel_tolerance);
	EXPECT_NEAR(wgh[18],  0.000000000248206,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.000000000000126, rel_tolerance);
	EXPECT_NEAR(wgh[19],  0.000000000000126,  abs_tolerance);
}
#endif //USE_GPL_SHOGUN
