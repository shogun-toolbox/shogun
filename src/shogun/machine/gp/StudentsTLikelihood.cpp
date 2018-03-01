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

/** Class of the probability density function of the non-standardized Student's
 * t-distribution
 */
class CStudentsTPDF : public CFunction
{
public:
	/** default constructor */
	CStudentsTPDF()
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
	CStudentsTPDF(float64_t sigma, float64_t nu, float64_t mu)
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
		float64_t lZ = CStatistics::lgamma((m_nu + 1.0) / 2.0) -
			           CStatistics::lgamma(m_nu / 2.0) -
			           std::log(m_nu * CMath::PI * CMath::sq(m_sigma)) / 2.0;
		return CMath::exp(
			lZ -
			((m_nu + 1.0) / 2.0) *
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

/** Class of the f(x)=x */
class CLinearFunction : public CFunction
{
public:
	/** default constructor */
	CLinearFunction() { }

	virtual ~CLinearFunction() { }

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
class CQuadraticFunction : public CFunction
{
public:
	/** default constructor */
	CQuadraticFunction() { }

	virtual ~CQuadraticFunction() { }

	/** returns value of the function at given point
	 *
	 * @param x argument
	 *
	 * @return f(x)=x^2
	 */
	virtual float64_t operator() (float64_t x)
	{
		return CMath::sq(x);
	}
};

#endif /* DOXYGEN_SHOULD_SKIP_THIS */

CStudentsTLikelihood::CStudentsTLikelihood() : CLikelihoodModel()
{
	init();
}

CStudentsTLikelihood::CStudentsTLikelihood(float64_t sigma, float64_t df)
		: CLikelihoodModel()
{
	init();
	set_sigma(sigma);
	set_degrees_freedom(df);
}

void CStudentsTLikelihood::init()
{
	m_log_sigma=0.0;
	m_log_df = std::log(2.0);
	SG_ADD(&m_log_df, "log_df", "Degrees of freedom in log domain", MS_AVAILABLE, GRADIENT_AVAILABLE);
	SG_ADD(&m_log_sigma, "log_sigma", "Scale parameter in log domain", MS_AVAILABLE, GRADIENT_AVAILABLE);
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

SGVector<float64_t> CStudentsTLikelihood::get_predictive_means(
		SGVector<float64_t> mu, SGVector<float64_t> s2, const CLabels* lab) const
{
	return SGVector<float64_t>(mu);
}

SGVector<float64_t> CStudentsTLikelihood::get_predictive_variances(
		SGVector<float64_t> mu, SGVector<float64_t> s2, const CLabels* lab) const
{
	SGVector<float64_t> result(s2);
	Map<VectorXd> eigen_result(result.vector, result.vlen);
	float64_t df=get_degrees_freedom();
	if (df<2.0)
		eigen_result=CMath::INFTY*VectorXd::Ones(result.vlen);
	else
	{
		eigen_result+=CMath::exp(m_log_sigma*2.0)*df/(df-2.0)*
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

	float64_t df=get_degrees_freedom();
	// compute lZ=log(gamma(df/2+1/2))-log(gamma(df/2))-log(df*pi*sigma^2)/2
	VectorXd eigen_lZ=(CStatistics::lgamma(df/2.0+0.5)-
		CStatistics::lgamma(df/2.0)-log(df*CMath::PI*CMath::exp(m_log_sigma*2.0))/2.0)*
		VectorXd::Ones(r.vlen);

	// compute log probability: lp=lZ-(df+1)*log(1+(y-f).^2./(df*sigma^2))/2
	eigen_r=eigen_y-eigen_f;
	eigen_r=eigen_r.cwiseProduct(eigen_r)/(df*CMath::exp(m_log_sigma*2.0));
	eigen_r=eigen_lZ-(df+1)*
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
	float64_t df=get_degrees_freedom();

	// compute a=(y-f).^2+df*sigma^2
	VectorXd a=eigen_r2+VectorXd::Ones(r.vlen)*df*CMath::exp(m_log_sigma*2.0);

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
		VectorXd b=eigen_r2-VectorXd::Ones(r.vlen)*df*CMath::exp(m_log_sigma*2.0);

		eigen_r=(df+1)*b.cwiseQuotient(a.cwiseProduct(a));
	}
	else if (i==3)
	{
		// compute third derivative of log probability wrt f:
		// d3lp=(f+1)*2*(y-f).*((y-f)^2-3*f*sigma^2)./a.^3
		VectorXd c=eigen_r2-VectorXd::Ones(r.vlen)*3*df*CMath::exp(m_log_sigma*2.0);
		VectorXd a2=a.cwiseProduct(a);

		eigen_r=(df+1)*2*eigen_r.cwiseProduct(c).cwiseQuotient(
			a2.cwiseProduct(a));
	}

	return r;
}

SGVector<float64_t> CStudentsTLikelihood::get_first_derivative(const CLabels* lab,
		SGVector<float64_t> func, const TParameter* param) const
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
	float64_t df=get_degrees_freedom();

	if (!strcmp(param->m_name, "log_df"))
	{
		// compute derivative of log probability wrt df:
		// lp_ddf=df*(dloggamma(df/2+1/2)-dloggamma(df/2))/2-1/2-
		// df*log(1+r2/(df*sigma^2))/2 +(df/2+1/2)*r2./(df*sigma^2+r2)
		eigen_r=(df*(CStatistics::dlgamma(df*0.5+0.5)-
			CStatistics::dlgamma(df*0.5))*0.5-0.5)*VectorXd::Ones(r.vlen);

		eigen_r-=df*(VectorXd::Ones(r.vlen)+
			eigen_r2/(df*CMath::exp(m_log_sigma*2.0))).array().log().matrix()/2.0;

		eigen_r+=(df/2.0+0.5)*eigen_r2.cwiseQuotient(
			eigen_r2+VectorXd::Ones(r.vlen)*(df*CMath::exp(m_log_sigma*2.0)));

		eigen_r*=(1.0-1.0/df);

		return r;
	}
	else if (!strcmp(param->m_name, "log_sigma"))
	{
		// compute derivative of log probability wrt sigma:
		// lp_dsigma=(df+1)*r2./a-1
		eigen_r=(df+1)*eigen_r2.cwiseQuotient(eigen_r2+
			VectorXd::Ones(r.vlen)*(df*CMath::exp(m_log_sigma*2.0)));
		eigen_r-=VectorXd::Ones(r.vlen);

		return r;
	}

	return SGVector<float64_t>();
}

SGVector<float64_t> CStudentsTLikelihood::get_second_derivative(const CLabels* lab,
		SGVector<float64_t> func, const TParameter* param) const
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
	float64_t df=get_degrees_freedom();

	// compute a=r+sigma^2*df and a2=a.^2
	VectorXd a=eigen_r2+CMath::exp(m_log_sigma*2.0)*df*VectorXd::Ones(r.vlen);
	VectorXd a2=a.cwiseProduct(a);

	if (!strcmp(param->m_name, "log_df"))
	{
		// compute derivative of first derivative of log probability wrt df:
		// dlp_ddf=df*r.*(a-sigma^2*(df+1))./a2
		eigen_r=df*eigen_r.cwiseProduct(a-CMath::exp(m_log_sigma*2.0)*(df+1.0)*
			VectorXd::Ones(r.vlen)).cwiseQuotient(a2);
		eigen_r*=(1.0-1.0/df);

		return r;
	}
	else if (!strcmp(param->m_name, "log_sigma"))
	{
		// compute derivative of first derivative of log probability wrt sigma:
		// dlp_dsigma=-(df+1)*2*df*sigma^2*r./a2
		eigen_r=-(df+1.0)*2*df*CMath::exp(m_log_sigma*2.0)*
			eigen_r.cwiseQuotient(a2);

		return r;
	}

	return SGVector<float64_t>();
}

SGVector<float64_t> CStudentsTLikelihood::get_third_derivative(const CLabels* lab,
		SGVector<float64_t> func, const TParameter* param) const
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
	float64_t df=get_degrees_freedom();

	// compute a=r+sigma^2*df and a3=a.^3
	VectorXd a=eigen_r2+CMath::exp(m_log_sigma*2.0)*df*VectorXd::Ones(r.vlen);
	VectorXd a3=(a.cwiseProduct(a)).cwiseProduct(a);

	if (!strcmp(param->m_name, "log_df"))
	{
		// compute derivative of second derivative of log probability wrt df:
		// d2lp_ddf=df*(r2.*(r2-3*sigma^2*(1+df))+df*sigma^4)./a3
		float64_t sigma2=CMath::exp(m_log_sigma*2.0);

		eigen_r=df*(eigen_r2.cwiseProduct(eigen_r2-3*sigma2*(1.0+df)*
			VectorXd::Ones(r.vlen))+(df*CMath::sq(sigma2))*VectorXd::Ones(r.vlen));
		eigen_r=eigen_r.cwiseQuotient(a3);

		eigen_r*=(1.0-1.0/df);

		return r;
	}
	else if (!strcmp(param->m_name, "log_sigma"))
	{
		// compute derivative of second derivative of log probability wrt sigma:
		// d2lp_dsigma=(df+1)*2*df*sigma^2*(a-4*r2)./a3
		eigen_r=(df+1.0)*2*df*CMath::exp(m_log_sigma*2.0)*
			(a-4.0*eigen_r2).cwiseQuotient(a3);

		return r;
	}

	return SGVector<float64_t>();
}

SGVector<float64_t> CStudentsTLikelihood::get_log_zeroth_moments(
		SGVector<float64_t> mu, SGVector<float64_t> s2, const CLabels* lab) const
{
	SGVector<float64_t> y;

	if (lab)
	{
		REQUIRE((mu.vlen==s2.vlen) && (mu.vlen==lab->get_num_labels()),
				"Length of the vector of means (%d), length of the vector of "
				"variances (%d) and number of labels (%d) should be the same\n",
				mu.vlen, s2.vlen, lab->get_num_labels())
		REQUIRE(lab->get_label_type()==LT_REGRESSION,
				"Labels must be type of CRegressionLabels\n")

		y=((CRegressionLabels*)lab)->get_labels();
	}
	else
	{
		REQUIRE(mu.vlen==s2.vlen, "Length of the vector of means (%d) and "
				"length of the vector of variances (%d) should be the same\n",
				mu.vlen, s2.vlen)

		y=SGVector<float64_t>(mu.vlen);
		y.set_const(1.0);
	}

	// create an object of normal pdf
	CNormalPDF* f=new CNormalPDF();

	// create an object of Student's t pdf
	CStudentsTPDF* g=new CStudentsTPDF();

	g->set_nu(get_degrees_freedom());
	g->set_sigma(CMath::exp(m_log_sigma));

	// create an object of product of Student's-t pdf and normal pdf
	CProductFunction* h=new CProductFunction(f, g);
	SG_REF(h);

	// compute probabilities using numerical integration
	SGVector<float64_t> r(mu.vlen);

	for (index_t i=0; i<mu.vlen; i++)
	{
		// set normal pdf parameters
		f->set_mu(mu[i]);
		f->set_sigma(CMath::sqrt(s2[i]));

		// set Stundent's-t pdf parameters
		g->set_mu(y[i]);

#ifdef USE_GPL_SHOGUN
		// evaluate integral on (-inf, inf)
		r[i]=CIntegration::integrate_quadgk(h, -CMath::INFTY, mu[i])+
			CIntegration::integrate_quadgk(h, mu[i], CMath::INFTY);
#else
			SG_ERROR("StudentsT likelihood moments only supported under GPL.\n")
#endif //USE_GPL_SHOGUN
	}

	SG_UNREF(h);

	for (index_t i=0; i<r.vlen; i++)
		r[i] = std::log(r[i]);

	return r;
}

float64_t CStudentsTLikelihood::get_first_moment(SGVector<float64_t> mu,
		SGVector<float64_t> s2, const CLabels *lab, index_t i) const
{
	// check the parameters
	REQUIRE(lab, "Labels are required (lab should not be NULL)\n")
	REQUIRE((mu.vlen==s2.vlen) && (mu.vlen==lab->get_num_labels()),
			"Length of the vector of means (%d), length of the vector of "
			"variances (%d) and number of labels (%d) should be the same\n",
			mu.vlen, s2.vlen, lab->get_num_labels())
	REQUIRE(i>=0 && i<=mu.vlen, "Index (%d) out of bounds!\n", i)
	REQUIRE(lab->get_label_type()==LT_REGRESSION,
			"Labels must be type of CRegressionLabels\n")

	SGVector<float64_t> y=((CRegressionLabels*)lab)->get_labels();

	// create an object of normal pdf
	CNormalPDF* f=new CNormalPDF(mu[i], CMath::sqrt(s2[i]));

	// create an object of Student's t pdf
	CStudentsTPDF* g=new CStudentsTPDF(CMath::exp(m_log_sigma), get_degrees_freedom(), y[i]);

	// create an object of h(x)=N(x|mu,sigma)*t(x|mu,sigma,nu)
	CProductFunction* h=new CProductFunction(f, g);

	// create an object of k(x)=x*N(x|mu,sigma)*t(x|mu,sigma,nu)
	CProductFunction* k=new CProductFunction(new CLinearFunction(), h);
	SG_REF(k);

	float64_t Ex=0;
#ifdef USE_GPL_SHOGUN
	// compute Z = \int N(x|mu,sigma)*t(x|mu,sigma,nu) dx
	float64_t Z=CIntegration::integrate_quadgk(h, -CMath::INFTY, mu[i])+
		CIntegration::integrate_quadgk(h, mu[i], CMath::INFTY);

	// compute 1st moment:
	// E[x] = Z^-1 * \int x*N(x|mu,sigma)*t(x|mu,sigma,nu)dx
	Ex=(CIntegration::integrate_quadgk(k, -CMath::INFTY, mu[i])+
			CIntegration::integrate_quadgk(k, mu[i], CMath::INFTY))/Z;
#else
			SG_ERROR("StudentsT likelihood moments only supported under GPL.\n")
#endif //USE_GPL_SHOGUN
	SG_UNREF(k);

	return Ex;
}

float64_t CStudentsTLikelihood::get_second_moment(SGVector<float64_t> mu,
		SGVector<float64_t> s2, const CLabels *lab, index_t i) const
{
	// check the parameters
	REQUIRE(lab, "Labels are required (lab should not be NULL)\n")
	REQUIRE((mu.vlen==s2.vlen) && (mu.vlen==lab->get_num_labels()),
			"Length of the vector of means (%d), length of the vector of "
			"variances (%d) and number of labels (%d) should be the same\n",
			mu.vlen, s2.vlen, lab->get_num_labels())
	REQUIRE(i>=0 && i<=mu.vlen, "Index (%d) out of bounds!\n", i)
	REQUIRE(lab->get_label_type()==LT_REGRESSION,
			"Labels must be type of CRegressionLabels\n")

	SGVector<float64_t> y=((CRegressionLabels*)lab)->get_labels();

	// create an object of normal pdf
	CNormalPDF* f=new CNormalPDF(mu[i], CMath::sqrt(s2[i]));

	// create an object of Student's t pdf
	CStudentsTPDF* g=new CStudentsTPDF(CMath::exp(m_log_sigma), get_degrees_freedom(), y[i]);

	// create an object of h(x)=N(x|mu,sigma)*t(x|mu,sigma,nu)
	CProductFunction* h=new CProductFunction(f, g);

	// create an object of k(x)=x*N(x|mu,sigma)*t(x|mu,sigma,nu)
	CProductFunction* k=new CProductFunction(new CLinearFunction(), h);
	SG_REF(k);

	// create an object of p(x)=x^2*N(x|mu,sigma^2)*t(x|mu,sigma,nu)
	CProductFunction* p=new CProductFunction(new CQuadraticFunction(), h);
	SG_REF(p);

	float64_t Ex=0;
	float64_t Ex2=0;
#ifdef USE_GPL_SHOGUN
	// compute Z = \int N(x|mu,sigma)*t(x|mu,sigma,nu) dx
	float64_t Z=CIntegration::integrate_quadgk(h, -CMath::INFTY, mu[i])+
		CIntegration::integrate_quadgk(h, mu[i], CMath::INFTY);

	// compute 1st moment:
	// E[x] = Z^-1 * \int x*N(x|mu,sigma)*t(x|mu,sigma,nu)dx
	Ex=(CIntegration::integrate_quadgk(k, -CMath::INFTY, mu[i])+
			CIntegration::integrate_quadgk(k, mu[i], CMath::INFTY))/Z;

	// compute E[x^2] = Z^-1 * \int x^2*N(x|mu,sigma)*t(x|mu,sigma,nu)dx
	Ex2=(CIntegration::integrate_quadgk(p, -CMath::INFTY, mu[i])+
			CIntegration::integrate_quadgk(p, mu[i], CMath::INFTY))/Z;
#else
	SG_GPL_ONLY
#endif //USE_GPL_SHOGUN
	SG_UNREF(k);
	SG_UNREF(p);

	// return 2nd moment: Var[x]=E[x^2]-E[x]^2
	return Ex2-CMath::sq(Ex);
}
}

