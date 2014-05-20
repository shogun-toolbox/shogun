/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (w) 2014 Wu Lin
 * Written (W) 2013 Roman Votyakov
 *
 * The abscissae and weights for Gauss-Kronrod rules are taken form
 * QUADPACK, which is in public domain.
 * http://www.netlib.org/quadpack/
 *
 * See header file for which functions are adapted from GNU Octave,
 * file quadgk.m: Copyright (C) 2008-2012 David Bateman under GPLv3
 * http://www.gnu.org/software/octave/
 *
 * See header file for which functions are adapted from
 * Gaussian Process Machine Learning Toolbox, file util/gauher.m,
 * http://www.gaussianprocess.org/gpml/code/matlab/doc/
 */

#include <shogun/mathematics/Integration.h>

#ifdef HAVE_EIGEN3

#include <shogun/mathematics/eigen3.h>
#include <shogun/lib/SGVector.h>

using namespace shogun;
using namespace Eigen;

namespace shogun
{

#ifndef DOXYGEN_SHOULD_SKIP_THIS

/** @brief Class of the function, which is used for standard infinite
 * to finite integral transformation
 *
 * \f[
 * \int_{-\infty}^{\infty}f(x)dx = \int_{-1}^{1}f(g(t))g'(t)dt
 * \f]
 *
 * where \f$g(t)=\frac{t}{1-t^2}\f$ and
 * \f$g'(t)=\frac{1+t^2}{(1-t^2)^2}\f$.
 */
class CITransformFunction : public CFunction
{
public:
	/** constructor
	 *
	 * @param f function \f$f(x)\f$
	 */
	CITransformFunction(CFunction* f)
	{
		SG_REF(f);
		m_f=f;
	}

	virtual ~CITransformFunction() { SG_UNREF(m_f); }

	/** return the real value of the function at given point
	 *
	 * @param x argument
	 *
	 * @return \f$f(g(x))*g'(x)\f$, where \f$g(x)=\frac{x}{1-x^2}\f$
	 * and \f$g'(t)=\frac{1+t^2}{(1-t^2)^2}\f$
	 */
	virtual float64_t operator() (float64_t x)
	{
		float64_t hx=1.0/(1.0-CMath::sq(x));
		float64_t gx=x*hx;
		float64_t dgx=(1.0+CMath::sq(x))*CMath::sq(hx);

		return (*m_f)(gx)*dgx;
	}

private:
	/** function \f$f(x)\f$ */
	CFunction* m_f;
};

/** @brief Class of the function, which is used for singularity
 * weakening transform on \f$(-\infty, b]\f$
 *
 * \f[
 * \int_{-\infty}^{b} f(x)dx=-\int_{-\infty}^{0} f(b-t^2)2tdt
 * \f]
 *
 * and the finite interval transform
 *
 * \f[
 * \int_{-\infty}^{0} f(b-t^2)2tdt = \int_{-1}^{0} f(b-g(s)^2)2g(s)g'(s)ds
 * \f]
 *
 * where \f$g(s)=\frac{s}{1+s}\f$ and \f$g'(s)=\frac{1}{(1+s)^2}\f$.
 */
class CILTransformFunction : public CFunction
{
public:
	/** constructor
	 *
	 * @param f function \f$f(x)\f$
	 * @param b upper bound
	 */
	CILTransformFunction(CFunction* f, float64_t b)
	{
		SG_REF(f);
		m_f=f;
		m_b=b;
	}

	virtual ~CILTransformFunction() { SG_UNREF(m_f); }

	/** return the real value of the function at given point
	 *
	 * @param x argument of a function
	 *
	 * @return \f$f(b-g(x)^2)2g(x)g'(x)dx\f$, where
	 * \f$g(x)=\frac{x}{1+x}\f$ and \f$g'(x)=\frac{1}{(1+x)^2}\f$
	 */
	virtual float64_t operator() (float64_t x)
	{
		float64_t hx=1.0/(1.0+x);
		float64_t gx=x*hx;
		float64_t dgx=CMath::sq(hx);

		return -(*m_f)(m_b-CMath::sq(gx))*2*gx*dgx;
	}

private:
	/** function \f$f(x)\f$ */
	CFunction* m_f;

	/** upper bound */
	float64_t m_b;
};

/** @brief Class of the function, which is used for singularity
 * weakening transform on \f$[a, \infty)\f$
 *
 * \f[
 * \int_{a}^{\infty} f(x)dx=\int_{0}^{\infty} f(a+t^2)2tdt
 * \f]
 *
 * and the finite interval transform
 *
 * \f[
 * \int_{0}^{\infty} f(a+t^2)2tdt = \int_{0}^{1} f(a+g(s)^2)2g(s)g'(s)ds
 * \f]
 *
 * where \f$g(s)=\frac{s}{1-s}\f$ and \f$g'(s)=\frac{1}{(1-s)^2}\f$.
 */
class CIUTransformFunction : public CFunction
{
public:
	/** constructor
	 *
	 * @param f function \f$f(x)\f$
	 * @param a lower bound
	 */
	CIUTransformFunction(CFunction* f, float64_t a)
	{
		SG_REF(f);
		m_f=f;
		m_a=a;
	}

	virtual ~CIUTransformFunction() { SG_UNREF(m_f); }

	/** return the real value of the function at given point
	 *
	 * @param x argument of a function
	 *
	 * @return \f$f(a+g(x)^2)2g(x)g'(x)\f$, where
	 * \f$g(x)=\frac{x}{1-x}\f$ and \f$g'(x)=\frac{1}{(1-x)^2}\f$
	 */
	virtual float64_t operator() (float64_t x)
	{
		float64_t hx=1.0/(1.0-x);
		float64_t gx=x*hx;
		float64_t dgx=CMath::sq(hx);

		return (*m_f)(m_a+CMath::sq(gx))*2*gx*dgx;
	}

private:
	/** function \f$f(x)\f$ */
	CFunction* m_f;

	/** lower bound */
	float64_t m_a;
};

/** @brief Class of a function, which is used for finite integral
 * transformation
 *
 * \f[
 * \int_{a}^{b}f(x)dx = \int_{-1}^{1} f(g(t))g'(t)dt
 * \f]
 *
 * where \f$g(t)=\frac{b-a}{2}(\frac{t}{2}(3-t^2))+\frac{b+a}{2}\f$
 * and \f$g'(t)=\frac{b-a}{4}(3-3t^2)\f$.
 */
class CTransformFunction : public CFunction
{
public:
	/** constructor
	 *
	 * @param f function \f$f(x)\f$
	 * @param a lower bound
	 * @param b upper bound
	 */
	CTransformFunction(CFunction* f, float64_t a, float64_t b)
	{
		SG_REF(f);
		m_f=f;
		m_a=a;
		m_b=b;
	}

	virtual ~CTransformFunction() { SG_UNREF(m_f); }

	/** return the real value of the function at given point
	 *
	 * @param x argument of a function
	 *
	 * @return \f$f(g(x))g'(x)\f$, where
	 * \f$g(t)=\frac{b-a}{2}(\frac{t}{2}(3-t^2))+\frac{b+a}{2}\f$ and
	 * \f$g'(t)=\frac{b-a}{4}(3-3t^2)\f$
	 */
	virtual float64_t operator() (float64_t x)
	{
		float64_t qw=(m_b-m_a)/4.0;
		float64_t gx=qw*(x*(3.0-CMath::sq(x)))+(m_b+m_a)/2.0;
		float64_t dgx=qw*3.0*(1.0-CMath::sq(x));

		return (*m_f)(gx)*dgx;
	}

private:
	/** function \f$f(x)\f$ */
	CFunction* m_f;

	/** lower bound */
	float64_t m_a;

	/** upper bound */
	float64_t m_b;
};

#endif /* DOXYGEN_SHOULD_SKIP_THIS */

float64_t CIntegration::integrate_quadgk(CFunction* f, float64_t a,
		float64_t b, float64_t abs_tol, float64_t rel_tol, uint32_t max_iter,
		index_t sn)
{
	// check the parameters
	REQUIRE(f, "Integrable function should not be NULL\n")
	REQUIRE(abs_tol>0.0, "Absolute tolerance must be positive, but is %f\n",
			abs_tol)
	REQUIRE(rel_tol>0.0, "Relative tolerance must be positive, but is %f\n",
			rel_tol)
	REQUIRE(max_iter>0, "Maximum number of iterations must be greater than 0, "
			"but is %d\n", max_iter)
	REQUIRE(sn>0, "Initial number of subintervals must be greater than 0, "
			"but is %d\n", sn)

	// integral evaluation function
	typedef void TQuadGKEvaluationFunction(CFunction* f,
		CDynamicArray<float64_t>* subs,	CDynamicArray<float64_t>* q,
		CDynamicArray<float64_t>* err);

	TQuadGKEvaluationFunction* evaluate_quadgk;

	CFunction* tf;
	float64_t ta;
	float64_t tb;
	float64_t q_sign;

	// negate integral value and swap a and b, if a>b
	if (a>b)
	{
		ta=b;
		tb=a;
		q_sign=-1.0;
	}
	else
	{
		ta=a;
		tb=b;
		q_sign=1.0;
	}

	// transform integrable function and domain of integration
	if (a==-CMath::INFTY && b==CMath::INFTY)
	{
		tf=new CITransformFunction(f);
		evaluate_quadgk=&evaluate_quadgk15;
		ta=-1.0;
		tb=1.0;
	}
	else if (a==-CMath::INFTY)
	{
		tf=new CILTransformFunction(f, b);
		evaluate_quadgk=&evaluate_quadgk15;
		ta=-1.0;
		tb=0.0;
	}
	else if (b==CMath::INFTY)
	{
		tf=new CIUTransformFunction(f, a);
		evaluate_quadgk=&evaluate_quadgk15;
		ta=0.0;
		tb=1.0;
	}
	else
	{
		tf=new CTransformFunction(f, a, b);
		evaluate_quadgk=&evaluate_quadgk21;
		ta=-1.0;
		tb=1.0;
	}

	// compute initial subintervals, by dividing domain [a, b] into sn
	// parts
	CDynamicArray<float64_t>* subs=new CDynamicArray<float64_t>();

	// width of each subinterval
	float64_t sw=(tb-ta)/sn;

	for (index_t i=0; i<sn; i++)
	{
		subs->push_back(ta+i*sw);
		subs->push_back(ta+(i+1)*sw);
	}

	// evaluate integrals on initial subintervals
	CDynamicArray<float64_t>* q_subs=new CDynamicArray<float64_t>();
	CDynamicArray<float64_t>* err_subs=new CDynamicArray<float64_t>();

	evaluate_quadgk(tf, subs, q_subs, err_subs);

	// compute value of integral and error on [a, b]
	float64_t q=0.0;
	float64_t err=0.0;

	for (index_t i=0; i<q_subs->get_num_elements(); i++)
		q+=(*q_subs)[i];

	for (index_t i=0; i<err_subs->get_num_elements(); i++)
		err+=(*err_subs)[i];

	// evaluate tolerance
	float64_t tol=CMath::max(abs_tol, rel_tol*CMath::abs(q));

	// number of iterations
	uint32_t iter=1;

	CDynamicArray<float64_t>* new_subs=new CDynamicArray<float64_t>();

	while (err>tol && iter<max_iter)
	{
		// choose and bisect subintervals with estimated error, which
		// is larger or equal to tolerance
		for (index_t i=0; i<subs->get_num_elements()/2; i++)
		{
			if (CMath::abs((*err_subs)[i])>=tol*CMath::abs((*subs)[2*i+1]-
				(*subs)[2*i])/(tb-ta))
			{
				// bisect subinterval
				float64_t mid=((*subs)[2*i]+(*subs)[2*i+1])/2.0;

				new_subs->push_back((*subs)[2*i]);
				new_subs->push_back(mid);
				new_subs->push_back(mid);
				new_subs->push_back((*subs)[2*i+1]);

				// subtract value of the integral and error on this
				// subinterval from total value and error
				q-=(*q_subs)[i];
				err-=(*err_subs)[i];
			}
		}

		subs->set_array(new_subs->get_array(), new_subs->get_num_elements(),
			new_subs->get_num_elements());

		new_subs->reset_array();

		// break if no new subintervals
		if (!subs->get_num_elements())
			break;

		// evaluate integrals on selected subintervals
		evaluate_quadgk(tf, subs, q_subs, err_subs);

		for (index_t i=0; i<q_subs->get_num_elements(); i++)
			q+=(*q_subs)[i];

		for (index_t i=0; i<err_subs->get_num_elements(); i++)
			err+=(*err_subs)[i];

		// evaluate tolerance
		tol=CMath::max(abs_tol, rel_tol*CMath::abs(q));

		iter++;
	}

	SG_UNREF(new_subs);

	if (err>tol)
	{
		SG_SWARNING("Error tolerance not met. Estimated error is equal to %g "
				"after %d iterations\n", err, iter)
	}

	// clean up
	SG_UNREF(subs);
	SG_UNREF(q_subs);
	SG_UNREF(err_subs);
	SG_UNREF(tf);

	return q_sign*q;
}

float64_t CIntegration::integrate_quadgh(CFunction* f)
{
	SG_REF(f);

	// evaluate integral using Gauss-Hermite 64-point rule
	float64_t q=evaluate_quadgh64(f);

	SG_UNREF(f);

	return q;
}

float64_t CIntegration::integrate_quadgh_customized(CFunction* f,
	SGVector<float64_t> xgh, SGVector<float64_t> wgh)
{
	REQUIRE(xgh.vlen == wgh.vlen,
		"The length of node array (%d) and weight array (%d) should be the same\n",
		xgh.vlen, wgh.vlen);

	SG_REF(f);

	float64_t q=evaluate_quadgh(f, xgh.vlen, xgh.vector, wgh.vector);

	SG_UNREF(f);

	return q;
}

void CIntegration::evaluate_quadgk(CFunction* f, CDynamicArray<float64_t>* subs,
		CDynamicArray<float64_t>* q, CDynamicArray<float64_t>* err, index_t n,
		float64_t* xgk, float64_t* wg, float64_t* wgk)
{
	// check the parameters
	REQUIRE(f, "Integrable function should not be NULL\n")
	REQUIRE(subs, "Array of subintervals should not be NULL\n")
	REQUIRE(!(subs->get_array_size()%2), "Size of the array of subintervals "
		"should be even\n")
	REQUIRE(q, "Array of values of integrals should not be NULL\n")
	REQUIRE(err, "Array of errors should not be NULL\n")
	REQUIRE(n%2, "Order of Gauss-Kronrod should be odd\n")
	REQUIRE(xgk, "Gauss-Kronrod nodes should not be NULL\n")
	REQUIRE(wgk, "Gauss-Kronrod weights should not be NULL\n")
	REQUIRE(wg, "Gauss weights should not be NULL\n")

	// create eigen representation of subs, xgk, wg, wgk
	Map<MatrixXd> eigen_subs(subs->get_array(), 2, subs->get_num_elements()/2);
	Map<VectorXd> eigen_xgk(xgk, n);
	Map<VectorXd> eigen_wg(wg, n/2);
	Map<VectorXd> eigen_wgk(wgk, n);

	// compute half width and centers of each subinterval
	VectorXd eigen_hw=(eigen_subs.row(1)-eigen_subs.row(0))/2.0;
	VectorXd eigen_center=eigen_subs.colwise().sum()/2.0;

	// compute Gauss-Kronrod nodes x for each subinterval: x=hw*xgk+center
	MatrixXd x=eigen_hw*eigen_xgk.adjoint()+eigen_center*
		(VectorXd::Ones(n)).adjoint();

	// compute ygk=f(x)
	MatrixXd ygk(x.rows(), x.cols());

	for (index_t i=0; i<ygk.rows(); i++)
		for (index_t j=0; j<ygk.cols(); j++)
			ygk(i,j)=(*f)(x(i,j));

	// compute value of definite integral on each subinterval
	VectorXd eigen_q=((ygk*eigen_wgk.asDiagonal()).rowwise().sum()).cwiseProduct(
		eigen_hw);
	q->set_array(eigen_q.data(), eigen_q.size());

	// choose function values for Gauss nodes
	MatrixXd yg(ygk.rows(), ygk.cols()/2);

	for (index_t i=1, j=0; i<ygk.cols(); i+=2, j++)
		yg.col(j)=ygk.col(i);

	// compute error on each subinterval
	VectorXd eigen_err=(((yg*eigen_wg.asDiagonal()).rowwise().sum()).cwiseProduct(
		eigen_hw)-eigen_q).array().abs();
	err->set_array(eigen_err.data(), eigen_err.size());
}

void CIntegration::generate_gauher(SGVector<float64_t> xgh, SGVector<float64_t> wgh)
{
	REQUIRE(xgh.vlen == wgh.vlen,
		"The length of node array (%d) and weight array (%d) should be the same\n",
		xgh.vlen, wgh.vlen);

	index_t n = xgh.vlen;

	if (n == 20)
	{
		generate_gauher20(xgh, wgh);
	}
	else
	{
		Map<VectorXd> eigen_xgh(xgh.vector, xgh.vlen);
		Map<VectorXd> eigen_wgh(wgh.vector, wgh.vlen);

		eigen_xgh = MatrixXd::Zero(n,1);
		eigen_wgh = MatrixXd::Ones(n,1);

		if (n > 1)
		{
			MatrixXd v = MatrixXd::Zero(n,n);

			//b = sqrt( (1:N-1)/2 )';    
			//[V,D] = eig( diag(b,1) + diag(b,-1) );
			v.block(0, 1, n-1, n-1).diagonal() = (0.5*ArrayXd::LinSpaced(n-1,1,n-1)).sqrt();
			v.block(1, 0, n-1, n-1).diagonal() = v.block(0, 1, n-1, n-1).diagonal();
			EigenSolver<MatrixXd> eig(v);

			//w = V(1,:)'.^2
			eigen_wgh = eig.eigenvectors().row(0).transpose().real().array().pow(2);

			//x = sqrt(2)*diag(D)
			eigen_xgh = eig.eigenvalues().real()*sqrt(2.0);
		}
	}
}

void CIntegration::generate_gauher20(SGVector<float64_t> xgh, SGVector<float64_t> wgh)
{
	REQUIRE(xgh.vlen == wgh.vlen,
		"The length of node array (%d) and weight array (%d) should be the same\n",
		xgh.vlen, wgh.vlen);
	REQUIRE(xgh.vlen == 20, "The length of xgh and wgh should be 20\n");

	static const index_t n = 20;
	static float64_t wgh_pre[n]=
	{
		0.0000000000001257800672437920121938444754,
		0.0000000002482062362315158465220083577413,
		0.0000000612749025998290679114578012251502,
		0.0000044021210902308611768963750310312832,
		0.0001288262799619289543807260089991473251,
		0.0018301031310804880686271545187082665507,
		0.0139978374471010288959682554832397727296,
		0.0615063720639768204967445797137770568952,
		0.1617393339840000332507941038784338161349,
		0.2607930634495548849471902030927594751120,
		0.2607930634495547739248877405771054327488,
		0.1617393339840003108065502601675689220428,
		0.0615063720639767788633811562704067910090,
		0.0139978374471010080792865437615546397865,
		0.0018301031310804856833823750505985117343,
		0.0001288262799619298488475183095403053812,
		0.0000044021210902308865878847926600414553,
		0.0000000612749025998294252534824241331057,
		0.0000000002482062362315177593771748866178,
		0.0000000000001257800672437921636551382778
	};

	static float64_t xgh_pre[n]=
	{
		-7.6190485416797573137159815814811736345291,
		-6.5105901570136559541879250900819897651672,
		-5.5787388058932032564030123467091470956802,
		-4.7345813340460569662582201999612152576447,
		-3.9439673506573176275935566081898286938667,
		-3.1890148165533904744961546384729444980621,
		-2.4586636111723669806394809711491689085960,
		-1.7452473208141270344384565760265104472637,
		-1.0429453488027506935509336472023278474808,
		-0.3469641570813560282893206476728664711118,
		0.3469641570813561393116231101885205134749,
		1.0429453488027513596847484222962521016598,
		1.7452473208141265903492467259638942778111,
		2.4586636111723669806394809711491689085960,
		3.1890148165533904744961546384729444980621,
		3.9439673506573162953259270580019801855087,
		4.7345813340460569662582201999612152576447,
		5.5787388058932014800461729464586824178696,
		6.5105901570136532896526659897062927484512,
		7.6190485416797573137159815814811736345291
		
	};

	for (index_t idx = 0; idx < n; idx++)
	{
		wgh[idx] = wgh_pre[idx];
		xgh[idx] = xgh_pre[idx];
	}

}

void CIntegration::evaluate_quadgk15(CFunction* f, CDynamicArray<float64_t>* subs,
		CDynamicArray<float64_t>* q, CDynamicArray<float64_t>* err)
{
	static const index_t n=15;

	// Gauss-Kronrod nodes
	static float64_t xgk[n]=
		{
			-0.991455371120812639206854697526329,
			-0.949107912342758524526189684047851,
			-0.864864423359769072789712788640926,
			-0.741531185599394439863864773280788,
			-0.586087235467691130294144838258730,
			-0.405845151377397166906606412076961,
			-0.207784955007898467600689403773245,
			0.000000000000000000000000000000000,
			0.207784955007898467600689403773245,
			0.405845151377397166906606412076961,
			0.586087235467691130294144838258730,
			0.741531185599394439863864773280788,
			0.864864423359769072789712788640926,
			0.949107912342758524526189684047851,
			0.991455371120812639206854697526329
		};

	// Gauss weights
	static float64_t wg[n/2]=
		{
			0.129484966168869693270611432679082,
			0.279705391489276667901467771423780,
			0.381830050505118944950369775488975,
			0.417959183673469387755102040816327,
			0.381830050505118944950369775488975,
			0.279705391489276667901467771423780,
			0.129484966168869693270611432679082
		};

	// Gauss-Kronrod weights
	static float64_t wgk[n]=
		{
			0.022935322010529224963732008058970,
			0.063092092629978553290700663189204,
			0.104790010322250183839876322541518,
			0.140653259715525918745189590510238,
			0.169004726639267902826583426598550,
			0.190350578064785409913256402421014,
			0.204432940075298892414161999234649,
			0.209482141084727828012999174891714,
			0.204432940075298892414161999234649,
			0.190350578064785409913256402421014,
			0.169004726639267902826583426598550,
			0.140653259715525918745189590510238,
			0.104790010322250183839876322541518,
			0.063092092629978553290700663189204,
			0.022935322010529224963732008058970
		};

	// evaluate definite integral on each subinterval using Gauss-Kronrod rule
	evaluate_quadgk(f, subs, q, err, n, xgk, wg, wgk);
}

void CIntegration::evaluate_quadgk21(CFunction* f, CDynamicArray<float64_t>* subs,
		CDynamicArray<float64_t>* q, CDynamicArray<float64_t>* err)
{
	static const index_t n=21;

	// Gauss-Kronrod nodes
	static float64_t xgk[n]=
		{
			-0.995657163025808080735527280689003,
			-0.973906528517171720077964012084452,
			-0.930157491355708226001207180059508,
			-0.865063366688984510732096688423493,
			-0.780817726586416897063717578345042,
			-0.679409568299024406234327365114874,
			-0.562757134668604683339000099272694,
			-0.433395394129247190799265943165784,
			-0.294392862701460198131126603103866,
			-0.148874338981631210884826001129720,
			0.000000000000000000000000000000000,
			0.148874338981631210884826001129720,
			0.294392862701460198131126603103866,
			0.433395394129247190799265943165784,
			0.562757134668604683339000099272694,
			0.679409568299024406234327365114874,
			0.780817726586416897063717578345042,
			0.865063366688984510732096688423493,
			0.930157491355708226001207180059508,
			0.973906528517171720077964012084452,
			0.995657163025808080735527280689003
		};

	// Gauss weights
	static float64_t wg[n/2]=
		{
			0.066671344308688137593568809893332,
			0.149451349150580593145776339657697,
			0.219086362515982043995534934228163,
			0.269266719309996355091226921569469,
			0.295524224714752870173892994651338,
			0.295524224714752870173892994651338,
			0.269266719309996355091226921569469,
			0.219086362515982043995534934228163,
			0.149451349150580593145776339657697,
			0.066671344308688137593568809893332
		};

	// Gauss-Kronrod weights
	static float64_t wgk[n]=
		{
			0.011694638867371874278064396062192,
			0.032558162307964727478818972459390,
			0.054755896574351996031381300244580,
			0.075039674810919952767043140916190,
			0.093125454583697605535065465083366,
			0.109387158802297641899210590325805,
			0.123491976262065851077958109831074,
			0.134709217311473325928054001771707,
			0.142775938577060080797094273138717,
			0.147739104901338491374841515972068,
			0.149445554002916905664936468389821,
			0.147739104901338491374841515972068,
			0.142775938577060080797094273138717,
			0.134709217311473325928054001771707,
			0.123491976262065851077958109831074,
			0.109387158802297641899210590325805,
			0.093125454583697605535065465083366,
			0.075039674810919952767043140916190,
			0.054755896574351996031381300244580,
			0.032558162307964727478818972459390,
			0.011694638867371874278064396062192
		};

	evaluate_quadgk(f, subs, q, err, n, xgk, wg, wgk);
}

float64_t CIntegration::evaluate_quadgh(CFunction* f, index_t n, float64_t* xgh,
		float64_t* wgh)
{
	// check the parameters
	REQUIRE(f, "Integrable function should not be NULL\n");
	REQUIRE(xgh, "Gauss-Hermite nodes should not be NULL\n");
	REQUIRE(wgh, "Gauss-Hermite weights should not be NULL\n");

	float64_t q=0.0;

	for (index_t i=0; i<n; i++)
		q+=wgh[i]*(*f)(xgh[i]);

	return q;
}

float64_t CIntegration::evaluate_quadgh64(CFunction* f)
{
	static const index_t n=64;

	// Gauss-Hermite nodes
	static float64_t xgh[n]=
	{
		-10.52612316796054588332682628381528,
		-9.895287586829539021204461477159608,
		-9.373159549646721162545652439723862,
		-8.907249099964769757295972885642943,
		-8.477529083379863090564166344821916,
		-8.073687285010225225858791140758144,
		-7.68954016404049682844780422986949,
		-7.321013032780949201189569363719477,
		-6.965241120551107529242642193492688,
		-6.620112262636027379036660108937914,
		-6.284011228774828235418093195070243,
		-5.955666326799486045344567180984366,
		-5.634052164349972147249920483307154,
		-5.318325224633270857323649515199378,
		-5.007779602198768196443702627184136,
		-4.701815647407499816097538015812822,
		-4.399917168228137647767932535438923,
		-4.101634474566656714970981238455522,
		-3.806571513945360461165972000460225,
		-3.514375935740906211539950586474333,
		-3.224731291992035725848171110188419,
		-2.93735082300462180968533902619139,
		-2.651972435430635011005457785998431,
		-2.368354588632401404111511265341516,
		-2.086272879881762020832563302363221,
		-1.805517171465544918903773574186889,
		-1.525889140209863662948970133151528,
		-1.24720015694311794069356453069359,
		-0.9692694230711780167435414890191023,
		-0.6919223058100445772682192875955947,
		-0.4149888241210786845769291291996859,
		-0.1383022449870097241150497679666744,
		0.1383022449870097241150497679666744,
		0.4149888241210786845769291291996859,
		0.6919223058100445772682192875955947,
		0.9692694230711780167435414890191023,
		1.24720015694311794069356453069359,
		1.525889140209863662948970133151528,
		1.805517171465544918903773574186889,
		2.086272879881762020832563302363221,
		2.368354588632401404111511265341516,
		2.651972435430635011005457785998431,
		2.93735082300462180968533902619139,
		3.224731291992035725848171110188419,
		3.514375935740906211539950586474333,
		3.806571513945360461165972000460225,
		4.101634474566656714970981238455522,
		4.399917168228137647767932535438923,
		4.701815647407499816097538015812822,
		5.007779602198768196443702627184136,
		5.318325224633270857323649515199378,
		5.634052164349972147249920483307154,
		5.955666326799486045344567180984366,
		6.284011228774828235418093195070243,
		6.620112262636027379036660108937914,
		6.965241120551107529242642193492688,
		7.321013032780949201189569363719477,
		7.68954016404049682844780422986949,
		8.073687285010225225858791140758144,
		8.477529083379863090564166344821916,
		8.907249099964769757295972885642943,
		9.373159549646721162545652439723862,
		9.895287586829539021204461477159608,
		10.52612316796054588332682628381528
	};

	// Gauss-Hermite weights
	static float64_t wgh[n]=
	{
		5.535706535856942820575463300987E-49,
		1.6797479901081592186662883306299E-43,
		3.4211380112557405043272218281457E-39,
		1.557390624629763802309335380265E-35,
		2.549660899112999256604766580441E-32,
		1.92910359546496685030196877906707E-29,
		7.8617977889259103690999914962788E-27,
		1.911706883300642829958456965534449E-24,
		2.982862784279851154478700702016E-22,
		3.15225456650378141612134668341E-20,
		2.35188471067581911695767591555844E-18,
		1.28009339132243804163956329526337E-16,
		5.218623726590847522957808513052588E-15,
		1.628340730709720362084307081240893E-13,
		3.95917776694772392723644586425458E-12,
		7.61521725014545135331529567531937E-11,
		1.1736167423215493435425064670822E-9,
		1.465125316476109354926622003804004E-8,
		1.495532936727247061102461692934817E-7,
		1.258340251031184576157842180019028E-6,
		8.7884992308503591814440474067043E-6,
		5.125929135786274660821911412739621E-5,
		2.509836985130624860823620179819094E-4,
		0.001036329099507577663456741746283101,
		0.00362258697853445876066812537162265,
		0.01075604050987913704946517278667313,
		0.0272031289536889184538348212614932,
		0.0587399819640994345496889462518317,
		0.1084983493061868406330258455060973,
		0.1716858423490837020007279701237768,
		0.2329947860626780466505660293325675,
		0.2713774249413039779456065084184279,
		0.2713774249413039779456065084184279,
		0.2329947860626780466505660293325675,
		0.1716858423490837020007279701237768,
		0.1084983493061868406330258455060973,
		0.0587399819640994345496889462518317,
		0.0272031289536889184538348212614932,
		0.01075604050987913704946517278667313,
		0.00362258697853445876066812537162265,
		0.001036329099507577663456741746283101,
		2.509836985130624860823620179819094E-4,
		5.125929135786274660821911412739621E-5,
		8.7884992308503591814440474067043E-6,
		1.258340251031184576157842180019028E-6,
		1.495532936727247061102461692934817E-7,
		1.465125316476109354926622003804004E-8,
		1.1736167423215493435425064670822E-9,
		7.61521725014545135331529567531937E-11,
		3.95917776694772392723644586425458E-12,
		1.628340730709720362084307081240893E-13,
		5.218623726590847522957808513052588E-15,
		1.28009339132243804163956329526337E-16,
		2.35188471067581911695767591555844E-18,
		3.15225456650378141612134668341E-20,
		2.982862784279851154478700702016E-22,
		1.911706883300642829958456965534449E-24,
		7.8617977889259103690999914962788E-27,
		1.92910359546496685030196877906707E-29,
		2.549660899112999256604766580441E-32,
		1.557390624629763802309335380265E-35,
		3.4211380112557405043272218281457E-39,
		1.6797479901081592186662883306299E-43,
		5.535706535856942820575463300987E-49
	};

	return evaluate_quadgh(f, n, xgh, wgh);
}
}

#endif /* HAVE_EIGEN3 */
