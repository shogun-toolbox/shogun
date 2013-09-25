/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 * 
 * Written (W) 2013 Soumyajit De
 * 
 * KRYLSTAT Copyright 2011 by Erlend Aune <erlenda@math.ntnu.no> under GPL2+
 * (few parts rewritten and adjusted for shogun)
 *
 * NOTE: For higher precision, the methods in this class rely on an external
 * library, ARPREC (http://crd-legacy.lbl.gov/~dhbailey/mpdist/), in absense of
 * which they fallback to shogun datatypes. To use it with shogun, configure 
 * ARPREC with `CXX="c++ -fPIC" ./configure' in order to link.
 */

#ifndef JACOBI_ELLIPTIC_FUNCTIONS_H_
#define JACOBI_ELLIPTIC_FUNCTIONS_H_

#include <shogun/lib/config.h>
#include <shogun/base/SGObject.h>
#include <limits>
#include <math.h>

#ifdef HAVE_ARPREC
#include <arprec/mp_real.h>
#include <arprec/mp_complex.h>
#endif //HAVE_ARPREC

namespace shogun
{

/** @brief Class that contains methods for computing Jacobi elliptic functions
 * related to complex analysis. These functions are inverse of the elliptic
 * integral of first kind, i.e.
 * \f[
 * u(k,m)=\int_{0}^{k}\frac{dt}{\sqrt{(1-t^{2})(1-m^{2}t^{2})}}
 * =\int_{0}^{\varphi}\frac{d\theta}{\sqrt{(1-m^{2}sin^{2}\theta)}}
 * \f]
 * where \f$k=sin\varphi\f$, \f$t=sin\theta\f$ and parameter \f$m, 0\le m
 * \le 1\f$ is called modulus. Three main Jacobi elliptic functions are defined
 * as \f$sn(u,m)=k=sin\theta\f$, \f$cn(u,m)=cos\theta=\sqrt{1-sn(u,m)^{2}}\f$
 * and \f$dn(u,m)=\sqrt{1-m^{2}sn(u,m)^{2}}\f$.
 * For \f$k=1\f$, i.e. \f$\varphi=\frac{\pi}{2}\f$, \f$u(1,m)=K(m)\f$ is known
 * as the complete elliptic integral of first kind. Similarly, \f$u(1,m'))=
 * K'(m')\f$, \f$m'=\sqrt{1-m^{2}}\f$ is called the complementary complete
 * elliptic integral of first kind. Jacobi functions are double periodic with
 * quardratic periods \f$K\f$ and \f$K'\f$.
 *
 * This class provides two sets of methods for computing \f$K,K'\f$, and
 * \f$sn,cn,dn\f$. Useful for computing rational approximation of matrix
 * functions given by Cauchy's integral formula, etc.
 */
class CJacobiEllipticFunctions: public CSGObject
{
#ifdef HAVE_ARPREC
	typedef mp_real Real;
	typedef mp_complex Complex;
#else
	typedef float64_t Real;
	typedef complex128_t Complex;
#endif //HAVE_ARPREC
private:
	static inline Real compute_quarter_period(Real b)
	{
#ifdef HAVE_ARPREC
		const Real eps=mp_real::_eps;
		const Real pi=mp_real::_pi;
#else
		const Real eps=std::numeric_limits<Real>::epsilon();
		const Real pi=M_PI;
#endif //HAVE_ARPREC
		Real a=1.0;
		Real mm=1.0;

		int64_t p=2;
		do
		{
			Real a_new=(a+b)*0.5;
			Real b_new=sqrt(a*b);
			Real c=(a-b)*0.5;
			mm=Real(p)*c*c;
			p<<=1;
			a=a_new;
			b=b_new;
		} while (mm>eps);
		return pi*0.5/a;
	}

	static inline Real poly_six(Real x)
	{
		return (132*pow(x,6)+42*pow(x,5)+14*pow(x,4)+5*pow(x,3)+2*pow(x,2)+x);
	}

public:
	/** Computes the quarter periods (K and K') of Jacobian elliptic functions
	 * (see class description).
	 * @param L
	 * @param K the quarter period (to be computed) on the Real axis
	 * @param Kp the quarter period (to be computed) on the Imaginary axis
	 * computed
	 */
	static void ellipKKp(Real L, Real &K, Real &Kp);

	/** Computes three main Jacobi elliptic functions, \f$sn(u,m)\f$,
	 * \f$cn(u,m)\f$ and \f$dn(u,m)\f$ (see class description).
	 * @param u the elliptic integral of the first kind \f$u(k,m)\f$
	 * @param m the modulus parameter, \f$0\le m \le 1\f$
	 * @param sn Jacobi elliptic function sn(u,m)
	 * @param cn Jacobi elliptic function cn(u,m)
	 * @param dn Jacobi elliptic function dn(u,m)
	 */
	static void ellipJC(Complex u, Real m, Complex &sn, Complex &cn,
		Complex &dn);

#ifdef HAVE_ARPREC
	/** Wrapper method for ellipKKp if ARPREC is present (for high precision)
	 * @param L
	 * @param K the quarter period (to be computed) on the Real axis
	 * @param Kp the quarter period (to be computed) on the Imaginary axis
	 * computed
	 */
	static void ellipKKp(float64_t L, float64_t &K, float64_t &Kp)
	{
		mp::mp_init(100, NULL, true);
		mp_real _K, _Kp;
		ellipKKp(mp_real(L), _K, _Kp);
		K=dble(_K);
		Kp=dble(_Kp);
		mp::mp_finalize();
	}
	
	/** Wrapper method for ellipJC if ARPREC is present (for high precision)
	 * @param u the elliptic integral of the first kind \f$u(k,m)\f$
	 * @param m the modulus parameter, \f$0\le m \le 1\f$
	 * @param sn Jacobi elliptic function sn(u,m)
	 * @param cn Jacobi elliptic function cn(u,m)
	 * @param dn Jacobi elliptic function dn(u,m)
	 */
	static void ellipJC(complex128_t u, float64_t m,
		complex128_t &sn, complex128_t &cn, complex128_t &dn)
	{
		mp::mp_init(100, NULL, true);
		mp_complex _sn, _cn, _dn;
		ellipJC(mp_complex(u.real(),u.imag()), mp_real(m), _sn, _cn, _dn);
		sn=complex128_t(dble(_sn.real),dble(_sn.imag));
		cn=complex128_t(dble(_cn.real),dble(_cn.imag));
		dn=complex128_t(dble(_dn.real),dble(_dn.imag));
		mp::mp_finalize();
	}
#endif //HAVE_ARPREC

	/** @return object name */
	virtual const char* get_name() const
	{
		return "JacobiEllipticFunctions";
	}
};

}

#endif /* JACOBI_ELLIPTIC_FUNCTIONS_H_ */
