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
 */

#include <mathematics/Math.h>
#include <mathematics/JacobiEllipticFunctions.h>

using namespace shogun;

void CJacobiEllipticFunctions::ellipKKp(Real L, Real &K, Real &Kp)
{
	REQUIRE(L>=0.0,
		"CJacobiEllipticFunctions::ellipKKp(): \
		Parameter L should be non-negative\n");
#ifdef HAVE_ARPREC
	const Real eps=Real(std::numeric_limits<float64_t>::epsilon());
	const Real pi=mp_real::_pi;
#else
	const Real eps=std::numeric_limits<Real>::epsilon();
	const Real pi=M_PI;
#endif //HAVE_ARPREC
	if (L>10.0)
	{
		K=pi*0.5;
		Kp=pi*L+log(4.0);
	}
	else
	{
		Real m=exp(-2.0*pi*L);
		Real mp=1.0-m;
		if (m<eps)
		{
			K=compute_quarter_period(sqrt(mp));
			Kp=Real(std::numeric_limits<float64_t>::max());
		}
		else if (mp<eps)
		{
			K=Real(std::numeric_limits<float64_t>::max());
			Kp=compute_quarter_period(sqrt(m));
		}
		else
		{
			K=compute_quarter_period(sqrt(mp));
			Kp=compute_quarter_period(sqrt(m));
		}
	}
}

void CJacobiEllipticFunctions
	::ellipJC(Complex u, Real m, Complex &sn, Complex &cn, Complex &dn)
{
	REQUIRE(m>=0.0 && m<=1.0,
		"CJacobiEllipticFunctions::ellipJC(): \
		Parameter m should be >=0 and <=1\n");

#ifdef HAVE_ARPREC
	const Real eps=sqrt(mp_real::_eps);
#else
	const Real eps=sqrt(std::numeric_limits<Real>::epsilon());
#endif //HAVE_ARPREC
	if (m>=(1.0-eps))
	{
#ifdef HAVE_ARPREC
		complex128_t _u(dble(u.real),dble(u.imag));
		complex128_t t=CMath::tanh(_u);
		complex128_t b=CMath::cosh(_u);
		complex128_t twon=b*CMath::sinh(_u);
		complex128_t ai=0.25*(1.0-dble(m));
		complex128_t _sn=t+ai*(twon-_u)/(b*b);
		complex128_t phi=1.0/b;
		complex128_t _cn=phi-ai*(twon-_u);
		complex128_t _dn=phi+ai*(twon+_u);
		sn=mp_complex(_sn.real(),_sn.imag());
		cn=mp_complex(_cn.real(),_cn.imag());
		dn=mp_complex(_dn.real(),_dn.imag());
#else
		Complex t=CMath::tanh(u);
		Complex b=CMath::cosh(u);
		Complex ai=0.25*(1.0-m);
		Complex twon=b*CMath::sinh(u);
		sn=t+ai*(twon-u)/(b*b);
		Complex phi=Real(1.0)/b;
		ai*=t*phi;
		cn=phi-ai*(twon-u);
		dn=phi+ai*(twon+u);
#endif //HAVE_ARPREC
	}
	else
	{
		const Real prec=4.0*eps;
		const index_t MAX_ITER=128;
		index_t i=0;
		Real kappa[MAX_ITER];

		while (i<MAX_ITER && m>prec)
		{
			Real k;
			if (m>0.001)
			{
				Real mp=sqrt(1.0-m);
				k=(1.0-mp)/(1.0+mp);
			}
			else
				k=poly_six(m/4.0);
			u/=(1.0+k);
			m=k*k;
			kappa[i++]=k;
		}
		Complex sin_u=sin(u);
		Complex cos_u=cos(u);
		Complex t=Real(0.25*m)*(u-sin_u*cos_u);
		sn=sin_u-t*cos_u;
		cn=cos_u+t*sin_u;
		dn=Real(1.0)+Real(0.5*m)*(cos_u*cos_u);

		i--;
		while (i>=0)
		{
			Real k=kappa[i--];
			Complex ksn2=k*(sn*sn);
			Complex d=Real(1.0)+ksn2;
			sn*=(1.0+k)/d;
			cn*=dn/d;
			dn=(Real(1.0)-ksn2)/d;
		}
	}
}
