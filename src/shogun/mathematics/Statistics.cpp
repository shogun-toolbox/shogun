/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011 Heiko Strathmann
 * Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society
 *
 * ALGLIB Copyright 1984, 1987, 1995, 2000 by Stephen L. Moshier under GPL2+
 * http://www.alglib.net/
 * See header file for which functions are taken from ALGLIB (with adjustments
 * for shogun)
 */

#include "mathematics/Statistics.h"
#include "mathematics/Math.h"

using namespace shogun;

float64_t CStatistics::mean(SGVector<float64_t> values)
{
	ASSERT(values.vlen);
	ASSERT(values.vector);

	float64_t sum=0;
	for (index_t i=0; i<values.vlen; ++i)
		sum+=values.vector[i];

	return sum/values.vlen;
}

float64_t CStatistics::variance(SGVector<float64_t> values)
{
	ASSERT(values.vlen>1);
	ASSERT(values.vector);

	float64_t mean=CStatistics::mean(values);

	float64_t sum_squared_diff=0;
	for (index_t i=0; i<values.vlen; ++i)
		sum_squared_diff+=CMath::pow(values.vector[i]-mean, 2);

	return sum_squared_diff/(values.vlen-1);
}

float64_t CStatistics::std_deviation(SGVector<float64_t> values)
{
	return CMath::sqrt(variance(values));
}

float64_t CStatistics::confidence_intervals_mean(SGVector<float64_t> values,
		float64_t alpha, float64_t& conf_int_low, float64_t& conf_int_up)
{
	ASSERT(values.vlen>1);
	ASSERT(values.vector);

	/* using one sided student t distribution evaluation */
	alpha=alpha/2;

	/* degrees of freedom */
	int32_t deg=values.vlen-1;

	/* compute t-value */
	float64_t t=inverse_student_t_distribution(deg, alpha);

	/* values for calculating confidence interval */
	float64_t std_dev=std_deviation(values);
	float64_t mean=CStatistics::mean(values);

	/* compute confidence interval */
	float64_t interval=t*std_dev/CMath::sqrt((float64_t)values.vlen);
	conf_int_low=mean-interval;
	conf_int_up=mean+interval;

	return mean;
}

float64_t CStatistics::student_t_distribution(int32_t k, float64_t t)
{
	float64_t x;
	float64_t rk;
	float64_t z;
	float64_t f;
	float64_t tz;
	float64_t p;
	float64_t xsqk;
	int32_t j;
	float64_t result;

	ASSERT(k>0);
	if (t==0)
	{
		result=0.5;
		return result;
	}
	if (t<-2.0)
	{
		rk=k;
		z=rk/(rk+t*t);
		result=0.5*incomplete_beta(0.5*rk, 0.5, z);
		return result;
	}
	if (t<0)
	{
		x=-t;
	}
	else
	{
		x=t;
	}
	rk=k;
	z=1.0+x*x/rk;
	if (k%2!=0)
	{
		xsqk=x/CMath::sqrt(rk);
		p=CMath::atan(xsqk);
		if (k>1)
		{
			f=1.0;
			tz=1.0;
			j=3;
			while (j<=k-2&&tz/f>CMath::MACHINE_EPSILON)
			{
				tz=tz*((j-1)/(z*j));
				f=f+tz;
				j=j+2;
			}
			p=p+f*xsqk/z;
		}
		p=p*2.0/CMath::PI;
	}
	else
	{
		f=1.0;
		tz=1.0;
		j=2;
		while (j<=k-2&&tz/f>CMath::MACHINE_EPSILON)
		{
			tz=tz*((j-1)/(z*j));
			f=f+tz;
			j=j+2;
		}
		p=f*x/CMath::sqrt(z*rk);
	}
	if (t<0)
	{
		p=-p;
	}
	result=0.5+0.5*p;
	return result;
}

float64_t CStatistics::incomplete_beta(float64_t a, float64_t b, float64_t x)
{
	float64_t t;
	float64_t xc;
	float64_t w;
	float64_t y;
	int32_t flag;
	float64_t big;
	float64_t biginv;
	float64_t maxgam;
	float64_t minlog;
	float64_t maxlog;
	float64_t result;

	big=4.503599627370496e15;
	biginv=2.22044604925031308085e-16;
	maxgam=171.624376956302725;
	minlog=CMath::log(CMath::MIN_REAL_NUMBER);
	maxlog=CMath::log(CMath::MAX_REAL_NUMBER);
	ASSERT(a>0&&b>0);
	ASSERT(x>=0&&x<=1);
	if (x==0)
	{
		result=0;
		return result;
	}
	if (x==1)
	{
		result=1;
		return result;
	}
	flag=0;
	if (b*x<=1.0&&x<=0.95)
	{
		result=ibetaf_incomplete_beta_ps(a, b, x, maxgam);
		return result;
	}
	w=1.0-x;
	if (x>a/(a+b))
	{
		flag=1;
		t=a;
		a=b;
		b=t;
		xc=x;
		x=w;
	}
	else
	{
		xc=w;
	}
	if ((flag==1&&b*x<=1.0)&&x<=0.95)
	{
		t=ibetaf_incomplete_beta_ps(a, b, x, maxgam);
		if (t<=CMath::MACHINE_EPSILON)
		{
			result=1.0-CMath::MACHINE_EPSILON;
		}
		else
		{
			result=1.0-t;
		}
		return result;
	}
	y=x*(a+b-2.0)-(a-1.0);
	if (y<0.0)
	{
		w=ibetaf_incomplete_beta_fe(a, b, x, big, biginv);
	}
	else
	{
		w=ibetaf_incomplete_beta_fe2(a, b, x, big, biginv)/xc;
	}
	y=a*CMath::log(x);
	t=b*CMath::log(xc);
	if ((a+b<maxgam&&CMath::abs(y)<maxlog)&&CMath::abs(t)<maxlog)
	{
		t=CMath::pow(xc, b);
		t=t*CMath::pow(x, a);
		t=t/a;
		t=t*w;
		t=t*(CMath::tgamma(a+b)/(CMath::tgamma(a)*CMath::tgamma(b)));
		if (flag==1)
		{
			if (t<=CMath::MACHINE_EPSILON)
			{
				result=1.0-CMath::MACHINE_EPSILON;
			}
			else
			{
				result=1.0-t;
			}
		}
		else
		{
			result=t;
		}
		return result;
	}
	y=y+t+CMath::lgamma(a+b)-CMath::lgamma(a)-CMath::lgamma(b);
	y=y+CMath::log(w/a);
	if (y<minlog)
	{
		t=0.0;
	}
	else
	{
		t=CMath::exp(y);
	}
	if (flag==1)
	{
		if (t<=CMath::MACHINE_EPSILON)
		{
			t=1.0-CMath::MACHINE_EPSILON;
		}
		else
		{
			t=1.0-t;
		}
	}
	result=t;
	return result;
}

float64_t CStatistics::ibetaf_incomplete_beta_ps(float64_t a, float64_t b,
		float64_t x, float64_t maxgam)
{
	float64_t s;
	float64_t t;
	float64_t u;
	float64_t v;
	float64_t n;
	float64_t t1;
	float64_t z;
	float64_t ai;
	float64_t result;

	ai=1.0/a;
	u=(1.0-b)*x;
	v=u/(a+1.0);
	t1=v;
	t=u;
	n=2.0;
	s=0.0;
	z=CMath::MACHINE_EPSILON*ai;
	while (CMath::abs(v)>z)
	{
		u=(n-b)*x/n;
		t=t*u;
		v=t/(a+n);
		s=s+v;
		n=n+1.0;
	}
	s=s+t1;
	s=s+ai;
	u=a*CMath::log(x);
	if (a+b<maxgam&&CMath::abs(u)<CMath::log(CMath::MAX_REAL_NUMBER))
	{
		t=CMath::tgamma(a+b)/(CMath::tgamma(a)*CMath::tgamma(b));
		s=s*t*CMath::pow(x, a);
	}
	else
	{
		t=CMath::lgamma(a+b)-CMath::lgamma(a)-CMath::lgamma(b)+u+CMath::log(s);
		if (t<CMath::log(CMath::MIN_REAL_NUMBER))
		{
			s=0.0;
		}
		else
		{
			s=CMath::exp(t);
		}
	}
	result=s;
	return result;
}

float64_t CStatistics::ibetaf_incomplete_beta_fe2(float64_t a, float64_t b,
		float64_t x, float64_t big, float64_t biginv)
{
	float64_t xk;
	float64_t pk;
	float64_t pkm1;
	float64_t pkm2;
	float64_t qk;
	float64_t qkm1;
	float64_t qkm2;
	float64_t k1;
	float64_t k2;
	float64_t k3;
	float64_t k4;
	float64_t k5;
	float64_t k6;
	float64_t k7;
	float64_t k8;
	float64_t r;
	float64_t t;
	float64_t ans;
	float64_t z;
	float64_t thresh;
	int32_t n;
	float64_t result;

	k1=a;
	k2=b-1.0;
	k3=a;
	k4=a+1.0;
	k5=1.0;
	k6=a+b;
	k7=a+1.0;
	k8=a+2.0;
	pkm2=0.0;
	qkm2=1.0;
	pkm1=1.0;
	qkm1=1.0;
	z=x/(1.0-x);
	ans=1.0;
	r=1.0;
	n=0;
	thresh=3.0*CMath::MACHINE_EPSILON;
	do
	{
		xk=-z*k1*k2/(k3*k4);
		pk=pkm1+pkm2*xk;
		qk=qkm1+qkm2*xk;
		pkm2=pkm1;
		pkm1=pk;
		qkm2=qkm1;
		qkm1=qk;
		xk=z*k5*k6/(k7*k8);
		pk=pkm1+pkm2*xk;
		qk=qkm1+qkm2*xk;
		pkm2=pkm1;
		pkm1=pk;
		qkm2=qkm1;
		qkm1=qk;
		if (qk!=0)
		{
			r=pk/qk;
		}
		if (r!=0)
		{
			t=CMath::abs((ans-r)/r);
			ans=r;
		}
		else
		{
			t=1.0;
		}
		if (t<thresh)
		{
			break;
		}
		k1=k1+1.0;
		k2=k2-1.0;
		k3=k3+2.0;
		k4=k4+2.0;
		k5=k5+1.0;
		k6=k6+1.0;
		k7=k7+2.0;
		k8=k8+2.0;
		if (CMath::abs(qk)+CMath::abs(pk)>big)
		{
			pkm2=pkm2*biginv;
			pkm1=pkm1*biginv;
			qkm2=qkm2*biginv;
			qkm1=qkm1*biginv;
		}
		if (CMath::abs(qk)<biginv||CMath::abs(pk)<biginv)
		{
			pkm2=pkm2*big;
			pkm1=pkm1*big;
			qkm2=qkm2*big;
			qkm1=qkm1*big;
		}
		n=n+1;
	} while (n!=300);
	result=ans;
	return result;
}

float64_t CStatistics::ibetaf_incomplete_beta_fe(float64_t a, float64_t b,
		float64_t x, float64_t big, float64_t biginv)
{
	float64_t xk;
	float64_t pk;
	float64_t pkm1;
	float64_t pkm2;
	float64_t qk;
	float64_t qkm1;
	float64_t qkm2;
	float64_t k1;
	float64_t k2;
	float64_t k3;
	float64_t k4;
	float64_t k5;
	float64_t k6;
	float64_t k7;
	float64_t k8;
	float64_t r;
	float64_t t;
	float64_t ans;
	float64_t thresh;
	int32_t n;
	float64_t result;

	k1=a;
	k2=a+b;
	k3=a;
	k4=a+1.0;
	k5=1.0;
	k6=b-1.0;
	k7=k4;
	k8=a+2.0;
	pkm2=0.0;
	qkm2=1.0;
	pkm1=1.0;
	qkm1=1.0;
	ans=1.0;
	r=1.0;
	n=0;
	thresh=3.0*CMath::MACHINE_EPSILON;
	do
	{
		xk=-x*k1*k2/(k3*k4);
		pk=pkm1+pkm2*xk;
		qk=qkm1+qkm2*xk;
		pkm2=pkm1;
		pkm1=pk;
		qkm2=qkm1;
		qkm1=qk;
		xk=x*k5*k6/(k7*k8);
		pk=pkm1+pkm2*xk;
		qk=qkm1+qkm2*xk;
		pkm2=pkm1;
		pkm1=pk;
		qkm2=qkm1;
		qkm1=qk;
		if (qk!=0)
		{
			r=pk/qk;
		}
		if (r!=0)
		{
			t=CMath::abs((ans-r)/r);
			ans=r;
		}
		else
		{
			t=1.0;
		}
		if (t<thresh)
		{
			break;
		}
		k1=k1+1.0;
		k2=k2+1.0;
		k3=k3+2.0;
		k4=k4+2.0;
		k5=k5+1.0;
		k6=k6-1.0;
		k7=k7+2.0;
		k8=k8+2.0;
		if (CMath::abs(qk)+CMath::abs(pk)>big)
		{
			pkm2=pkm2*biginv;
			pkm1=pkm1*biginv;
			qkm2=qkm2*biginv;
			qkm1=qkm1*biginv;
		}
		if (CMath::abs(qk)<biginv||CMath::abs(pk)<biginv)
		{
			pkm2=pkm2*big;
			pkm1=pkm1*big;
			qkm2=qkm2*big;
			qkm1=qkm1*big;
		}
		n=n+1;
	} while (n!=300);
	result=ans;
	return result;
}

float64_t CStatistics::inverse_student_t_distribution(int32_t k, float64_t p)
{
	float64_t t;
	float64_t rk;
	float64_t z;
	int32_t rflg;
	float64_t result;

	ASSERT(k>0&&p>0&&p<1);
	rk=k;
	if (p>0.25&&p<0.75)
	{
		if (p==0.5)
		{
			result=0;
			return result;
		}
		z=1.0-2.0*p;
		z=inverse_incomplete_beta(0.5, 0.5*rk, CMath::abs(z));
		t=CMath::sqrt(rk*z/(1.0-z));
		if (p<0.5)
		{
			t=-t;
		}
		result=t;
		return result;
	}
	rflg=-1;
	if (p>=0.5)
	{
		p=1.0-p;
		rflg=1;
	}
	z=inverse_incomplete_beta(0.5*rk, 0.5, 2.0*p);
	if (CMath::MAX_REAL_NUMBER*z<rk)
	{
		result=rflg*CMath::MAX_REAL_NUMBER;
		return result;
	}
	t=CMath::sqrt(rk/z-rk);
	result=rflg*t;
	return result;
}

float64_t CStatistics::inverse_incomplete_beta(float64_t a, float64_t b,
		float64_t y)
{
	float64_t aaa;
	float64_t bbb;
	float64_t y0;
	float64_t d;
	float64_t yyy;
	float64_t x;
	float64_t x0;
	float64_t x1;
	float64_t lgm;
	float64_t yp;
	float64_t di;
	float64_t dithresh;
	float64_t yl;
	float64_t yh;
	float64_t xt;
	int32_t i;
	int32_t rflg;
	int32_t dir;
	int32_t nflg;
	int32_t mainlooppos;
	int32_t ihalve;
	int32_t ihalvecycle;
	int32_t newt;
	int32_t newtcycle;
	int32_t breaknewtcycle;
	int32_t breakihalvecycle;
	float64_t result;

	i=0;
	ASSERT(y>=0&&y<=1);

	/*
	 * special cases
	 */
	if (y==0)
	{
		result=0;
		return result;
	}
	if (y==1.0)
	{
		result=1;
		return result;
	}

	/*
	 * these initializations are not really necessary,
	 * but without them compiler complains about 'possibly uninitialized variables'.
	 */
	dithresh=0;
	rflg=0;
	aaa=0;
	bbb=0;
	y0=0;
	x=0;
	yyy=0;
	lgm=0;
	dir=0;
	di=0;

	/*
	 * normal initializations
	 */
	x0=0.0;
	yl=0.0;
	x1=1.0;
	yh=1.0;
	nflg=0;
	mainlooppos=0;
	ihalve=1;
	ihalvecycle=2;
	newt=3;
	newtcycle=4;
	breaknewtcycle=5;
	breakihalvecycle=6;

	/*
	 * main loop
	 */
	for (;;)
	{

		/*
		 * start
		 */
		if (mainlooppos==0)
		{
			if (a<=1.0||b<=1.0)
			{
				dithresh=1.0e-6;
				rflg=0;
				aaa=a;
				bbb=b;
				y0=y;
				x=aaa/(aaa+bbb);
				yyy=incomplete_beta(aaa, bbb, x);
				mainlooppos=ihalve;
				continue;
			}
			else
			{
				dithresh=1.0e-4;
			}
			yp=-inverse_normal_distribution(y);
			if (y>0.5)
			{
				rflg=1;
				aaa=b;
				bbb=a;
				y0=1.0-y;
				yp=-yp;
			}
			else
			{
				rflg=0;
				aaa=a;
				bbb=b;
				y0=y;
			}
			lgm=(yp*yp-3.0)/6.0;
			x=2.0/(1.0/(2.0*aaa-1.0)+1.0/(2.0*bbb-1.0));
			d=yp*CMath::sqrt(x+lgm)/x
					-(1.0/(2.0*bbb-1.0)-1.0/(2.0*aaa-1.0))
							*(lgm+5.0/6.0-2.0/(3.0*x));
			d=2.0*d;
			if (d<CMath::log(CMath::MIN_REAL_NUMBER))
			{
				x=0;
				break;
			}
			x=aaa/(aaa+bbb*CMath::exp(d));
			yyy=incomplete_beta(aaa, bbb, x);
			yp=(yyy-y0)/y0;
			if (CMath::abs(yp)<0.2)
			{
				mainlooppos=newt;
				continue;
			}
			mainlooppos=ihalve;
			continue;
		}

		/*
		 * ihalve
		 */
		if (mainlooppos==ihalve)
		{
			dir=0;
			di=0.5;
			i=0;
			mainlooppos=ihalvecycle;
			continue;
		}

		/*
		 * ihalvecycle
		 */
		if (mainlooppos==ihalvecycle)
		{
			if (i<=99)
			{
				if (i!=0)
				{
					x=x0+di*(x1-x0);
					if (x==1.0)
					{
						x=1.0-CMath::MACHINE_EPSILON;
					}
					if (x==0.0)
					{
						di=0.5;
						x=x0+di*(x1-x0);
						if (x==0.0)
						{
							break;
						}
					}
					yyy=incomplete_beta(aaa, bbb, x);
					yp=(x1-x0)/(x1+x0);
					if (CMath::abs(yp)<dithresh)
					{
						mainlooppos=newt;
						continue;
					}
					yp=(yyy-y0)/y0;
					if (CMath::abs(yp)<dithresh)
					{
						mainlooppos=newt;
						continue;
					}
				}
				if (yyy<y0)
				{
					x0=x;
					yl=yyy;
					if (dir<0)
					{
						dir=0;
						di=0.5;
					}
					else
					{
						if (dir>3)
						{
							di=1.0-(1.0-di)*(1.0-di);
						}
						else
						{
							if (dir>1)
							{
								di=0.5*di+0.5;
							}
							else
							{
								di=(y0-yyy)/(yh-yl);
							}
						}
					}
					dir=dir+1;
					if (x0>0.75)
					{
						if (rflg==1)
						{
							rflg=0;
							aaa=a;
							bbb=b;
							y0=y;
						}
						else
						{
							rflg=1;
							aaa=b;
							bbb=a;
							y0=1.0-y;
						}
						x=1.0-x;
						yyy=incomplete_beta(aaa, bbb, x);
						x0=0.0;
						yl=0.0;
						x1=1.0;
						yh=1.0;
						mainlooppos=ihalve;
						continue;
					}
				}
				else
				{
					x1=x;
					if (rflg==1&&x1<CMath::MACHINE_EPSILON)
					{
						x=0.0;
						break;
					}
					yh=yyy;
					if (dir>0)
					{
						dir=0;
						di=0.5;
					}
					else
					{
						if (dir<-3)
						{
							di=di*di;
						}
						else
						{
							if (dir<-1)
							{
								di=0.5*di;
							}
							else
							{
								di=(yyy-y0)/(yh-yl);
							}
						}
					}
					dir=dir-1;
				}
				i=i+1;
				mainlooppos=ihalvecycle;
				continue;
			}
			else
			{
				mainlooppos=breakihalvecycle;
				continue;
			}
		}

		/*
		 * breakihalvecycle
		 */
		if (mainlooppos==breakihalvecycle)
		{
			if (x0>=1.0)
			{
				x=1.0-CMath::MACHINE_EPSILON;
				break;
			}
			if (x<=0.0)
			{
				x=0.0;
				break;
			}
			mainlooppos=newt;
			continue;
		}

		/*
		 * newt
		 */
		if (mainlooppos==newt)
		{
			if (nflg!=0)
			{
				break;
			}
			nflg=1;
			lgm=CMath::lgamma(aaa+bbb)-CMath::lgamma(aaa)-CMath::lgamma(bbb);
			i=0;
			mainlooppos=newtcycle;
			continue;
		}

		/*
		 * newtcycle
		 */
		if (mainlooppos==newtcycle)
		{
			if (i<=7)
			{
				if (i!=0)
				{
					yyy=incomplete_beta(aaa, bbb, x);
				}
				if (yyy<yl)
				{
					x=x0;
					yyy=yl;
				}
				else
				{
					if (yyy>yh)
					{
						x=x1;
						yyy=yh;
					}
					else
					{
						if (yyy<y0)
						{
							x0=x;
							yl=yyy;
						}
						else
						{
							x1=x;
							yh=yyy;
						}
					}
				}
				if (x==1.0||x==0.0)
				{
					mainlooppos=breaknewtcycle;
					continue;
				}
				d=(aaa-1.0)*CMath::log(x)+(bbb-1.0)*CMath::log(1.0-x)+lgm;
				if (d<CMath::log(CMath::MIN_REAL_NUMBER))
				{
					break;
				}
				if (d>CMath::log(CMath::MAX_REAL_NUMBER))
				{
					mainlooppos=breaknewtcycle;
					continue;
				}
				d=CMath::exp(d);
				d=(yyy-y0)/d;
				xt=x-d;
				if (xt<=x0)
				{
					yyy=(x-x0)/(x1-x0);
					xt=x0+0.5*yyy*(x-x0);
					if (xt<=0.0)
					{
						mainlooppos=breaknewtcycle;
						continue;
					}
				}
				if (xt>=x1)
				{
					yyy=(x1-x)/(x1-x0);
					xt=x1-0.5*yyy*(x1-x);
					if (xt>=1.0)
					{
						mainlooppos=breaknewtcycle;
						continue;
					}
				}
				x=xt;
				if (CMath::abs(d/x)<128.0*CMath::MACHINE_EPSILON)
				{
					break;
				}
				i=i+1;
				mainlooppos=newtcycle;
				continue;
			}
			else
			{
				mainlooppos=breaknewtcycle;
				continue;
			}
		}

		/*
		 * breaknewtcycle
		 */
		if (mainlooppos==breaknewtcycle)
		{
			dithresh=256.0*CMath::MACHINE_EPSILON;
			mainlooppos=ihalve;
			continue;
		}
	}

	/*
	 * done
	 */
	if (rflg!=0)
	{
		if (x<=CMath::MACHINE_EPSILON)
		{
			x=1.0-CMath::MACHINE_EPSILON;
		}
		else
		{
			x=1.0-x;
		}
	}
	result=x;
	return result;
}

float64_t CStatistics::inverse_normal_distribution(float64_t y0)
{
	float64_t expm2;
	float64_t s2pi;
	float64_t x;
	float64_t y;
	float64_t z;
	float64_t y2;
	float64_t x0;
	float64_t x1;
	int32_t code;
	float64_t p0;
	float64_t q0;
	float64_t p1;
	float64_t q1;
	float64_t p2;
	float64_t q2;
	float64_t result;

	expm2=0.13533528323661269189;
	s2pi=2.50662827463100050242;
	if (y0<=0)
	{
		result=-CMath::MAX_REAL_NUMBER;
		return result;
	}
	if (y0>=1)
	{
		result=CMath::MAX_REAL_NUMBER;
		return result;
	}
	code=1;
	y=y0;
	if (y>1.0-expm2)
	{
		y=1.0-y;
		code=0;
	}
	if (y>expm2)
	{
		y=y-0.5;
		y2=y*y;
		p0=-59.9633501014107895267;
		p0=98.0010754185999661536+y2*p0;
		p0=-56.6762857469070293439+y2*p0;
		p0=13.9312609387279679503+y2*p0;
		p0=-1.23916583867381258016+y2*p0;
		q0=1;
		q0=1.95448858338141759834+y2*q0;
		q0=4.67627912898881538453+y2*q0;
		q0=86.3602421390890590575+y2*q0;
		q0=-225.462687854119370527+y2*q0;
		q0=200.260212380060660359+y2*q0;
		q0=-82.0372256168333339912+y2*q0;
		q0=15.9056225126211695515+y2*q0;
		q0=-1.18331621121330003142+y2*q0;
		x=y+y*y2*p0/q0;
		x=x*s2pi;
		result=x;
		return result;
	}
	x=CMath::sqrt(-2.0*CMath::log(y));
	x0=x-CMath::log(x)/x;
	z=1.0/x;
	if (x<8.0)
	{
		p1=4.05544892305962419923;
		p1=31.5251094599893866154+z*p1;
		p1=57.1628192246421288162+z*p1;
		p1=44.0805073893200834700+z*p1;
		p1=14.6849561928858024014+z*p1;
		p1=2.18663306850790267539+z*p1;
		p1=-1.40256079171354495875*0.1+z*p1;
		p1=-3.50424626827848203418*0.01+z*p1;
		p1=-8.57456785154685413611*0.0001+z*p1;
		q1=1;
		q1=15.7799883256466749731+z*q1;
		q1=45.3907635128879210584+z*q1;
		q1=41.3172038254672030440+z*q1;
		q1=15.0425385692907503408+z*q1;
		q1=2.50464946208309415979+z*q1;
		q1=-1.42182922854787788574*0.1+z*q1;
		q1=-3.80806407691578277194*0.01+z*q1;
		q1=-9.33259480895457427372*0.0001+z*q1;
		x1=z*p1/q1;
	}
	else
	{
		p2=3.23774891776946035970;
		p2=6.91522889068984211695+z*p2;
		p2=3.93881025292474443415+z*p2;
		p2=1.33303460815807542389+z*p2;
		p2=2.01485389549179081538*0.1+z*p2;
		p2=1.23716634817820021358*0.01+z*p2;
		p2=3.01581553508235416007*0.0001+z*p2;
		p2=2.65806974686737550832*0.000001+z*p2;
		p2=6.23974539184983293730*0.000000001+z*p2;
		q2=1;
		q2=6.02427039364742014255+z*q2;
		q2=3.67983563856160859403+z*q2;
		q2=1.37702099489081330271+z*q2;
		q2=2.16236993594496635890*0.1+z*q2;
		q2=1.34204006088543189037*0.01+z*q2;
		q2=3.28014464682127739104*0.0001+z*q2;
		q2=2.89247864745380683936*0.000001+z*q2;
		q2=6.79019408009981274425*0.000000001+z*q2;
		x1=z*p2/q2;
	}
	x=x0-x1;
	if (code!=0)
	{
		x=-x;
	}
	result=x;
	return result;
}
