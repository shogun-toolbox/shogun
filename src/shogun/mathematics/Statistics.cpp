/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2014 Wu Lin
 * Written (W) 2011-2013 Heiko Strathmann
 * Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society
 *
 * ALGLIB Copyright 1984, 1987, 1995, 2000 by Stephen L. Moshier under GPL2+
 * http://www.alglib.net/
 * See header file for which functions are taken from ALGLIB (with adjustments
 * for shogun)
 *
 * lnormal_cdf and normal_cdf are adapted from
 * Gaussian Process Machine Learning Toolbox file logphi.m
 * http://www.gaussianprocess.org/gpml/code/matlab/doc/
 * and PraatLib 0.3 (GPL v2) file gsl_specfunc__erfc.c
 */

#include <shogun/mathematics/Statistics.h>
#include <shogun/mathematics/Math.h>
#include <shogun/lib/SGMatrix.h>
#include <shogun/lib/SGVector.h>
#include <shogun/lib/SGSparseMatrix.h>
#include <shogun/lib/SGSparseVector.h>

#ifdef HAVE_LAPACK
#include <shogun/mathematics/lapack.h>
#endif //HAVE_LAPACK

#ifdef HAVE_EIGEN3
#include <shogun/mathematics/eigen3.h>
using namespace Eigen;
#endif //HAVE_EIGEN3

using namespace shogun;

float64_t CStatistics::median(SGVector<float64_t> values, bool modify,
			bool in_place)
{
	float64_t result;
	if (modify)
	{
		/* use QuickSelect method
		 * This Quickselect routine is based on the algorithm described in
		 * "Numerical recipes in C", Second Edition,
		 * Cambridge University Press, 1992, Section 8.5, ISBN 0-521-43108-5
		 * This code by Nicolas Devillard - 1998. Public domain.
		 * Adapted to SHOGUN by Heiko Strathmann
		 */
		int32_t low;
		int32_t high;
		int32_t median;
		int32_t middle;
		int32_t l;
		int32_t h;

		low=0;
		high=values.vlen-1;
		median=(low+high)/2;

		while (true)
		{
			if (high<=low)
			{
				result=values[median];
				break;
			}

			if (high==low+1)
			{
				if (values[low]>values[high])
					CMath::CMath::swap(values[low], values[high]);
				result=values[median];
				break;
			}

			middle=(low+high)/2;
			if (values[middle]>values[high])
				CMath::swap(values[middle], values[high]);
			if (values[low]>values[high])
				CMath::swap(values[low], values[high]);
			if (values[middle]>values[low])
				CMath::swap(values[middle], values[low]);

			CMath::swap(values[middle], values[low+1]);

			l=low+1;
			h=high;
			for (;;)
			{
				do
					l++;
				while (values[low]>values[l]);
				do
					h--;
				while (values[h]>values[low]);
				if (h<l)
					break;
				CMath::swap(values[l], values[h]);
			}

			CMath::swap(values[low], values[h]);
			if (h<=median)
				low=l;
			if (h>=median)
				high=h-1;
		}

	}
	else
	{
		if (in_place)
		{
			/* use Torben method
			 * The following code is public domain.
			 * Algorithm by Torben Mogensen, implementation by N. Devillard.
			 * This code in public domain.
			 * Adapted to SHOGUN by Heiko Strathmann
			 */
			int32_t i;
			int32_t less;
			int32_t greater;
			int32_t equal;
			float64_t min;
			float64_t max;
			float64_t guess;
			float64_t maxltguess;
			float64_t mingtguess;
			min=max=values[0];
			for (i=1; i<values.vlen; i++)
			{
				if (values[i]<min)
					min=values[i];
				if (values[i]>max)
					max=values[i];
			}
			while (1)
			{
				guess=(min+max)/2;
				less=0;
				greater=0;
				equal=0;
				maxltguess=min;
				mingtguess=max;
				for (i=0; i<values.vlen; i++)
				{
					if (values[i]<guess)
					{
						less++;
						if (values[i]>maxltguess)
							maxltguess=values[i];
					}
					else if (values[i]>guess)
					{
						greater++;
						if (values[i]<mingtguess)
							mingtguess=values[i];
					}
					else
						equal++;
				}
				if (less<=(values.vlen+1)/2&&greater<=(values.vlen+1)/2)
					break;
				else if (less>greater)
					max=maxltguess;
				else
					min=mingtguess;
			}

			if (less>=(values.vlen+1)/2)
				result=maxltguess;
			else if (less+equal>=(values.vlen+1)/2)
				result=guess;
			else
				result=mingtguess;
		}
		else
		{
			/* copy vector and do recursive call which modifies copy */
			SGVector<float64_t> copy(values.vlen);
			memcpy(copy.vector, values.vector, sizeof(float64_t)*values.vlen);
			result=median(copy, true);
		}
	}

	return result;
}

float64_t CStatistics::matrix_median(SGMatrix<float64_t> values,
		bool modify, bool in_place)
{
	/* create a vector that uses the matrix data, dont do reference counting */
	SGVector<float64_t> as_vector(values.matrix,
			values.num_rows*values.num_cols, false);

	/* return vector median method */
	return median(as_vector, modify, in_place);
}


float64_t CStatistics::variance(SGVector<float64_t> values)
{
	ASSERT(values.vlen>1)
	ASSERT(values.vector)

	float64_t mean=CStatistics::mean(values);

	float64_t sum_squared_diff=0;
	for (index_t i=0; i<values.vlen; ++i)
		sum_squared_diff+=CMath::pow(values.vector[i]-mean, 2);

	return sum_squared_diff/(values.vlen-1);
}

SGVector<float64_t> CStatistics::matrix_mean(SGMatrix<float64_t> values,
		bool col_wise)
{
	ASSERT(values.num_rows>0)
	ASSERT(values.num_cols>0)
	ASSERT(values.matrix)

	SGVector<float64_t> result;

	if (col_wise)
	{
		result=SGVector<float64_t>(values.num_cols);
		for (index_t j=0; j<values.num_cols; ++j)
		{
			result[j]=0;
			for (index_t i=0; i<values.num_rows; ++i)
				result[j]+=values(i,j);

			result[j]/=values.num_rows;
		}
	}
	else
	{
		result=SGVector<float64_t>(values.num_rows);
		for (index_t j=0; j<values.num_rows; ++j)
		{
			result[j]=0;
			for (index_t i=0; i<values.num_cols; ++i)
				result[j]+=values(j,i);

			result[j]/=values.num_cols;
		}
	}

	return result;
}

SGVector<float64_t> CStatistics::matrix_variance(SGMatrix<float64_t> values,
		bool col_wise)
{
	ASSERT(values.num_rows>0)
	ASSERT(values.num_cols>0)
	ASSERT(values.matrix)

	/* first compute mean */
	SGVector<float64_t> mean=CStatistics::matrix_mean(values, col_wise);

	SGVector<float64_t> result;

	if (col_wise)
	{
		result=SGVector<float64_t>(values.num_cols);
		for (index_t j=0; j<values.num_cols; ++j)
		{
			result[j]=0;
			for (index_t i=0; i<values.num_rows; ++i)
				result[j]+=CMath::pow(values(i,j)-mean[j], 2);

			result[j]/=(values.num_rows-1);
		}
	}
	else
	{
		result=SGVector<float64_t>(values.num_rows);
		for (index_t j=0; j<values.num_rows; ++j)
		{
			result[j]=0;
			for (index_t i=0; i<values.num_cols; ++i)
				result[j]+=CMath::pow(values(j,i)-mean[j], 2);

			result[j]/=(values.num_cols-1);
		}
	}

	return result;
}

float64_t CStatistics::std_deviation(SGVector<float64_t> values)
{
	return CMath::sqrt(variance(values));
}

SGVector<float64_t> CStatistics::matrix_std_deviation(
		SGMatrix<float64_t> values, bool col_wise)
{
	SGVector<float64_t> var=CStatistics::matrix_variance(values, col_wise);
	for (index_t i=0; i<var.vlen; ++i)
		var[i]=CMath::sqrt(var[i]);

	return var;
}

#ifdef HAVE_LAPACK
SGMatrix<float64_t> CStatistics::covariance_matrix(
		SGMatrix<float64_t> observations, bool in_place)
{
	SGMatrix<float64_t> centered=
			in_place ?
					observations :
					SGMatrix<float64_t>(observations.num_rows,
							observations.num_cols);

	if (!in_place)
	{
		memcpy(centered.matrix, observations.matrix,
				sizeof(float64_t)*observations.num_rows*observations.num_cols);
	}
	centered.remove_column_mean();

	/* compute 1/(m-1) * X' * X */
	SGMatrix<float64_t> cov=SGMatrix<float64_t>::matrix_multiply(centered,
			centered, true, false, 1.0/(observations.num_rows-1));

	return cov;
}
#endif //HAVE_LAPACK

float64_t CStatistics::confidence_intervals_mean(SGVector<float64_t> values,
		float64_t alpha, float64_t& conf_int_low, float64_t& conf_int_up)
{
	ASSERT(values.vlen>1)
	ASSERT(values.vector)

	/* using one sided student t distribution evaluation */
	alpha=alpha/2;

	/* degrees of freedom */
	int32_t deg=values.vlen-1;

	/* compute absolute value of t-value */
	float64_t t=CMath::abs(inverse_student_t(deg, alpha));

	/* values for calculating confidence interval */
	float64_t std_dev=CStatistics::std_deviation(values);
	float64_t mean=CStatistics::mean(values);

	/* compute confidence interval */
	float64_t interval=t*std_dev/CMath::sqrt((float64_t)values.vlen);
	conf_int_low=mean-interval;
	conf_int_up=mean+interval;

	return mean;
}

float64_t CStatistics::inverse_student_t(int32_t k, float64_t p)
{
	float64_t t;
	float64_t rk;
	float64_t z;
	int32_t rflg;
	float64_t result;

	if (!(k>0 && greater(p, 0)) && less(p, 1))
	{
		SG_SERROR("CStatistics::inverse_student_t_distribution(): "
		"Domain error\n");
	}
	rk=k;
	if (greater(p, 0.25) && less(p, 0.75))
	{
		if (equal(p, 0.5))
		{
			result=0;
			return result;
		}
		z=1.0-2.0*p;
		z=boost::math::ibeta_inv(0.5, 0.5*rk, CMath::abs(z));
		t=CMath::sqrt(rk*z/(1.0-z));
		if (less(p, 0.5))
		{
			t=-t;
		}
		result=t;
		return result;
	}
	rflg=-1;
	if (greater_equal(p, 0.5))
	{
		p=1.0-p;
		rflg=1;
	}
	z=boost::math::ibeta_inv(0.5*rk, 0.5, 2.0*p);
	if (less(CMath::MAX_REAL_NUMBER*z, rk))
	{
		result=rflg*CMath::MAX_REAL_NUMBER;
		return result;
	}
	t=CMath::sqrt(rk/z-rk);
	result=rflg*t;
	return result;
}

float64_t CStatistics::inverse_normal_cdf(float64_t y0, float64_t mean,
		float64_t std_dev)
{
	return inverse_normal_cdf(y0)*std_dev+mean;
}

float64_t CStatistics::inverse_normal_cdf(float64_t y0)
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
	if (less_equal(y0, 0))
	{
		result=-CMath::MAX_REAL_NUMBER;
		return result;
	}
	if (greater_equal(y0, 1))
	{
		result=CMath::MAX_REAL_NUMBER;
		return result;
	}
	code=1;
	y=y0;
	if (greater(y, 1.0-expm2))
	{
		y=1.0-y;
		code=0;
	}
	if (greater(y, expm2))
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
	if (less(x, 8.0))
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

float64_t CStatistics::ibetaf_incompletebetaps(float64_t a, float64_t b,
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
	while (greater(CMath::abs(v), z))
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
	if (less(a+b, maxgam)
			 && less(CMath::abs(u), CMath::log(CMath::MAX_REAL_NUMBER)))
	{
		t=tgamma(a+b)/(tgamma(a)*tgamma(b));
		s=s*t*CMath::pow(x, a);
	}
	else
	{
		t=lgamma(a+b)-lgamma(a)-lgamma(b)+u+CMath::log(s);
		if (less(t, CMath::log(CMath::MIN_REAL_NUMBER)))
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

float64_t CStatistics::ibetaf_incompletebetafe(float64_t a, float64_t b,
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
		if (not_equal(qk, 0))
		{
			r=pk/qk;
		}
		if (not_equal(r, 0))
		{
			t=CMath::abs((ans-r)/r);
			ans=r;
		}
		else
		{
			t=1.0;
		}
		if (less(t, thresh))
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
		if (greater(CMath::abs(qk)+CMath::abs(pk), big))
		{
			pkm2=pkm2*biginv;
			pkm1=pkm1*biginv;
			qkm2=qkm2*biginv;
			qkm1=qkm1*biginv;
		}
		if (less(CMath::abs(qk), biginv) || less(CMath::abs(pk), biginv))
		{
			pkm2=pkm2*big;
			pkm1=pkm1*big;
			qkm2=qkm2*big;
			qkm1=qkm1*big;
		}
		n=n+1;
	}
	while (n!=300);
	result=ans;
	return result;
}

float64_t CStatistics::ibetaf_incompletebetafe2(float64_t a, float64_t b,
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
		if (not_equal(qk, 0))
		{
			r=pk/qk;
		}
		if (not_equal(r, 0))
		{
			t=CMath::abs((ans-r)/r);
			ans=r;
		}
		else
		{
			t=1.0;
		}
		if (less(t, thresh))
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
		if (greater(CMath::abs(qk)+CMath::abs(pk), big))
		{
			pkm2=pkm2*biginv;
			pkm1=pkm1*biginv;
			qkm2=qkm2*biginv;
			qkm1=qkm1*biginv;
		}
		if (less(CMath::abs(qk), biginv) || less(CMath::abs(pk), biginv))
		{
			pkm2=pkm2*big;
			pkm1=pkm1*big;
			qkm2=qkm2*big;
			qkm1=qkm1*big;
		}
		n=n+1;
	}
	while (n!=300);
	result=ans;
	return result;
}

float64_t CStatistics::gamma_cdf(float64_t x, float64_t a, float64_t b)
{
	/* definition of wikipedia: incomplete gamma devised by true gamma */
	return boost::math::gamma_p(a, x/b);
}

float64_t CStatistics::inverse_gamma_cdf(float64_t p, float64_t a,
		float64_t b)
{
	/* inverse of gamma(a,b) CDF is
	 * inverse_incomplete_gamma_completed(a, 1. - p) * b */
	return boost::math::gamma_q_inv(a, 1-p)*b;
}

float64_t CStatistics::normal_cdf(float64_t x, float64_t std_dev)
{
	return 0.5*(boost::math::erfc(-x/std_dev/std::sqrt(2)));
}


const float64_t CStatistics::ERFC_CASE1=0.0492;

const float64_t CStatistics::ERFC_CASE2=-11.3137;

float64_t CStatistics::lnormal_cdf(float64_t x)
{
	const float64_t sqrt_of_2=1.41421356237309514547;
	const float64_t log_of_2=0.69314718055994528623;
	const float64_t sqrt_of_pi=1.77245385090551588192;

	const index_t c_len=14;
	static float64_t c_array[c_len]=
	{
		0.00048204,
		-0.00142906,
		0.0013200243174,
		0.0009461589032,
		-0.0045563339802,
		0.00556964649138,
		0.00125993961762116,
		-0.01621575378835404,
		0.02629651521057465,
		-0.001829764677455021,
		2.0*(1.0-CMath::PI/3.0),
		(4.0-CMath::PI)/3.0,
		1.0,
		1.0
	};

	if (x*x<ERFC_CASE1)
	{
		//id1 = z.*z<0.0492;                               
		//lp0 = -z(id1)/sqrt(2*pi);
		//c = [ 0.00048204; -0.00142906; 0.0013200243174; 0.0009461589032;
		//-0.0045563339802; 0.00556964649138; 0.00125993961762116;
		//-0.01621575378835404; 0.02629651521057465; -0.001829764677455021;
		//2*(1-pi/3); (4-pi)/3; 1; 1];
		//f = 0; for i=1:14, f = lp0.*(c(i)+f); end, lp(id1) = -2*f-log(2);
		float64_t f = 0.0;
		float64_t lp0 = -x/(sqrt_of_2*sqrt_of_pi);
		for (index_t i=0; i<c_len; i++)
			f=lp0*(c_array[i]+f);
		return -2.0*f-log_of_2;
	}
	else if (x<ERFC_CASE2)
	{
		//id2 = z<-11.3137;                           
		//r = [ 1.2753666447299659525; 5.019049726784267463450;
		//6.1602098531096305441; 7.409740605964741794425;
		//2.9788656263939928886 ];
		//q = [ 2.260528520767326969592;  9.3960340162350541504;
		//12.048951927855129036034; 17.081440747466004316; 
		//9.608965327192787870698;  3.3690752069827527677 ];
		//num = 0.5641895835477550741; for i=1:5, num = -z(id2).*num/sqrt(2) + r(i); end
		//den = 1.0;                   for i=1:6, den = -z(id2).*den/sqrt(2) + q(i); end
		//e = num./den; lp(id2) = log(e/2) - z(id2).^2/2;

		return CMath::log(erfc8_weighted_sum(x))-log_of_2-x*x*0.5;
	}

	//id3 = ~id2 & ~id1; lp(id3) = log(erfc(-z(id3)/sqrt(2))/2);
	return CMath::log(normal_cdf(x));
}

float64_t CStatistics::chi2_cdf(float64_t x, float64_t k)
{
	/* F(x,k) = incomplete_gamma(k/2,x/2) divided by true gamma(k/2) */
	return boost::math::gamma_p(k/2.0,x/2.0);
}

float64_t CStatistics::fdistribution_cdf(float64_t x, float64_t d1, float64_t d2)
{
	/* F(x;d1,d2) = incomplete_beta(d1/2, d2/2, d1*x/(d1*x+d2)) divided by beta(d1/2,d2/2)*/
	return boost::math::ibeta(d1/2.0,d2/2.0,d1*x/(d1*x+d2));
}

float64_t CStatistics::erfc8_weighted_sum(float64_t x)
{
	/* This is based on index 5725 in Hart et al */

	const float64_t sqrt_of_2=1.41421356237309514547;

	static float64_t P[]=
	{
		0.5641895835477550741253201704,
		1.275366644729965952479585264,
		5.019049726784267463450058,
		6.1602098531096305440906,
		7.409740605964741794425,
		2.97886562639399288862
	};

	static float64_t Q[]=
	{
		1.0,
		2.260528520767326969591866945,
		9.396034016235054150430579648,
		12.0489519278551290360340491,
		17.08144074746600431571095,
		9.608965327192787870698,
		3.3690752069827527677
	};

	float64_t num=0.0, den=0.0;

	num = P[0];
	for (index_t i=1; i<6; i++)
	{
		num=-x*num/sqrt_of_2+P[i];
	}

	den = Q[0];
	for (index_t i=1; i<7; i++)
	{
		den=-x*den/sqrt_of_2+Q[i];
	}

	return num/den;
}

SGVector<float64_t> CStatistics::fishers_exact_test_for_multiple_2x3_tables(
		SGMatrix<float64_t> tables)
{
	SGMatrix<float64_t> table(NULL, 2, 3, false);
	int32_t len=tables.num_cols/3;

	SGVector<float64_t> v(len);
	for (int32_t i=0; i<len; i++)
	{
		table.matrix=&tables.matrix[2*3*i];
		v.vector[i]=fishers_exact_test_for_2x3_table(table);
	}
	return v;
}

float64_t CStatistics::fishers_exact_test_for_2x3_table(
		SGMatrix<float64_t> table)
{
	ASSERT(table.num_rows==2)
	ASSERT(table.num_cols==3)

	int32_t m_len=3+2;
	float64_t* m=SG_MALLOC(float64_t, 3+2);
	m[0]=table.matrix[0]+table.matrix[2]+table.matrix[4];
	m[1]=table.matrix[1]+table.matrix[3]+table.matrix[5];
	m[2]=table.matrix[0]+table.matrix[1];
	m[3]=table.matrix[2]+table.matrix[3];
	m[4]=table.matrix[4]+table.matrix[5];

	float64_t n=SGVector<float64_t>::sum(m, m_len)/2.0;
	int32_t x_len=2*3*CMath::sq(CMath::max(m, m_len));
	float64_t* x=SG_MALLOC(float64_t, x_len);
	SGVector<float64_t>::fill_vector(x, x_len, 0.0);

	float64_t log_nom=0.0;
	for (int32_t i=0; i<3+2; i++)
		log_nom+=lgamma(m[i]+1);
	log_nom-=lgamma(n+1.0);

	float64_t log_denomf=0;
	floatmax_t log_denom=0;

	for (int32_t i=0; i<3*2; i++)
	{
		log_denom+=lgammal((floatmax_t)table.matrix[i]+1);
		log_denomf+=lgammal((floatmax_t)table.matrix[i]+1);
	}

	floatmax_t prob_table_log=log_nom-log_denom;

	int32_t dim1=CMath::min(m[0], m[2]);

	//traverse all possible tables with given m
	int32_t counter=0;
	for (int32_t k=0; k<=dim1; k++)
	{
		for (int32_t l=CMath::max(0.0, m[0]-m[4]-k);
				l<=CMath::min(m[0]-k, m[3]); l++)
		{
			x[0+0*2+counter*2*3]=k;
			x[0+1*2+counter*2*3]=l;
			x[0+2*2+counter*2*3]=m[0]-x[0+0*2+counter*2*3]-x[0+1*2+counter*2*3];
			x[1+0*2+counter*2*3]=m[2]-x[0+0*2+counter*2*3];
			x[1+1*2+counter*2*3]=m[3]-x[0+1*2+counter*2*3];
			x[1+2*2+counter*2*3]=m[4]-x[0+2*2+counter*2*3];

			counter++;
		}
	}

//#define DEBUG_FISHER_TABLE
#ifdef DEBUG_FISHER_TABLE
	SG_SPRINT("counter=%d\n", counter)
	SG_SPRINT("dim1=%d\n", dim1)
	SG_SPRINT("l=%g...%g\n", CMath::max(0.0,m[0]-m[4]-0), CMath::min(m[0]-0, m[3]))
	SG_SPRINT("n=%g\n", n)
	SG_SPRINT("prob_table_log=%.18Lg\n", prob_table_log)
	SG_SPRINT("log_denomf=%.18g\n", log_denomf)
	SG_SPRINT("log_denom=%.18Lg\n", log_denom)
	SG_SPRINT("log_nom=%.18g\n", log_nom)
	display_vector(m, m_len, "marginals");
	display_vector(x, 2*3*counter, "x");
#endif // DEBUG_FISHER_TABLE

	floatmax_t* log_denom_vec=SG_MALLOC(floatmax_t, counter);
	SGVector<floatmax_t>::fill_vector(log_denom_vec, counter, (floatmax_t)0.0);

	for (int32_t k=0; k<counter; k++)
	{
		for (int32_t j=0; j<3; j++)
		{
			for (int32_t i=0; i<2; i++)
				log_denom_vec[k]+=lgammal(x[i+j*2+k*2*3]+1.0);
		}
	}

	for (int32_t i=0; i<counter; i++)
		log_denom_vec[i]=log_nom-log_denom_vec[i];

#ifdef DEBUG_FISHER_TABLE
	display_vector(log_denom_vec, counter, "log_denom_vec");
#endif // DEBUG_FISHER_TABLE

	float64_t nonrand_p=-CMath::INFTY;
	for (int32_t i=0; i<counter; i++)
	{
		if (log_denom_vec[i]<=prob_table_log)
			nonrand_p=CMath::logarithmic_sum(nonrand_p, log_denom_vec[i]);
	}

#ifdef DEBUG_FISHER_TABLE
	SG_SPRINT("nonrand_p=%.18g\n", nonrand_p)
	SG_SPRINT("exp_nonrand_p=%.18g\n", CMath::exp(nonrand_p))
#endif // DEBUG_FISHER_TABLE
	nonrand_p=CMath::exp(nonrand_p);

	SG_FREE(log_denom_vec);
	SG_FREE(x);
	SG_FREE(m);

	return nonrand_p;
}

float64_t CStatistics::mutual_info(float64_t* p1, float64_t* p2, int32_t len)
{
	double e=0;

	for (int32_t i=0; i<len; i++)
		for (int32_t j=0; j<len; j++)
			e+=exp(p2[j*len+i])*(p2[j*len+i]-p1[i]-p1[j]);

	return (float64_t)e;
}

float64_t CStatistics::relative_entropy(float64_t* p, float64_t* q, int32_t len)
{
	double e=0;

	for (int32_t i=0; i<len; i++)
		e+=exp(p[i])*(p[i]-q[i]);

	return (float64_t)e;
}

float64_t CStatistics::entropy(float64_t* p, int32_t len)
{
	double e=0;

	for (int32_t i=0; i<len; i++)
		e-=exp(p[i])*p[i];

	return (float64_t)e;
}

SGVector<int32_t> CStatistics::sample_indices(int32_t sample_size, int32_t N)
{
	REQUIRE(sample_size<N,
			"sample size should be less than number of indices\n");
	int32_t* idxs=SG_MALLOC(int32_t,N);
	int32_t i, rnd;
	int32_t* permuted_idxs=SG_MALLOC(int32_t,sample_size);

	// reservoir sampling
	for (i=0; i<N; i++)
		idxs[i]=i;
	for (i=0; i<sample_size; i++)
		permuted_idxs[i]=idxs[i];
	for (i=sample_size; i<N; i++)
	{
		rnd=CMath::random(1, i);
		if (rnd<sample_size)
			permuted_idxs[rnd]=idxs[i];
	}
	SG_FREE(idxs);

	SGVector<int32_t> result=SGVector<int32_t>(permuted_idxs, sample_size);
	CMath::qsort(result);
	return result;
}

float64_t CStatistics::dlgamma(float64_t x)
{
	float64_t result=0.0;

	if (x<0.0)
	{
		// use reflection formula
		x=1.0-x;
		result=CMath::PI/CMath::tan(CMath::PI*x);
	}

	// make x>7 for approximation
	// (use reccurent formula: psi(x+1) = psi(x) + 1/x)
	while (x<=7.0)
	{
		result-=1.0/x;
		x++;
	}

	// perform approximation
	x-=0.5;
	result+=log(x);

	float64_t coeff[10]={
		0.04166666666666666667,
		-0.00729166666666666667,
		0.00384424603174603175,
		-0.00413411458333333333,
		0.00756096117424242424,
		-0.02108249687595390720,
		0.08332316080729166666,
		-0.44324627670587277880,
		3.05393103044765369366,
		-26.45616165999210241989};

	float64_t power=1.0;
	float64_t ix2=1.0/CMath::sq(x);

	// perform approximation
	for (index_t i=0; i<10; i++)
	{
		power*=ix2;
		result+=coeff[i]*power;
	}

	return result;
}

#ifdef HAVE_EIGEN3
float64_t CStatistics::log_det_general(const SGMatrix<float64_t> A)
{
	Map<MatrixXd> eigen_A(A.matrix, A.num_rows, A.num_cols);
	REQUIRE(eigen_A.rows()==eigen_A.cols(),
		"Input matrix should be a sqaure matrix row(%d) col(%d)\n",
		eigen_A.rows(), eigen_A.cols());

	PartialPivLU<MatrixXd> lu(eigen_A);
	VectorXd tmp(eigen_A.rows());

	for (index_t idx=0; idx<tmp.rows(); idx++)
		tmp[idx]=idx+1;

	VectorXd p=lu.permutationP()*tmp;
	int detP=1;

	for (index_t idx=0; idx<p.rows(); idx++)
	{
		if (p[idx]!=idx+1)
		{
			detP*=-1;
			index_t j=idx+1;
			while(j<p.rows())
			{
				if (p[j]==idx+1)
					break;
				j++;
			}
			p[j]=p[idx];
		}
	}

	VectorXd u=lu.matrixLU().diagonal();
	int check_u=1;

	for (int idx=0; idx<u.rows(); idx++)
	{
		if (u[idx]<0)
			check_u*=-1;
		else if (u[idx]==0)
		{
			check_u=0;
			break;
		}
	}

	float64_t result=CMath::INFTY;

	if (check_u==detP)
		result=u.array().abs().log().sum();

	return result;
}

float64_t CStatistics::log_det(SGMatrix<float64_t> m)
{
	/* map the matrix to eigen3 to perform cholesky */
	Map<MatrixXd> M(m.matrix, m.num_rows, m.num_cols);

	/* computing the cholesky decomposition */
	LLT<MatrixXd> llt;
	llt.compute(M);

	/* the lower triangular matrix */
	MatrixXd l = llt.matrixL();

	/* calculate the log-determinant */
	VectorXd diag = l.diagonal();
	float64_t retval = 0.0;
	for( int32_t i = 0; i < diag.rows(); ++i ) {
		retval += log(diag(i));
	}
	retval *= 2;

	return retval;
}

float64_t CStatistics::log_det(const SGSparseMatrix<float64_t> m)
{
	typedef SparseMatrix<float64_t> MatrixType;
	const MatrixType &M=EigenSparseUtil<float64_t>::toEigenSparse(m);

	SimplicialLLT<MatrixType> llt;

	// factorize using cholesky with amd permutation
	llt.compute(M);
	MatrixType L=llt.matrixL();

	// calculate the log-determinant
	float64_t retval=0.0;
	for( index_t i=0; i<M.rows(); ++i )
		retval+=log(L.coeff(i,i));
	retval*=2;

	return retval;
}

SGMatrix<float64_t> CStatistics::sample_from_gaussian(SGVector<float64_t> mean,
	SGMatrix<float64_t> cov, int32_t N, bool precision_matrix)
{
	REQUIRE(cov.num_rows>0, "Number of covariance rows must be positive!\n");
	REQUIRE(cov.num_cols>0,"Number of covariance cols must be positive!\n");
	REQUIRE(cov.matrix, "Covariance is not initialized!\n");
	REQUIRE(cov.num_rows==cov.num_cols, "Covariance should be square matrix!\n");
	REQUIRE(mean.vlen==cov.num_rows, "Mean and covariance dimension mismatch!\n");

	int32_t dim=mean.vlen;
	Map<VectorXd> mu(mean.vector, mean.vlen);
	Map<MatrixXd> c(cov.matrix, cov.num_rows, cov.num_cols);

	// generate samples, z,  from N(0, I), DxN
	SGMatrix<float64_t> S(dim, N);
	for( int32_t j=0; j<N; ++j )
		for( int32_t i=0; i<dim; ++i )
			S(i,j)=CMath::randn_double();

	// the cholesky factorization c=L*U
	MatrixXd U=c.llt().matrixU();

	// generate samples, x, from N(mean, cov) or N(mean, cov^-1)
	// return samples of dimension NxD
	if( precision_matrix )
	{
		// here we have U*x=z, to solve this, we use cholesky again
		Map<MatrixXd> s(S.matrix, S.num_rows, S.num_cols);
		LDLT<MatrixXd> ldlt;
		ldlt.compute(U);
		s=ldlt.solve(s);
	}

	SGMatrix<float64_t>::transpose_matrix(S.matrix, S.num_rows, S.num_cols);

	if( !precision_matrix )
	{
		// here we need to find x=L*z, so, x'=z'*L' i.e. x'=z'*U
		Map<MatrixXd> s(S.matrix, S.num_rows, S.num_cols);
		s=s*U;
	}

	// add the mean
	Map<MatrixXd> x(S.matrix, S.num_rows, S.num_cols);
	for( int32_t i=0; i<N; ++i )
		x.row(i)+=mu;

	return S;
}

SGMatrix<float64_t> CStatistics::sample_from_gaussian(SGVector<float64_t> mean,
 SGSparseMatrix<float64_t> cov, int32_t N, bool precision_matrix)
{
	REQUIRE(cov.num_vectors>0,
		"CStatistics::sample_from_gaussian(): \
		Number of covariance rows must be positive!\n");
	REQUIRE(cov.num_features>0,
		"CStatistics::sample_from_gaussian(): \
		Number of covariance cols must be positive!\n");
	REQUIRE(cov.sparse_matrix,
		"CStatistics::sample_from_gaussian(): \
		Covariance is not initialized!\n");
	REQUIRE(cov.num_vectors==cov.num_features,
		"CStatistics::sample_from_gaussian(): \
		Covariance should be square matrix!\n");
	REQUIRE(mean.vlen==cov.num_vectors,
		"CStatistics::sample_from_gaussian(): \
		Mean and covariance dimension mismatch!\n");

	int32_t dim=mean.vlen;
	Map<VectorXd> mu(mean.vector, mean.vlen);

	typedef SparseMatrix<float64_t> MatrixType;
	const MatrixType &c=EigenSparseUtil<float64_t>::toEigenSparse(cov);

	SimplicialLLT<MatrixType> llt;

	// generate samples, z,  from N(0, I), DxN
	SGMatrix<float64_t> S(dim, N);
	for( int32_t j=0; j<N; ++j )
		for( int32_t i=0; i<dim; ++i )
			S(i,j)=CMath::randn_double();

	Map<MatrixXd> s(S.matrix, S.num_rows, S.num_cols);

	// the cholesky factorization P*c*P^-1 = LP*UP, with LP=P*L, UP=U*P^-1
	llt.compute(c);
	MatrixType LP=llt.matrixL();
	MatrixType UP=llt.matrixU();

	// generate samples, x, from N(mean, cov) or N(mean, cov^-1)
	// return samples of dimension NxD
	if( precision_matrix )
	{
		// here we have UP*xP=z, to solve this, we use cholesky again
		SimplicialLLT<MatrixType> lltUP;
		lltUP.compute(UP);
		s=lltUP.solve(s);
	}
	else
	{
		// here we need to find xP=LP*z
		s=LP*s;
	}

	// permute the samples back with x=P^-1*xP
	s=llt.permutationPinv()*s;

	SGMatrix<float64_t>::transpose_matrix(S.matrix, S.num_rows, S.num_cols);
	// add the mean
	Map<MatrixXd> x(S.matrix, S.num_rows, S.num_cols);
	for( int32_t i=0; i<N; ++i )
		x.row(i)+=mu;

	return S;
}

#endif //HAVE_EIGEN3

CStatistics::SigmoidParamters CStatistics::fit_sigmoid(SGVector<float64_t> scores)
{
	SG_SDEBUG("entering CStatistics::fit_sigmoid()\n")

	REQUIRE(scores.vector, "CStatistics::fit_sigmoid() requires "
			"scores vector!\n");

	/* count prior0 and prior1 if needed */
	int32_t prior0=0;
	int32_t prior1=0;
	SG_SDEBUG("counting number of positive and negative labels\n")
	{
		for (index_t i=0; i<scores.vlen; ++i)
		{
			if (scores[i]>0)
				prior1++;
			else
				prior0++;
		}
	}
	SG_SDEBUG("%d pos; %d neg\n", prior1, prior0)

	/* parameter setting */
	/* maximum number of iterations */
	index_t maxiter=100;

	/* minimum step taken in line search */
	float64_t minstep=1E-10;

	/* for numerically strict pd of hessian */
	float64_t sigma=1E-12;
	float64_t eps=1E-5;

	/* construct target support */
	float64_t hiTarget=(prior1+1.0)/(prior1+2.0);
	float64_t loTarget=1/(prior0+2.0);
	index_t length=prior1+prior0;

	SGVector<float64_t> t(length);
	for (index_t i=0; i<length; ++i)
	{
		if (scores[i]>0)
			t[i]=hiTarget;
		else
			t[i]=loTarget;
	}

	/* initial Point and Initial Fun Value */
	/* result parameters of sigmoid */
	float64_t a=0;
	float64_t b=CMath::log((prior0+1.0)/(prior1+1.0));
	float64_t fval=0.0;

	for (index_t i=0; i<length; ++i)
	{
		float64_t fApB=scores[i]*a+b;
		if (fApB>=0)
			fval+=t[i]*fApB+CMath::log(1+CMath::exp(-fApB));
		else
			fval+=(t[i]-1)*fApB+CMath::log(1+CMath::exp(fApB));
	}

	index_t it;
	float64_t g1;
	float64_t g2;
	for (it=0; it<maxiter; ++it)
	{
		SG_SDEBUG("Iteration %d, a=%f, b=%f, fval=%f\n", it, a, b, fval)

		/* Update Gradient and Hessian (use H' = H + sigma I) */
		float64_t h11=sigma; //Numerically ensures strict PD
		float64_t h22=h11;
		float64_t h21=0;
		g1=0;
		g2=0;

		for (index_t i=0; i<length; ++i)
		{
			float64_t fApB=scores[i]*a+b;
			float64_t p;
			float64_t q;
			if (fApB>=0)
			{
				p=CMath::exp(-fApB)/(1.0+CMath::exp(-fApB));
				q=1.0/(1.0+CMath::exp(-fApB));
			}
			else
			{
				p=1.0/(1.0+CMath::exp(fApB));
				q=CMath::exp(fApB)/(1.0+CMath::exp(fApB));
			}

			float64_t d2=p*q;
			h11+=scores[i]*scores[i]*d2;
			h22+=d2;
			h21+=scores[i]*d2;
			float64_t d1=t[i]-p;
			g1+=scores[i]*d1;
			g2+=d1;
		}

		/* Stopping Criteria */
		if (CMath::abs(g1)<eps && CMath::abs(g2)<eps)
			break;

		/* Finding Newton direction: -inv(H') * g */
		float64_t det=h11*h22-h21*h21;
		float64_t dA=-(h22*g1-h21*g2)/det;
		float64_t dB=-(-h21*g1+h11*g2)/det;
		float64_t gd=g1*dA+g2*dB;

		/* Line Search */
		float64_t stepsize=1;

		while (stepsize>=minstep)
		{
			float64_t newA=a+stepsize*dA;
			float64_t newB=b+stepsize*dB;

			/* New function value */
			float64_t newf=0.0;
			for (index_t i=0; i<length; ++i)
			{
				float64_t fApB=scores[i]*newA+newB;
				if (fApB>=0)
					newf+=t[i]*fApB+CMath::log(1+CMath::exp(-fApB));
				else
					newf+=(t[i]-1)*fApB+CMath::log(1+CMath::exp(fApB));
			}

			/* Check sufficient decrease */
			if (newf<fval+0.0001*stepsize*gd)
			{
				a=newA;
				b=newB;
				fval=newf;
				break;
			}
			else
				stepsize=stepsize/2.0;
		}

		if (stepsize<minstep)
		{
			SG_SWARNING("CStatistics::fit_sigmoid(): line search fails, A=%f, "
					"B=%f, g1=%f, g2=%f, dA=%f, dB=%f, gd=%f\n",
					a, b, g1, g2, dA, dB, gd);
		}
	}

	if (it>=maxiter-1)
	{
		SG_SWARNING("CStatistics::fit_sigmoid(): reaching maximal iterations,"
				" g1=%f, g2=%f\n", g1, g2);
	}

	SG_SDEBUG("fitted sigmoid: a=%f, b=%f\n", a, b)

	CStatistics::SigmoidParamters result;
	result.a=a;
	result.b=b;

	SG_SDEBUG("leaving CStatistics::fit_sigmoid()\n")
	return result;
}
