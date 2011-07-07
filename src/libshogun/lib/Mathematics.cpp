/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2009 Soeren Sonnenburg
 * Written (W) 1999-2008 Gunnar Raetsch
 * Copyright (C) 1999-2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

// Math.cpp: implementation of the CMath class.
//
//////////////////////////////////////////////////////////////////////


#include "lib/common.h"
#include "lib/Mathematics.h"
#include "lib/lapack.h"
#include "lib/io.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

using namespace shogun;

#ifdef USE_LOGCACHE
#ifdef USE_HMMDEBUG
#define MAX_LOG_TABLE_SIZE 10*1024*1024
#define LOG_TABLE_PRECISION 1e-6
#else //USE_HMMDEBUG
#define MAX_LOG_TABLE_SIZE 123*1024*1024
#define LOG_TABLE_PRECISION 1e-15
#endif //USE_HMMDEBUG
int32_t CMath::LOGACCURACY         = 0; // 100000 steps per integer
#endif // USE_LOGCACHE

int32_t CMath::LOGRANGE            = 0; // range for logtable: log(1+exp(x))  -25 <= x <= 0

const float64_t CMath::INFTY            =  -log(0.0);	// infinity
const float64_t CMath::ALMOST_INFTY		=  +1e+20;		//a large number
const float64_t CMath::ALMOST_NEG_INFTY =  -1000;	
#ifdef USE_LOGCACHE
float64_t* CMath::logtable = NULL;
#endif
char* CMath::rand_state = NULL;
uint32_t CMath::seed = 0;

CMath::CMath()
: CSGObject()
{
	CMath::rand_state=new char[RNG_SEED_SIZE];
	init_random();

#ifdef USE_LOGCACHE
    LOGRANGE=CMath::determine_logrange();
    LOGACCURACY=CMath::determine_logaccuracy(LOGRANGE);
    CMath::logtable=new float64_t[LOGRANGE*LOGACCURACY];
    init_log_table();
#else
	int32_t i=0;
	while ((float64_t)log(1+((float64_t)exp(-float64_t(i)))))
		i++;

	LOGRANGE=i;
#endif 
}

CMath::~CMath()
{
	delete[] CMath::rand_state;
	CMath::rand_state=NULL;
#ifdef USE_LOGCACHE
	delete[] CMath::logtable;
	CMath::logtable=NULL;
#endif
}

#ifdef USE_LOGCACHE
int32_t CMath::determine_logrange()
{
    int32_t i;
    float64_t acc=0;
    for (i=0; i<50; i++)
    {
	acc=((float64_t)log(1+((float64_t)exp(-float64_t(i)))));
	if (acc<=(float64_t)LOG_TABLE_PRECISION)
	    break;
    }

    SG_SINFO( "determined range for x in table log(1+exp(-x)) is:%d (error:%G)\n",i,acc);
    return i;
}

int32_t CMath::determine_logaccuracy(int32_t range)
{
    range=MAX_LOG_TABLE_SIZE/range/((int)sizeof(float64_t));
    SG_SINFO( "determined accuracy for x in table log(1+exp(-x)) is:%d (error:%G)\n",range,1.0/(double) range);
    return range;
}

SGVector<float64_t> CMath::fishers_exact_test_for_multiple_3x2_tables(SGMatrix<float64_t> tables, float64_t epsilon)
{
	SGMatrix<float64_t> table(3,2);



}

float64_t CMath::fishers_exact_test_for_3x2_table(SGMatrix<float64_t> table, float64_t epsilon)
{

function [nonrand_p rand_p, prob_table] = fisher_p23_fast(obs, epsilon)

%build marginals = (n1., n2., n.1, n.2, n.3)
marginals = [sum(obs, 2)', sum(obs, 1)];

n = sum(marginals) / 2;

x = zeros(2, 3, max(marginals)*max(marginals));

%build log nominator statistic
log_nom = sum(sum(gammaln(marginals+1))) - gammaln(n+1);

%build log denominator statistic
log_denom = sum(sum(gammaln(obs+1)));


%compute probability of observed table
prob_table = exp(log_nom - log_denom);


nonrand_p = 0.0;
rand_count = 0;

dim1 = min(marginals(1), marginals(3));

%traverse all possible tables with given marginals
counter = 0;
for k=0:dim1
    for l=max(0,marginals(1)-marginals(5)-k):min(marginals(1)-k, marginals(4))
        counter = counter+1;
        x(1, 1, counter) = k;
        x(1, 2, counter) = l;
        x(1, 3, counter) = marginals(1) - x(1, 1, counter) - x(1, 2, counter);
        x(2, 1, counter) = marginals(3) - x(1, 1, counter);
        x(2, 2, counter) = marginals(4) - x(1, 2, counter);
        x(2, 3, counter) = marginals(5) - x(1, 3, counter);
    end
end
log_denom = sum(sum(gammaln(x(:, :, 1:counter)+1)));
log_denom = reshape(log_denom, 1, counter);
prob_lauf = exp(log_nom - log_denom);
nonrand_p = sum(prob_lauf(find(prob_lauf <= prob_table)));
rand_count = length(find(abs(exp(prob_lauf) - exp(prob_table)) < epsilon));

u = rand(1);
rand_p = max(0.0, nonrand_p - u*rand_count*prob_table);
% End of function 'fisher_p23_fast'. 

}

//init log table of form log(1+exp(x))
void CMath::init_log_table()
{
  for (int32_t i=0; i< LOGACCURACY*LOGRANGE; i++)
    logtable[i]=log(1+exp(float64_t(-i)/float64_t(LOGACCURACY)));
}
#endif

void CMath::sort(int32_t *a, int32_t cols, int32_t sort_col)
{
  int32_t changed=1;
  if (a[0]==-1) return ;
  while (changed)
  {
      changed=0; int32_t i=0 ;
      while ((a[(i+1)*cols]!=-1) && (a[(i+1)*cols+1]!=-1)) // to be sure
	  {
		  if (a[i*cols+sort_col]>a[(i+1)*cols+sort_col])
		  {
			  for (int32_t j=0; j<cols; j++)
				  CMath::swap(a[i*cols+j],a[(i+1)*cols+j]) ;
			  changed=1 ;
		  } ;
		  i++ ;
	  } ;
  } ;
} 

void CMath::sort(float64_t *a, int32_t* idx, int32_t N) 
{
	int32_t changed=1;
	while (changed)
	{
		changed=0;
		for (int32_t i=0; i<N-1; i++)
		{
			if (a[i]>a[i+1])
			{
				swap(a[i],a[i+1]) ;
				swap(idx[i],idx[i+1]) ;
				changed=1 ;
			} ;
		} ;
	} ;
 
}

float64_t CMath::Align(
	char* seq1, char* seq2, int32_t l1, int32_t l2, float64_t gapCost)
{
  float64_t actCost=0 ;
  int32_t i1, i2 ;
  float64_t* const gapCosts1 = new float64_t[ l1 ];
  float64_t* const gapCosts2 = new float64_t[ l2 ];
  float64_t* costs2_0 = new float64_t[ l2 + 1 ];
  float64_t* costs2_1 = new float64_t[ l2 + 1 ];

  // initialize borders
  for( i1 = 0; i1 < l1; ++i1 ) {
    gapCosts1[ i1 ] = gapCost * i1;
  }
  costs2_1[ 0 ] = 0;
  for( i2 = 0; i2 < l2; ++i2 ) {
    gapCosts2[ i2 ] = gapCost * i2;
    costs2_1[ i2+1 ] = costs2_1[ i2 ] + gapCosts2[ i2 ];
  }
  // compute alignment
  for( i1 = 0; i1 < l1; ++i1 ) {
    swap( costs2_0, costs2_1 );
    actCost = costs2_0[ 0 ] + gapCosts1[ i1 ];
    costs2_1[ 0 ] = actCost;
    for( i2 = 0; i2 < l2; ++i2 ) {
      const float64_t actMatch = costs2_0[ i2 ] + ( seq1[i1] == seq2[i2] );
      const float64_t actGap1 = costs2_0[ i2+1 ] + gapCosts1[ i1 ];
      const float64_t actGap2 = actCost + gapCosts2[ i2 ];
      const float64_t actGap = min( actGap1, actGap2 );
      actCost = min( actMatch, actGap );
      costs2_1[ i2+1 ] = actCost;
    }
  }

  delete [] gapCosts1;
  delete [] gapCosts2;
  delete [] costs2_0;
  delete [] costs2_1;
  
  // return the final cost
  return actCost;
}

float64_t CMath::mutual_info(float64_t* p1, float64_t* p2, int32_t len)
{
	double e=0;

	for (int32_t i=0; i<len; i++)
		for (int32_t j=0; j<len; j++)
			e+=exp(p2[j*len+i])*(p2[j*len+i]-p1[i]-p1[j]);

	return (float64_t) e;
}

float64_t CMath::relative_entropy(float64_t* p, float64_t* q, int32_t len)
{
	double e=0;

	for (int32_t i=0; i<len; i++)
		e+=exp(p[i])*(p[i]-q[i]);

	return (float64_t) e;
}

float64_t CMath::entropy(float64_t* p, int32_t len)
{
	double e=0;

	for (int32_t i=0; i<len; i++)
		e-=exp(p[i])*p[i];

	return (float64_t) e;
}


//Howto construct the pseudo inverse (from "The Matrix Cookbook")
//
//Assume A does not have full rank, i.e. A is n \times m and rank(A) = r < min(n;m).
//
//The matrix A+ known as the pseudo inverse is unique and does always exist.
//
//The pseudo inverse A+ can be constructed from the singular value
//decomposition A = UDV^T , by  A^+ = V(D+)U^T.

#ifdef HAVE_LAPACK
float64_t* CMath::pinv(
	float64_t* matrix, int32_t rows, int32_t cols, float64_t* target)
{
	if (!target)
		target=new float64_t[rows*cols];

	char jobu='A';
	char jobvt='A';
	int m=rows; /* for calling external lib */
	int n=cols; /* for calling external lib */
	int lda=m; /* for calling external lib */
	int ldu=m; /* for calling external lib */
	int ldvt=n; /* for calling external lib */
	int info=-1; /* for calling external lib */
	int32_t lsize=CMath::min((int32_t) m, (int32_t) n);
	double* s=new double[lsize];
	double* u=new double[m*m];
	double* vt=new double[n*n];

	wrap_dgesvd(jobu, jobvt, m, n, matrix, lda, s, u, ldu, vt, ldvt, &info);
	ASSERT(info==0);

	for (int32_t i=0; i<n; i++)
	{
		for (int32_t j=0; j<lsize; j++)
			vt[i*n+j]=vt[i*n+j]/s[j];
	}

	cblas_dgemm(CblasColMajor, CblasTrans, CblasTrans, m, n, m, 1.0, vt, ldvt, u, ldu, 0, target, m);

	delete[] u;
	delete[] vt;
	delete[] s;

	return target;
}
#endif

namespace shogun
{
template <>
void CMath::display_vector(const uint8_t* vector, int32_t n, const char* name)
{
	ASSERT(n>=0);
	SG_SPRINT("%s=[", name);
	for (int32_t i=0; i<n; i++)
		SG_SPRINT("%d%s", vector[i], i==n-1? "" : ",");
	SG_SPRINT("]\n");
}

template <>
void CMath::display_vector(const int32_t* vector, int32_t n, const char* name)
{
	ASSERT(n>=0);
	SG_SPRINT("%s=[", name);
	for (int32_t i=0; i<n; i++)
		SG_SPRINT("%d%s", vector[i], i==n-1? "" : ",");
	SG_SPRINT("]\n");
}

template <>
void CMath::display_vector(const int64_t* vector, int32_t n, const char* name)
{
	ASSERT(n>=0);
	SG_SPRINT("%s=[", name);
	for (int32_t i=0; i<n; i++)
		SG_SPRINT("%lld%s", vector[i], i==n-1? "" : ",");
	SG_SPRINT("]\n");
}

template <>
void CMath::display_vector(const uint64_t* vector, int32_t n, const char* name)
{
	ASSERT(n>=0);
	SG_SPRINT("%s=[", name);
	for (int32_t i=0; i<n; i++)
		SG_SPRINT("%llu%s", vector[i], i==n-1? "" : ",");
	SG_SPRINT("]\n");
}

template <>
void CMath::display_vector(const float32_t* vector, int32_t n, const char* name)
{
	ASSERT(n>=0);
	SG_SPRINT("%s=[", name);
	for (int32_t i=0; i<n; i++)
		SG_SPRINT("%10.10f%s", vector[i], i==n-1? "" : ",");
	SG_SPRINT("]\n");
}

template <>
void CMath::display_vector(const float64_t* vector, int32_t n, const char* name)
{
	ASSERT(n>=0);
	SG_SPRINT("%s=[", name);
	for (int32_t i=0; i<n; i++)
		SG_SPRINT("%10.10f%s", vector[i], i==n-1? "" : ",");
	SG_SPRINT("]\n");
}

template <>
void CMath::display_matrix(
	const int32_t* matrix, int32_t rows, int32_t cols, const char* name)
{
	ASSERT(rows>=0 && cols>=0);
	SG_SPRINT("%s=[\n", name);
	for (int32_t i=0; i<rows; i++)
	{
		SG_SPRINT("[");
		for (int32_t j=0; j<cols; j++)
			SG_SPRINT("\t%d%s", matrix[j*rows+i],
				j==cols-1? "" : ",");
		SG_SPRINT("]%s\n", i==rows-1? "" : ",");
	}
	SG_SPRINT("]\n");
}

template <>
void CMath::display_matrix(
	const float64_t* matrix, int32_t rows, int32_t cols, const char* name)
{
	ASSERT(rows>=0 && cols>=0);
	SG_SPRINT("%s=[\n", name);
	for (int32_t i=0; i<rows; i++)
	{
		SG_SPRINT("[");
		for (int32_t j=0; j<cols; j++)
			SG_SPRINT("\t%lf%s", (double) matrix[j*rows+i],
				j==cols-1? "" : ",");
		SG_SPRINT("]%s\n", i==rows-1? "" : ",");
	}
	SG_SPRINT("]\n");
}
}
