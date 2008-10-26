/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2008 Soeren Sonnenburg
 * Written (W) 1999-2008 Gunnar Raetsch
 * Copyright (C) 1999-2008 Fraunhofer Institute FIRST and Max-Planck-Society
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

//////////////////////////////////////////////////////////////////////
// Construction/Destruction
//////////////////////////////////////////////////////////////////////

#ifdef USE_LOGCACHE
//gene/math specific constants
#ifdef USE_HMMDEBUG
#define MAX_LOG_TABLE_SIZE 10*1024*1024
#define LOG_TABLE_PRECISION 1e-6
#else
#define MAX_LOG_TABLE_SIZE 123*1024*1024
#define LOG_TABLE_PRECISION 1e-15
#endif

int32_t CMath::LOGACCURACY         = 0; // 100000 steps per integer
#endif

int32_t CMath::LOGRANGE            = 0; // range for logtable: log(1+exp(x))  -25 <= x <= 0

const DREAL CMath::INFTY            =  -log(0.0);	// infinity
const DREAL CMath::ALMOST_INFTY		=  +1e+20;		//a large number
const DREAL CMath::ALMOST_NEG_INFTY =  -1000;	
#ifdef USE_LOGCACHE
DREAL* CMath::logtable = NULL;
#endif
char* CMath::rand_state = NULL;
uint32_t CMath::seed = 0;

CMath::CMath()
: CSGObject()
{
#ifndef HAVE_SWIG
	CSGObject::version.print_version();
#endif
	CMath::rand_state=new char[RNG_SEED_SIZE];
	init_random();
#ifndef HAVE_SWIG
	SG_PRINT("( seeding random number generator with %u (seed size %d))\n", seed, RNG_SEED_SIZE);
#endif

#ifdef USE_LOGCACHE
    LOGRANGE=CMath::determine_logrange();
    LOGACCURACY=CMath::determine_logaccuracy(LOGRANGE);
#ifndef HAVE_SWIG
    SG_PRINT( "initializing log-table (size=%i*%i*%i=%2.1fMB) ... ) ",LOGRANGE,LOGACCURACY,sizeof(DREAL),LOGRANGE*LOGACCURACY*sizeof(DREAL)/(1024.0*1024.0)) ;
#endif 
   
    CMath::logtable=new DREAL[LOGRANGE*LOGACCURACY];
    init_log_table();
#else
	int32_t i=0;
	while ((DREAL)log(1+((DREAL)exp(-DREAL(i)))))
		i++;
#ifndef HAVE_SWIG
    SG_PRINT("determined range for x in log(1+exp(-x)) is:%d )\n", i);
#endif 
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
    DREAL acc=0;
    for (i=0; i<50; i++)
    {
	acc=((DREAL)log(1+((DREAL)exp(-DREAL(i)))));
	if (acc<=(DREAL)LOG_TABLE_PRECISION)
	    break;
    }

    SG_INFO( "determined range for x in table log(1+exp(-x)) is:%d (error:%G)\n",i,acc);
    return i;
}

int32_t CMath::determine_logaccuracy(int32_t range)
{
    range=MAX_LOG_TABLE_SIZE/range/((int)sizeof(DREAL));
    SG_INFO( "determined accuracy for x in table log(1+exp(-x)) is:%d (error:%G)\n",range,1.0/(double) range);
    return range;
}

//init log table of form log(1+exp(x))
void CMath::init_log_table()
{
  for (int32_t i=0; i< LOGACCURACY*LOGRANGE; i++)
    logtable[i]=log(1+exp(DREAL(-i)/DREAL(LOGACCURACY)));
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

void CMath::sort(DREAL *a, int32_t* idx, int32_t N) 
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

DREAL CMath::Align(char * seq1, char* seq2, int32_t l1, int32_t l2, DREAL gapCost)
{
  DREAL actCost=0 ;
  int32_t i1, i2 ;
  DREAL* const gapCosts1 = new DREAL[ l1 ];
  DREAL* const gapCosts2 = new DREAL[ l2 ];
  DREAL* costs2_0 = new DREAL[ l2 + 1 ];
  DREAL* costs2_1 = new DREAL[ l2 + 1 ];

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
      const DREAL actMatch = costs2_0[ i2 ] + ( seq1[i1] == seq2[i2] );
      const DREAL actGap1 = costs2_0[ i2+1 ] + gapCosts1[ i1 ];
      const DREAL actGap2 = actCost + gapCosts2[ i2 ];
      const DREAL actGap = min( actGap1, actGap2 );
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

//plot x- axis false positives (fp) 1-Specificity
//plot y- axis true positives (tp) Sensitivity
int32_t CMath::calcroc(DREAL* fp, DREAL* tp, DREAL* output, int32_t* label, int32_t& size, int32_t& possize, int32_t& negsize, DREAL& tresh, FILE* rocfile)
{
	int32_t left=0;
	int32_t right=size-1;
	int32_t i;

	for (i=0; i<size; i++)
	{
		if (!(label[i]== -1 || label[i]==1))
			return -1;
	}

	//sort data such that -1 labels first +1 behind
	while (left<right)
	{
		while ((label[left] < 0) && (left<right))
			left++;
		while ((label[right] > 0) && (left<right))	//warning: label must be either -1 or +1
			right--;

		swap(output[left],output[right]);
		swap(label[left], label[right]);
	}

	// neg/pos sizes
	negsize=left;
	possize=size-left;
	DREAL* negout=output;
	DREAL* posout=&output[left];

	// sort the pos and neg outputs separately
	qsort(negout, negsize);
	qsort(posout, possize);

	// set minimum+maximum values for decision-treshhold
	DREAL minimum=min(negout[0], posout[0]);
	DREAL maximum=minimum;
	if (negsize>0)
		maximum=max(maximum,negout[negsize-1]);
	if (possize>0)
		maximum=max(maximum,posout[possize-1]);

	DREAL treshhold=minimum;
	DREAL old_treshhold=treshhold;

	//clear array.
	for (i=0; i<size; i++)
	{
		fp[i]=1.0;
		tp[i]=1.0;
	}

	//start with fp=1.0 tp=1.0 which is posidx=0, negidx=0
	//everything right of {pos,neg}idx is considered to beLONG to +1
	int32_t posidx=0;
	int32_t negidx=0;
	int32_t iteration=1;
	int32_t returnidx=-1;

	DREAL minerr=10;

	while (iteration < size && treshhold<=maximum)
	{
		old_treshhold=treshhold;

		while (treshhold==old_treshhold && treshhold<=maximum)
		{
			if (posidx<possize && negidx<negsize)
			{
				if (posout[posidx]<negout[negidx])
				{
					if (posout[posidx]==treshhold)
						posidx++;
					else
						treshhold=posout[posidx];
				}
				else
				{
					if (negout[negidx]==treshhold)
						negidx++;
					else
						treshhold=negout[negidx];
				}
			}
			else
			{
				if (posidx>=possize && negidx<negsize-1)
				{
					negidx++;
					treshhold=negout[negidx];
				}
				else if (negidx>=negsize && posidx<possize-1)
				{
					posidx++;
					treshhold=posout[posidx];
				}
				else if (negidx<negsize && treshhold!=negout[negidx])
					treshhold=negout[negidx];
				else if (posidx<possize && treshhold!=posout[posidx])
					treshhold=posout[posidx];
				else
				{
					treshhold=2*(maximum+100); // force termination
					posidx=possize;
					negidx=negsize;
					break;
				}
			}
		}

		//calc tp,fp
		tp[iteration]=(possize-posidx)/(DREAL (possize));
		fp[iteration]=(negsize-negidx)/(DREAL (negsize));

		//choose point with minimal error
		if (minerr > negsize*fp[iteration]/size+(1-tp[iteration])*possize/size )
		{
			minerr=negsize*fp[iteration]/size+(1-tp[iteration])*possize/size;
			tresh=(old_treshhold+treshhold)/2;
			returnidx=iteration;
		}

		iteration++;
	}

	// set new size
	size=iteration;

	if (rocfile)
	{
		const char id[]="ROC";
		fwrite(id, sizeof(char), sizeof(id), rocfile);
		fwrite(fp, sizeof(DREAL), size, rocfile);
		fwrite(tp, sizeof(DREAL), size, rocfile);
	}

	return returnidx;
}

uint32_t CMath::crc32(uint8_t *data, int32_t len)
{
	uint32_t result;
	int32_t i,j;
	uint8_t octet;

	result = 0-1;
	for (i=0; i<len; i++)
	{
	octet = *(data++);
	for (j=0; j<8; j++)
	{
	    if ((octet >> 7) ^ (result >> 31))
	    {
		result = (result << 1) ^ 0x04c11db7;
	    }
	    else
	    {
		result = (result << 1);
	    }
	    octet <<= 1;
	}
    }

    return ~result; 
}

double CMath::mutual_info(DREAL* p1, DREAL* p2, int32_t len)
{
	double e=0;

	for (int32_t i=0; i<len; i++)
		for (int32_t j=0; j<len; j++)
			e+=exp(p2[j*len+i])*(p2[j*len+i]-p1[i]-p1[j]);

	return e;
}

double CMath::relative_entropy(DREAL* p, DREAL* q, int32_t len)
{
	double e=0;

	for (int32_t i=0; i<len; i++)
		e+=exp(p[i])*(p[i]-q[i]);

	return e;
}

double CMath::entropy(DREAL* p, int32_t len)
{
	double e=0;

	for (int32_t i=0; i<len; i++)
		e-=exp(p[i])*p[i];

	return e;
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
DREAL* CMath::pinv(DREAL* matrix, int32_t rows, int32_t cols, DREAL* target)
{
	if (!target)
		target=new DREAL[rows*cols];

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

template <>
void CMath::display_vector(uint8_t* vector, int32_t n, const char* name)
{
	ASSERT(n>=0);
	SG_SPRINT("%s=[", name);
	for (int32_t i=0; i<n; i++)
		SG_SPRINT("%d%s", vector[i], i==n-1? "" : ",");
	SG_SPRINT("]\n");
}

template <>
void CMath::display_vector(int32_t* vector, int32_t n, const char* name)
{
	ASSERT(n>=0);
	SG_SPRINT("%s=[", name);
	for (int32_t i=0; i<n; i++)
		SG_SPRINT("%d%s", vector[i], i==n-1? "" : ",");
	SG_SPRINT("]\n");
}

template <>
void CMath::display_vector(LONG* vector, int32_t n, const char* name)
{
	ASSERT(n>=0);
	SG_SPRINT("%s=[", name);
	for (int32_t i=0; i<n; i++)
		SG_SPRINT("%lld%s", vector[i], i==n-1? "" : ",");
	SG_SPRINT("]\n");
}

template <>
void CMath::display_vector(SHORTREAL* vector, int32_t n, const char* name)
{
	ASSERT(n>=0);
	SG_SPRINT("%s=[", name);
	for (int32_t i=0; i<n; i++)
		SG_SPRINT("%10.10f%s", vector[i], i==n-1? "" : ",");
	SG_SPRINT("]\n");
}

template <>
void CMath::display_vector(DREAL* vector, int32_t n, const char* name)
{
	ASSERT(n>=0);
	SG_SPRINT("%s=[", name);
	for (int32_t i=0; i<n; i++)
		SG_SPRINT("%10.10f%s", vector[i], i==n-1? "" : ",");
	SG_SPRINT("]\n");
}

template <>
void CMath::display_matrix(int32_t* matrix, int32_t rows, int32_t cols, const char* name)
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
void CMath::display_matrix(DREAL* matrix, int32_t rows, int32_t cols, const char* name)
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

