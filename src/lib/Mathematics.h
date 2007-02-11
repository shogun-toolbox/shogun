/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2007 Soeren Sonnenburg
 * Written (W) 1999-2007 Gunnar Raetsch
 * Copyright (C) 1999-2007 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef __MATHMATICS_H_
#define __MATHMATICS_H_

#include "lib/common.h"
#include "lib/io.h"
#include "base/SGObject.h"

#include <math.h>
#include <stdio.h>
#include <float.h>

#ifdef HAVE_LAPACK
extern "C" {
#include <cblas.h>
#ifndef DARWIN
#include <clapack.h>
#endif
}

#endif

//define finite for win32
#ifdef _WIN32
#include <float.h>
#ifndef finite
#define finite _finite
#endif

#ifndef isnan
#define isnan _isnan
#endif
#endif

//define finite/nan for CYGWIN
#ifdef CYGWIN
#ifndef finite
#include <ieeefp.h>
#endif
#endif

//fall back to double precision pow if powl is not
//available
#ifndef powl
#define powl(x,y) pow((double) (x),(double) (y))
#endif

#ifndef NAN
#include <stdlib.h>
#define NAN (strtod("NAN",NULL))
#endif

#ifdef SUNOS
extern "C" int	finite(double);
#endif

/** Mathematical Functions.
 * Class which collects generic mathematical functions
 */
class CMath : public CSGObject
{
public:
	/**@name Constructor/Destructor.
	 */
	//@{
	///Constructor - initializes log-table
	CMath();

	///Destructor - frees logtable
	virtual ~CMath();
	//@}

	/**@name min/max/abs functions.
	*/
	//@{

	///return the minimum of two integers
	//
	template <class T>
	static inline T min(T a, T b)
	{
		return ((a)<=(b))?(a):(b);
	}

	///return the maximum of two integers
	template <class T>
	static inline T max(T a, T b) 
	{
		return ((a)>=(b))?(a):(b);
	}

	///return the maximum of two integers
	template <class T>
	static inline T abs(T a)
	{
		return ((a)>=(0))?(a):(-a);
	}
	//@}

	/**@name misc functions */
	//@{

	/// crc32
	static UINT crc32(BYTE *data, INT len);

	static inline DREAL round(DREAL d)
	{
		return ::floor(d+0.5);
	}

	static inline DREAL floor(DREAL d)
	{
		return ::floor(d);
	}

	static inline DREAL ceil(DREAL d)
	{
		return ::ceil(d);
	}

	/// signum of type T variable a 
	template <class T>
	static inline INT sign(DREAL a)
	{
		if (a==0)
			return 0;
		else return (a<0) ? (-1) : (+1);
	}

	/// swap floats a and b
	template <class T>
	static inline void swap(T & a,T &b)
	{
		T c=a ;
		a=b; b=c ;
	}

	/// x^2
	template <class T>
	static inline T sq(T x)
	{
		return x*x;
	}

	/// x^n
	static inline long double powl(long double x, long double n)
	{
//#ifdef
		return ::powl(x, n);
	}

	static inline DREAL log10(DREAL v)
	{
		return ::log(v)/::log(10.0);
	}

	static inline DREAL log(DREAL v)
	{
		return ::log(v);
	}

#ifdef HAVE_LAPACK
	/// return the pseudo inverse for matrix
	/// when matrix has shape (rows, cols) the pseudo inverse has (cols, rows)
	static DREAL* pinv(DREAL* matrix, INT rows, INT cols, DREAL* target=NULL);
#endif

	static inline LONG factorial(INT n)
	{
		LONG res=1 ;
		for (int i=2; i<=n; i++)
			res*=i ;
		return res ;
	}
	
	static INT random()
	{
#ifdef CYGWIN
		return rand();
#else
		return ::random();
#endif
	}

	static INT random(INT min_value, INT max_value)
	{
		DREAL ret = min_value + (max_value+1-min_value)*(1.*rand())/(RAND_MAX*1.) ;
		INT iret = (INT) ::floor(ret - 1e-10) ;
		ASSERT(iret >= min_value && iret<=max_value) ;
		return iret ;
	}

	static DREAL random(DREAL min_value, DREAL max_value)
	{
		DREAL ret = min_value + (max_value-min_value)*(1.*rand())/(RAND_MAX*1.) ;
		if (!(ret >= min_value-1e-6 && ret<=max_value+1e-6))
			fprintf(stderr, "%f %f %f\n", ret, min_value, max_value) ;
		
		ASSERT(ret >= min_value-1e-6 && ret<=max_value+1e-6) ;
		return ret ;
	}

	static inline INT* randperm(INT n)
	{
		INT* perm = new INT[n];

		if (perm)
		{
			for (INT i=0; i<n; i++)
				perm[i]=i;

			for (INT i=0; i<n; i++)
			{
				swap( perm[(n*rand())/(RAND_MAX+1)], perm[i]);
			}
		}

		return perm;
	}

	static inline LONG nchoosek(INT n, INT k)
	{
		long res=1 ;

		for (INT i=n-k+1; i<=n; i++)
			res*=i ;

		return res/factorial(k) ;
	}

	static inline DREAL dot(DREAL* v1, DREAL* v2, INT n)
	{
		DREAL r=0;
#ifdef HAVE_LAPACK
		INT skip=1;
		r = cblas_ddot(n, v1, skip, v2, skip);
#else
		for (INT i=0; i<n; i++)
			r+=v1[i]*v2[i];
#endif
		return r;
	}

	static inline DREAL mean(DREAL* vec, INT len)
	{
		ASSERT(vec);
		ASSERT(len>0);

		DREAL mean=0;
		for (INT i=0; i<len; i++)
			mean+=vec[i];
		return mean/len;
	}

	static inline DREAL trace(DREAL* mat, INT cols, INT rows)
	{
		DREAL trace=0;
		for (INT i=0; i<rows; i++)
			trace+=mat[i*cols+i];
		return trace;
	}

	/** performs a bubblesort on a given matrix a.
	 * it is sorted from in ascending order from top to bottom
	 * and left to right */
	static void sort(INT *a, INT cols, INT sort_col=0) ;
	static void sort(DREAL *a, INT*idx, INT N) ;
	
	/** performs a quicksort on an array output of length size
	 * it is sorted from in ascending (for type T) */
	//template <class T>
	//static void qsort(T* output, INT size) ;
	template <class T>
	static void qsort(T* output, INT size)
	{
		if (size==2)
		{
			if (output[0] > output [1])
				swap(output[0],output[1]);
		}
		else
		{
			T split=output[(size*rand())/(RAND_MAX+1)];
			//T split=output[size/2];

			INT left=0;
			INT right=size-1;

			while (left<=right)
			{
				while (output[left] < split)
					left++;
				while (output[right] > split)
					right--;

				if (left<=right)
				{
					swap(output[left],output[right]);
					left++;
					right--;
				}
			}

			if (right+1> 1)
				qsort(output,right+1);

			if (size-left> 1)
				qsort(&output[left],size-left);
		}
	}

	/// display vector (useful for debugging)
	template <class T> static void display_vector(T* vector, INT n, const char* name="vector");

	/// display matrix (useful for debugging)
	template <class T> static void display_matrix(T* matrix, INT rows, INT cols, const char* name="matrix");

	/** performs a quicksort on an array output of length size
	 * it is sorted from in ascending order 
	 * (for type T1) and returns the index (type T2)
	 * matlab alike [sorted,index]=sort(output) 
	 */
	template <class T1,class T2>
	static void qsort(T1* output, T2* index, INT size);

	/** performs a quicksort on an array output of length size
	 * it is sorted from in ascending order
	 * (for type T1) and returns the index (type T2)
	 * matlab alike [sorted,index]=sort(output) 
	 */
	template <class T1,class T2>
	static void qsort_backward(T1* output, T2* index, INT size);

     /* finds the smallest element in output and puts that element as the 
		first element  */
	template <class T>
	static void min(DREAL* output, T* index, INT size);

     /* finds the n smallest elements in output and puts these elements as the 
		first n elements  */
	template <class T>
	static void nmin(DREAL* output, T* index, INT size, INT n);

	/* performs a inplace unique of a vector of type T using quicksort 
	 * returns the new number of elements */
	template <class T>
	static INT unique(T* output, INT size) 
	{
		qsort(output, size) ;
		INT i,j=0 ;
		for (i=0; i<size; i++)
			if (i==0 || output[i]!=output[i-1])
				output[j++]=output[i] ;
		return j ;
	}

	/* finds an element in a sorted array via binary search
     * returns -1 if not found */
	template <class T>
	static inline INT binary_search(T* output, INT size, T elem)
	{
		INT start=0;
		INT end=size-1;

		if (size<1)
			return -1;

		while (start<end)
		{
			INT middle=start+(end-start)/2; 

			if (output[middle]>elem)
				end=middle-1;
			else if (output[middle]<elem)
				start=middle+1;
			else
				return middle;
		}

		if (start<size && output[start]==elem)
			return start;
		else
			return -1;
	}

	/* finds an element in a sorted array via binary search 
	 * if it exists, else the index the largest smaller element
	 * is returned
	 * note: a successor is not mandatory */
	template <class T>
	static inline INT binary_search_max_lower_equal(T* output, INT size, T elem)
	{
		INT start=0;
		INT end=size-1;

		if (size<1)
			return -1;

		while (start<end)
		{
			INT middle=start+(end-start)/2; 

			if (output[middle]>elem)
				end=middle-1;
			else if (output[middle]<elem)
				start=middle+1;
			else
				return middle;
		}

		if (start<size && output[start]<=elem)
			return start;

		if (start>0 && output[start-1]<=elem)
			return start-1;

		return -1;
	}

	/** calculates ROC into (fp,tp)
	 * from output and label of length size 
	 * returns index with smallest error=fp+fn
	 */
	static INT calcroc(DREAL* fp, DREAL* tp, DREAL* output, INT* label, INT& size, INT& possize, INT& negsize, DREAL& tresh, FILE* rocfile);
	//@}

	/// returns the mutual information of p which is given in logspace
	/// where p,q are given in logspace
	static double mutual_info(DREAL* p1, DREAL* p2, INT len);

	/// returns the relative entropy H(P||Q), 
	/// where p,q are given in logspace
	static double relative_entropy(DREAL* p, DREAL* q, INT len);

	/// returns entropy of p which is given in logspace
	static double entropy(DREAL* p, INT len);

	/// returns number generator seed
	inline static UINT get_seed()
	{
		return CMath::seed;
	}


	/**@name summing functions */
	//@{ 
	/** sum logarithmic probabilities.
	 * Probability measures are summed up but are now given in logspace where
	 * direct summation of exp(operand) is not possible due to numerical problems, i.e. eg. exp(-1000)=0. Therefore
	 * we do log( exp(a) + exp(b)) = a + log (1 + exp (b-a)) where a is max(p,q) and b min(p,q). */
#ifdef USE_LOGCACHE
	static inline DREAL logarithmic_sum(DREAL p, DREAL q)
	{
		if (finite(p))
		{
			if (finite(q))
			{

				register DREAL diff=p-q;

				if (diff>0)		//p>q
				{
					if (diff > LOGRANGE)
						return p;
					else
						return  p + logtable[(int)(diff*LOGACCURACY)];
				}
				else			//p<=q
				{
					if (-diff > LOGRANGE)
						return  q;
					else
						return  q + logtable[(int)(-diff*LOGACCURACY)];
				}
			}
			SG_SWARNING(("INVALID second operand to logsum(%f,%f) expect undefined results\n", p, q);
			return NAN;
		}
		else 
			return q;
	}

	///init log table of form log(1+exp(x))
	static void init_log_table();
	
	/// determine INT x for that log(1+exp(-x)) == 0
	static INT determine_logrange();

	/// determine accuracy, such that the thing fits into MAX_LOG_TABLE_SIZE, needs logrange as argument
	static INT determine_logaccuracy(INT range);
#else
	static inline DREAL logarithmic_sum(DREAL p, DREAL q)
	{
		if (finite(p))
		{
			if (finite(q))
			{

				register DREAL diff=p-q;
				if (diff>0)		//p>q
				{
					if (diff > LOGRANGE)
						return p;
					else
						return  p + log(1+exp(-diff));
				}
				else			//p<=q
				{
					if (-diff > LOGRANGE)
						return  q;
					else
						return  q + log(1+exp(diff));
				}
			}
			return p;
		}
		else 
			return q;
	}
#endif
#ifdef LOG_SUM_ARRAY
	/** sum up a whole array of values in logspace.
	 * This function addresses the numeric instabiliy caused by simply summing up N elements by adding 
	 * each of the elements to some variable. Instead array neighbours are summed up until one element remains.
	 * Whilst the number of additions remains the same, the error is only in the order of log(N) instead N.
	 */
	static inline DREAL logarithmic_sum_array(DREAL *p, INT len)
	  {
	    if (len<=2)
	      {
		if (len==2)
		  return logarithmic_sum(p[0],p[1]) ;
		if (len==1)
		  return p[0];
		return -INFTY ;
	      }
	    else
	      {
		register DREAL *pp=p ;
		if (len%2==1) pp++ ;
		for (register INT j=0; j < len>>1; j++)
		  pp[j]=logarithmic_sum(pp[j<<1], pp[1+(j<<1)]) ;
	      }
	    return logarithmic_sum_array(p,len%2+len>>1) ;
	  } 
#endif
	//@}
public:
	/**@name constants*/
	//@{
	/// infinity
	static const DREAL INFTY;
	
	/// almost neg (log) infinity
	static const DREAL ALMOST_NEG_INFTY;

	/// range for logtable: log(1+exp(x))  -LOGRANGE <= x <= 0
	static INT LOGRANGE;

	/// random generator seed
	static UINT seed;
	
#ifdef USE_LOGCACHE	

	/// number of steps per integer
	static INT LOGACCURACY;
	//@}
protected:
	///table with log-values
	static DREAL* logtable;	
#endif
	static CHAR rand_state[256];
};


//implementations of template functions
template <class T1,class T2>
void CMath::qsort(T1* output, T2* index, INT size)
{
	if (size==2)
	{
		if (output[0] > output [1]){
			swap(output[0],output[1]);
			swap(index[0],index[1]);
		}
		
	}
	else
	{
		DREAL split=output[(size*rand())/(RAND_MAX+1)];
		//DREAL split=output[size/2];
		
		INT left=0;
		INT right=size-1;
		
		while (left<=right)
		{
			while (output[left] < split)
				left++;
			while (output[right] > split)
				right--;
			
			if (left<=right)
			{
				swap(output[left],output[right]);
				swap(index[left],index[right]);
				left++;
				right--;
			}
		}
		
		if (right+1> 1)
			qsort(output,index,right+1);
		
		if (size-left> 1)
			qsort(&output[left],&index[left], size-left);
	}
}

template <class T1,class T2>
void CMath::qsort_backward(T1* output, T2* index, INT size)
{
	if (size==2)
	{
		if (output[0] < output [1]){
			swap(output[0],output[1]);
			swap(index[0],index[1]);
		}
		
	}
	else
	{
		DREAL split=output[(size*rand())/(RAND_MAX+1)];
		//DREAL split=output[size/2];
		
		INT left=0;
		INT right=size-1;
		
		while (left<=right)
		{
			while (output[left] > split)
				left++;
			while (output[right] < split)
				right--;
			
			if (left<=right)
			{
				swap(output[left],output[right]);
				swap(index[left],index[right]);
				left++;
				right--;
			}
		}
		
		if (right+1> 1)
			qsort(output,index,right+1);
		
		if (size-left> 1)
			qsort(&output[left],&index[left], size-left);
	}
}

template <class T> 
void CMath::nmin(DREAL* output, T* index, INT size, INT n)
{
	if (6*n*size<13*size*CMath::log(size))
		for (INT i=0; i<n; i++)
			min(&output[i], &index[i], size-i) ;
	else
		qsort(output, index, size) ;
}

template <class T>
void CMath::min(DREAL* output, T* index, INT size)
{
	if (size<=0)
		return ;
	DREAL min_elem = output[0] ;
	INT min_index = 0 ;
	for (INT i=1; i<size; i++)
		if (output[i]<min_elem)
		{
			min_index=i ;
			min_elem=output[i] ;
		}
	swap(output[0], output[min_index]) ;
	swap(index[0], index[min_index]) ;
}

template <class T>
void CMath::display_vector(T* vector, INT n, const char* name)
{
	SG_SPRINT("%s=[", name);
	for (INT i=0; i<n-1; i++)
		SG_SPRINT("%f,", (double) vector[i]);
	for (INT i=n-1; i<n; i++)
		SG_SPRINT("%f]\n", (double) vector[i]);
}

template <class T>
void CMath::display_matrix(T* matrix, INT rows, INT cols, const char* name)
{
	SG_SPRINT("%s=[\n", name);
	for (INT j=0; j<rows-1; j++)
	{
		SG_SPRINT("[");
		for (INT i=0; i<cols-1; i++)
			SG_SPRINT("\t%f,", (double) matrix[j+i*rows]);
		for (INT i=cols-1; i<cols; i++)
			SG_SPRINT("\t%f],\n", (double) matrix[j+i*rows]);
	}
	for (INT j=rows-1; j<rows; j++)
	{
		SG_SPRINT("[");
		for (INT i=0; i<cols-1; i++)
			SG_SPRINT("\t%f,", (double) matrix[j+i*rows]);
		for (INT i=cols-1; i<cols; i++)
			SG_SPRINT("\t%f]\n]\n", (double) matrix[j+i*rows]);
	}
}

#endif
