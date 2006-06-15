/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2006 Soeren Sonnenburg
 * Written (W) 1999-2006 Gunnar Raetsch
 * Written (W) 1999-2006 Fabio De Bona
 * Copyright (C) 1999-2006 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef __MATHMATICS_H_
#define __MATHMATICS_H_

#include "lib/common.h"
#include "lib/io.h"

#include <math.h>
#include <stdio.h>

#ifdef HAVE_LAPACK
extern "C" {
#include <cblas.h>
}
#endif

//define finite for win32
#ifdef _WIN32
#include <float.h>
#define finite _finite
#define isnan _isnan
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
class CMath  
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
		return floor(d+0.5);
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

	static inline DREAL log10(DREAL v)
	{
		return log(v)/log(10);
	}

	static DREAL* pinv(DREAL* matrix, INT rows, INT cols, DREAL* target=NULL);

	static inline LONG factorial(INT n)
	{
		LONG res=1 ;
		for (int i=2; i<=n; i++)
			res*=i ;
		return res ;
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
			CIO::message("INVALID second operand to logsum(%f,%f) expect undefined results\n", p, q);
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
	/*
	inline DREAL logarithmic_sum(DREAL p, DREAL q)
	{
	    double result=comp_logarithmic_sum(p,q);

	    printf("diff:%f <-> %f\n",p-q, result);
	    return result;
	}*/

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
	if (6*n*size<13*size*log(size))
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
#endif
