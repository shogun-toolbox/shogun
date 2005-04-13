#ifndef __MATHMATICS_H_
#define __MATHMATICS_H_

#include "lib/common.h"
#include "lib/io.h"

#include <math.h>
#include <stdio.h>

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

	static inline REAL round(REAL d)
	{
		return floor(d+0.5);
	}

	/// signum of type T variable a 
	template <class T>
	static inline REAL sign(REAL a)
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

	static inline REAL log10(REAL v)
	{
		return log(v)/log(10);
	}

	static inline LONG factorial(INT n)
	{
		LONG res=1 ;
		for (int i=2; i<=n; i++)
			res*=i ;
		return res ;
	}

	static inline LONG nchoosek(INT n, INT k)
	{
		long res=1 ;

		for (INT i=n-k+1; i<=n; i++)
			res*=i ;

		return res/factorial(k) ;
	}

	/** performs a bubblesort on a given matrix a.
	 * it is sorted from in ascending order from top to bottom
	 * and left to right */
	static void sort(INT *a, INT cols, INT sort_col=0) ;
	static void sort(REAL *a, INT*idx, INT N) ;
	
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
			REAL split=output[(size*rand())/(RAND_MAX+1)];
			//REAL split=output[size/2];

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
	static void min(REAL* output, T* index, INT size);

     /* finds the n smallest elements in output and puts these elements as the 
		first n elements  */
	template <class T>
	static void nmin(REAL* output, T* index, INT size, INT n);

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
	static inline INT fast_find(T* output, INT size, T elem)
	{
		INT start=0, end=size-1, middle=size/2 ;

		if (output[start]>elem || output[end]<elem)
			return -1 ;

		while (1)
		{
			if (output[middle]>elem)
			{
				end = middle ;
				middle=start+(end-start)/2 ;
			} ;
			if (output[middle]<elem)
			{
				start = middle ;
				middle=start+(end-start)/2 ;
			}
			if (output[middle]==elem)
				return middle ;
			if (end-start<=1)
			{
				if (output[start]==elem)
					return start ;
				if (output[end]==elem)
					return end ;
				return -1 ;
			}
		}
	}

	/* finds an element in a sorted array via binary search
	 * that is smaller-equal elem und the next element is larger-equal  */
	static inline INT fast_find_range(REAL* output, INT size, REAL elem)
	{
		INT start=0, end=size-2, middle=(size-2)/2 ;

		if (output[start]>elem)
			return -1 ;
		if (output[end]<=elem)
			return size-1 ;

		while (1)
		{
			if (output[middle]>elem)
			{
				end = middle ;
				middle=start+(end-start)/2 ;
			} ;
			if (output[middle]<elem)
			{
				start = middle ;
				middle=start+(end-start)/2 ;
			}
			if (output[middle]<=elem && output[middle+1]>=elem)
				return middle ;
			if (end-start<=1)
			{
				if (output[start]<=elem && output[start+1]>=elem)
					return start ;
				return end ;
			}
		}
	}

	/** calculates ROC into (fp,tp)
	 * from output and label of length size 
	 * returns index with smallest error=fp+fn
	 */
	static INT calcroc(REAL* fp, REAL* tp, REAL* output, INT* label, INT& size, INT& possize, INT& negsize, REAL& tresh, FILE* rocfile);
	//@}

	/// returns the mutual information of p which is given in logspace
	/// where p,q are given in logspace
	static double mutual_info(REAL* p1, REAL* p2, INT len);

	/// returns the relative entropy H(P||Q), 
	/// where p,q are given in logspace
	static double relative_entropy(REAL* p, REAL* q, INT len);

	/// returns entropy of p which is given in logspace
	static double entropy(REAL* p, INT len);

	/**@name summing functions */
	//@{ 
	/** sum logarithmic probabilities.
	 * Probability measures are summed up but are now given in logspace where
	 * direct summation of exp(operand) is not possible due to numerical problems, i.e. eg. exp(-1000)=0. Therefore
	 * we do log( exp(a) + exp(b)) = a + log (1 + exp (b-a)) where a is max(p,q) and b min(p,q). */
#ifdef USE_LOGCACHE
	static inline REAL logarithmic_sum(REAL p, REAL q)
	{
		if (finite(p))
		{
			if (finite(q))
			{

				register REAL diff=p-q;

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
	inline REAL logarithmic_sum(REAL p, REAL q)
	{
	    double result=comp_logarithmic_sum(p,q);

	    printf("diff:%f <-> %f\n",p-q, result);
	    return result;
	}*/

	static inline REAL logarithmic_sum(REAL p, REAL q)
	{
		if (finite(p))
		{
			if (finite(q))
			{

				register REAL diff=p-q;
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
	static inline REAL logarithmic_sum_array(REAL *p, INT len)
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
		register REAL *pp=p ;
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
	static const REAL INFTY;
	
	/// almost neg (log) infinity
	static const REAL ALMOST_NEG_INFTY;

	/// range for logtable: log(1+exp(x))  -LOGRANGE <= x <= 0
	static INT LOGRANGE;
	
#ifdef USE_LOGCACHE	
	/// number of steps per integer
	static INT LOGACCURACY;
	//@}
protected:
	///table with log-values
	static REAL* logtable;	
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
		REAL split=output[(size*rand())/(RAND_MAX+1)];
		//REAL split=output[size/2];
		
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
		REAL split=output[(size*rand())/(RAND_MAX+1)];
		//REAL split=output[size/2];
		
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
void CMath::nmin(REAL* output, T* index, INT size, INT n)
{
	if (6*n*size<13*size*log(size))
		for (INT i=0; i<n; i++)
			min(&output[i], &index[i], size-i) ;
	else
		qsort(output, index, size) ;
}

template <class T>
void CMath::min(REAL* output, T* index, INT size)
{
	if (size<=0)
		return ;
	REAL min_elem = output[0] ;
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
