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
	inline INT min(INT a, INT b) 
	{
		return ((a)<=(b))?(a):(b);
	}

	///return the maximum of two integers
	inline INT max(INT a, INT b) 
	{
		return ((a)>=(b))?(a):(b);
	}

	///return the maximum of two integers
	inline INT abs(INT a)
	{
		return ((a)>=(0))?(a):(-a);
	}

	///return the minimum of two integers
	inline LONG min(LONG a, LONG b) 
	{
		return ((a)<=(b))?(a):(b);
	}

	///return the maximum of two integers
	inline LONG max(LONG a, LONG b) 
	{
		return ((a)>=(b))?(a):(b);
	}

	///return the maximum of two integers
	inline LONG abs(LONG a)
	{
		return ((a)>=(0))?(a):(-a);
	}

	///return the minimum of two integers
	inline REAL min(REAL a, REAL b) 
	{
		return ((a)<=(b))?(a):(b);
	}

	///return the maximum of two integers
	inline REAL max(REAL a, REAL b) 
	{
		return ((a)>=(b))?(a):(b);
	}

	///return the maximum of two integers
	inline REAL abs(REAL a)
	{
		return ((a)>=(0))?(a):(-a);
	}
	//@}

	/**@name misc functions */
	//@{

	/// crc32
	UINT crc32(BYTE *data, INT len);

	inline REAL round(REAL d)
	{
		return floor(d+0.5);
	}

	/// signum of int a 
	inline INT sign(INT a)
	  {
	      if (a==0)
		  return 0;
	      else return (a<0) ? -1 : +1;
	  }

	/// signum of float a 
	inline REAL sign(REAL a)
	  {
	      if (a==0)
		  return 0;
	      else return (a<0) ? -1 : +1;
	  }

	/// swap floats a and b
	inline void swap(REAL & a,REAL &b)
	  {
	    REAL c=a ;
	    a=b; b=c ;
	  }

	/// swap float vectors a and b
	inline void swap(REAL * & a,REAL *&b)
	  {
	    REAL *c=a ;
	    a=b; b=c ;
	  }

	/// swap INT vectors a and b
	inline void swap(INT * & a, INT *&b)
	  {
	    INT *c=a ;
	    a=b; b=c ;
	  }

	/// swap integers a and b
	inline void swap(INT & a, INT &b)
	  {
	    INT c=a ;
	    a=b; b=c ;
	  } 

	/// swap integers a and b
	inline void swap(WORD &a, WORD &b)
	  {
	    WORD c=a ;
	    a=b; b=c ;
	  } 

	/// x^2
	inline REAL sq(REAL x)
	{
		return x*x;
	}

	inline LONG factorial(INT n)
		{
			LONG res=1 ;
			for (int i=2; i<=n; i++)
				res*=i ;
			return res ;
		} ;

	inline LONG nchoosek(INT n, INT k)
		{
			long res=1 ;
			
			for (INT i=n-k+1; i<=n; i++)
				res*=i ;
			
			return res/factorial(k) ;
		} ;
	
	
	 

	/** performs a bubblesort on a given matrix a.
	 * it is sorted from in ascending order from top to bottom
	 * and left to right */
	void sort(INT *a, INT cols, INT sort_col=0) ;
	void sort(REAL *a, INT*idx, INT N) ;
	
	/** performs a quicksort on an array output of length size
	 * it is sorted from in ascending order from top to bottom
	 * and left to right (for REAL) */
	void qsort(REAL* output, INT size) ;

	/** performs a quicksort on an array output of length size
	 * it is sorted from in ascending order from top to bottom
	 * and left to right (for WORD) */
	void qsort(WORD* output, INT size) ;
	void qsort(REAL* output, INT* index, INT size) ;
	void qsort_backward(REAL* output, INT* index, INT size) ;

	/* performs a inplace unique of a WORD vector using quicksort 
	 * returns the new number of elements */
	INT unique(WORD* output, INT size) 
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
	INT fast_find(WORD* output, INT size, WORD elem) ;

	/** calculates ROC into (fp,tp)
	 * from output and label of length size 
	 * returns index with smallest error=fp+fn
	 */
	INT calcroc(REAL* fp, REAL* tp, REAL* output, INT* label, INT& size, INT& possize, INT& negsize, REAL& tresh, FILE* rocfile);
	//@}

	/// returns the mutual information of p which is given in logspace
	/// where p,q are given in logspace
	double mutual_info(REAL* p1, REAL* p2, INT len);

	/// returns the relative entropy H(P||Q), 
	/// where p,q are given in logspace
	double relative_entropy(REAL* p, REAL* q, INT len);

	/// returns entropy of p which is given in logspace
	double entropy(REAL* p, INT len);

	/**@name summing functions */
	//@{ 
	/** sum logarithmic probabilities.
	 * Probability measures are summed up but are now given in logspace where
	 * direct summation of exp(operand) is not possible due to numerical problems, i.e. eg. exp(-1000)=0. Therefore
	 * we do log( exp(a) + exp(b)) = a + log (1 + exp (b-a)) where a is max(p,q) and b min(p,q). */
#ifdef USE_LOGCACHE
	inline REAL logarithmic_sum(REAL p, REAL q)
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
	void init_log_table();
	
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

	inline REAL logarithmic_sum(REAL p, REAL q)
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
	inline REAL logarithmic_sum_array(REAL *p, INT len)
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
	REAL* logtable;	
#endif
	static CHAR rand_state[256];
};

#endif

extern CMath math;
