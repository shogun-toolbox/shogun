#ifndef __MATH_H_
#define __MATH_H_

#include "lib/common.h"

#include <math.h>
#include <stdio.h>

//define finite for win32
#ifdef _WIN32
	#include <float.h>
	#define finite _finite
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

	/**@name min/max functions.
	*/
	//@{
	///return the maximum of two integers
	inline int max(int a, int b) 
	{
	  return ((a)>=(b))?(a):(b) ;
	} ;

	///return the minimum of two integers
	inline int min(int a, int b) 
	{
	  return ((a)<=(b))?(a):(b) ;
	}
	///return the maximum of two floating point numbers
	inline REAL max(REAL a, REAL b) 
	{
	  return ((a)>=(b))?(a):(b) ;
	} ;

	///return the minimum of two floating point numbers
	inline REAL min(REAL a, REAL b) 
	{
	  return ((a)<=(b))?(a):(b) ;
	}
	
	///return the maximum of two long
	inline long max(long a, long b) 
	{
	  return ((a)>=(b))?(a):(b) ;
	} ;

	///return the minimum of two longs
	inline long min(long a, long b) 
	{
	  return ((a)<=(b))?(a):(b) ;
	}
	
	//@}

#ifdef _WIN32
	inline int isnan(REAL a)
	{
	    return(_isnan(a));
	}
#else
	inline int isnan(REAL a)
	{
	    return(isnan(a));
	}
#endif

	/**@name misc functions */
	//@{

	/// crc32
	unsigned int crc32(unsigned char *data, int len);

	/// signum of int a 
	inline int sign(int a)
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
	/// swap integers a and b
	inline void swap(int & a,int &b)
	  {
	    int c=a ;
	    a=b; b=c ;
	  } 

	/** performs a bubblesort on a given matrix a.
	 * it is sorted from in ascending order from top to bottom
	 * and left to right */
	void sort(int *a, int cols, int sort_col=0) ;
	
	/** performs a quicksort on an array output of length size
	 * it is sorted from in ascending order from top to bottom
	 * and left to right */
	void qsort(REAL* output, int size) ;

	/** calculates ROC into (fp,tp)
	 * from output and label of length size 
	 * returns index with smallest error=fp+fn
	 */
	int calcroc(REAL* fp, REAL* tp, REAL* output, int* label, int size, int& possize, int& negsize, FILE* rocfile);
	//@}

	/**@name summing functions */
	//@{ 
	/** sum logarithmic probabilities.
	 * Probability measures are summed up but are now given in logspace where
	 * direct summation of exp(operand) is not possible due to numerical problems, i.e. eg. exp(-1000)=0. Therefore
	 * we do log( exp(a) + exp(b)) = a + log (1 + exp (b-a)) where a is max(p,q) and b min(p,q). */
#ifndef NO_LOG_CACHE
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
			return p;
		}
		else 
			return q;
	}

	///init log table of form log(1+exp(x))
	void init_log_table();
	
	/// determine int x for that log(1+exp(-x)) == 0
	static int determine_logrange();

	/// determine accuracy, such that the thing fits into MAX_LOG_TABLE_SIZE, needs logrange as argument
	static int determine_logaccuracy(int range);
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
					if (diff > 73)
						return p;
					else
						return  p + log(1+exp(-diff));
				}
				else			//p<=q
				{
					if (-diff > 73)
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
	inline REAL logarithmic_sum_array(REAL *p, int len)
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
		for (register int j=0; j < len>>1; j++)
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
#ifndef NO_LOG_CACHE	
	/// range for logtable: log(1+exp(x))  -LOGRANGE <= x <= 0
	static int LOGRANGE;
	
	/// number of steps per integer
	static int LOGACCURACY;
	//@}
protected:
	///table with log-values
	REAL* logtable;	
#endif
};

#endif

extern CMath math;
