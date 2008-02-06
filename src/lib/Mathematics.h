/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2008 Soeren Sonnenburg
 * Written (W) 1999-2008 Gunnar Raetsch
 * Written (W) 2007 Konrad Rieck
 * Copyright (C) 1999-2008 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef __MATHMATICS_H_
#define __MATHMATICS_H_

#include "lib/common.h"
#include "lib/io.h"
#include "base/SGObject.h"
#include "base/Parallel.h"

#include <math.h>
#include <stdio.h>
#include <float.h>
#include <pthread.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/time.h>
#include <time.h>

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

#ifndef NAN
#include <stdlib.h>
#define NAN (strtod("NAN",NULL))
#endif

#ifdef SUNOS
extern "C" int	finite(double);
#endif

/* Maximum stack size */
#define RADIX_STACK_SIZE	    512

/* Stack macros */
#define radix_push(a, n, i)	    sp->sa = a, sp->sn = n, (sp++)->si = i
#define radix_pop(a, n, i)	    a = (--sp)->sa, n = sp->sn, i = sp->si

/** Stack structure */
template <class T> struct radix_stack_t
{
	/** Pointer to pile */
	T *sa;
	/** Number of grams in pile */
	size_t sn;
	/** Byte in current focus */
	unsigned short si;
};

/** pair */
struct pair
{
	/** index 1 */
	int idx1;
	/** index 2 */
	int idx2;
};

/** thread qsort */
struct thread_qsort
{
	/** output */
	DREAL* output;
	/** index */
	INT* index;
	/** size */
	INT size;

	/** qsort threads */
	INT* qsort_threads;
	/** sort limit */
	INT sort_limit;
	/** number of threads */
	INT num_threads;
};

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

		///return the value clamped to interval [lb,ub]
		template <class T>
			static inline T clamp(T value, T lb, T ub) 
			{
				if (value<=lb)
					return lb;
				else if (value>=ub)
					return ub;
				else
					return value;
			}

		///return the maximum of two integers
		template <class T>
			static inline T abs(T a)
			{
				// can't be a>=0?(a):(-a), because compiler complains about
				// 'comparison always true' when T is unsigned
				if (a==0)
					return 0;
				else if (a>0)
					return a;
				else
					return -a;
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
			static inline T sign(T a)
			{
				if (a==0)
					return 0;
				else return (a<0) ? (-1) : (+1);
			}

		/// swap e.g. floats a and b
		template <class T>
			static inline void swap(T & a,T &b)
			{
				T c=a;
				a=b;
				b=c;
			}

		/// x^2
		template <class T>
			static inline T sq(T x)
			{
				return x*x;
			}

		/// x^0.5
		static inline SHORTREAL sqrt(SHORTREAL x)
		{
			return ::sqrtf(x);
		}

		/// x^0.5
		static inline DREAL sqrt(DREAL x)
		{
			return ::sqrt(x);
		}

		/// x^0.5
		static inline LONGREAL sqrt(LONGREAL x)
		{
			return ::sqrtl(x);
		}


		/// x^n
		static inline long double powl(long double x, long double n)
		{
			//fall back to double precision pow if powl is not
			//available
#ifdef HAVE_POWL
			return ::powl(x, n);
#else
			return ::pow(x, n);
#endif
		}

		static inline INT pow(INT x, INT n)
		{
			ASSERT(n>=0);
			INT result=1;
			while (n--)
				result*=x;

			return result;
		}

		static inline DREAL pow(DREAL x, DREAL n)
		{
			return ::pow(x, n);
		}

		static inline DREAL log10(DREAL v)
		{
			return ::log(v)/::log(10.0);
		}

#ifdef HAVE_LOG2
		static inline DREAL log2(DREAL v)
		{
			return ::log2(v);
		}
#else
		static inline DREAL log2(DREAL v)
		{
			return ::log(v)/::log(2.0);
		}
#endif //HAVE_LOG2

		static inline DREAL log(DREAL v)
		{
			return ::log(v);
		}

#ifdef HAVE_LAPACK
		/// return the pseudo inverse for matrix
		/// when matrix has shape (rows, cols) the pseudo inverse has (cols, rows)
		static DREAL* pinv(DREAL* matrix, INT rows, INT cols, DREAL* target=NULL);


		//C := alpha*op( A )*op( B ) + beta*C
		//op( X ) = X   or   op( X ) = X',
		static inline void dgemm(double alpha, const double* A, int rows, int cols, CBLAS_TRANSPOSE transposeA,
				double *B, int cols_B, CBLAS_TRANSPOSE transposeB,
				double beta, double *C)
		{
			cblas_dgemm(CblasColMajor, transposeA, transposeB, rows, cols, cols_B,
					alpha, A, cols, B, cols_B, beta, C, cols);
		}

		//y := alpha*A*x + beta*y,   or   y := alpha*A'*x + beta*y,
		static inline void dgemv(double alpha, const double *A, int rows, int cols, const enum CBLAS_TRANSPOSE transposeA,
				const double* X, double beta, double* Y)
		{
			cblas_dgemv(CblasColMajor, transposeA,
					rows, cols, alpha, A, cols,
					X, 1, beta, Y, 1);
		}
#endif

		static inline LONG factorial(INT n)
		{
			LONG res=1 ;
			for (int i=2; i<=n; i++)
				res*=i ;
			return res ;
		}

		static void init_random(UINT initseed=0)
		{
			if (initseed==0)
			{
				struct timeval tv;
				gettimeofday(&tv, NULL);
				seed=(UINT) (4223517*getpid()*tv.tv_sec*tv.tv_usec);
			}
			else
				seed=initseed;
#if !defined(CYGWIN) && !defined(__INTERIX)
			//seed=42;
			//SG_SPRINT("initializing random number generator with %d\n", seed);
			initstate(seed, CMath::rand_state, sizeof(CMath::rand_state));
#endif
		}

		static inline LONG random()
		{
#if defined(CYGWIN) || defined(__INTERIX)
			return rand();
#else
			return ::random();
#endif
		}

		static inline INT random(INT min_value, INT max_value)
		{
			INT ret = min_value + (INT) ((max_value-min_value+1) * (random() / (RAND_MAX+1.0)));
			ASSERT(ret >= min_value && ret<=max_value);
			return ret ;
		}

		static inline SHORTREAL random(SHORTREAL min_value, SHORTREAL max_value)
		{
			SHORTREAL ret = min_value + ((max_value-min_value) * (random() / (1.0*RAND_MAX)));

			if (ret<min_value || ret>max_value)
				SG_SPRINT("min_value:%10.10f value: %10.10f max_value:%10.10f", min_value, ret, max_value);
			ASSERT(ret >= min_value && ret<=max_value);
			return ret;
		}

		static inline DREAL random(DREAL min_value, DREAL max_value)
		{
			DREAL ret = min_value + ((max_value-min_value) * (random() / (1.0*RAND_MAX)));

			if (ret<min_value || ret>max_value)
				SG_SPRINT("min_value:%10.10f value: %10.10f max_value:%10.10f", min_value, ret, max_value);
			ASSERT(ret >= min_value && ret<=max_value);
			return ret;
		}

		template <class T>
			static void fill_vector(T* vec, INT len, T value)
			{
				for (INT i=0; i<len; i++)
					vec[i]=value;
			}
		template <class T>
			static void range_fill_vector(T* vec, INT len, T start=0)
			{
				for (INT i=0; i<len; i++)
					vec[i]=i+start;
			}

		template <class T>
			static void random_vector(T* vec, INT len, T min_value, T max_value)
			{
				for (INT i=0; i<len; i++)
					vec[i]=CMath::random(min_value, max_value);
			}

		static inline INT* randperm(INT n)
		{
			INT* perm = new INT[n];

			if (!perm)
				return NULL;
			for (INT i = 0; i < n; i++)
				perm[i] = i;
			for (INT i = 0; i < n; i++)
				swap(perm[random(0, n - 1)], perm[i]);
			return perm;
		}

		static inline LONG nchoosek(INT n, INT k)
		{
			long res=1 ;

			for (INT i=n-k+1; i<=n; i++)
				res*=i ;

			return res/factorial(k) ;
		}

		/// x=x+alpha*y
		template <class T>
			static inline void vec1_plus_scalar_times_vec2(T* vec1,
					T scalar, const T* vec2, INT n)
			{
				for (INT i=0; i<n; i++)
					vec1[i]+=scalar*vec2[i];
			}

		/// compute dot product between v1 and v2 (blas optimized)
		static inline DREAL dot(const DREAL* v1, const DREAL* v2, INT n)
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

		/// compute dot product between v1 and v2 (blas optimized)
		static inline SHORTREAL dot(const SHORTREAL* v1, const SHORTREAL* v2, INT n)
		{
			DREAL r=0;
#ifdef HAVE_LAPACK
			INT skip=1;
			r = cblas_sdot(n, v1, skip, v2, skip);
#else
			for (INT i=0; i<n; i++)
				r+=v1[i]*v2[i];
#endif
			return r;
		}

		/// target=alpha*vec1 + beta*vec2
		template <class T>
			static inline void add(T* target, T alpha, const T* v1, T beta, const T* v2, INT len)
			{
				for (INT i=0; i<len; i++)
					target[i]=alpha*v1[i]+beta*v2[i];
			}

		/// add scalar to vector inplace
		template <class T>
			static inline void add_scalar(T alpha, T* vec, INT len)
			{
				for (INT i=0; i<len; i++)
					vec[i]+=alpha;
			}

		/// scale vector inplace
		template <class T>
			static inline void scale_vector(T alpha, T* vec, INT len)
			{
				for (INT i=0; i<len; i++)
					vec[i]*=alpha;
			}

		/// return sum(vec)
		template <class T>
			static inline T sum(T* vec, INT len)
			{
				T result=0;
				for (INT i=0; i<len; i++)
					result+=vec[i];

				return result;
			}

		/// return max(vec)
		template <class T>
			static inline T max(T* vec, INT len)
			{
				ASSERT(len>0);
				T maxv=vec[0];

				for (INT i=1; i<len; i++)
					maxv=CMath::max(vec[i], maxv);

				return maxv;
			}

		/// return sum(abs(vec))
		template <class T>
			static inline T sum_abs(T* vec, INT len)
			{
				T result=0;
				for (INT i=0; i<len; i++)
					result+=CMath::abs(vec[i]);

				return result;
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
		 * it is sorted in ascending order from top to bottom
		 * and left to right */
		static void sort(INT *a, INT cols, INT sort_col=0) ;
		static void sort(DREAL *a, INT*idx, INT N) ;

		/*
		 * Inline function to extract the byte at position p (from left)
		 * of an 64 bit integer. The function is somewhat identical to 
		 * accessing an array of characters via [].
		 */

		/** performs a in-place radix sort in ascending order */
		template <class T>
			inline static void radix_sort(T* array, size_t size)
			{
				radix_sort_helper(array,size,0);
			}

		template <class T>
			static inline BYTE byte(T word, unsigned short p)
			{
				return (word >> (sizeof(T)-p-1) * 8) & 0xff;
			}

		template <class T>
			static void radix_sort_helper(T* array, size_t size, unsigned short i)
			{
				static size_t count[256], nc, cmin;
				T *ak;
				BYTE c=0;
				radix_stack_t<T> s[RADIX_STACK_SIZE], *sp, *olds, *bigs;
				T *an, *aj, *pile[256];
				size_t *cp, cmax;

				/* Push initial array to stack */
				sp = s;
				radix_push(array, size, i);

				/* Loop until all digits have been sorted */
				while (sp>s) {
					radix_pop(array, size, i);
					an = array + size;

					/* Make character histogram */
					if (nc == 0) {
						cmin = 0xff;
						for (ak = array; ak < an; ak++) {
							c = byte(*ak, i);
							count[c]++;
							if (count[c] == 1) {
								/* Determine smallest character */
								if (c < cmin)
									cmin = c;
								nc++;
							}
						}

						/* Sort recursively if stack size too small */
						if (sp + nc > s + RADIX_STACK_SIZE) {
							radix_sort_helper(array, size, i);
							continue;
						}
					}

					/*
					 * Set pile[]; push incompletely sorted bins onto stack.
					 * pile[] = pointers to last out-of-place element in bins.
					 * Before permuting: pile[c-1] + count[c] = pile[c];
					 * during deal: pile[c] counts down to pile[c-1].
					 */
					olds = bigs = sp;
					cmax = 2;
					ak = array;
					pile[0xff] = an;
					for (cp = count + cmin; nc > 0; cp++) {
						/* Find next non-empty pile */
						while (*cp == 0)
							cp++;
						/* Pile with several entries */
						if (*cp > 1) {
							/* Determine biggest pile */
							if (*cp > cmax) {
								cmax = *cp;
								bigs = sp;
							}

							if (i < sizeof(T)-1)
								radix_push(ak, *cp, (unsigned short) (i + 1));
						}
						pile[cp - count] = ak += *cp;
						nc--;
					}

					/* Play it safe -- biggest bin last. */
					swap(*olds, *bigs);

					/*
					 * Permute misplacements home. Already home: everything
					 * before aj, and in pile[c], items from pile[c] on.
					 * Inner loop:
					 *      r = next element to put in place;
					 *      ak = pile[r[i]] = location to put the next element.
					 *      aj = bottom of 1st disordered bin.
					 * Outer loop:
					 *      Once the 1st disordered bin is done, ie. aj >= ak,
					 *      aj<-aj + count[c] connects the bins in array linked list;
					 *      reset count[c].
					 */
					aj = array;
					while (aj<an)
					{
						T r;

						for (r = *aj; aj < (ak = --pile[c = byte(r, i)]);)
							swap(*ak, r);

						*aj = r;
						aj += count[c];
						count[c] = 0;
					}
				}
			}
		/** performs insertion sort of an array output of length size
		 * it is sorted from in ascending (for type T) */
		template <class T>
			static void insertion_sort(T* output, INT size)
			{
				for (INT i=0; i<size-1; i++)
				{
					INT j=i-1;
					T value=output[i];
					while (j >= 0 && output[j] > value)
					{
						output[j+1] = output[j];
						j--;
					}
					output[j+1]=value;
				}
			}


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
					return;
				}
				T split=output[random(0,size-1)];

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

		/// display vector (useful for debugging)
		template <class T> static void display_vector(T* vector, INT n, const char* name="vector");

		/// display matrix (useful for debugging)
		template <class T> static void display_matrix(T* matrix, INT rows, INT cols, const char* name="matrix");

		/** performs a quicksort on an array output of length size
		 * it is sorted in ascending order 
		 * (for type T1) and returns the index (type T2)
		 * matlab alike [sorted,index]=sort(output) 
		 */
		template <class T1,class T2>
			static void qsort_index(T1* output, T2* index, UINT size);

		template <class T1,class T2>
			static void* parallel_qsort_index(void* p);

		/** performs a quicksort on an array output of length size
		 * it is sorted in ascending order
		 * (for type T1) and returns the index (type T2)
		 * matlab alike [sorted,index]=sort(output) 
		 */
		template <class T1,class T2>
			static void qsort_backward_index(T1* output, T2* index, INT size);

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
				qsort(output, size);
				INT i,j=0 ;
				for (i=0; i<size; i++)
					if (i==0 || output[i]!=output[i-1])
						output[j++]=output[i];
				return j ;
			}

		/* finds an element in a sorted array via binary search
		 * returns -1 if not found */
		template <class T>
			static INT binary_search_helper(T* output, INT size, T elem)
			{
				INT start=0;
				INT end=size-1;

				if (size<1)
					return -1;

				while (start<end)
				{
					INT middle=(start+end)/2; 

					if (output[middle]>elem)
						end=middle-1;
					else if (output[middle]<elem)
						start=middle+1;
					else
						return middle;
				}

				if (output[start]==elem)
					return start;
				else
					return -1;
			}

		/* finds an element in a sorted array via binary search
		 *     * returns -1 if not found */
		template <class T>
			static inline INT binary_search(T* output, INT size, T elem)
			{
				INT ind = binary_search_helper(output, size, elem);
				if (ind >= 0 && output[ind] == elem)
					return ind;
				return -1;
			}

		/* finds an element in a sorted array via binary search 
		 * if it exists, else the index the largest smaller element
		 * is returned
		 * note: a successor is not mandatory */
		template <class T>
			static INT binary_search_max_lower_equal(T* output, INT size, T elem)
			{
				INT ind = binary_search_helper(output, size, elem);

				if (output[ind]<=elem)
					return ind;
				if (ind>0 && output[ind-1] <= elem)
					return ind-1;
				return -1;
			}

		/// align two sequences seq1 & seq2 of length l1 and l2 using gapCost
		/// return alignment cost
		static DREAL Align(CHAR * seq1, CHAR* seq2, INT l1, INT l2, DREAL gapCost);

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
		/**
		 * sum logarithmic probabilities.
		 * Probability measures are summed up but are now given in logspace
		 * where direct summation of exp(operand) is not possible due to
		 * numerical problems, i.e. eg. exp(-1000)=0. Therefore we do
		 * log( exp(a) + exp(b)) = a + log (1 + exp (b-a)) where a = max(p,q)
		 * and b min(p,q).
		 */
#ifdef USE_LOGCACHE
		static inline DREAL logarithmic_sum(DREAL p, DREAL q)
		{
			DREAL diff;

			if (!finite(p))
				return q;
			if (!finite(q))
			{
				SG_SWARNING(("INVALID second operand to logsum(%f,%f) expect undefined results\n", p, q);
						return NAN;
						}
						diff = p - q;
						if (diff > 0)
						return diff > LOGRANGE? p : p + logtable[(int)(diff * LOGACCURACY)];
						return -diff > LOGRANGE? q : q + logtable[(int)(-diff * LOGACCURACY)];
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
						DREAL diff;

						if (!finite(p))
							return q;
						if (!finite(q))
							return p;
						diff = p - q;
						if (diff > 0)
							return diff > LOGRANGE? p : p + log(1 + exp(-diff));
						return -diff > LOGRANGE? q : q + log(1 + exp(diff));
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
				static const DREAL ALMOST_INFTY;

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
				static CHAR* rand_state;
};

	template <class T1,class T2>
void* CMath::parallel_qsort_index(void* p)
{
	struct thread_qsort* ps=(thread_qsort*) p;
	T1* output=ps->output;
	T2* index=ps->index;
	INT size=ps->size;
	INT* qsort_threads=ps->qsort_threads;
	INT sort_limit=ps->sort_limit;
	INT num_threads=ps->num_threads;

	if (size==2)
	{
		if (output[0] > output [1])
		{
			swap(output[0], output[1]);
			swap(index[0], index[1]);
		}
		return NULL;
	}
	/*double split=output[(((uint64_t) size)*rand())/(((uint64_t)RAND_MAX)+1)];*/
	double split=output[size/2];

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
			swap(output[left], output[right]);
			swap(index[left], index[right]);
			left++;
			right--;
		}
	}
	bool lthread_start=false;
	bool rthread_start=false;
	pthread_t lthread;
	pthread_t rthread;
	struct thread_qsort t1;
	struct thread_qsort t2;

	if (right+1> 1 && (right+1< sort_limit || *qsort_threads >= num_threads-1))
		qsort_index(output,index,right+1);
	else if (right+1> 1)
	{
		*qsort_threads++;
		lthread_start=true;
		t1.output=output;
		t1.index=index;
		t1.size=right+1;
		if (pthread_create(&lthread, NULL, parallel_qsort_index<T1,T2>, (void*) &t1) != 0)
		{
			lthread_start=false;
			*qsort_threads--;
			qsort_index(output,index,right+1);
		}
	}


	if (size-left> 1 && (size-left< sort_limit || *qsort_threads >= num_threads-1))
		qsort_index(&output[left],&index[left], size-left);
	else if (size-left> 1)
	{
		*qsort_threads++;
		rthread_start=true;
		t2.output=&output[left];
		t2.index=&index[left];
		t2.size=size-left;
		if (pthread_create(&rthread, NULL, parallel_qsort_index<T1,T2>, (void*)&t2) != 0)
		{
			rthread_start=false;
			*qsort_threads--;
			qsort_index(&output[left],&index[left], size-left);
		}
	}

	if (lthread_start)
	{
		pthread_join(lthread, NULL);
		*qsort_threads--;
	}

	if (rthread_start)
	{
		pthread_join(rthread, NULL);
		*qsort_threads--;
	}

	return NULL;
}


//implementations of template functions
	template <class T1,class T2>
void CMath::qsort_index(T1* output, T2* index, UINT size)
{
	if (size==2)
	{
		if (output[0] > output [1])
		{
			swap(output[0],output[1]);
			swap(index[0],index[1]);
		}
		return;
	}
	T1 split=output[(size*rand())/(RAND_MAX+1)];

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
		qsort_index(output,index,right+1);

	if (size-left> 1)
		qsort_index(&output[left],&index[left], size-left);
}

	template <class T1,class T2>
void CMath::qsort_backward_index(T1* output, T2* index, INT size)
{
	if (size==2)
	{
		if (output[0] < output [1])
		{
			swap(output[0],output[1]);
			swap(index[0],index[1]);
		}
		return;
	}

	T1 split=output[(size*rand())/(RAND_MAX+1)];

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
		qsort_backward_index(output,index,right+1);

	if (size-left> 1)
		qsort_backward_index(&output[left],&index[left], size-left);
}

	template <class T> 
void CMath::nmin(DREAL* output, T* index, INT size, INT n)
{
	if (6*n*size<13*size*CMath::log(size))
		for (INT i=0; i<n; i++)
			min(&output[i], &index[i], size-i) ;
	else
		qsort_index(output, index, size) ;
}

/* move the smallest entry in the array to the beginning */
	template <class T>
void CMath::min(DREAL* output, T* index, INT size)
{
	if (size<=1)
		return;
	DREAL min_elem=output[0];
	INT min_index=0;
	for (INT i=1; i<size; i++)
	{
		if (output[i]<min_elem)
		{
			min_index=i;
			min_elem=output[i];
		}
	}
	swap(output[0], output[min_index]);
	swap(index[0], index[min_index]);
}
#endif
