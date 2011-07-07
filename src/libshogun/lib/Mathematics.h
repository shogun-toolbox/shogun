/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011 Siddharth Kherada
 * Written (W) 2011 Justin Patera
 * Written (W) 2011 Alesis Novik
 * Written (W) 1999-2009 Soeren Sonnenburg
 * Written (W) 1999-2008 Gunnar Raetsch
 * Written (W) 2007 Konrad Rieck
 * Copyright (C) 1999-2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef __MATHEMATICS_H_
#define __MATHEMATICS_H_

#include "lib/common.h"
#include "lib/io.h"
#include "lib/lapack.h"
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

#ifdef SUNOS
#include <ieeefp.h>
#endif

/// workaround for log2 being a define on cygwin
#ifdef log2
#define cygwin_log2 log2
#undef log2
#endif



/// workaround a bug in std cmath
#ifdef _GLIBCXX_CMATH
#if _GLIBCXX_USE_C99_MATH
#if !_GLIBCXX_USE_C99_FP_MACROS_DYNAMIC

  /// Function template definitions [8.16.3].
  using std::signbit;

  using std::fpclassify;

  using std::isfinite;
  using std::isinf;
  using std::isnan;
  using std::isnormal;

  using std::isgreater;
  using std::isgreaterequal;
  using std::isless;
  using std::islessequal;
  using std::islessgreater;
  using std::isunordered;
#endif
#endif
#endif
/// end of workaround a bug in std cmath

#ifdef _WIN32
#ifndef isnan
#define isnan _isnan
#endif

#ifndef isfinite
#define isfinite _isfinite
#endif
#endif //_WIN32

#ifndef NAN
#include <stdlib.h>
#define NAN (strtod("NAN",NULL))
#endif

/* Size of RNG seed */
#define RNG_SEED_SIZE 256

/* Maximum stack size */
#define RADIX_STACK_SIZE	    512

/* Stack macros */
#define radix_push(a, n, i)	    sp->sa = a, sp->sn = n, (sp++)->si = i
#define radix_pop(a, n, i)	    a = (--sp)->sa, n = sp->sn, i = sp->si

#ifndef DOXYGEN_SHOULD_SKIP_THIS
/** Stack structure */
template <class T> struct radix_stack_t
{
	/** Pointer to pile */
	T *sa;
	/** Number of grams in pile */
	size_t sn;
	/** Byte in current focus */
	uint16_t si;
};

///** pair */

/** thread qsort */
template <class T1, class T2> struct thread_qsort
{
	/** output */
	T1* output;
	/** index */
	T2* index;
	/** size */
	uint32_t size;

	/** qsort threads */
	int32_t* qsort_threads;
	/** sort limit */
	int32_t sort_limit;
	/** number of threads */
	int32_t num_threads;
};
#endif // DOXYGEN_SHOULD_SKIP_THIS

namespace shogun
{
/** @brief Class which collects generic mathematical functions
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
				return (a<=b) ? a : b;
			}

		///return the maximum of two integers
		template <class T>
			static inline T max(T a, T b) 
			{
				return (a>=b) ? a : b;
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

		///return the absolute value of a number
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

		static inline float64_t round(float64_t d)
		{
			return ::floor(d+0.5);
		}

		static inline float64_t floor(float64_t d)
		{
			return ::floor(d);
		}

		static inline float64_t ceil(float64_t d)
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
			static inline void swap(T &a,T &b)
			{
				T c=a;
				a=b;
				b=c;
			}

		/** resize array from old_size to new_size (keeping as much array
		 * content as possible intact)
		 */
		template <class T>
			static inline void resize(T* &data, int64_t old_size, int64_t new_size)
			{
				if (old_size==new_size)
					return;
				T* new_data = new T[new_size];
				for (int64_t i=0; i<old_size && i<new_size; i++)
					new_data[i]=data[i];
				delete[] data;
				data=new_data;
			}

		/// || x ||_2
		template <class T>
			static inline T twonorm(T* x, int32_t len)
			{
				float64_t result=0;
				for (int32_t i=0; i<len; i++)
					result+=x[i]*x[i];

				return CMath::sqrt(result);
			}

		/// || x ||_q^q
		template <class T>
			static inline T qsq(T* x, int32_t len, float64_t q)
			{
				float64_t result=0;
				for (int32_t i=0; i<len; i++)
					result+=CMath::pow(x[i], q);

				return result;
			}

		/// || x ||_q
		template <class T>
			static inline T qnorm(T* x, int32_t len, float64_t q)
			{
				ASSERT(q!=0);
				return CMath::pow(qsq(x, len, q), 1/q);
			}

		/// x^2
		template <class T>
			static inline T sq(T x)
			{
				return x*x;
			}

		/// x^0.5
		static inline float32_t sqrt(float32_t x)
		{
			return ::sqrtf(x);
		}

		/// x^0.5
		static inline float64_t sqrt(float64_t x)
		{
			return ::sqrt(x);
		}

		/// x^0.5
		static inline floatmax_t sqrt(floatmax_t x)
		{
			//fall back to double precision sqrt if sqrtl is not
			//available
#ifdef HAVE_SQRTL
			return ::sqrtl(x);
#else
			return ::sqrt(x);
#endif
		}


		/// x^n
		static inline floatmax_t powl(floatmax_t x, floatmax_t n)
		{
			//fall back to double precision pow if powl is not
			//available
#ifdef HAVE_POWL
			return ::powl((long double) x, (long double) n);
#else
			return ::pow((double) x, (double) n);
#endif
		}

		static inline int32_t pow(int32_t x, int32_t n)
		{
			ASSERT(n>=0);
			int32_t result=1;
			while (n--)
				result*=x;

			return result;
		}

		static inline float64_t pow(float64_t x, int32_t n)
		{
			ASSERT(n>=0);
			float64_t result=1;
			while (n--)
				result*=x;

			return result;
		}

		static inline float64_t pow(float64_t x, float64_t n)
		{
			return ::pow((double) x, (double) n);
		}

		static inline float64_t exp(float64_t x)
		{
			return ::exp((double) x);
		}

		static inline float64_t log10(float64_t v)
		{
			return ::log(v)/::log(10.0);
		}

		static inline float64_t log2(float64_t v)
		{
#ifdef HAVE_LOG2
			return ::log2(v);
#else
			return ::log(v)/::log(2.0);
#endif //HAVE_LOG2
		}

		static inline float64_t log(float64_t v)
		{
			return ::log(v);
		}

		static float64_t area_under_curve(float64_t* xy, int32_t len, bool reversed)
		{
			ASSERT(len>0 && xy);

			float64_t area = 0.0;
			
			if (!reversed)
			{
				for (int i=1; i<len; i++)
					area += 0.5*(xy[2*i]-xy[2*(i-1)])*(xy[2*i+1]+xy[2*(i-1)+1]);
			} 
			else
			{
				for (int i=1; i<len; i++)
					area += 0.5*(xy[2*i+1]-xy[2*(i-1)+1])*(xy[2*i]+xy[2*(i-1)]);	
			}

			return area;
		}

		template <class T>
		static void transpose_matrix(
			T*& matrix, int32_t& num_feat, int32_t& num_vec)
		{
			T* transposed=new T[num_vec*num_feat];
			for (int32_t i=0; i<num_vec; i++)
			{
				for (int32_t j=0; j<num_feat; j++)
					transposed[i+j*num_vec]=matrix[i*num_feat+j];
			}

			delete[] matrix;
			matrix=transposed;

			CMath::swap(num_feat, num_vec);
		}

#ifdef HAVE_LAPACK
		/// return the pseudo inverse for matrix
		/// when matrix has shape (rows, cols) the pseudo inverse has (cols, rows)
		static float64_t* pinv(
			float64_t* matrix, int32_t rows, int32_t cols,
			float64_t* target=NULL);


		//C := alpha*op( A )*op( B ) + beta*C
		//op( X ) = X   or   op( X ) = X',
		static inline void dgemm(
			double alpha, const double* A, int rows, int cols,
			CBLAS_TRANSPOSE transposeA, double *B, int cols_B,
			CBLAS_TRANSPOSE transposeB, double beta, double *C)
		{
			cblas_dgemm(CblasColMajor, transposeA, transposeB, rows, cols, cols_B,
					alpha, A, cols, B, cols_B, beta, C, cols);
		}

		//y := alpha*A*x + beta*y,   or   y := alpha*A'*x + beta*y,
		static inline void dgemv(
			double alpha, const double *A, int rows, int cols,
			const CBLAS_TRANSPOSE transposeA, const double* X, double beta,
			double* Y)
		{
			cblas_dgemv(CblasColMajor, transposeA,
					rows, cols, alpha, A, cols,
					X, 1, beta, Y, 1);
		}
#endif

		static inline int64_t factorial(int32_t n)
		{
			int64_t res=1;
			for (int i=2; i<=n; i++)
				res*=i ;
			return res ;
		}

		static void init_random(uint32_t initseed=0)
		{
			if (initseed==0)
			{
				struct timeval tv;
				gettimeofday(&tv, NULL);
				seed=(uint32_t) (4223517*getpid()*tv.tv_sec*tv.tv_usec);
			}
			else
				seed=initseed;
#if !defined(CYGWIN) && !defined(__INTERIX)
			//seed=42
			//SG_SPRINT("initializing random number generator with %d (seed size %d)\n", seed, RNG_SEED_SIZE);
			initstate(seed, CMath::rand_state, RNG_SEED_SIZE);
#endif
		}

		static inline int64_t random()
		{
#if defined(CYGWIN) || defined(__INTERIX)
			return rand();
#else
			return ::random();
#endif
		}

		static inline int32_t random(int32_t min_value, int32_t max_value)
		{
			int32_t ret = min_value + (int32_t) ((max_value-min_value+1) * (random() / (RAND_MAX+1.0)));
			ASSERT(ret>=min_value && ret<=max_value);
			return ret ;
		}

		static inline float32_t random(float32_t min_value, float32_t max_value)
		{
			float32_t ret = min_value + ((max_value-min_value) * (random() / (1.0*RAND_MAX)));

			if (ret<min_value || ret>max_value)
				SG_SPRINT("min_value:%10.10f value: %10.10f max_value:%10.10f", min_value, ret, max_value);
			ASSERT(ret>=min_value && ret<=max_value);
			return ret;
		}

		static inline float64_t random(float64_t min_value, float64_t max_value)
		{
			float64_t ret = min_value + ((max_value-min_value) * (random() / (1.0*RAND_MAX)));

			if (ret<min_value || ret>max_value)
				SG_SPRINT("min_value:%10.10f value: %10.10f max_value:%10.10f", min_value, ret, max_value);
			ASSERT(ret>=min_value && ret<=max_value);
			return ret;
		}

		/// Returns a Gaussian or Normal random number.
		/// Using the polar form of the Box-Muller transform.
		/// http://en.wikipedia.org/wiki/Box%E2%80%93Muller_transform#Polar_form
		static inline float32_t normal_random(float32_t mean, float32_t std_dev)
		{
			// sets up variables & makes sure rand_s.range == (0,1)
			float32_t ret;
			float32_t rand_u;
			float32_t rand_v;
			float32_t rand_s;
			do
			{
				rand_u = random(-1.0, 1.0);
				rand_v = random(-1.0, 1.0);
				rand_s = rand_u*rand_u + rand_v*rand_v;
			} while ((rand_s == 0) || (rand_s >= 1));

			// the meat & potatos, and then the mean & standard deviation shifting...
			ret = rand_u*sqrt(-2.0*log(rand_s)/rand_s);
			ret = std_dev*ret + mean;
			return ret;
		}

		/// Returns a Gaussian or Normal random number.
		/// Using the polar form of the Box-Muller transform.
		/// http://en.wikipedia.org/wiki/Box%E2%80%93Muller_transform#Polar_form
		static inline float64_t normal_random(float64_t mean, float64_t std_dev)
		{
			float64_t ret;
			float64_t rand_u;
			float64_t rand_v;
			float64_t rand_s;
			do
			{
				rand_u = random(-1.0, 1.0);
				rand_v = random(-1.0, 1.0);
				rand_s = rand_u*rand_u + rand_v*rand_v;
			} while ((rand_s == 0) || (rand_s >= 1));

			ret = rand_u*sqrt(-2.0*log(rand_s)/rand_s);
			ret = std_dev*ret + mean;
			return ret;
		}

		/// Convenience method for generating Standard Normal random numbers
		/// Float: Mean = 0 and Standard Deviation = 1
		static inline float32_t randn_float()
		{
			return normal_random(0.0, 1.0);
		}

		/// Convenience method for generating Standard Normal random numbers
		/// Double: Mean = 0 and Standard Deviation = 1
		static inline float64_t randn_double()
		{
			return normal_random(0.0, 1.0);
		}

		template <class T>
			static T* clone_vector(const T* vec, int32_t len)
			{
				T* result = new T[len];
				for (int32_t i=0; i<len; i++)
					result[i]=vec[i];

				return result;
			}
		template <class T>
			static void fill_vector(T* vec, int32_t len, T value)
			{
				for (int32_t i=0; i<len; i++)
					vec[i]=value;
			}
		template <class T>
			static void range_fill_vector(T* vec, int32_t len, T start=0)
			{
				for (int32_t i=0; i<len; i++)
					vec[i]=i+start;
			}

		template <class T>
			static void random_vector(T* vec, int32_t len, T min_value, T max_value)
			{
				for (int32_t i=0; i<len; i++)
					vec[i]=CMath::random(min_value, max_value);
			}

		static inline int32_t* randperm(int32_t n)
		{
			int32_t* perm = new int32_t[n];

			if (!perm)
				return NULL;
			for (int32_t i = 0; i < n; i++)
				perm[i] = i;
			for (int32_t i = 0; i < n; i++)
				swap(perm[random(0, n - 1)], perm[i]);
			return perm;
		}

		static inline int64_t nchoosek(int32_t n, int32_t k)
		{
			int64_t res=1;

			for (int32_t i=n-k+1; i<=n; i++)
				res*=i;

			return res/factorial(k);
		}

		/// x=x+alpha*y
		template <class T>
			static inline void vec1_plus_scalar_times_vec2(T* vec1,
					T scalar, const T* vec2, int32_t n)
			{
				for (int32_t i=0; i<n; i++)
					vec1[i]+=scalar*vec2[i];
			}

		/// compute dot product between v1 and v2 (blas optimized)
		static inline float64_t dot(const bool* v1, const bool* v2, int32_t n)
		{
			float64_t r=0;
			for (int32_t i=0; i<n; i++)
				r+=((v1[i]) ? 1 : 0) * ((v2[i]) ? 1 : 0);
			return r;
		}

		/// compute dot product between v1 and v2 (blas optimized)
		static inline floatmax_t dot(const floatmax_t* v1, const floatmax_t* v2, int32_t n)
		{
			floatmax_t r=0;
			for (int32_t i=0; i<n; i++)
				r+=v1[i]*v2[i];
			return r;
		}
		

		/// compute dot product between v1 and v2 (blas optimized)
		static inline float64_t dot(const float64_t* v1, const float64_t* v2, int32_t n)
		{
			float64_t r=0;
#ifdef HAVE_LAPACK
			int32_t skip=1;
			r = cblas_ddot(n, v1, skip, v2, skip);
#else
			for (int32_t i=0; i<n; i++)
				r+=v1[i]*v2[i];
#endif
			return r;
		}
		
		/// compute dot product between v1 and v2 (blas optimized)
		static inline float32_t dot(
			const float32_t* v1, const float32_t* v2, int32_t n)
		{
			float64_t r=0;
#ifdef HAVE_LAPACK
			int32_t skip=1;
			r = cblas_sdot(n, v1, skip, v2, skip);
#else
			for (int32_t i=0; i<n; i++)
				r+=v1[i]*v2[i];
#endif
			return r;
		}

		/// compute dot product between v1 and v2 (for 64bit unsigned ints)
		static inline float64_t dot(
			const uint64_t* v1, const uint64_t* v2, int32_t n)
		{
			float64_t r=0;
			for (int32_t i=0; i<n; i++)
				r+=((float64_t) v1[i])*v2[i];

			return r;
		}
		/// compute dot product between v1 and v2 (for 64bit ints)
		static inline float64_t dot(
			const int64_t* v1, const int64_t* v2, int32_t n)
		{
			float64_t r=0;
			for (int32_t i=0; i<n; i++)
				r+=((float64_t) v1[i])*v2[i];

			return r;
		}

		/// compute dot product between v1 and v2 (for 32bit ints)
		static inline float64_t dot(
			const int32_t* v1, const int32_t* v2, int32_t n)
		{
			float64_t r=0;
			for (int32_t i=0; i<n; i++)
				r+=((float64_t) v1[i])*v2[i];

			return r;
		}

		/// compute dot product between v1 and v2 (for 32bit unsigned ints)
		static inline float64_t dot(
			const uint32_t* v1, const uint32_t* v2, int32_t n)
		{
			float64_t r=0;
			for (int32_t i=0; i<n; i++)
				r+=((float64_t) v1[i])*v2[i];

			return r;
		}

		/// compute dot product between v1 and v2 (for 16bit unsigned ints)
		static inline float64_t dot(
			const uint16_t* v1, const uint16_t* v2, int32_t n)
		{
			float64_t r=0;
			for (int32_t i=0; i<n; i++)
				r+=((float64_t) v1[i])*v2[i];

			return r;
		}

		/// compute dot product between v1 and v2 (for 16bit unsigned ints)
		static inline float64_t dot(
			const int16_t* v1, const int16_t* v2, int32_t n)
		{
			float64_t r=0;
			for (int32_t i=0; i<n; i++)
				r+=((float64_t) v1[i])*v2[i];

			return r;
		}

		/// compute dot product between v1 and v2 (for 8bit (un)signed ints)
		static inline float64_t dot(
			const char* v1, const char* v2, int32_t n)
		{
			float64_t r=0;
			for (int32_t i=0; i<n; i++)
				r+=((float64_t) v1[i])*v2[i];

			return r;
		}

		/// compute dot product between v1 and v2 (for 8bit (un)signed ints)
		static inline float64_t dot(
			const uint8_t* v1, const uint8_t* v2, int32_t n)
		{
			float64_t r=0;
			for (int32_t i=0; i<n; i++)
				r+=((float64_t) v1[i])*v2[i];

			return r;
		}

		/// compute dot product between v1 and v2 (for 8bit (un)signed ints)
		static inline float64_t dot(
			const int8_t* v1, const int8_t* v2, int32_t n)
		{
			float64_t r=0;
			for (int32_t i=0; i<n; i++)
				r+=((float64_t) v1[i])*v2[i];

			return r;
		}

		/// compute dot product between v1 and v2
		static inline float64_t dot(
			const float64_t* v1, const char* v2, int32_t n)
		{
			float64_t r=0;
			for (int32_t i=0; i<n; i++)
				r+=((float64_t) v1[i])*v2[i];

			return r;
		}

		/// compute vector multiplication
		template <class T>
			static inline void vector_multiply(
				T* target, const T* v1, const T* v2,int32_t len)
			{
				for (int32_t i=0; i<len; i++)
					target[i]=v1[i]*v2[i];
			}


		/// target=alpha*vec1 + beta*vec2
		template <class T>
			static inline void add(
				T* target, T alpha, const T* v1, T beta, const T* v2,
				int32_t len)
			{
				for (int32_t i=0; i<len; i++)
					target[i]=alpha*v1[i]+beta*v2[i];
			}

		/// add scalar to vector inplace
		template <class T>
			static inline void add_scalar(T alpha, T* vec, int32_t len)
			{
				for (int32_t i=0; i<len; i++)
					vec[i]+=alpha;
			}

		/// scale vector inplace
		template <class T>
			static inline void scale_vector(T alpha, T* vec, int32_t len)
			{
				for (int32_t i=0; i<len; i++)
					vec[i]*=alpha;
			}

		/// return sum(vec)
		template <class T>
			static inline T sum(T* vec, int32_t len)
			{
				T result=0;
				for (int32_t i=0; i<len; i++)
					result+=vec[i];

				return result;
			}

		/// return max(vec)
		template <class T>
			static inline T max(T* vec, int32_t len)
			{
				ASSERT(len>0);
				T maxv=vec[0];

				for (int32_t i=1; i<len; i++)
					maxv=CMath::max(vec[i], maxv);

				return maxv;
			}

		/// return sum(abs(vec))
		template <class T>
			static inline T sum_abs(T* vec, int32_t len)
			{
				T result=0;
				for (int32_t i=0; i<len; i++)
					result+=CMath::abs(vec[i]);

				return result;
			}

		/// return sum(abs(vec))
		template <class T>
			static inline bool fequal(T x, T y, float64_t precision=1e-6)
			{
				return CMath::abs(x-y)<precision;
			}

		static inline float64_t mean(float64_t* vec, int32_t len)
		{
			ASSERT(vec);
			ASSERT(len>0);

			float64_t mean=0;
			for (int32_t i=0; i<len; i++)
				mean+=vec[i];
			return mean/len;
		}

		static inline float64_t trace(
			float64_t* mat, int32_t cols, int32_t rows)
		{
			float64_t trace=0;
			for (int32_t i=0; i<rows; i++)
				trace+=mat[i*cols+i];
			return trace;
		}

		/** performs a bubblesort on a given matrix a.
		 * it is sorted in ascending order from top to bottom
		 * and left to right */
		static void sort(int32_t *a, int32_t cols, int32_t sort_col=0);
		static void sort(float64_t *a, int32_t*idx, int32_t N);

		/*
		 * Inline function to extract the byte at position p (from left)
		 * of an 64 bit integer. The function is somewhat identical to 
		 * accessing an array of characters via [].
		 */

		/** performs a in-place radix sort in ascending order */
		template <class T>
			inline static void radix_sort(T* array, int32_t size)
			{
				radix_sort_helper(array,size,0);
			}

		template <class T>
			static inline uint8_t byte(T word, uint16_t p)
			{
				return (word >> (sizeof(T)-p-1) * 8) & 0xff;
			}

		template <class T>
			static void radix_sort_helper(T* array, int32_t size, uint16_t i)
			{
				static size_t count[256], nc, cmin;
				T *ak;
				uint8_t c=0;
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
								radix_push(ak, *cp, (uint16_t) (i + 1));
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
			static void insertion_sort(T* output, int32_t size)
			{
				for (int32_t i=0; i<size-1; i++)
				{
					int32_t j=i-1;
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
		template <class T>
			static void qsort(T* output, int32_t size)
			{
				if (size==1)
					return;

				if (size==2)
				{
					if (output[0] > output [1])
						swap(output[0],output[1]);
					return;
				}
				//T split=output[random(0,size-1)];
				T split=output[size/2];

				int32_t left=0;
				int32_t right=size-1;

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

		/// display bits (useful for debugging)
		template <class T> static void display_bits(T word, int32_t width=8*sizeof(T))
		{
			ASSERT(width>=0);
			for (int i=0; i<width; i++)
			{
				T mask = ((T) 1)<<(sizeof(T)*8-1);
				while (mask)
				{
					if (mask & word)
						SG_SPRINT("1");
					else
						SG_SPRINT("0");

					mask>>=1;
				}
			}
		}

		/// display vector (useful for debugging)
		template <class T> static void display_vector(
			const T* vector, int32_t n, const char* name="vector");

		/// display matrix (useful for debugging)
		template <class T> static void display_matrix(
			const T* matrix, int32_t rows, int32_t cols, const char* name="matrix");

		/** performs a quicksort on an array output of length size
		 * it is sorted in ascending order 
		 * (for type T1) and returns the index (type T2)
		 * matlab alike [sorted,index]=sort(output) 
		 */
		template <class T1,class T2>
			static void qsort_index(T1* output, T2* index, uint32_t size);

		/** performs a quicksort on an array output of length size
		 * it is sorted in descending order
		 * (for type T1) and returns the index (type T2)
		 * matlab alike [sorted,index]=sort(output) 
		 */
		template <class T1,class T2>
			static void qsort_backward_index(
				T1* output, T2* index, int32_t size);

		/** performs a quicksort on an array output of length size
		 * it is sorted in ascending order
		 * (for type T1) and returns the index (type T2)
		 * matlab alike [sorted,index]=sort(output) 
		 *
		 * parallel version
		 */
		template <class T1,class T2>
			inline static void parallel_qsort_index(T1* output, T2* index, uint32_t size, int32_t n_threads, int32_t limit=262144)
			{
				int32_t n=0;
				thread_qsort<T1,T2> t;
				t.output=output;
				t.index=index;
				t.size=size;
				t.qsort_threads=&n;
				t.sort_limit=limit;
				t.num_threads=n_threads;
				parallel_qsort_index<T1,T2>(&t);
			}


		template <class T1,class T2>
			static void* parallel_qsort_index(void* p);


		/* finds the smallest element in output and puts that element as the 
		   first element  */
		template <class T>
			static void min(float64_t* output, T* index, int32_t size);

		/* finds the n smallest elements in output and puts these elements as the 
		   first n elements  */
		template <class T>
			static void nmin(
				float64_t* output, T* index, int32_t size, int32_t n);

		/* performs a inplace unique of a vector of type T using quicksort
		 * returns the new number of elements */
		template <class T>
			static int32_t unique(T* output, int32_t size)
			{
				qsort(output, size);
				int32_t i,j=0 ;
				for (i=0; i<size; i++)
					if (i==0 || output[i]!=output[i-1])
						output[j++]=output[i];
				return j ;
			}

		/** compute eigenvalues and eigenvectors of symmetric matrix
		 *
		 * @param matrix is overwritten and contains n orthonormal eigenvectors
		 * @return eigenvalues (array of length n, to be deleted[])
		 * */
		static double* compute_eigenvectors(double* matrix, int n, int m)
		{
#ifdef HAVE_LAPACK
			ASSERT(n == m);

			char V='V';
			char U='U';
			int info;
			int ord=n;
			int lda=n;
			double* eigenvalues=new float64_t[n+1];
			fill_vector(eigenvalues, n+1, 0.0);

			// lapack sym matrix eigenvalues+vectors
			wrap_dsyev(V, U,  ord, matrix, lda,
					eigenvalues, &info);

			if (info!=0)
				SG_SERROR("DSYEV failed with code %d\n", info);

			return eigenvalues;
#else
			SG_SERROR("Function not available - Lapack/Atlas not enabled at compile time!\n");
			return NULL;
#endif
		}

		/* Sums up all rows of a matrix and returns the resulting rowvector */
		template <class T>
			static T* get_row_sum(T* matrix, int32_t m, int32_t n)
			{
				T* rowsums=new T[n];
				fill_vector(rowsums, n, (T) 0);

				for (int32_t i=0; i<n; i++)
				{
					for (int32_t j=0; j<m; j++)
						rowsums[i]+=matrix[j+int64_t(i)*m];
				}
				return rowsums;
			}

		/* Sums up all columns of a matrix and returns the resulting columnvector */
		template <class T>
			static T* get_column_sum(T* matrix, int32_t m, int32_t n)
			{
				T* colsums=new T[m];
				fill_vector(colsums, m, (T) 0);

				for (int32_t i=0; i<n; i++)
				{
					for (int32_t j=0; j<m; j++)
						colsums[j]+=matrix[j+int64_t(i)*m];
				}
				return colsums;
			}

		/* Centers  matrix (e.g. kernel matrix in feature space INPLACE */
		template <class T>
			static void center_matrix(T* matrix, int32_t m, int32_t n)
			{
				float64_t num_data=n;

				T* colsums=get_column_sum(matrix, m,n);
				T* rowsums=get_row_sum(matrix, m,n);

				for (int32_t i=0; i<m; i++)
					colsums[i]/=num_data;
				for (int32_t j=0; j<n; j++)
					rowsums[j]/=num_data;

				T s=sum(rowsums, n)/num_data;

				for (int32_t i=0; i<n; i++)
				{
					for (int32_t j=0; j<m; j++)
						matrix[int64_t(i)*m+j]+=s-colsums[j]-rowsums[i];
				}

				delete[] rowsums;
				delete[] colsums;
			}

		/* finds an element in a sorted array via binary search
		 * returns -1 if not found */
		template <class T>
			static int32_t binary_search_helper(T* output, int32_t size, T elem)
			{
				int32_t start=0;
				int32_t end=size-1;

				if (size<1)
					return 0;

				while (start<end)
				{
					int32_t middle=(start+end)/2; 

					if (output[middle]>elem)
						end=middle-1;
					else if (output[middle]<elem)
						start=middle+1;
					else
						return middle;
				}

				return start;
			}

		/* finds an element in a sorted array via binary search
		 *     * returns -1 if not found */
		template <class T>
			static inline int32_t binary_search(T* output, int32_t size, T elem)
			{
				int32_t ind = binary_search_helper(output, size, elem);
				if (ind >= 0 && output[ind] == elem)
					return ind;
				return -1;
			}

		/* finds an element in a sorted array via binary search 
		 * if it exists, else the index the largest smaller element
		 * is returned
		 * note: a successor is not mandatory */
		template <class T>
			static int32_t binary_search_max_lower_equal(
				T* output, int32_t size, T elem)
			{
				int32_t ind = binary_search_helper(output, size, elem);

				if (output[ind]<=elem)
					return ind;
				if (ind>0 && output[ind-1] <= elem)
					return ind-1;
				return -1;
			}

		/// align two sequences seq1 & seq2 of length l1 and l2 using gapCost
		/// return alignment cost
		static float64_t Align(
			char * seq1, char* seq2, int32_t l1, int32_t l2, float64_t gapCost);


		//@}

		/// returns the mutual information of p which is given in logspace
		/// where p,q are given in logspace
		static float64_t mutual_info(float64_t* p1, float64_t* p2, int32_t len);

		/// returns the relative entropy H(P||Q), 
		/// where p,q are given in logspace
		static float64_t relative_entropy(
			float64_t* p, float64_t* q, int32_t len);

		/// returns entropy of p which is given in logspace
		static float64_t entropy(float64_t* p, int32_t len);

		/// returns number generator seed
		inline static uint32_t get_seed()
		{
			return CMath::seed;
		}

		/// returns range of logtable
		inline static uint32_t get_log_range()
		{
			return CMath::LOGRANGE;
		}

		/// checks whether a float is finite
		inline static int is_finite(double f)
		{
#if defined(isfinite) && !defined(SUNOS)
			return isfinite(f);
#else
			return finite(f);
#endif
		}

		/// checks whether a float is infinity
		inline static int is_infinity(double f)
		{
#ifdef SUNOS
			if (fpclass(f) == FP_NINF || fpclass(f) == FP_PINF)
				return 1;
			else
				return 0;
#else
            return isinf(f);
#endif
		}

		/// checks whether a float is nan
		inline static int is_nan(double f)
		{
#ifdef SUNOS
			return isnand(f);
#else
            return isnan(f);
#endif
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
		static inline float64_t logarithmic_sum(float64_t p, float64_t q)
		{
			float64_t diff;

			if (!CMath::finite(p))
				return q;

			if (!CMath::finite(q))
			{
				SG_SWARNING("INVALID second operand to logsum(%f,%f) expect undefined results\n", p, q);
				return NAN;
			}
			diff = p - q;
			if (diff > 0)
				return diff > LOGRANGE? p : p + logtable[(int)(diff * LOGACCURACY)];
			return -diff > LOGRANGE? q : q + logtable[(int)(-diff * LOGACCURACY)];
		}

		
		static float64_t fishers_exact_test_for_3x2_table(SGMatrix<float64_t> table);

		///init log table of form log(1+exp(x))
		static void init_log_table();

		/// determine int32_t x for that log(1+exp(-x)) == 0
		static int32_t determine_logrange();

		/// determine accuracy, such that the thing fits into MAX_LOG_TABLE_SIZE, needs logrange as argument
		static int32_t determine_logaccuracy(int32_t range);
#else
		static inline float64_t logarithmic_sum(
				float64_t p, float64_t q)
		{
			float64_t diff;

			if (!CMath::is_finite(p))
				return q;
			if (!CMath::is_finite(q))
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
				static inline float64_t logarithmic_sum_array(
					float64_t *p, int32_t len)
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
						register float64_t *pp=p ;
						if (len%2==1) pp++ ;
						for (register int32_t j=0; j < len>>1; j++)
							pp[j]=logarithmic_sum(pp[j<<1], pp[1+(j<<1)]) ;
					}
					return logarithmic_sum_array(p,len%2+len>>1) ;
				} 
#endif
				//@}

				/** @return object name */
				inline virtual const char* get_name() const { return "Mathematics"; }
	public:
				/**@name constants*/
				//@{
				/// infinity
				static const float64_t INFTY;
				static const float64_t ALMOST_INFTY;

				/// almost neg (log) infinity
				static const float64_t ALMOST_NEG_INFTY;

	protected:
				/// range for logtable: log(1+exp(x))  -LOGRANGE <= x <= 0
				static int32_t LOGRANGE;

				/// random generator seed
				static uint32_t seed;
				static char* rand_state;

#ifdef USE_LOGCACHE	

				/// number of steps per integer
				static int32_t LOGACCURACY;
				//@}
				///table with log-values
				static float64_t* logtable;	
#endif
};

//implementations of template functions
template <class T1,class T2>
void* CMath::parallel_qsort_index(void* p)
	{
		struct thread_qsort<T1,T2>* ps=(thread_qsort<T1,T2>*) p;
		T1* output=ps->output;
		T2* index=ps->index;
		uint32_t size=ps->size;
		int32_t* qsort_threads=ps->qsort_threads;
		int32_t sort_limit=ps->sort_limit;
		int32_t num_threads=ps->num_threads;

		if (size==2)
		{
			if (output[0] > output [1])
			{
				swap(output[0], output[1]);
				swap(index[0], index[1]);
			}
			return NULL;
		}
		//T1 split=output[random(0,size-1)];
		T1 split=output[size/2];

		int32_t left=0;
		int32_t right=size-1;

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
		struct thread_qsort<T1,T2> t1;
		struct thread_qsort<T1,T2> t2;

		if (right+1> 1 && (right+1< sort_limit || *qsort_threads >= num_threads-1))
			qsort_index(output,index,right+1);
		else if (right+1> 1)
		{
			(*qsort_threads)++;
			lthread_start=true;
			t1.output=output;
			t1.index=index;
			t1.size=right+1;
			t1.qsort_threads=qsort_threads;
			t1.sort_limit=sort_limit;
			t1.num_threads=num_threads;
			if (pthread_create(&lthread, NULL, parallel_qsort_index<T1,T2>, &t1) != 0)
			{
				lthread_start=false;
				(*qsort_threads)--;
				qsort_index(output,index,right+1);
			}
		}


		if (size-left> 1 && (size-left< sort_limit || *qsort_threads >= num_threads-1))
			qsort_index(&output[left],&index[left], size-left);
		else if (size-left> 1)
		{
			(*qsort_threads)++;
			rthread_start=true;
			t2.output=&output[left];
			t2.index=&index[left];
			t2.size=size-left;
			t2.qsort_threads=qsort_threads;
			t2.sort_limit=sort_limit;
			t2.num_threads=num_threads;
			if (pthread_create(&rthread, NULL, parallel_qsort_index<T1,T2>, &t2) != 0)
			{
				rthread_start=false;
				(*qsort_threads)--;
				qsort_index(&output[left],&index[left], size-left);
			}
		}

		if (lthread_start)
		{
			pthread_join(lthread, NULL);
			(*qsort_threads)--;
		}

		if (rthread_start)
		{
			pthread_join(rthread, NULL);
			(*qsort_threads)--;
		}

		return NULL;
	}

	template <class T1,class T2>
void CMath::qsort_index(T1* output, T2* index, uint32_t size)
{
	if (size==1)
		return;

	if (size==2)
	{
		if (output[0] > output [1])
		{
			swap(output[0],output[1]);
			swap(index[0],index[1]);
		}
		return;
	}
	//T1 split=output[random(0,size-1)];
	T1 split=output[size/2];

	int32_t left=0;
	int32_t right=size-1;

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
void CMath::qsort_backward_index(T1* output, T2* index, int32_t size)
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

	//T1 split=output[random(0,size-1)];
	T1 split=output[size/2];

	int32_t left=0;
	int32_t right=size-1;

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
void CMath::nmin(float64_t* output, T* index, int32_t size, int32_t n)
{
	if (6*n*size<13*size*CMath::log(size))
		for (int32_t i=0; i<n; i++)
			min(&output[i], &index[i], size-i) ;
	else
		qsort_index(output, index, size) ;
}

/* move the smallest entry in the array to the beginning */
	template <class T>
void CMath::min(float64_t* output, T* index, int32_t size)
{
	if (size<=1)
		return;
	float64_t min_elem=output[0];
	int32_t min_index=0;
	for (int32_t i=1; i<size; i++)
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
}
#endif
