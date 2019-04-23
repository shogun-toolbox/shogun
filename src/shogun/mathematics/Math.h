/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Heiko Strathmann, Viktor Gal, Fernando Iglesias,
 *          Sergey Lisitsyn, Sanuj Sharma, Soumyajit De, Shashwat Lal Das,
 *          Thoralf Klein, Wu Lin, Chiyuan Zhang, Harshit Syal, Evan Shelhamer,
 *          Philippe Tillet, Bjoern Esser, Yuyu Zhang, Abhinav Agarwalla,
 *          Saurabh Goyal
 */

#ifndef __MATHEMATICS_H_
#define __MATHEMATICS_H_

#include <shogun/lib/config.h>

#include <shogun/lib/common.h>
#include <shogun/base/Parallel.h>
#include <shogun/base/ShogunEnv.h>
#include <shogun/base/SGObject.h>
#include <shogun/lib/SGVector.h>
#include <algorithm>
#include <numeric>

#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif
#include <cmath>
#include <cfloat>

#ifndef _WIN32
#include <unistd.h>
#endif

#ifdef HAVE_PTHREAD
#include <pthread.h>
#endif

#ifdef SUNOS
#include <ieeefp.h>
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
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

#define COMPLEX128_ERROR_ONEARG(function)	\
static inline complex128_t function(complex128_t a)	\
{	\
	error("Math::{}():: Not supported for complex128_t",\
		#function);\
	return complex128_t(0.0, 0.0);	\
}

#define COMPLEX128_STDMATH(function)	\
static inline complex128_t function(complex128_t a)	\
{	\
	return std::function(a);	\
}

namespace shogun
{
/** @brief Class which collects generic mathematical functions
 */
class Math : public SGObject
{
	public:
		/**@name Constructor/Destructor.
		*/
		//@{
		///Constructor - initializes log-table
		Math();

		///Destructor - frees logtable
		virtual ~Math();
		//@}

#ifndef SWIG // SWIG should skip this part
		/**@name min/max/abs functions.
		*/
		//@{

		/** Returns the smallest element amongst two input values
		 * @param a first value
		 * @param b second value
		 * @return minimum value amongst a and b
		 */
		template <class T, class = typename std::enable_if<std::is_arithmetic<T>::value>::type>
			static inline T min(T a, T b)
			{
				return std::min(a, b);
			}

		/** Returns the greatest element amongst two input values
		 * @param a first value
		 * @param b second value
		 * @return maximum value amongst a and b
		 */
		template <class T, class = typename std::enable_if<std::is_arithmetic<T>::value>::type>
			static inline T max(T a, T b)
			{
				return std::max(a, b);
			}
#endif

		/** Returns the absolute value of a number, that is
		 * if a>0, output is a; if a<0 ,output is -a
		 * @param a complex number
		 * @return the corresponding absolute value
		 */
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

		/** Returns the absolute value of a complex number
		 * @param a complex number
		 * @return the corresponding absolute value
		 */
		static inline float64_t abs(complex128_t a)
		{
			float64_t a_real=a.real();
			float64_t a_imag=a.imag();
			return (std::sqrt(a_real * a_real + a_imag * a_imag));
		}
		//@}

		/** Returns the smallest element in the vector
		 * @param vec input vector
		 * @param len length of vector
		 * @return minimum value in the vector
		 */
		template <class T>
			static T min(T* vec, int32_t len)
			{
				ASSERT(len>0)
				return *std::min_element(vec, vec+len);
			}

		/** Returns the greatest element in the vector
		 * @param vec input vector
		 * @param len length of vector
		 * @return maximum value in the vector
		 */
		template <class T>
			static T max(T* vec, int32_t len)
			{
				ASSERT(len>0)
				return *std::max_element(vec, vec+len);
			}

#ifndef SWIG // SWIG should skip this part
		/** Returns the value clamped to interval [lb,ub]
		 * @param value input value
		 * @param lb lower bound
		 * @param ub upper bound
		 * @return the corresponding clamped value
		 */
		template <class T, class = typename std::enable_if<std::is_arithmetic<T>::value>::type>
			static inline T clamp(T value, T lb, T ub)
			{
				if (value<=lb)
					return lb;
				else if (value>=ub)
					return ub;
				else
					return value;
			}

		/** Returns the index of the maximum value
		 * @param vec input vector
		 * @param inc increment factor
		 * @param len length of the input vector
		 * @param maxv_ptr pointer to store the maximum value
		 * @return index of the maximum value
		 */
		template <class T, class = typename std::enable_if<std::is_arithmetic<T>::value>::type>
			static int32_t arg_max(T * vec, int32_t inc, int32_t len, T * maxv_ptr = NULL)
			{
				ASSERT(len > 0 || inc > 0)

				T maxv = vec[0];
				int32_t maxIdx = 0;

				for (int32_t i = 1, j = inc ; i < len ; i++, j += inc)
				{
					if (vec[j] > maxv)
					maxv = vec[j], maxIdx = i;
				}

				if (maxv_ptr != NULL)
					*maxv_ptr = maxv;

				return maxIdx;
			}

		/** Returns the index of the minimum value
		 * @param vec input vector
		 * @param inc increment factor
		 * @param len length of the input vector
		 * @param minv_ptr pointer to store the minimum value
		 * @return index of the minimum value
		 */
		template <class T, class = typename std::enable_if<std::is_arithmetic<T>::value>::type>
			static int32_t arg_min(T * vec, int32_t inc, int32_t len, T * minv_ptr = NULL)
		{
			ASSERT(len > 0 || inc > 0)

			T minv = vec[0];
			int32_t minIdx = 0;

			for (int32_t i = 1, j = inc ; i < len ; i++, j += inc)
			{
				if (vec[j] < minv)
				minv = vec[j], minIdx = i;
			}

			if (minv_ptr != NULL)
				*minv_ptr = minv;

			return minIdx;
		}

		/**@name misc functions */
		//@{

		/** Compares the value of two floats based on eps only
		 * @param a first value to compare
		 * @param b second value to compare
		 * @param eps threshold for values to be equal/different
		 * @return true if values are equal within eps accuracy, false if not.
		 */
		template <class T, class = typename std::enable_if<std::is_floating_point<T>::value>::type>
			static inline bool fequals_abs(const T& a, const T& b,
				const float64_t eps)
			{
				const T diff = Math::abs<T>((a-b));
				return (diff < eps);
			}

		/** Compares the value of two floats (handles special cases, such as NaN, Inf etc.)
		 * Note: returns true if a == b == NAN
		 * Implementation inspired by http://floating-point-gui.de/errors/comparison/
		 * @param a first value to compare
		 * @param b second value to compare
		 * @param eps threshold for values to be equal/different
		 * @return true if values are equal within eps accuracy, false if not.
		 */
		    template <class T, class = typename std::enable_if<
		                           std::is_floating_point<T>::value>::type>
		    static inline bool
		    fequals(const T& a, const T& b, const float64_t eps_)
		    {
			    // global fequals epsilon might override passed one
			    // hack for lossy serialization formats
			    float64_t eps = std::max(
					eps_,env()->fequals_epsilon());

			    const T absA = Math::abs<T>(a);
			    const T absB = Math::abs<T>(b);
			    const T diff = Math::abs<T>((a - b));

			    // Handle this separately since NAN is unordered
			    if (Math::is_nan((float64_t)a) && Math::is_nan((float64_t)b))
				    return true;

				// Required for JSON Serialization Tests
			    if (env()->fequals_tolerant())
				    return Math::fequals_abs<T>(a, b, eps);

			    // handles float32_t and float64_t separately
			    T comp = (std::is_same<float32_t, T>::value)
			                 ? Math::F_MIN_NORM_VAL32
			                 : Math::F_MIN_NORM_VAL64;

			    if (a == b)
				    return true;

			    // both a and b are 0 and relative error is less meaningful
				else if ((a == 0) || (b == 0) || (diff < comp))
					return (diff < (eps * comp));
				// use max(relative error, diff) to handle large eps
				else
				{
					T check = ((diff/(absA + absB)) > diff)?
						(diff/(absA + absB)):diff;
					return (check < eps);
				}
			}
#endif

		/* Get the corresponding absolute tolerance for unit test given a relative tolerance
		 *
		 * Note that a unit test will be passed only when
		 * \f[
		 * |v_\text{true} - v_\text{predict}| \leq tol_\text{relative} * |v_\text{true}|
		 * \f] which is equivalent to
		 * \f[
		 * |v_\text{true} - v_\text{predict}| \leq tol_\text{absolute}
		 * \f] where
		 * \f[
		 * tol_\text{absolute} = tol_\text{relative} * |v_\text{true}|
		 * \f]
		 *
		 * @param true_value true value should be finite (neither NAN nor INF)
		 * @param rel_tolerance relative tolerance should be positive and less than 1.0
		 *
		 * @return the corresponding absolute tolerance
		 */
		static float64_t get_abs_tolerance(float64_t true_value, float64_t rel_tolerance);

		/** Rounds off the input value to it's nearest integer (as a floating-point value)
		 * @param d input decimal value
		 * @return rounded off value
		 */
		static inline float64_t round(float64_t d)
		{
			return std::floor(d+0.5);
		}

		/** The value of x rounded downward (as a floating-point value)
		 * @param d input decimal value
		 * @return rounded off value
		 */
		static inline float64_t floor(float64_t d)
		{
			return std::floor(d);
		}

		/** Signum of input value
		 * @param a input value
		 * @return 1 if a>0, -1 if a<0
		 */
		template <class T>
			static inline T sign(T a)
			{
				if (a==0)
					return 0;
				else return (a<0) ? (-1) : (+1);
			}

		/** Swaps a and b
		 * @param a first input value
		 * @param b second input value
		 */
		template <class T>
			static inline void swap(T &a,T &b)
			{
				T c=a;
				a=b;
				b=c;
			}

		/** Computes square of the input
		 * @param x input value
		 * @return x*x (x^2)
		 */
		template <class T>
			static inline T sq(T x)
			{
				return x*x;
			}

		/** Computes square-root of the input
		 * @param x input value
		 * @return x^0.5
		 */
		template <class T>
			static inline T sqrt(T x)
			{
				return std::sqrt(x);
			}

		/// x^0.5, x being a complex128_t
		COMPLEX128_STDMATH(sqrt)

		/** Computes inverse square-root of the input
		 * @param x input value
		 * @return x^-0.5
		 */
		static inline float32_t invsqrt(float32_t x)
		{
			union float_to_int
			{
				float32_t f;
				int32_t i;
			};

			float_to_int tmp;
			tmp.f=x;

			float32_t xhalf = 0.5f * x;
			// store floating-point bits in integer tmp.i
			// and do initial guess for Newton's method
			tmp.i = 0x5f3759d5 - (tmp.i >> 1);
			x = tmp.f; // convert new bits into float
			x = x*(1.5f - xhalf*x*x); // One round of Newton's method
			return x;
		}

		/**
		 * @name Exponential methods (x^n)
 		 */
		//@{
		/** Computes x raised to the power n
		 * @param x base
		 * @param n exponent
		 * @return x^n
		 */
		static inline floatmax_t powl(floatmax_t x, floatmax_t n)
		{
			return std::pow(x, n);
		}

		static inline int32_t pow(bool x, int32_t n)
		{
			return 0;
		}

		/**
		 * @param x base (integer)
		 * @param n exponent (integer)
		 */
		template<typename T, std::enable_if_t<std::numeric_limits<T>::is_integer, T>* = nullptr>
		static inline T pow(T x, T n)
		{
			ASSERT(n>=0)
			// power of integer 2 is basically a bitshift...
			if (x == 2)
				return (1 << n);

			T result = 1;
			while (n--)
				result *= x;

			return result;
		}

		/**
		 * @param x base (decimal)
		 * @param n exponent (integer)
		 */
		static inline float64_t pow(float64_t x, int32_t n)
		{
			return std::pow(x, n);
		}

		/**
		 * @param x base (decimal)
		 * @param n exponent (decimal)
		 */
		static inline float64_t pow(float64_t x, float64_t n)
		{
			return std::pow(x, n);
		}

		/**
		 * @param x base (complex)
		 * @param n exponent (integer)
		 */
		static inline complex128_t pow(complex128_t x, int32_t n)
		{
			return std::pow(x, n);
		}

		/**
		 * @param x base (complex)
		 * @param n exponent (complex)
		 */
		static inline complex128_t pow(complex128_t x, complex128_t n)
		{
			return std::pow(x, n);
		}

		/**
		 * @param x base (complex)
		 * @param n exponent (decimal)
		 */
		static inline complex128_t pow(complex128_t x, float64_t n)
		{
			return std::pow(x, n);
		}

		/**
		 * @param x base (decimal)
		 * @param n exponent (complex)
		 */
		static inline complex128_t pow(float64_t x, complex128_t n)
		{
			return std::pow(x, n);
		}
		//@}

		/**
		 * @name Logarithmic functions
		 */
		//@{
		/** Computes logarithm base 10 of input
		 * @param v input
		 * @return log base 10 of v
		 */
		static inline float64_t log10(float64_t v)
		{
			return std::log10(v);
		}

		/// log10(x), x being a complex128_t
		COMPLEX128_STDMATH(log10)

		/** Computes logarithm base 2 of input
		 * @param v input
		 * @return log base 2 of v
		 */
		static inline float64_t log2(float64_t v)
		{
			return std::log2(v);
		}


		static inline index_t floor_log(index_t n)
		{
			index_t i;
			for (i = 0; n != 0; i++)
				n >>= 1;

			return i;
		}
		//@}

		/** Computes area under the curve
		 * @param xy
		 * @param len length
		 * @param reversed boolean
		 * @return area
		 */
		static float64_t area_under_curve(float64_t* xy, int32_t len, bool reversed)
		{
			ASSERT(len>0 && xy)

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

		/** Converts string to float
		 * @param str input string
		 * @param float_result float pointer
		 * @return returns true if successful, else returns false
		 */
		static bool strtof(const char* str, float32_t* float_result);

		/** Converts string to double
		 * @param str input string
		 * @param double_result double pointer
		 * @return returns true if successful, else returns false
		 */
		static bool strtod(const char* str, float64_t* double_result);

		/** Converts string to long double
		 * @param str input string
		 * @param long_double_result long double pointer
		 * @return returns true if successful, else returns false
		 */
		static bool strtold(const char* str, floatmax_t* long_double_result);

		/** Computes factorial of input
		 * @param n input
		 * @return factorial of n (n!)
		 */
		static inline int64_t factorial(int32_t n)
		{
			int64_t res=1;
			for (int i=2; i<=n; i++)
				res*=i ;
			return res ;
		}

		/** Implements the greatest common divisor (gcd) via modulo operations.
		 * Requires that either a>0 and b>=0 or vice versa.
		 *
		 * @param a first number
		 * @param b second number
		 * @return gcd between the two numbers
		 */
		static int32_t gcd(int32_t a, int32_t b)
		{
			require((a>=0 && b>0) || (b>=0 && a>0), "gcd({},{}) is not defined.",
					a, b);

			if (1 == a || 1 == b)
				return 1;

			while (0 < a && 0 < b)
			{
				if (a > b)
					a %= b;
				else
					b %= a;
			}

			return 0 == a ? b : a;
		}

		/** Computes sum of non-zero elements
		 * @param vec vector
		 * @param len length
		 * @return non-zero sum
		 */
		template <class T>
			static int32_t get_num_nonzero(T* vec, int32_t len)
			{
				int32_t nnz = 0;
				for (index_t i=0; i<len; ++i)
					nnz += vec[i] != 0;

				return nnz;
			}

		/** Computes sum of non-zero elements
		 * @param vec vector of complex numbers
		 * @param len length
		 * @return non-zero sum
		 */
		static int32_t get_num_nonzero(complex128_t* vec, int32_t len)
		{
				int32_t nnz=0;
				for (index_t i=0; i<len; ++i)
					nnz+=vec[i]!=0.0;
				return nnz;
		}

		/** Computes binomial coefficient or all possible combinations
		 * of n items taken k at a time.
		 * @param n total number of items
		 * @param k number of items to be chosen
		 */
		static inline int64_t nchoosek(int32_t n, int32_t k)
		{
			int64_t res=1;

			for (int32_t i=n-k+1; i<=n; i++)
				res*=i;

			return res/factorial(k);
		}

		/** Builds an array with n linearly spaced elements between start and end.
		 * @param output array with linearly spaced elements within the interval
		 * @param start beginning of the interval to divide
		 * @param end upper bound of the interval to divide
		 * @param n number of elements used to divide the interval
		 */
		static void linspace(float64_t* output, float64_t start, float64_t end, int32_t n = 100);

#ifndef SWIG // SWIG should skip this part
		/** Returns an array with n linearly spaced elements between start and end.
		 * @param start beginning of the interval to divide
		 * @param end upper bound of the interval to divide
		 * @param n number of elements used to divide the interval
		 * @return array with linearly spaced elements within the interval
		 */
		template <class T, class = typename std::enable_if<std::is_arithmetic<T>::value>::type>
			static float64_t* linspace(T start, T end, int32_t n)
			{
				float64_t* output = SG_MALLOC(float64_t, n);
				linspace(output, start, end, n);

				return output;
			}
#endif

		/** Returns a vector with n linearly spaced elements between start and end.
		 * @param start beginning of the interval to divide
		 * @param end upper bound of the interval to divide
		 * @param n number of elements used to divide the interval
		 * @return vector with linearly spaced elements within the interval
		 */
		template <class T>
			static SGVector<float64_t> linspace_vec(T start, T end, int32_t n)
			{
				return SGVector<float64_t>(linspace(start, end, n), n);
			}

		/** Computes \f$\log(\sum_{i=1}^n \exp(x_i))\f$ for given \f$x_i\f$
		 * using the log-sum-exp trick which avoids numerical problems.
		 * @param values the vector of \f$x_i\f$
		 * @return \f$\log(\sum_{i=1}^n \exp(x_i))\f$ for given \f$x_i\f$
		 */
		template <class T>
		static T log_sum_exp(SGVector<T> values)
		{
			require(values.vector, "Values are empty");
			require(values.vlen>0,"number of values supplied is 0");

			/* find minimum element index */
			index_t min_index=0;
			T X0=values[0];
			if (values.vlen>1)
			{
				for (index_t i=1; i<values.vlen; ++i)
				{
					if (values[i]<X0)
					{
						X0=values[i];
						min_index=i;
					}
				}
			}

			/* remove element from vector copy and compute log sum exp */
			SGVector<T> values_without_X0(values.vlen-1);
			index_t from_idx=0;
			index_t to_idx=0;
			for (from_idx=0; from_idx<values.vlen; ++from_idx)
			{
				if (from_idx!=min_index)
				{
					values_without_X0[to_idx] = std::exp(values[from_idx] - X0);
					to_idx++;
				}
			}

			return X0 + std::log(SGVector<T>::sum(values_without_X0) + 1);
		}

		/** Computes \f$\log(\frac{1}{n}\sum_{i=1}^n \exp(x_i))\f$ for given
		 * \f$x_i\f$ using the log-sum-exp trick which avoids numerical problems.
		 *
		 * @param values the vector of \f$x_i\f$
		 * @return \f$\log(\frac{1}{n}\sum_{i=1}^n \exp(x_i))\f$
		 */
		template <class T>
		static T log_mean_exp(SGVector<T> values)
		{
			return log_sum_exp(values) - std::log(values.vlen);
		}

		/** Performs a bubblesort on a given matrix a.
		 * it is sorted in ascending order from top to bottom
		 * and left to right
		 * @param a matrix to be sorted
		 * @param cols number of columns
		 * @param sort_col
		 */
		static void sort(int32_t *a, int32_t cols, int32_t sort_col=0);

		/** Performs a bubblesort on a given array a.
		 * it is sorted in ascending order from left to right
		 * @param a array to be sorted
		 * @param idx index array
		 * @param N length of array
		 */
		static void sort(float64_t *a, int32_t*idx, int32_t N);

#ifndef SWIG // SWIG should skip this part
		/** Performs a quicksort on an array output of length size
		 * it is sorted from in ascending (for type T)
		 * @param output array to be sorted
		 * @param size size of array
		 */
		template <class T, class = typename std::enable_if<std::is_arithmetic<T>::value>::type>
			static void qsort(T* output, int32_t size)
			{
				if (size<=1)
					return;

				if (size==2)
				{
					if (output[0] > output [1])
						Math::swap(output[0],output[1]);
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
						Math::swap(output[left],output[right]);
						left++;
						right--;
					}
				}

				if (right+1> 1)
					qsort(output,right+1);

				if (size-left> 1)
					qsort(&output[left],size-left);
			}

		/** Performs insertion sort of an array output of length size
		 * it is sorted from in ascending (for type T)
		 * @param output array to be sorted
		 * @param size size of array
		 */
		template <class T, class = typename std::enable_if<std::is_arithmetic<T>::value>::type>
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

		/** Performs a in-place radix sort in ascending order
		 * @param array array to be sorted
		 * @param size size of array
		 */
		template <class T, class = typename std::enable_if<std::is_arithmetic<T>::value>::type>
			inline static void radix_sort(T* array, int32_t size)
			{
				radix_sort_helper(array,size,0);
			}
#endif

		/** Extract the byte at position p (from left)
		 * of a 64 bit integer. The function is somewhat identical to
		 * accessing an array of characters via [].
		 * @param word input from which byte is extracted
		 * @param p position
		 * @return byte
		 */
		template <class T>
			static inline uint8_t byte(T word, uint16_t p)
			{
				return (word >> (sizeof(T)-p-1) * 8) & 0xff;
			}

		/// byte not implemented for complex128_t
		static inline uint8_t byte(complex128_t word, uint16_t p)
		{
			error("Math::byte():: Not supported for complex128_t");
			return uint8_t(0);
		}

		/** Helper function for radix sort.
		 * @param array array to be sorted
		 * @param size size of array
		 * @param i index
		 */
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

		/// radix_sort_helper not implemented for complex128_t
		static void radix_sort_helper(complex128_t* array, int32_t size, uint16_t i)
		{
			error("Math::radix_sort_helper():: Not supported for complex128_t");
		}

#ifndef SWIG // SWIG should skip this part
		/** Performs a quicksort on an array of pointers.
		 * It is sorted from in ascending (for type T)
		 * Every element is dereferenced once before being compared
		 * @param vector array of pointers to sort
		 * @param length length of array
		 */
		template <class T, class = typename std::enable_if<std::is_arithmetic<T>::value>::type>
			static void qsort(T** vector, index_t length)
			{
				if (length<=1)
					return;

				if (length==2)
				{
					if (*vector[0]>*vector[1])
						swap(vector[0],vector[1]);
					return;
				}
				T* split=vector[length/2];

				int32_t left=0;
				int32_t right=length-1;

				while (left<=right)
				{
					while (*vector[left]<*split)
						++left;
					while (*vector[right]>*split)
						--right;

					if (left<=right)
					{
						swap(vector[left],vector[right]);
						++left;
						--right;
					}
				}

				if (right+1>1)
					qsort(vector,right+1);

				if (length-left>1)
					qsort(&vector[left],length-left);
			}

		/** Quicksort the vector in ascending order (for type T)
		  * @param vector vector to be sorted
		  */
		template <class T, class = typename std::enable_if<std::is_arithmetic<T>::value>::type>
			static void qsort(SGVector<T> vector)
			{
				qsort<T>(vector, vector.size());
			}
#endif

#ifndef SWIG // SWIG should skip this part
		/** Get sorted index.
		 *
		 * idx = v.argsort() is similar to Matlab [~, idx] = sort(v)
		 *
		 * @param v vector to be sorted
		 * @return sorted index for this vector
		 */
		template<class T, class = typename std::enable_if<std::is_arithmetic<T>::value>::type>
			static SGVector<index_t> argsort(SGVector<T> v)
			{
				SGVector<index_t> idx(v.vlen);
				std::iota(idx.begin(), idx.end(), 0);
				std::sort(idx.begin(), idx.end(),
					[&v](index_t i1, index_t i2)
					{
						return std::abs(v[i1]-v[i2])>std::numeric_limits<T>::epsilon()
							&& v[i1]<v[i2];
					});
				return idx;
			}

		/** Check if vector is sorted
		 *
		 * @param vector input vector
		 * @return true if vector is sorted, false otherwise
		 */
		template <class T, class = typename std::enable_if<std::is_arithmetic<T>::value>::type>
			static bool is_sorted(SGVector<T> vector)
			{
				if (vector.size() < 2)
					return true;

				for(int32_t i=1; i<vector.size(); i++)
				{
					if (vector[i-1] > vector[i])
						return false;
				}

				return true;
			}
#endif

		/** Display bits (useful for debugging)
		 * @param word input to be displayed as bits
		 * @param width number of bits displayed
		 */
		template <class T> static void display_bits(T word, int32_t width=8*sizeof(T))
		{
			ASSERT(width>=0)
			for (int i=0; i<width; i++)
			{
				T mask = ((T) 1)<<(sizeof(T)*8-1);
				while (mask)
				{
					if (mask & word)
						io::print("1");
					else
						io::print("0");

					mask>>=1;
				}
			}
		}

		/// disply_bits not implemented for complex128_t
		static void display_bits(complex128_t word,
			int32_t width=8*sizeof(complex128_t))
		{
			error("Math::display_bits():: Not supported for complex128_t");
		}

		/** Performs a quicksort on an array output of length size
		 * it is sorted in ascending order
		 * (for type T1) and returns the index (type T2)
		 * matlab alike [sorted,index]=sort(output)
		 * @param output array to be sorted
		 * @param index index array
		 * @param size size of arrays
		 */
		template <class T1,class T2>
			static void qsort_index(T1* output, T2* index, uint32_t size);

		/// qsort_index not implemented for complex128_t
		template <class T>
			static void qsort_index(complex128_t* output, T* index, uint32_t size)
			{
				error("Math::qsort_index():: Not supported for complex128_t");
			}

		/** Performs a quicksort on an array output of length size
		 * it is sorted in descending order
		 * (for type T1) and returns the index (type T2)
		 * matlab alike [sorted,index]=sort(output)
		 * @param output array to be sorted
		 * @param index index array
		 * @param size size of array
		 */
		template <class T1,class T2>
			static void qsort_backward_index(
				T1* output, T2* index, int32_t size);

		/// qsort_backword_index not implemented for complex128_t
		template <class T>
			static void qsort_backword_index(
				complex128_t* output, T* index, uint32_t size)
			{
				error("Math::qsort_backword_index():: \
					Not supported for complex128_t");
			}

		/** Performs a quicksort on an array output of length size
		 * it is sorted in ascending order
		 * (for type T1) and returns the index (type T2)
		 * matlab alike [sorted,index]=sort(output)
		 * parallel version
		 * @param output input array
		 * @param index index array
		 * @param size size of the array
		 * @param n_threads number of threads
		 * @param limit sort limit
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

		/// parallel_qsort_index not implemented for complex128_t
		template <class T>
			inline static void parallel_qsort_index(complex128_t* output, T* index,
				uint32_t size, int32_t n_threads, int32_t limit=0)
			{
				error("Math::parallel_qsort_index():: Not supported for complex128_t");
			}

		/// helper function for parallel_qsort_index.
		template <class T1,class T2>
			static void* parallel_qsort_index(void* p);

		/** Finds the smallest element in output and puts that element as the
		 * first element
		 * @param output element array
		 * @param index index array
		 * @param size size of arrays
		 */
		template <class T>
			static void min(float64_t* output, T* index, int32_t size);

		/// complex128_t cannot be used as index
		static void min(float64_t* output, complex128_t* index, int32_t size)
		{
			error("Math::min():: Not supported for complex128_t");
		}

		/** Finds the n smallest elements in output and puts these elements as the
		 * first n elements
		 */
		template <class T>
			static void nmin(
				float64_t* output, T* index, int32_t size, int32_t n);

		/// complex128_t cannot be used as index
		static void nmin(float64_t* output, complex128_t* index,
			int32_t size, int32_t n)
		{
			error("Math::nmin():: Not supported for complex128_t");
		}



		/** Finds an element in a sorted array via binary search
		 * returns -1 if not found
		 */
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

		/// binary_search_helper not implemented for complex128_t
		static int32_t binary_search_helper(complex128_t* output, int32_t size, complex128_t elem)
		{
			error("Math::binary_search_helper():: Not supported for complex128_t");
			return int32_t(0);
		}

		/** Finds an element in a sorted array via binary search
		 * @param output array to search
		 * @param size size of array
		 * @param elem element to search
		 * @return -1 if not found
		 */
		template <class T>
			static inline int32_t binary_search(T* output, int32_t size, T elem)
			{
				int32_t ind = binary_search_helper(output, size, elem);
				if (ind >= 0 && output[ind] == elem)
					return ind;
				return -1;
			}

		/// binary_search not implemented for complex128_t
		static inline int32_t binary_search(complex128_t* output, int32_t size, complex128_t elem)
		{
			error("Math::binary_search():: Not supported for complex128_t");
			return int32_t(-1);
		}


		/** Finds an element in a sorted array of pointers via binary search
		 * Every element is dereferenced once before being compared
		 * @param vector array of pointers to search in (assumed being sorted)
		 * @param length length of array
		 * @param elem pointer to element to search for
		 * @return index of elem, -1 if not found
		 */
		template<class T>
			static inline int32_t binary_search(T** vector, index_t length,
					T* elem)
			{
				int32_t start=0;
				int32_t end=length-1;

				if (length<1)
					return -1;

				while (start<end)
				{
					int32_t middle=(start+end)/2;

					if (*vector[middle]>*elem)
						end=middle-1;
					else if (*vector[middle]<*elem)
						start=middle+1;
					else
					{
						start=middle;
						break;
					}
				}

				if (start>=0&&*vector[start]==*elem)
					return start;

				return -1;
			}

		/// binary_search not implemented for complex128_t
		static inline int32_t binary_search(complex128_t** vector, index_t length, complex128_t* elem)
		{
			error("Math::binary_search():: Not supported for complex128_t");
			return int32_t(-1);
		}

		/** Finds greatest element which is less than or equal to the
		 * searched element (query).
		 * @param output array to search
		 * @param size size of the array
		 * @param elem element that needs to be searched
		 */
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

		/// binary_search_max_lower_equal not implemented for complex128_t
		static int32_t binary_search_max_lower_equal(complex128_t* output,
			int32_t size, complex128_t elem)
		{
			error("Math::binary_search_max_lower_equal():: \
				Not supported for complex128_t");
			return int32_t(-1);
		}

		/** Align two sequences seq1 & seq2 of length l1 and l2 using gapCost
		 * @param seq1 first sequence
		 * @param seq2 second sequence
		 * @param l1 length of first sequence
		 * @param l2 length of second sequence
		 * @param gapCost
		 * @return alignment cost
		 */
		static float64_t Align(
			char * seq1, char* seq2, int32_t l1, int32_t l2, float64_t gapCost);


		//@}

		/** Returns real part of a complex128_t number
		 * @param c complex number
		 * @return real part of the complex number
		 */
		static float64_t real(complex128_t c)
		{
			return c.real();
		}

		/** Returns imaginary part of a complex128_t number
		 * @param c complex number
		 * @return imaginary part of the complex number
		 */
		static float64_t imag(complex128_t c)
		{
			return c.imag();
		}

		/// returns range of logtable
		inline static uint32_t get_log_range()
		{
			return Math::LOGRANGE;
		}

#ifdef USE_LOGCACHE
		/// returns range of logtable
		inline static uint32_t get_log_accuracy()
		{
			return Math::LOGACCURACY;
		}
#endif

		/// checks whether a float is finite
		static int is_finite(double f);

		/// checks whether a float is nan
		static int is_nan(double f);

		/**@name summing functions */
		//@{
		/** Sum logarithmic probabilities.
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

			if (!Math::is_finite(p))
				return q;

			if (!Math::is_finite(q))
			{
				io::warn("INVALID second operand to logsum({},{}) expect undefined results", p, q);
				return NOT_A_NUMBER;
			}
			diff = p - q;
			if (diff > 0)
				return diff > LOGRANGE? p : p + logtable[(int)(diff * LOGACCURACY)];
			return -diff > LOGRANGE? q : q + logtable[(int)(-diff * LOGACCURACY)];
		}

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

			if (!Math::is_finite(p))
				return q;
			if (!Math::is_finite(q))
				return p;
			diff = p - q;
			if (diff > 0)
				return diff > LOGRANGE ? p : p + std::log(1 + std::exp(-diff));
			return -diff > LOGRANGE ? q : q + std::log(1 + std::exp(diff));
		}
#endif
#ifdef USE_LOGSUMARRAY
				/** Sum up a whole array of values in logspace.
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
						float64_t *pp=p ;
						if (len%2==1) pp++ ;
						for (int32_t j=0; j < len>>1; j++)
							pp[j]=logarithmic_sum(pp[j<<1], pp[1+(j<<1)]) ;
					}
					return logarithmic_sum_array(p,len%2 + (len>>1)) ;
				}
#endif //USE_LOGSUMARRAY
				//@}

				/** @return object name */
				virtual const char* get_name() const { return "Math"; }
	public:
				/**@name constants*/
				//@{
				/// not a number
				static const float64_t NOT_A_NUMBER;
				/// infinity
				static const float64_t INFTY;
				static const float64_t ALMOST_INFTY;

				/// almost neg (log) infinity
				static const float64_t ALMOST_NEG_INFTY;

				/** The number pi */
				static const float64_t PI;

				/** Machine epsilon for float64_t */
				static const float64_t MACHINE_EPSILON;

				/* Largest and smallest possible float64_t */
				static const float64_t MAX_REAL_NUMBER;
				static const float64_t MIN_REAL_NUMBER;

				/* Floating point Limits, Normalized */
				static const float32_t F_MAX_VAL32;
				static const float32_t F_MIN_NORM_VAL32;
				static const float64_t F_MAX_VAL64;
				static const float64_t F_MIN_NORM_VAL64;

				/* Floating point limits, Denormalized */
				static const float32_t F_MIN_VAL32;
				static const float64_t F_MIN_VAL64;

	protected:
				/// range for logtable: log(1+exp(x))  -LOGRANGE <= x <= 0
				static int32_t LOGRANGE;

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
void* Math::parallel_qsort_index(void* p)
	{
		struct thread_qsort<T1,T2>* ps=(thread_qsort<T1,T2>*) p;
		T1* output=ps->output;
		T2* index=ps->index;
		int32_t size=ps->size;
		int32_t* qsort_threads=ps->qsort_threads;
		int32_t sort_limit=ps->sort_limit;
		int32_t num_threads=ps->num_threads;

		if (size<2)
		{
			return NULL;
		}

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
void Math::qsort_index(T1* output, T2* index, uint32_t size)
{
	if (size<=1)
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
void Math::qsort_backward_index(T1* output, T2* index, int32_t size)
{
	if (size<=1)
		return;

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
void Math::nmin(float64_t* output, T* index, int32_t size, int32_t n)
{
	if (6 * n * size < 13 * size * std::log(size))
		for (int32_t i=0; i<n; i++)
			min(&output[i], &index[i], size-i);
	else
		qsort_index(output, index, size);
}

/* move the smallest entry in the array to the beginning */
	template <class T>
void Math::min(float64_t* output, T* index, int32_t size)
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

#define COMPLEX128_ERROR_ONEARG_T(function)	\
template <> \
inline complex128_t Math::function<complex128_t>(complex128_t a)	\
{	\
	error("Math::{}():: Not supported for complex128_t",\
		#function);\
	return complex128_t(0.0, 0.0);	\
}

/// signum not implemented for complex128_t, returns (0.0)+i(0.0) instead
// COMPLEX128_ERROR_ONEARG_T(sign)

}
#undef COMPLEX128_ERROR_ONEARG
#undef COMPLEX128_ERROR_ONEARG_T
#undef COMPLEX128_STDMATH
#endif /** __MATHEMATICS_H_ */
