/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Heiko Strathmann, Viktor Gal, Fernando Iglesias,
 *          Sergey Lisitsyn, Sanuj Sharma, Soumyajit De, Shashwat Lal Das,
 *          Thoralf Klein, Wu Lin, Chiyuan Zhang, Harshit Syal, Evan Shelhamer,
 *          Philippe Tillet, Björn Esser, Yuyu Zhang, Abhinav Agarwalla,
 *          Saurabh Goyal
 */

#ifndef __MATHEMATICS_H_
#define __MATHEMATICS_H_

#include <shogun/lib/config.h>

#include <shogun/lib/common.h>
#include <shogun/base/Parallel.h>
#include <shogun/mathematics/Random.h>
#include <shogun/lib/SGVector.h>
#include <algorithm>

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
	SG_SERROR("CMath::%s():: Not supported for complex128_t\n",\
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
	/** random number generator */
	extern CRandom* sg_rand;
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
			return (CMath::sqrt(a_real*a_real+a_imag*a_imag));
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
				const T diff = CMath::abs<T>((a-b));
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
			    float64_t eps = std::max(eps_, get_global_fequals_epsilon());

			    const T absA = CMath::abs<T>(a);
			    const T absB = CMath::abs<T>(b);
			    const T diff = CMath::abs<T>((a - b));

			    // Handle this separately since NAN is unordered
			    if (CMath::is_nan((float64_t)a) && CMath::is_nan((float64_t)b))
				    return true;

				// Required for JSON Serialization Tests
			    if (get_global_fequals_tolerant())
				    return CMath::fequals_abs<T>(a, b, eps);

			    // handles float32_t and float64_t separately
			    T comp = (std::is_same<float32_t, T>::value)
			                 ? CMath::F_MIN_NORM_VAL32
			                 : CMath::F_MIN_NORM_VAL64;

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

		/** The value of x rounded upward (as a floating-point value)
		 * @param d input decimal value
		 * @return rounded off value
		 */
		static inline float64_t ceil(float64_t d)
		{
			return std::ceil(d);
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
		static inline int32_t pow(int32_t x, int32_t n)
		{
			ASSERT(n>=0)
			int32_t result=1;
			while (n--)
				result*=x;

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

		/** Computes e^x where e=2.71828 approx.
		 * @param x exponent
		 */
		static inline float64_t exp(float64_t x)
		{
			return std::exp(x);
		}

		/// exp(x), x being a complex128_t
		COMPLEX128_STDMATH(exp)

		/**
		 * @name Trignometric and Hyperbolic Functions
		 */
		//@{
		/** Computes tangent of input
		 * @param x input
		 * @return tan(x)
		 */
		static inline float64_t tan(float64_t x)
		{
			return std::tan(x);
		}

		/// tan(x), x being a complex128_t
		COMPLEX128_STDMATH(tan)

		/** Computes arc tangent of input
		 * @param x input
		 * @return arctan(x)
		 */
		static inline float64_t atan(float64_t x)
		{
			return std::atan(x);
		}

		/// atan(x), x being a complex128_t not implemented
		COMPLEX128_ERROR_ONEARG(atan)

		/** Computes arc tangent with 2 parameters
		 * @param y input(numerator)
		 * @param x input(denominator)
		 * @return arctan(y/x)
		 */
		static inline float64_t atan2(float64_t y, float64_t x)
		{
			return std::atan2(y, x);
		}

		/// atan2(x), x being a complex128_t not implemented
		COMPLEX128_ERROR_ONEARG(atan2)

		/** Computes hyperbolic tangent of input
		 * @param x input
		 * @return tanh(x)
		 */
		static inline float64_t tanh(float64_t x)
		{
			return std::tanh(x);
		}

		/// tanh(x), x being a complex128_t
		COMPLEX128_STDMATH(tanh)

		/** Computes sine of input
		 * @param x input
		 * @return sin(x)
		 */
		static inline float64_t sin(float64_t x)
		{
			return std::sin(x);
		}

		/// sin(x), x being a complex128_t
		COMPLEX128_STDMATH(sin)

		/** Computes arc sine of input
		 * @param x input
		 * @return arcsin(x)
		 */
		static inline float64_t asin(float64_t x)
		{
			return std::asin(x);
		}

		/// asin(x), x being a complex128_t not implemented
		COMPLEX128_ERROR_ONEARG(asin)

		/** Computes hyperbolic sine of input
		 * @param x input
		 * @return sinh(x)
		 */
		static inline float64_t sinh(float64_t x)
		{
			return std::sinh(x);
		}

		/// sinh(x), x being a complex128_t
		COMPLEX128_STDMATH(sinh)

		/** Computes cosine of input
		 * @param x input
		 * @return cos(x)
		 */
		static inline float64_t cos(float64_t x)
		{
			return std::cos(x);
		}

		/// cos(x), x being a complex128_t
		COMPLEX128_STDMATH(cos)

		/** Computes arc cosine of input
		 * @param x input
		 * @return arccos(x)
		 */
		static inline float64_t acos(float64_t x)
		{
			return std::acos(x);
		}

		/// acos(x), x being a complex128_t not implemented
		COMPLEX128_ERROR_ONEARG(acos)

		/** Computes hyperbolic cosine of input
		 * @param x input
		 * @return cosh(x)
		 */
		static inline float64_t cosh(float64_t x)
		{
			return std::cosh(x);
		}

		/// cosh(x), x being a complex128_t
		COMPLEX128_STDMATH(cosh)
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

		/** Computes natural logarithm input
		 * @param v input
		 * @return log base e of v or ln(v)
		 */


		/// log(x), x being a complex128_t
		COMPLEX128_STDMATH(log)

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

		/**
		 * @name Random Functions
		 */
		//@{
		/** Initiates seed for pseudo random generator
		 * @param initseed value of seed
		 */
		static void init_random(uint32_t initseed=0)
		{
			if (initseed==0)
				seed = CRandom::generate_seed();
			else
				seed=initseed;

			sg_rand->set_seed(seed);
		}

		/** Returns random number
		 * @return unsigned 64 bit integer
		 */
		static inline uint64_t random()
		{
			return sg_rand->random_64();
		}

		/** Returns random number
		 * @return unsigned 64 bit integer
		 */
		static inline uint64_t random(uint64_t min_value, uint64_t max_value)
		{
			return sg_rand->random(min_value, max_value);
		}

		/** Returns random number between minimum and maximum value
		 * @param min_value minimum value (64 bit integer)
		 * @param max_value maximum value (64 bit integer)
		 * @return signed 64 bit integer
		 */
		static inline int64_t random(int64_t min_value, int64_t max_value)
		{
			return sg_rand->random(min_value, max_value);
		}

		/** Returns random number between minimum and maximum value
		 * @param min_value minimum value (32 bit unsigned integer)
		 * @param max_value maximum value (32 bit unsigned integer)
		 * @return unsigned 32 bit integer
		 */
		static inline uint32_t random(uint32_t min_value, uint32_t max_value)
		{
			return sg_rand->random(min_value, max_value);
		}

		/** Returns random number between minimum and maximum value
		 * @param min_value minimum value (32 bit signed integer)
		 * @param max_value maximum value (32 bit signed integer)
		 * @return signed 32 bit integer
		 */
		static inline int32_t random(int32_t min_value, int32_t max_value)
		{
			return sg_rand->random(min_value, max_value);
		}

		/** Returns random number between minimum and maximum value
		 * @param min_value minimum value (32 bit float)
		 * @param max_value maximum value (32 bit float)
		 * @return 32 bit float
		 */
		static inline float32_t random(float32_t min_value, float32_t max_value)
		{
			return sg_rand->random(min_value, max_value);
		}

		/** Returns random number between minimum and maximum value
		 * @param min_value minimum value (64 bit float)
		 * @param max_value maximum value (64 bit float)
		 * @return 64 bit float
		 */
		static inline float64_t random(float64_t min_value, float64_t max_value)
		{
			return sg_rand->random(min_value, max_value);
		}

		/** Returns random number between minimum and maximum value
		 * @param min_value minimum value (128 bit float)
		 * @param max_value maximum value (128 bit float)
		 * @return 128 bit float
		 */
		static inline floatmax_t random(floatmax_t min_value, floatmax_t max_value)
		{
			return sg_rand->random(min_value, max_value);
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
				rand_u = static_cast<float32_t>(CMath::random(-1.0, 1.0));
				rand_v = static_cast<float32_t>(CMath::random(-1.0, 1.0));
				rand_s = rand_u*rand_u + rand_v*rand_v;
			} while ((rand_s == 0) || (rand_s >= 1));

			// the meat & potatos, and then the mean & standard deviation shifting...
			ret = static_cast<float32_t>(
			    rand_u * CMath::sqrt(-2.0 * std::log(rand_s) / rand_s));
			ret = std_dev*ret + mean;
			return ret;
		}

		/// Returns a Gaussian or Normal random number.
		/// Using the polar form of the Box-Muller transform.
		/// http://en.wikipedia.org/wiki/Box%E2%80%93Muller_transform#Polar_form
		static inline float64_t normal_random(float64_t mean, float64_t std_dev)
		{
			return sg_rand->normal_distrib(mean, std_dev);
		}

		/// Convenience method for generating Standard Normal random numbers
		/// Float: Mean = 0 and Standard Deviation = 1
		static inline float32_t randn_float()
		{
			return static_cast<float32_t>(normal_random(0.0, 1.0));
		}

		/// Convenience method for generating Standard Normal random numbers
		/// Double: Mean = 0 and Standard Deviation = 1
		static inline float64_t randn_double()
		{
			return sg_rand->std_normal_distrib();
		}
		//@}

		/** Implements the greatest common divisor (gcd) via modulo operations.
		 * Requires that either a>0 and b>=0 or vice versa.
		 *
		 * @param a first number
		 * @param b second number
		 * @return gcd between the two numbers
		 */
		static int32_t gcd(int32_t a, int32_t b)
		{
			REQUIRE((a>=0 && b>0) || (b>=0 && a>0), "gcd(%d,%d) is not defined.\n",
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

		/** Permute randomly the elements of the vector. If provided, use the
		 * random object to generate the permutations.
		 * @param v the vector to permute.
		 * @param rand random object that might be used to generate the permutations.
		 */
		template <class T>
			static void permute(SGVector<T> v, CRandom* rand=NULL)
			{
				if (rand)
				{
					for (index_t i=0; i<v.vlen; ++i)
						swap(v[i], v[rand->random(i, v.vlen-1)]);
				}
				else
				{
					for (index_t i=0; i<v.vlen; ++i)
						swap(v[i], v[random(i, v.vlen-1)]);
				}
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
			REQUIRE(values.vector, "Values are empty");
			REQUIRE(values.vlen>0,"number of values supplied is 0\n");

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
					values_without_X0[to_idx]=exp(values[from_idx]-X0);
					to_idx++;
				}
			}

			return X0+std::log(SGVector<T>::sum(values_without_X0)+1);
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
						CMath::swap(output[0],output[1]);
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
						CMath::swap(output[left],output[right]);
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
			SG_SERROR("CMath::byte():: Not supported for complex128_t\n");
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
			SG_SERROR("CMath::radix_sort_helper():: Not supported for complex128_t\n");
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

		/** Helper functor for the function argsort */
		template<class T>
			struct IndexSorter
			{
				/** constructor */
				IndexSorter(const SGVector<T> *vec) { data = vec->vector; }

				/** access operator */
				bool operator() (index_t i, index_t j) const
				{
					return abs(data[i]-data[j])>std::numeric_limits<T>::epsilon()
					&& data[i]<data[j];
				}

				/** data */
				const T* data;
			};

#ifndef SWIG // SWIG should skip this part
		/** Get sorted index.
		 *
		 * idx = v.argsort() is similar to Matlab [~, idx] = sort(v)
		 *
		 * @param vector vector to be sorted
		 * @return sorted index for this vector
		 */
		template<class T, class = typename std::enable_if<std::is_arithmetic<T>::value>::type>
			static SGVector<index_t> argsort(SGVector<T> vector)
			{
				IndexSorter<T> cmp(&vector);
				SGVector<index_t> idx(vector.size());
				for (index_t i=0; i < vector.size(); ++i)
					idx[i] = i;

				std::sort(idx.vector, idx.vector+vector.size(), cmp);

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
						SG_SPRINT("1")
					else
						SG_SPRINT("0")

					mask>>=1;
				}
			}
		}

		/// disply_bits not implemented for complex128_t
		static void display_bits(complex128_t word,
			int32_t width=8*sizeof(complex128_t))
		{
			SG_SERROR("CMath::display_bits():: Not supported for complex128_t\n");
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
				SG_SERROR("CMath::qsort_index():: Not supported for complex128_t\n");
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
				SG_SERROR("CMath::qsort_backword_index():: \
					Not supported for complex128_t\n");
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
				SG_SERROR("CMath::parallel_qsort_index():: Not supported for complex128_t\n");
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
			SG_SERROR("CMath::min():: Not supported for complex128_t\n");
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
			SG_SERROR("CMath::nmin():: Not supported for complex128_t\n");
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
			SG_SERROR("CMath::binary_search_helper():: Not supported for complex128_t\n");
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
			SG_SERROR("CMath::binary_search():: Not supported for complex128_t\n");
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
			SG_SERROR("CMath::binary_search():: Not supported for complex128_t\n");
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
			SG_SERROR("CMath::binary_search_max_lower_equal():: \
				Not supported for complex128_t\n");
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

#ifdef USE_LOGCACHE
		/// returns range of logtable
		inline static uint32_t get_log_accuracy()
		{
			return CMath::LOGACCURACY;
		}
#endif

		/// checks whether a float is finite
		static int is_finite(double f);

		/// checks whether a float is infinity
		static int is_infinity(double f);

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

			if (!CMath::is_finite(p))
				return q;

			if (!CMath::is_finite(q))
			{
				SG_SWARNING("INVALID second operand to logsum(%f,%f) expect undefined results\n", p, q)
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

			if (!CMath::is_finite(p))
				return q;
			if (!CMath::is_finite(q))
				return p;
			diff = p - q;
			if (diff > 0)
				return diff > LOGRANGE? p : p + std::log(1 + exp(-diff));
			return -diff > LOGRANGE? q : q + std::log(1 + exp(diff));
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

				/// random generator seed
				static uint32_t seed;

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
void CMath::qsort_index(T1* output, T2* index, uint32_t size)
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
void CMath::qsort_backward_index(T1* output, T2* index, int32_t size)
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
void CMath::nmin(float64_t* output, T* index, int32_t size, int32_t n)
{
	if (6 * n * size < 13 * size * std::log(size))
		for (int32_t i=0; i<n; i++)
			min(&output[i], &index[i], size-i);
	else
		qsort_index(output, index, size);
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

#define COMPLEX128_ERROR_ONEARG_T(function)	\
template <> \
inline complex128_t CMath::function<complex128_t>(complex128_t a)	\
{	\
	SG_SERROR("CMath::%s():: Not supported for complex128_t\n",\
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
