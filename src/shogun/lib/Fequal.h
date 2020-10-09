
#ifndef __FEQUAL_H__
#define __FEQUAL_H__

#include <shogun/base/ShogunEnv.h>
#include <shogun/mathematics/Math.h>


namespace shogun
{

#ifndef SWIG

	/** Compares the value of two floats based on eps only
	 * @param a first value to compare
	 * @param b second value to compare
	 * @param eps threshold for values to be equal/different
	 * @return true if values are equal within eps accuracy, false if not.
	 */
	template <class T, class = typename std::enable_if<std::is_floating_point<T>::value>::type>
	static inline bool 
	fequals_abs(const T& a, const T& b,const float64_t eps)
	{
		const T diff = Math::abs<T>((a-b));
		return (diff < eps);
	}

	/** Compares the value of two floats (handles special cases, such as NaN, Inf
	 * etc.) Note: returns true if a == b == NAN Implementation inspired by
	 * http://floating-point-gui.de/errors/comparison/
	 * @param a first value to compare
	 * @param b second value to compare
	 * @param eps threshold for values to be equal/different
	 * @return true if values are equal within eps accuracy, false if not.
	 */
	template <class T, class = typename std::enable_if<std::is_floating_point<T>::value>::type>
	static inline bool 
	fequals(const T& a, const T& b, const float64_t eps_)
	{
		// global fequals epsilon might override passed one
		// hack for lossy serialization formats
		float64_t eps = std::max(eps_, env()->fequals_epsilon());

		const T absA = Math::abs<T>(a);
		const T absB = Math::abs<T>(b);
		const T diff = Math::abs<T>((a - b));

		// Handle this separately since NAN is unordered
		if (Math::is_nan((float64_t)a) && Math::is_nan((float64_t)b))
			return true;

		// Required for JSON Serialization Tests
		if (env()->fequals_tolerant())
			return fequals_abs<T>(a, b, eps);

		// handles float32_t and float64_t separately
		T comp = (std::is_same<float32_t, T>::value) ? Math::F_MIN_NORM_VAL32
		                                             : Math::F_MIN_NORM_VAL64;

		if (a == b)
			return true;

		// both a and b are 0 and relative error is less meaningful
		else if ((a == 0) || (b == 0) || (diff < comp))
			return (diff < (eps * comp));
		// use max(relative error, diff) to handle large eps
		else
		{
			T check =
			    ((diff / (absA + absB)) > diff) ? (diff / (absA + absB)) : diff;
			return (check < eps);
		}
	}

#endif

}	//namespace shogun

#endif	// __FEQUAL_H__