/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Viktor Gal
 */
#ifndef __UTIL_CONVERTERS_H__
#define __UTIL_CONVERTERS_H__

#include <limits>
#include <stdexcept>
#include <type_traits>

#include <shogun/lib/common.h>

namespace shogun
{
	namespace utils
	{
		template<typename T>
		using try_make_unsigned =
			typename std::conditional_t<
				std::is_integral<T>::value,
				std::make_unsigned<T>,
				std::common_type<T>
			>;

		template <typename I, typename J>
		static typename std::enable_if_t<
		    std::is_signed<I>::value && std::is_signed<J>::value, I>
		safe_convert(J value)
		{
			if (std::isfinite(value) &&
			    (value < std::numeric_limits<I>::lowest() ||
			     value > std::numeric_limits<I>::max()))
				throw std::overflow_error(
				    "value cannot be stored in a variable of type requested");
			return static_cast<I>(value);
		}

		template <typename I, typename J>
		static typename std::enable_if_t<
		    std::is_signed<I>::value && std::is_unsigned<J>::value, I>
		safe_convert(J value)
		{
			if (value > static_cast<typename try_make_unsigned<I>::type>(
			                std::numeric_limits<I>::max()))
				throw std::overflow_error(
				    "value cannot be stored in a variable of type requested");
			return static_cast<I>(value);
		}

		template <typename I, typename J>
		static typename std::enable_if_t<
		    std::is_unsigned<I>::value && std::is_signed<J>::value, I>
		safe_convert(J value)
		{
			if (value < 0 ||
			    static_cast<typename try_make_unsigned<J>::type>(value) >
			        std::numeric_limits<I>::max())
				throw std::overflow_error(
				    "value cannot be stored in a variable of type requested");
			return static_cast<I>(value);
		}

		template <typename I, typename J>
		static typename std::enable_if_t<
		    std::is_unsigned<I>::value && std::is_unsigned<J>::value, I>
		safe_convert(J value)
		{
			if (value > std::numeric_limits<I>::max())
				throw std::overflow_error(
				    "value cannot be stored in a variable of type requested");
			return static_cast<I>(value);
		}
	}
}

#endif
