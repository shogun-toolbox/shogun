/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Viktor Gal
 */

#ifndef SHOGUN_ERRORS_H__
#define SHOGUN_ERRORS_H__

#include <system_error>

#include <shogun/lib/common.h>

namespace shogun
{
	namespace io
	{
		enum class ShogunErrc: std::int32_t
		{
			// no 0
			OutOfRange = 10,
			Unknown,
		};

		const std::error_category& category();

		SG_FORCED_INLINE std::error_condition make_error_condition(ShogunErrc e)
		{
			return {static_cast<int>(e), shogun::io::category()};
		}

		SG_FORCED_INLINE std::error_code make_error_code(ShogunErrc e)
		{
			return {static_cast<int>(e), shogun::io::category()};
		}

		SG_FORCED_INLINE bool is_out_of_range(const std::error_condition& ec)
		{
			return ((ec.value() == static_cast<int>(ShogunErrc::OutOfRange)) &&
				(shogun::io::category() == ec.category()));
		}

		SG_FORCED_INLINE std::system_error to_system_error(const std::error_condition& ec)
		{
			return std::system_error(ec.value(), ec.category());
		}
	}
}

namespace std
{
	template <>
		struct is_error_code_enum<shogun::io::ShogunErrc> : true_type {};
}

#endif
