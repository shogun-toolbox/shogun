/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Viktor Gal
 */

#include <shogun/io/ShogunErrc.h>

using namespace std;
using namespace shogun::io;

namespace shogun
{
	namespace io
	{
		namespace detail
		{
			struct shogun_category: std::error_category
			{
				const char* name() const noexcept override
				{
					return "sg";
				}

				error_condition default_error_condition(int ev) const noexcept override
				{
					switch (ev)
					{
						case static_cast<int>(ShogunErrc::OutOfRange):
							return make_error_condition(ShogunErrc::OutOfRange);
						default:
							return make_error_condition(ShogunErrc::Unknown);
					}
				}

				bool equivalent(const error_code& code, int condition) const noexcept override
				{
					return *this==code.category() &&
							static_cast<int>(default_error_condition(code.value()).value()) == condition;

				}

				bool equivalent(int code, const error_condition& condition) const noexcept override
				{
					return default_error_condition(code) == condition;
				}

				string message(int ev) const override
				{
					switch (ev)
					{
						case static_cast<int>(ShogunErrc::OutOfRange):
							return "Read less bytes than requested.";
						default:
							return "Unknown error!";
					}
				}
			};
		}

		const std::error_category& category()
		{
			static detail::shogun_category instance;
			return instance;
		}
	}
}
