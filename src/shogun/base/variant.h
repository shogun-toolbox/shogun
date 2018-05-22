#ifndef __SG_VARIANT_H__
#define __SG_VARIANT_H__

#include <shogun/lib/config.h>

#if HAVE_STD_VARIANT

#include <variant>

#else

#include <shogun/third_party/variant/variant.hpp>

#endif

namespace shogun
{
#if HAVE_STD_VARIANT
	using std::variant;
	using std::get;
	using std::get_if;
	using std::visit;
	using std::holds_alternative;
	using std::monostate;
	using std::bad_variant_access;
	using std::variant_size;
	using std::variant_size_v;
	using std::variant_alternative;
	using std::variant_alternative_t;
#else
	using mpark::variant;
	using mpark::get;
	using mpark::get_if;
	using mpark::visit;
	using mpark::holds_alternative;
	using mpark::monostate;
	using mpark::bad_variant_access;
	using mpark::variant_size;
	using mpark::variant_size_v;
	using mpark::variant_alternative;
	using mpark::variant_alternative_t;
#endif
}

#endif // __SG_VARIANT_H__
