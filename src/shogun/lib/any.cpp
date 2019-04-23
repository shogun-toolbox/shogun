/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Heiko Strathmann, Sergey Lisitsyn, Gil Hoben
 */

#include <shogun/lib/any.h>
#include <shogun/mathematics/Math.h>
#ifdef HAVE_CXA_DEMANGLE
#include <cxxabi.h>
#endif

namespace shogun
{
	namespace any_detail {
		std::string demangled_type_helper(const char *name) {
#ifdef HAVE_CXA_DEMANGLE
			size_t length;
			int status;
			char *demangled = abi::__cxa_demangle(name, nullptr, &length, &status);
			std::string demangled_string(demangled);
			free(demangled);
#else
			std::string demangled_string(name);
#endif
			return demangled_string;
		}
	}

	namespace any_detail
	{

#ifndef REAL_COMPARE_IMPL
#define REAL_COMPARE_IMPL(real_t)                                              \
	template <>                                                                \
	bool compare_impl_eq(const real_t& lhs, const real_t& rhs)                 \
	{                                                                          \
		SG_DEBUG("Comparing using fequals<" #real_t ">(lhs, rhs).");        \
		return Math::fequals(                                                 \
		    lhs, rhs, std::numeric_limits<real_t>::epsilon());                 \
	}

		REAL_COMPARE_IMPL(float32_t)
		REAL_COMPARE_IMPL(float64_t)
		REAL_COMPARE_IMPL(floatmax_t)
#undef REAL_COMPARE_IMPL
#endif // REAL_COMPARE_IMPL

		template <>
		bool compare_impl_eq(const complex128_t& lhs, const complex128_t& rhs)
		{
			SG_DEBUG("Comparing using fequals<complex128_t>(lhs, rhs).");
			return Math::fequals(lhs.real(), rhs.real(), LDBL_EPSILON) &&
			       Math::fequals(lhs.imag(), rhs.imag(), LDBL_EPSILON);
		}

		void free_object(SGObject* obj)
		{
			//FIXME
			//SG_UNREF(obj);
		}
	}

	void Any::set_or_inherit(const Any& other)
	{
		if (other.policy->should_inherit_storage())
		{
			storage = other.storage;
		}
		else
		{
			policy->set(&storage, other.storage);
		}
	}

	Any::CastingRegistry Any::casting_registry = {};

	Any::VisitorRegistry Any::visitor_registry = {};

	Any::Any() : Any(owning_policy<Empty>(), nullptr)
	{
	}

	Any::Any(const BaseAnyPolicy* the_policy, void* the_storage)
		: policy(the_policy), storage(the_storage)
	{
	}

	Any::Any(const Any& other) : Any(other.policy, nullptr)
	{
		set_or_inherit(other);
	}

	/** Move constructor */
	Any::Any(Any&& other) : Any(other.policy, nullptr)
	{
		set_or_inherit(other);
	}

	Any::~Any()
	{
		policy->clear(&storage);
	}

	Any& Any::operator=(const Any& other)
	{
		if (empty())
		{
			policy = other.policy;
			set_or_inherit(other);
			return *(this);
		}
		if (!policy->matches_policy(other.policy))
		{
			throw TypeMismatchException(
				other.policy->type(), policy->type());
		}
		policy->clear(&storage);
		if (other.policy->should_inherit_storage())
		{
			policy = other.policy;
		}
		set_or_inherit(other);
		return *(this);
	}

	Any& Any::clone_from(const Any& other)
	{
		if (!other.cloneable())
		{
			throw std::logic_error("Tried to clone non-cloneable Any");
		}
		if (empty())
		{
			policy = other.policy;
			set_or_inherit(other);
			return *(this);
		}
		if (!policy->matches_policy(other.policy))
		{
			throw TypeMismatchException(
				other.policy->type(), policy->type());
		}
		policy->clone(&storage, other.storage);
		return *(this);
	}

	bool Any::empty() const
	{
		return has_type<Empty>();
	}

	bool Any::cloneable() const
	{
		return !policy->is_functional();
	}

	bool Any::comparable() const
	{
		return !policy->is_functional();
	}

	bool Any::visitable() const
	{
		return !policy->is_functional() || (policy->is_functional() && !policy->is_void());
	}

	bool Any::safe_visitable() const
	{
		const bool is_safe = policy->is_functional() ? policy->is_nothrow() : true;
		return visitable() && is_safe;
	}

	bool Any::hashable() const
	{
		return !policy->is_functional();
	}

	const std::type_info& Any::type_info() const
	{
		return policy->type_info();
	}

	std::string Any::type() const
	{
		return policy->type();
	}

	size_t Any::hash() const
	{
		return policy->hash(storage);
	}

	void Any::visit(AnyVisitor* visitor) const
	{
		if (!visitable())
		{
			throw std::logic_error("Tried to visit non-visitable Any");
		}
		policy->visit(storage, visitor);
	}

	bool operator==(const Any& lhs, const Any& rhs)
	{
		if (lhs.empty() || rhs.empty())
		{
			return lhs.empty() && rhs.empty();
		}
		if (!lhs.policy->matches_policy(rhs.policy))
		{
			return false;
		}
		void* lhs_storage = lhs.storage;
		void* rhs_storage = rhs.storage;
		return lhs.policy->equals(lhs_storage, rhs_storage);
	}

	bool operator!=(const Any& lhs, const Any& rhs)
	{
		return !(lhs == rhs);
	}

}
