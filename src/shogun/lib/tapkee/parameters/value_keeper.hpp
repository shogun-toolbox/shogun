/* This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Copyright (c) 2012-2013 Sergey Lisitsyn
 */

#ifndef TAPKEE_VALUE_KEEPER_H_
#define TAPKEE_VALUE_KEEPER_H_

/* Tapkee includes */
#include <shogun/lib/tapkee/parameters/policy.hpp>
/* End of Tapkee includes */

namespace tapkee
{
namespace tapkee_internal
{

struct EmptyType
{
};

class ValueKeeper
{

public:
	template <typename T>
	explicit ValueKeeper(const T& value) : 
		policy(get_policy<T>()), checker(get_checker_policy<T>()), value_ptr(NULL) 
	{
		policy->copy_from_value(&value, &value_ptr);
	}

	ValueKeeper() :
		policy(get_policy<EmptyType>()), checker(get_checker_policy<EmptyType>()), value_ptr(NULL) 
	{
	}

	~ValueKeeper()
	{
		policy->free(&value_ptr);
	}

	ValueKeeper(const ValueKeeper& v) : policy(v.policy), checker(v.checker), value_ptr(NULL)
	{
		policy->clone(&(v.value_ptr), &value_ptr);
	}

	ValueKeeper& operator=(const ValueKeeper& v)
	{
		policy->free(&value_ptr);
		policy = v.policy;
		checker = v.checker;
		policy->clone(&(v.value_ptr), &value_ptr);
		return *this;
	}

	template <typename T>
	inline T get_value() const
	{
		T* v;
		if (!is_initialized())
		{
			throw missed_parameter_error("Parameter is missed");
		}
		if (is_type_correct<T>())
		{
			void* vv = policy->get_value(const_cast<void**>(&value_ptr));
			v = reinterpret_cast<T*>(vv);
		}
		else
			throw wrong_parameter_type_error("Wrong value type");
		return *v;
	}

	template <typename T>
	inline bool is_type_correct() const
	{
		return get_policy<T>() == policy;
	}

	inline bool is_initialized() const
	{
		return get_policy<EmptyType>() != policy;
	}

	template <typename T>
	inline bool in_range(T lower, T upper) const
	{
		if (!is_type_correct<T>() && is_initialized())
			throw std::domain_error("Wrong range bounds type");
		return checker->is_in_range(&value_ptr,&lower,&upper);
	}

	template <typename T>
	inline bool equal(T value) const
	{
		if (!is_type_correct<T>() && is_initialized())
			throw std::domain_error("Wrong equality value type");
		return checker->is_equal(&value_ptr,&value);
	}

	template <typename T>
	inline bool not_equal(T value) const
	{
		if (!is_type_correct<T>() && is_initialized())
			throw std::domain_error("Wrong non-equality value type");
		return checker->is_not_equal(&value_ptr,&value);
	}

	inline bool positive() const
	{
		return checker->is_positive(&value_ptr);
	}

	inline bool negative() const
	{
		return checker->is_negative(&value_ptr);
	}

	template <typename T>
	inline bool greater(T lower) const
	{
		if (!is_type_correct<T>() && is_initialized())
			throw std::domain_error("Wrong greater check bound type");
		return checker->is_greater(&value_ptr,&lower);
	}

	template <typename T>
	inline bool lesser(T upper) const
	{
		if (!is_type_correct<T>() && is_initialized())
			throw std::domain_error("Wrong lesser check bound type");
		return checker->is_lesser(&value_ptr,&upper);
	}

private:
	TypePolicyBase* policy;
	CheckerPolicyBase* checker;
	void* value_ptr;

};

}
}
#endif
