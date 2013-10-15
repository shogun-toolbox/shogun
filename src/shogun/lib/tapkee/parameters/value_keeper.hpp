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
		policy(getPolicy<T>()), checker(getCheckerPolicy<T>()), value_ptr(NULL)
	{
		policy->copyFromValue(&value, &value_ptr);
	}

	ValueKeeper() :
		policy(getPolicy<EmptyType>()), checker(getCheckerPolicy<EmptyType>()), value_ptr(NULL)
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
	inline T getValue() const
	{
		T* v;
		if (!isInitialized())
		{
			throw missed_parameter_error("Parameter is missed");
		}
		if (isTypeCorrect<T>())
		{
			void* vv = policy->getValue(const_cast<void**>(&value_ptr));
			v = reinterpret_cast<T*>(vv);
		}
		else
			throw wrong_parameter_type_error("Wrong value type");
		return *v;
	}

	template <typename T>
	inline bool isTypeCorrect() const
	{
		return getPolicy<T>() == policy;
	}

	inline bool isInitialized() const
	{
		return getPolicy<EmptyType>() != policy;
	}

	template <typename T>
	inline bool inRange(T lower, T upper) const
	{
		if (!isTypeCorrect<T>() && isInitialized())
			throw std::domain_error("Wrong range bounds type");
		return checker->isInRange(&value_ptr,&lower,&upper);
	}

	template <typename T>
	inline bool equal(T value) const
	{
		if (!isTypeCorrect<T>() && isInitialized())
			throw std::domain_error("Wrong equality value type");
		return checker->isEqual(&value_ptr,&value);
	}

	template <typename T>
	inline bool notEqual(T value) const
	{
		if (!isTypeCorrect<T>() && isInitialized())
			throw std::domain_error("Wrong non-equality value type");
		return checker->isNotEqual(&value_ptr,&value);
	}

	inline bool positive() const
	{
		return checker->isPositive(&value_ptr);
	}

	inline bool nonNegative() const
	{
		return checker->isNonNegative(&value_ptr);
	}

	inline bool negative() const
	{
		return checker->isNegative(&value_ptr);
	}

	inline bool nonPositive() const
	{
		return checker->isNonPositive(&value_ptr);
	}

	template <typename T>
	inline bool greater(T lower) const
	{
		if (!isTypeCorrect<T>() && isInitialized())
			throw std::domain_error("Wrong greater check bound type");
		return checker->isGreater(&value_ptr,&lower);
	}

	template <typename T>
	inline bool lesser(T upper) const
	{
		if (!isTypeCorrect<T>() && isInitialized())
			throw std::domain_error("Wrong lesser check bound type");
		return checker->isLesser(&value_ptr,&upper);
	}

private:

	TypePolicyBase* policy;
	CheckerPolicyBase* checker;
	void* value_ptr;

};

}
}
#endif
