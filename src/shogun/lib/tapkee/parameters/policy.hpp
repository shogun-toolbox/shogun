/* This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Copyright (c) 2012-2013 Sergey Lisitsyn
 */

#ifndef TAPKEE_POLICY_H_
#define TAPKEE_POLICY_H_

namespace tapkee
{
namespace tapkee_internal
{

struct TypePolicyBase
{
	virtual ~TypePolicyBase() {}
	virtual void copyFromValue(void const*, void**) const = 0;
	virtual void* getValue(void**) const = 0;
	virtual void free(void**) const = 0;
	virtual void clone(void* const*, void**) const = 0;
	virtual void move(void* const*, void**) const = 0;
};

template <typename T>
struct PointerTypePolicyImpl : public TypePolicyBase
{
	inline virtual void copyFromValue(void const* src, void** dest) const
	{
		*dest = new T(*reinterpret_cast<T const*>(src));
	}
	inline virtual void* getValue(void** src) const
	{
		return *src;
	}
	inline virtual void free(void** src) const
	{
		if (*src)
			delete (*reinterpret_cast<T**>(src));
		*src = NULL;
	}
	virtual void clone(void* const* src, void** dest) const
	{
		if (*dest)
			(*reinterpret_cast<T**>(dest))->~T();
		*dest = new T(**reinterpret_cast<T* const*>(src));
	}
	inline virtual void move(void* const* src, void** dest) const
	{
		(*reinterpret_cast<T**>(dest))->~T();
		**reinterpret_cast<T**>(dest) = **reinterpret_cast<T* const*>(src);
	}
};

template <typename T>
TypePolicyBase* getPolicy()
{
	static PointerTypePolicyImpl<T> policy;
	return &policy;
}

struct CheckerPolicyBase
{
	virtual ~CheckerPolicyBase() {}
	virtual bool isInRange(void* const*, void*, void*) const = 0;
	virtual bool isEqual(void* const*, void*) const = 0;
	virtual bool isNotEqual(void* const*, void*) const = 0;
	virtual bool isPositive(void* const*) const = 0;
	virtual bool isNonNegative(void * const*) const = 0;
	virtual bool isNegative(void* const*) const = 0;
	virtual bool isNonPositive(void * const*) const = 0;
	virtual bool isGreater(void* const*, void*) const = 0;
	virtual bool isLesser(void* const*, void*) const = 0;
};

template <typename T>
struct PointerCheckerPolicyImpl : public CheckerPolicyBase
{
	virtual bool isInRange(void* const*, void*, void*) const
	{
		return false;
	}
	virtual bool isEqual(void* const*, void*) const
	{
		return false;
	}
	virtual bool isNotEqual(void* const*, void*) const
	{
		return false;
	}
	virtual bool isPositive(void* const*) const
	{
		return false;
	}
	virtual bool isNonNegative(void* const*) const
	{
		return false;
	}
	virtual bool isNegative(void* const*) const
	{
		return false;
	}
	virtual bool isNonPositive(void* const*) const
	{
		return false;
	}
	virtual bool isGreater(void* const*, void*) const
	{
		return false;
	}
	virtual bool isLesser(void* const*, void*) const
	{
		return false;
	}
};

#define default_policy_for(Type) \
template <>																		\
struct PointerCheckerPolicyImpl<Type> : public CheckerPolicyBase				\
{																				\
	inline Type value(void* v) const											\
	{																			\
		return *reinterpret_cast<Type*>(v);										\
	}																			\
	virtual bool isInRange(void* const* src, void* lower, void* upper) const	\
	{																			\
		Type v = value(*src);													\
		Type l = value(lower);													\
		Type u = value(upper);													\
		return (v>=l) && (v<u);													\
	}																			\
	virtual bool isEqual(void* const* src, void* other_src) const				\
	{																			\
		Type v = value(*src);													\
		Type ov = value(other_src);												\
		return (v==ov);															\
	}																			\
	virtual bool isNotEqual(void* const* src, void* other_src) const			\
	{																			\
		Type v = value(*src);													\
		Type ov = value(other_src);												\
		return (v!=ov);															\
	}																			\
	virtual bool isPositive(void* const* src) const								\
	{																			\
		Type v = value(*src);													\
		return (v>0);															\
	}																			\
	virtual bool isNonNegative(void* const* src) const							\
	{																			\
		Type v = value(*src);													\
		return (v>=0);															\
	}																			\
	virtual bool isNegative(void* const* src) const								\
	{																			\
		Type v = value(*src);													\
		return (v<0);															\
	}																			\
	virtual bool isNonPositive(void* const* src) const							\
	{																			\
		Type v = value(*src);													\
		return (v<=0);															\
	}																			\
	virtual bool isGreater(void* const* src, void* lower) const					\
	{																			\
		Type v = value(*src);													\
		return (v>value(lower));												\
	}																			\
	virtual bool isLesser(void* const* src, void* upper) const					\
	{																			\
		Type v = value(*src);													\
		return (v<value(upper));												\
	}																			\
}

default_policy_for(double);
default_policy_for(float);
default_policy_for(int);

template <typename T>
CheckerPolicyBase* getCheckerPolicy()
{
	static PointerCheckerPolicyImpl<T> policy;
	return &policy;
}

}
}
#endif
