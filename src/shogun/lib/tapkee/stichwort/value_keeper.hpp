/** Stichwort
 *
 * Copyright (c) 2013, Sergey Lisitsyn <lisitsyn.s.o@gmail.com>
 * All rights reserved.
 *
 * Distributed under the BSD 2-clause license:
 *
 * Redistribution and use in source and binary forms, with or without modification,
 * are permitted provided that the following conditions are met:
 *
 * * Redistributions of source code must retain the above copyright notice,
 *   this list of conditions and the following disclaimer.
 * * Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
 * OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT
 * SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA,
 * OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 * WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef STICHWORT_KEEPER_H_
#define STICHWORT_KEEPER_H_

#include <shogun/lib/tapkee/stichwort/policy.hpp>
#include <shogun/lib/tapkee/stichwort/exceptions.hpp>

namespace stichwort
{
namespace stichwort_internal
{

struct EmptyType
{
};

class ValueKeeper
{

public:
	template <typename T>
	explicit ValueKeeper(const T& value) :
		policy(getPolicy<T>()), value_ptr(NULL)
	{
		policy->copyFromValue(&value, &value_ptr);
	}

	ValueKeeper() :
		policy(getPolicy<EmptyType>()), value_ptr(NULL)
	{
	}

	~ValueKeeper()
	{
		policy->free(&value_ptr);
	}

	ValueKeeper(const ValueKeeper& v) : policy(v.policy), value_ptr(NULL)
	{
		policy->clone(&(v.value_ptr), &value_ptr);
	}

	ValueKeeper& operator=(const ValueKeeper& v)
	{
		policy->free(&value_ptr);
		policy = v.policy;
		policy->clone(&(v.value_ptr), &value_ptr);
		return *this;
	}

	template <typename T>
	inline T getValue() const
	{
		T* v;
		if (!isInitialized())
			throw missed_parameter_error("Parameter is missed");

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

	template <template<class> class F, class Q>
	inline bool isCondition(F<Q> cond) const
	{
		Q value = getValue<Q>();
		return cond(value);
	}

	inline std::string repr() const
	{
		return policy->repr(const_cast<void**>(&value_ptr));
	}

private:

	TypePolicyBase* policy;
	void* value_ptr;

};

}
}
#endif
