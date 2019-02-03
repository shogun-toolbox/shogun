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

#ifndef STICHWORT_PARAMETER_H_
#define STICHWORT_PARAMETER_H_

#include <shogun/lib/tapkee/stichwort/value_keeper.hpp>

#include <sstream>
#include <vector>
#include <map>
#include <iostream>
#include <list>

namespace stichwort
{

class ParametersSet;
class CheckedParameter;

class Parameter
{
	friend class CheckedParameter;

	typedef std::string ParameterName;

private:

	template <typename T>
	Parameter(const ParameterName& pname, const T& value) :
		valid(true), invalidity_reasons(),
		parameter_name(pname), keeper(stichwort_internal::ValueKeeper(value))
	{
	}

public:

	template <typename T>
	static Parameter create(const std::string& name, const T& value)
	{
		return Parameter(name, value);
	}

	Parameter() :
		valid(true), invalidity_reasons(),
		parameter_name("unknown"), keeper(stichwort_internal::ValueKeeper())
	{
	}

	Parameter(const Parameter& p) :
		valid(p.valid), invalidity_reasons(p.invalidity_reasons),
		parameter_name(p.name()), keeper(p.keeper)
	{
	}

	~Parameter()
	{
	}

	template <typename T>
	inline Parameter withDefault(T value)
	{
		if (!isInitialized())
		{
			keeper = stichwort_internal::ValueKeeper(value);
		}
		return *this;
	}

	template <typename T>
	inline operator T()
	{
		if (!valid)
		{
			throw wrong_parameter_error(invalidity_reasons);
		}
		try
		{
			return getValue<T>();
		}
		catch (const missed_parameter_error&)
		{
			throw missed_parameter_error(parameter_name + " is missed");
		}
	}

	operator ParametersSet();

	template <typename T>
	bool is(T v)
	{
		if (!isTypeCorrect<T>())
			return false;
		T kv = keeper.getValue<T>();
		if (v == kv)
			return true;
		return false;
	}

	template <typename T>
	bool operator==(T v) const
	{
		return is<T>(v);
	}

	CheckedParameter checked();

	bool isInitialized() const
	{
		return keeper.isInitialized();
	}

	template <template<class> class F, class Q>
	inline bool isCondition(F<Q> cond) const
	{
		return keeper.isCondition(cond);
	}

	ParameterName name() const
	{
		return parameter_name;
	}

	std::string repr() const
	{
		return keeper.repr();
	}

	ParametersSet operator,(const Parameter& p);

private:

	template <typename T>
	inline T getValue() const
	{
		return keeper.getValue<T>();
	}

	template <typename T>
	inline bool isTypeCorrect() const
	{
		return keeper.isTypeCorrect<T>();
	}

	inline void invalidate(const std::string& reason)
	{
		if (valid)
			valid = false;

		invalidity_reasons += reason;
	}

private:

	bool valid;
	std::string invalidity_reasons;
	ParameterName parameter_name;
	stichwort_internal::ValueKeeper keeper;

};

class CheckedParameter
{

public:

	explicit CheckedParameter(Parameter& p) : parameter(p)
	{
	}

	inline operator const Parameter&()
	{
		return parameter;
	}

	template <typename T>
	bool is(T v)
	{
		return parameter.is<T>(v);
	}

	template <template<class> class F, class Q>
	inline Parameter& satisfies(const F<Q>& cond) const
	{
		if (!parameter.isCondition(cond))
			parameter.invalidate(cond.failureMessage(parameter));

		return parameter;
	}

	template <typename T>
	bool operator==(T v)
	{
		return is<T>(v);
	}

private:

	Parameter& parameter;

};

inline CheckedParameter Parameter::checked()
{
	return CheckedParameter(*this);
}

class ParametersSet
{
public:

	typedef std::map<std::string, Parameter> ParametersMap;
	typedef std::list<std::string> DuplicatesList;

	ParametersSet() : pmap(), dups()
	{
	}
	ParametersSet(const ParametersSet& other) : pmap(other.pmap), dups(other.dups)
	{
	}
	ParametersSet& operator=(const ParametersSet& other)
	{
		this->pmap = other.pmap;
		this->dups = other.dups;
		return *this;
	}
	void check()
	{
		if (!dups.empty())
		{
			std::stringstream ss;
			ss << "The following parameters are set more than once: ";
			for (DuplicatesList::const_iterator iter=dups.begin(); iter!=dups.end(); ++iter)
				ss << *iter << " ";

			throw multiple_parameter_error(ss.str());
		}
	}
	void add(const Parameter& p)
	{
		if (pmap.count(p.name()))
			dups.push_back(p.name());

		pmap[p.name()] = p;
	}
	bool contains(const std::string& name) const
	{
		return pmap.count(name) > 0;
	}
	void merge(const ParametersSet& pg)
	{
		typedef ParametersMap::const_iterator MapIter;
		for (MapIter iter = pg.pmap.begin(); iter!=pg.pmap.end(); ++iter)
		{
			if (!pmap.count(iter->first))
			{
				pmap[iter->first] = iter->second;
			}
		}
	}
	Parameter operator[](const std::string& name) const
	{
		ParametersMap::const_iterator it = pmap.find(name);
		if (it != pmap.end())
		{
			return it->second;
		}
		else
		{
			throw missed_parameter_error(name + " is missed");
		}
	}
	ParametersSet& operator,(const Parameter& p)
	{
		add(p);
		return *this;
	}

private:

	ParametersMap pmap;
	DuplicatesList dups;
};

inline ParametersSet Parameter::operator,(const Parameter& p)
{
	ParametersSet pg;
	pg.add(*this);
	pg.add(p);
	return pg;
}

inline Parameter::operator ParametersSet()
{
	ParametersSet pg;
	pg.add(*this);
	return pg;
}


}

#endif
