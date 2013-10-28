/* This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Copyright (c) 2012-2013 Sergey Lisitsyn
 */

#ifndef TAPKEE_PARAMETERS_H_
#define TAPKEE_PARAMETERS_H_

/* Tapkee includes */
#include <shogun/lib/tapkee/parameters/value_keeper.hpp>
/* End of Tapkee includes */

#include <sstream>
#include <vector>
#include <map>

namespace tapkee
{

struct Message
{
	Message() : ss()
	{
	}
	template <typename T>
	Message& operator<<(const T& data)
	{
		ss << data;
		return *this;
	}
	operator std::string()
	{
		return ss.str();
	}

	std::stringstream ss;
};

class ParametersSet;
class CheckedParameter;

class Parameter
{
	friend class CheckedParameter;

	typedef std::string ParameterName;

private:

	template <typename T>
	Parameter(const ParameterName& pname, const T& value) :
		valid(true), invalidity_reason(),
		parameter_name(pname), keeper(tapkee_internal::ValueKeeper(value))
	{
	}

public:

	template <typename T>
	static Parameter create(const std::string& name, const T& value)
	{
		return Parameter(name, value);
	}

	Parameter() :
		valid(false), invalidity_reason(),
		parameter_name("unknown"), keeper(tapkee_internal::ValueKeeper())
	{
	}

	Parameter(const Parameter& p) :
		valid(p.valid), invalidity_reason(p.invalidity_reason),
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
			keeper = tapkee_internal::ValueKeeper(value);
		}
		return *this;
	}

	template <typename T>
	inline operator T()
	{
		if (!valid)
		{
			throw wrong_parameter_error(invalidity_reason);
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

	template <typename T>
	bool isInRange(T lower, T upper) const
	{
		return keeper.inRange<T>(lower, upper);
	}

	template <typename T>
	bool isEqual(T value) const
	{
		return keeper.equal<T>(value);
	}

	template <typename T>
	bool isNotEqual(T value) const
	{
		return keeper.notEqual<T>(value);
	}

	bool isPositive() const
	{
		return keeper.positive();
	}

	bool isNonNegative() const
	{
		return keeper.nonNegative();
	}

	bool isNegative() const
	{
		return keeper.negative();
	}

	template <typename T>
	bool isGreater(T lower) const
	{
		return keeper.greater<T>(lower);
	}

	template <typename T>
	bool isLesser(T upper) const
	{
		return keeper.lesser<T>(upper);
	}

	bool isInitialized() const
	{
		return keeper.isInitialized();
	}

	ParameterName name() const
	{
		return parameter_name;
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
		valid = false;
		invalidity_reason = reason;
	}

private:

	bool valid;
	std::string invalidity_reason;

	ParameterName parameter_name;

	tapkee_internal::ValueKeeper keeper;

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

	template <typename T>
	bool operator==(T v)
	{
		return is<T>(v);
	}

	template <typename T>
	CheckedParameter& inRange(T lower, T upper)
	{
		if (!parameter.isInRange(lower, upper))
		{
			std::string reason =
				(Message() << "Value of " << parameter.name() << " "
				 << parameter.getValue<T>() << " doesn't fit the range [" <<
				    lower << ", " << upper << ")");
			parameter.invalidate(reason);
		}
		return *this;
	}

	template <typename T>
	CheckedParameter& inClosedRange(T lower, T upper)
	{
		if (!parameter.isInRange(lower, upper) && !parameter.is(upper))
		{
			std::string reason =
				(Message() << "Value of " << parameter.name() << " "
				 << parameter.getValue<T>() << " doesn't fit the range [" <<
				    lower << ", " << upper << "]");
			parameter.invalidate(reason);
		}
		return *this;
	}

	CheckedParameter& positive()
	{
		if (!parameter.isPositive())
		{
			std::string reason =
				(Message() << "Value of " << parameter.name() << " is not positive");
			parameter.invalidate(reason);
		}
		return *this;
	}

	CheckedParameter& nonNegative()
	{
		if (!parameter.isNonNegative())
		{
			std::string reason =
				(Message() << "Value of " << parameter.name() << " is negative");
			parameter.invalidate(reason);
		}
		return *this;
	}


private:

	Parameter& parameter;

};

CheckedParameter Parameter::checked()
{
	return CheckedParameter(*this);
}

class ParametersSet
{
public:

	typedef std::map<std::string, Parameter> ParametersMap;

	ParametersSet() : pmap()
	{
	}
	void add(const Parameter& p)
	{
		if (pmap.count(p.name()))
			throw multiple_parameter_error(Message() << "Parameter " << p.name() << " is set more than once");

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
	Parameter operator()(const std::string& name) const
	{
		ParametersMap::const_iterator it = pmap.find(name);
		if (it != pmap.end())
		{
			return it->second;
		}
		else
		{
			throw missed_parameter_error(Message() << "Parameter " << name << " is missed");
		}
	}
	ParametersSet& operator,(const Parameter& p)
	{
		add(p);
		return *this;
	}

private:

	ParametersMap pmap;
};

ParametersSet Parameter::operator,(const Parameter& p)
{
	ParametersSet pg;
	pg.add(*this);
	pg.add(p);
	return pg;
}

Parameter::operator ParametersSet()
{
	ParametersSet pg;
	pg.add(*this);
	return pg;
}


}

#endif
