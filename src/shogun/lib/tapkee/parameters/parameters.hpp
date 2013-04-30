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

using std::vector;
using std::string;
using std::stringstream;

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
	operator string() 
	{
		return ss.str();
	}

	stringstream ss;
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
		parameter_name(pname), keeper(tapkee_internal::ValueKeeper(value))
	{
	}

public:

	template <typename T>
	static Parameter create(const std::string& name, const T& value) 
	{
		return Parameter(name, value);
	}

	Parameter() : parameter_name("unknown"), keeper(tapkee_internal::ValueKeeper())
	{
	}

	Parameter(const Parameter& p) : parameter_name(p.name()), keeper(p.keeper)
	{
	}

	~Parameter()
	{
	}

	template <typename T>
	inline Parameter with_default(T value)
	{
		if (!is_initialized())
		{
			keeper = tapkee_internal::ValueKeeper(value);
		}
		return *this;
	}

	template <typename T>
	inline operator T()
	{
		try 
		{
			return get_value<T>();
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
		if (!is_type_correct<T>())
			return false;
		T kv = keeper.get_value<T>();
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
	bool in_range(T lower, T upper) const
	{
		return keeper.in_range<T>(lower, upper);
	}

	template <typename T>
	bool equal(T value) const
	{
		return keeper.equal<T>(value);
	}

	template <typename T>
	bool not_equal(T value) const
	{
		return keeper.not_equal<T>(value);
	}

	bool positive() const 
	{
		return keeper.positive();
	}

	bool negative() const
	{
		return keeper.negative();
	}

	template <typename T>
	bool greater(T lower) const
	{
		return keeper.greater<T>(lower);
	}

	template <typename T>
	bool is_lesser(T upper) const
	{
		return keeper.lesser<T>(upper);
	}

	bool is_initialized() const
	{
		return keeper.is_initialized();
	}

	ParameterName name() const 
	{
		return parameter_name;
	}

	ParametersSet operator,(const Parameter& p);

private:

	template <typename T>
	inline T get_value() const
	{
		return keeper.get_value<T>();
	}
	
	template <typename T>
	inline bool is_type_correct() const
	{
		return keeper.is_type_correct<T>();
	}

private:

	ParameterName parameter_name;

	tapkee_internal::ValueKeeper keeper; 

};

class CheckedParameter
{

public:

	explicit CheckedParameter(const Parameter& p) : parameter(p)
	{
	}

	template <typename T>
	inline operator T() const
	{
		return parameter.get_value<T>();
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
	CheckedParameter& in_range(T lower, T upper)
	{
		if (!parameter.in_range(lower, upper))
		{
			std::string error_message = 
				(Message() << "Value " << parameter.name() << " " 
				 << parameter.get_value<T>() << " doesn't fit the range [" << 
				    lower << ", " << upper << ")");
			throw tapkee::wrong_parameter_error(error_message);
		}
		return *this;
	}
	
	CheckedParameter& positive()
	{
		if (!parameter.positive())
		{
			std::string error_message = 
				(Message() << "Value of " << parameter.name() << " is not positive");
			throw tapkee::wrong_parameter_error(error_message);
		}
		return *this;
	}

private:

	Parameter parameter;

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
	bool contains(const string& name) const
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
	Parameter operator()(const string& name) const
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
