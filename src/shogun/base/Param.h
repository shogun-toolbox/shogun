/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Yuhui Liu
 */
#ifndef __PARAM_H__
#define __PARAM_H__

namespace shogun
{
template<typename T>
class Param{
public:
	template<typename ... Args>
	Param(Args &&...args) 
	{
		if constexpr(sizeof...(Args) == 2 && 
			std::is_same_v<float64_t, std::common_type_t<Args...>>)
		{
			m_parameters = std::pair{make_any(args)...};
		}
		else 
		{
			m_parameters = std::vector{make_any(args)...};
		}
	}
	using ParameterRange = std::variant<std::pair<Any, Any>, std::vector<Any>>;

	ParameterRange get_paramter_range()
	{
		return m_parameters;
	}

	const ParameterRange get_paramter_range() const
	{
		return m_parameters;
	}
private:
	
	ParameterRange m_parameters;
};

template<typename ... Args> Param(Args&& ...args)
	-> Param<typename std::tuple_element_t<0, std::tuple<Args...>>>;
}

#endif