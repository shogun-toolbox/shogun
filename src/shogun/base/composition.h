/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Gil Hoben
 */

#ifndef __COMPOSITION_H__
#define __COMPOSITION_H__

#include <tuple>

namespace shogun {

	template <typename T, typename ...Args, std::size_t... Idx>
	bool apply_helper(T&& val, std::tuple<Args...> funcs, std::index_sequence<Idx...>)
	{
		return (std::get<Idx>(funcs)(std::forward<T>(val)) && ...);
	}

	/**
	 * Compile time composition with tuples.
	 */
	template <typename T, typename ...Args>
	bool apply(T&& val, std::tuple<Args...> funcs)
	{
		return apply_helper(val, funcs, std::index_sequence_for<Args...> {});
	}

	/**
 	 *
 	 */
	template <typename T>
	struct generic_checker
	{
	public:
		generic_checker(T val): m_val(val) {};
		bool operator()(T val) {return check(val);};

	protected:
		T m_val;
		virtual bool check(T val) = 0;
	};

	template <typename T>
	struct less_than: generic_checker<T>
	{
	public:
		less_than(T val): generic_checker<T>(val) {};
	protected:
		bool check(T val) override {return val < this->m_val;}
	};

	template <typename T>
	struct greater_than: generic_checker<T>
	{
	public:
		greater_than(T val): generic_checker<T>(val) {};
	protected:
		bool check(T val) override {return val > this->m_val;}
	};

	template <typename T = double>
	struct positive: greater_than<T>
	{
	public:
		positive(): greater_than<T>(0) {};
	};

	template <typename T = double>
	struct negative: less_than<T>
	{
	public:
		negative(): less_than<T>(0) {};
	};

	/**
 	  * Composer helper class that invokes apply using class member functions m_funcs.
	  */
	template <typename ...Args>
	class Composer
	{
	public:
		Composer(std::tuple<Args...> funcs): m_funcs(funcs) {}

		template <typename T>
		bool run(T val)
		{
			return apply(val, m_funcs);
		}

	private:
		std::tuple<Args...> m_funcs;
	};

	template <typename ...Args>
	Composer<Args...> make_composer(Args... args)
	{
		return Composer<Args...>(std::forward_as_tuple(args...));
	}
}

#endif // __COMPOSITION_H__