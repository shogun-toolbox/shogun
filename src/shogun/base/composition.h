/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Gil Hoben
 */

#ifndef __COMPOSITION_H__
#define __COMPOSITION_H__

#include <tuple>

namespace shogun
{
#ifdef __cpp_fold_expressions
	template <typename T, typename... Args, std::size_t... Idx>
	bool apply_helper(
	    T&& val, std::tuple<Args...> funcs, std::index_sequence<Idx...>)
	{
		return (std::get<Idx>(funcs)(std::forward<T>(val)) && ...);
	}
	/**
	 * Compile time composition with tuples.
	 */
	template <typename T, typename... Args>
	bool apply(T&& val, std::tuple<Args...> funcs)
	{
		return apply_helper(val, funcs, std::index_sequence_for<Args...>{});
	}
#else
	// taken from
	// https://stackoverflow.com/questions/10626856/how-to-split-a-tuple
	template <typename T, typename... Ts>
	auto head(std::tuple<T, Ts...> t)
	{
		return std::get<0>(t);
	}

	template <std::size_t... Ns, typename... Ts>
	auto tail_impl(std::index_sequence<Ns...>, std::tuple<Ts...> t)
	{
		return std::make_tuple(std::get<Ns + 1u>(t)...);
	}

	template <typename... Ts>
	auto tail(std::tuple<Ts...> t)
	{
		return tail_impl(std::make_index_sequence<sizeof...(Ts) - 1u>(), t);
	}

	template <
	    size_t N, typename T1, typename T2, std::enable_if_t<N == 1>* = nullptr>
	bool apply_helper(T1&& val, T2&& func)
	{
		return head(func)(val);
	}

	template <
	    size_t N, typename T1, typename T2,
	    std::enable_if_t<(N > 1)>* = nullptr>
	bool apply_helper(T1&& val, T2&& func)
	{
		return head(func)(val) &&
		       apply_helper<std::tuple_size<T2>::value - 1>(val, tail(func));
	}

	/**
	 * Compile time composition with tuples.
	 */
	template <typename T, typename... Args>
	bool apply(T&& val, std::tuple<Args...> funcs)
	{
		return apply_helper<std::tuple_size<std::tuple<Args...>>::value>(
		    val, funcs);
	}
#endif

	/**
	 *
	 */
	template <typename T>
	struct generic_checker
	{
	public:
		generic_checker(T val) : m_val(val){};
		bool operator()(T val)
		{
			return check(val);
		};

	protected:
		T m_val;
		virtual bool check(T val) = 0;
	};

	template <typename T>
	struct less_than : generic_checker<T>
	{
	public:
		less_than(T val) : generic_checker<T>(val){};

	protected:
		bool check(T val) override
		{
			return val < this->m_val;
		}
	};

	template <typename T>
	struct greater_than : generic_checker<T>
	{
	public:
		greater_than(T val) : generic_checker<T>(val){};

	protected:
		bool check(T val) override
		{
			return val > this->m_val;
		}
	};

	template <typename T = double>
	struct positive : greater_than<T>
	{
	public:
		positive() : greater_than<T>(0){};
	};

	template <typename T = double>
	struct negative : less_than<T>
	{
	public:
		negative() : less_than<T>(0){};
	};

	/**
	 * Composer helper class that invokes apply using class member functions
	 * m_funcs.
	 */
	template <typename... Args>
	class Composer
	{
	public:
		Composer(std::tuple<Args...> funcs) : m_funcs(funcs)
		{
		}

		template <typename T>
		bool run(T val)
		{
			return apply(val, m_funcs);
		}

	private:
		std::tuple<Args...> m_funcs;
	};

	template <typename... Args>
	Composer<Args...> make_composer(Args... args)
	{
		return Composer<Args...>(std::forward_as_tuple(args...));
	}
} // namespace shogun

#endif // __COMPOSITION_H__