/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Gil Hoben
 */

#ifndef __CONSTRAINT_H__
#define __CONSTRAINT_H__

#include <shogun/io/SGIO.h>
#include <shogun/util/traits.h>

#include <string>
#include <tuple>

namespace shogun
{
	class SGObject;

	namespace constraint_detail
	{
		template <typename T, typename... Args, std::size_t... Idx>
		bool apply_helper(
		    T&& val, const std::tuple<Args...>& funcs, std::index_sequence<Idx...>)
		{
			return (std::get<Idx>(funcs)(std::forward<T>(val)) && ...);
		}
		/**
		 * Compile time composition with tuples.
		 */
		template <typename T, typename... Args>
		bool apply(T&& val, const std::tuple<Args...>& funcs)
		{
			return apply_helper(val, funcs, std::index_sequence_for<Args...>{});
		}

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

		template <size_t N, typename T, std::enable_if_t<N == 1>* = nullptr>
		void get_error_helper(T&& func, std::string& result)
		{
			result += head(func).error_msg();
		}

		template <size_t N, typename T, std::enable_if_t<N == 2>* = nullptr>
		void get_error_helper(const T& func, std::string& result)
		{
			result += head(func).error_msg() + " and ";
            get_error_helper<1>(tail(func), result);
		}

		template <size_t N, typename T, std::enable_if_t<(N > 2)>* = nullptr>
		void get_error_helper(const T& func, std::string& result)
		{
			result += head(func).error_msg() + ", ";
            get_error_helper<std::tuple_size<T>::value - 1u>(tail(func), result);
		}

		template <typename... Args>
		std::string get_error(const std::tuple<Args...>& funcs)
		{
			std::string result;
			get_error_helper<sizeof...(Args)>(funcs, result);
			return result;
		}
	} // namespace constraint_detail
	/**
	 * The base class of all constraints. The call operator calls the check pure
	 * virtual class member with a value to be checked. See derived classes for
	 * examples. In addition there is also a error_msg method to retrieve a
	 * custom error message.
	 */
	template <typename T>
	struct generic_checker
	{
		generic_checker() = default;
		bool operator()(const T& val) const
		{
			return check(val);
		};

		virtual std::string error_msg() const = 0;

	protected:
		virtual bool check(const T& val) const = 0;
	};

	template <typename T>
	struct custom_constraint: generic_checker<T>
	{
		template <typename Functor>
		custom_constraint(Functor&& func): m_func(func) 
		{
		}

		std::string error_msg() const override
		{
			return msg;
		}

	protected:
		bool check(const T& val) const override 
		{
			try
			{
				m_func(val);
			}
			catch (const std::exception& e)
			{
				msg = std::string(e.what());
				return false;
			}

			return true;
		}

	private:
		std::string msg;
		std::function<void(const T&)> m_func;
	};


	template <typename DerivedType>
	struct castable: generic_checker<std::shared_ptr<SGObject>>
	{
		castable(): generic_checker<std::shared_ptr<SGObject>>()
		{
		}

		std::string error_msg() const override
		{
			return "of type " + demangled_type<DerivedType>();
		}

	protected:
		bool check(const std::shared_ptr<SGObject>& ptr) const override 
		{
			return static_cast<bool>(std::dynamic_pointer_cast<DerivedType>(ptr));
		}
	};

	template <typename T>
	struct comparisson_checker: generic_checker<T>
	{
		comparisson_checker(T val): generic_checker<T>() {
			m_val = val;
		}
	protected:
		T m_val;
	};

	/**
	 * Checks if a value is less than val.
	 *
	 * @tparam T the type of val
	 */
	template <typename T>
	struct less_than : comparisson_checker<T>
	{
		less_than(T val) : comparisson_checker<T>(val){};

		std::string error_msg() const override
		{
			return "less than " + std::to_string(this->m_val);
		}

	protected:
		bool check(const T& val) const override
		{
			return val < this->m_val;
		}
	};

	/**
	 * Checks if a value is less than or equal to val.
	 *
	 * @tparam T the type of val
	 */
	template <typename T>
	struct less_than_or_equal : comparisson_checker<T>
	{
		less_than_or_equal(T val) : comparisson_checker<T>(val){};

		std::string error_msg() const override
		{
			return "less than " + std::to_string(this->m_val);
		}

	protected:
		bool check(const T& val) const override
		{
			return val <= this->m_val;
		}
	};

	/**
	 * Checks if a value is greater than val.
	 *
	 * @tparam T the type of val
	 */
	template <typename T>
	struct greater_than : comparisson_checker<T>
	{
		greater_than(T val) : comparisson_checker<T>(val){};
		std::string error_msg() const override
		{
			return "greater than " + std::to_string(this->m_val);
		}

	protected:
		bool check(const T& val) const override
		{
			return val > this->m_val;
		}
	};

	/**
	 * Checks if a value is greater than or equal to val.
	 *
	 * @tparam T the type of val
	 */
	template <typename T>
	struct greater_than_or_equal : comparisson_checker<T>
	{
		greater_than_or_equal(T val) : comparisson_checker<T>(val){};

		std::string error_msg() const override
		{
			return "less than " + std::to_string(this->m_val);
		}

	protected:
		bool check(const T& val) const override
		{
			return val >= this->m_val;
		}
	};

	/**
	 * Checks if a value is positive.
	 *
	 * @tparam T the type of zero
	 */
	template <typename T = double>
	struct positive : greater_than<T>
	{
	public:
		positive() : greater_than<T>(0){};
		std::string error_msg() const override
		{
			return "positive";
		}
	};

	/**
	 * Checks if a value is negative.
	 *
	 * @tparam T the type of zero
	 */
	template <typename T = double>
	struct negative : less_than<T>
	{
	public:
		negative() : less_than<T>(0){};
		std::string error_msg() const override
		{
			return "negative";
		}
	};

	/**
	 * Constraint helper class that invokes apply using class member functions
	 * m_funcs. If any of the functions returns false then it retrieves
	 * the error and passes it to the buffer.
	 */
	template <typename... Args>
	class Constraint
	{
	public:
		Constraint(std::tuple<Args...>&& funcs) : m_funcs(std::move(funcs))
		{
		}

		template <typename T>
		std::optional<std::string> check(const T& val) const
		{
			if (!constraint_detail::apply(val, m_funcs))
			{
				return constraint_detail::get_error(m_funcs);
			}
			return std::nullopt;
		}

	private:
		std::tuple<Args...> m_funcs;
	};

	/**
	 * A helper function to make a new Constraint instance.
	 * @tparam Args the types of args
	 * @param args the constraints
	 * @return the new Constraint instance
	 */
	template <typename... Args>
	Constraint<Args...> make_constraint(Args&&... args)
	{
		return Constraint<Args...>(std::forward_as_tuple(args...));
	}

} // namespace shogun

#define SG_CONSTRAINT(...) make_constraint(__VA_ARGS__)

#endif // __CONSTRAIN_H__