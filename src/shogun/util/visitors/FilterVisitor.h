#ifndef SHOGUN_FILTERVISITOR_H
#define SHOGUN_FILTERVISITOR_H

#include <shogun/base/SGObject.h>

#include <functional>

namespace shogun
{
	template <typename T>
	class FilterVisitor : public AnyVisitor
	{
	public:
		FilterVisitor(std::function<void(const std::string&, T*)> operation)
		    : AnyVisitor(), m_operation(operation)
		{
		}

		template <typename U>
		void on_impl(U* v)
		{
			if constexpr (std::is_same_v<T, U>)
				m_operation(m_name, v);

			if constexpr (std::is_pointer_v<U>)
			{
				if constexpr (std::is_base_of_v<
					typename std::remove_pointer_t<U>,
					typename std::remove_pointer_t<T>>)
				{
					// this is misleading, since it returns a pointer to this local
					// variable, instead of the original one
					auto obj = dynamic_cast<T>(*v);
					if (obj)
						m_operation(m_name, &obj);
				}
			}

			if constexpr (traits::is_shared_ptr<U>::value)
			{
				if constexpr (std::is_base_of_v<
					typename std::remove_pointer_t<typename U::element_type>,
					typename std::remove_pointer_t<T>>)
				{
					// this is misleading, since it returns a pointer to this local
					// variable, instead of the original one
					auto obj = std::dynamic_pointer_cast<std::remove_pointer_t<T>>(*v).get();
					if (obj)
						m_operation(m_name, &obj);

				}
			}

			return;
		}

		virtual void on(bool* v) override
		{
			on_impl(v);
		}

		virtual void on(std::vector<bool>::reference* v) override
		{
			on_impl(v);
		}

		virtual void on(int8_t* v) override
		{
			on_impl(v);
		}

		virtual void on(int16_t* v) override
		{
			on_impl(v);
		}

		virtual void on(int32_t* v) override
		{
			on_impl(v);
		}

		virtual void on(int64_t* v) override
		{
			on_impl(v);
		}

		virtual void on(float32_t* v) override
		{
			on_impl(v);
		}

		virtual void on(float64_t* v) override
		{
			on_impl(v);
		}

		virtual void on(floatmax_t* v) override
		{
			on_impl(v);
		}

		virtual void on(std::shared_ptr<SGObject>* v) override
		{
			if(*v)
				on_impl(v);
		}

		virtual void on(std::string* v) override
		{
			on_impl(v);
		}

		virtual void on(char* v) override
		{
			on_impl(v);
		}

		virtual void on(uint8_t* v) override
		{
			on_impl(v);
		}

		virtual void on(uint16_t* v) override
		{
			on_impl(v);
		}

		virtual void on(uint32_t* v) override
		{
			on_impl(v);
		}

		virtual void on(uint64_t* v) override
		{
			on_impl(v);
		}

		virtual void on(complex128_t* v) override
		{
			on_impl(v);
		}

		virtual void enter_matrix(index_t* rows, index_t* cols) override
		{
		}

		virtual void enter_vector(index_t* size) override
		{
		}

		virtual void enter_std_vector(size_t* size) override
		{
		}

		virtual void enter_map(size_t* size) override
		{
		}

		virtual void exit_matrix(index_t* rows, index_t* cols) override
		{
		}

		virtual void exit_vector(index_t* size) override
		{
		}

		virtual void exit_std_vector(size_t* size) override
		{
		}

		virtual void exit_map(size_t* size) override
		{
		}

		virtual void enter_matrix_row(index_t* rows, index_t* cols) override
		{
		}

		virtual void exit_matrix_row(index_t* rows, index_t* cols) override
		{
		}

		void set_name(const std::string name)
		{
			m_name = name;
		}

	private:
		std::function<void(const std::string&, T*)> m_operation;
		std::string m_name;
	};
} // namespace shogun

#endif // SHOGUN_FILTERVISITOR_H
