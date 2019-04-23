#ifndef SHOGUN_FILTERVISITOR_H
#define SHOGUN_FILTERVISITOR_H

#include <shogun/base/SGObject.h>
#include <shogun/lib/DynamicObjectArray.h>

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
			if constexpr (std::is_same<T, U>::value)
				m_operation(m_name, v);

			if constexpr (std::is_pointer<U>::value)
			{
				if constexpr (std::is_base_of<
								typename std::remove_pointer<U>::type,
								typename std::remove_pointer<T>::type>::value)
				{
					// this is misleading, since it returns a pointer to this local
					// variable, instead of the original one
					auto obj = dynamic_cast<T>(*v);
					if (obj)
						m_operation(m_name, &obj);
				}

				auto* dyn_obj_array = dynamic_cast<DynamicObjectArray*>(*v);
				if(dyn_obj_array)
				{
					for(auto sub_object : *dyn_obj_array)
						on(&sub_object);
				}
			}
			return;
		}

		virtual void on(bool* v) override
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
