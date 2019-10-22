/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Gil Hoben
 */

%{
	#include <shogun/lib/any.h>
	#include <shogun/io/SGIO.h>

	namespace shogun
	{
		enum class SG_TYPE_TO_INTERFACE
		{
			NONE = 0,
			SCALAR = 1,
			VECTOR = 2,
			MATRIX = 3,
			STD_VECTOR = 4,
			STD_MAP = 5
		};

		template <typename Derived, typename InterfaceBaseType>
		class GetterVisitorInterface: public AnyVisitor {

		public:

			GetterVisitorInterface(InterfaceBaseType& obj) : AnyVisitor(), 
					m_interface_obj(&obj), 
					m_type(SG_TYPE_TO_INTERFACE::NONE),
					m_nested_type(SG_TYPE_TO_INTERFACE::NONE) 
			{
			}

			void on(bool *v) final
			{
				handle_sg(v);
			}

			void on(std::vector<bool>::reference *v) final
			{
				handle_sg(v);
			}

			void on(int8_t *v) final
			{
				handle_sg(v);
			}

			void on(int16_t *v) final
			{
				handle_sg(v);
			}

			void on(int32_t *v) final
			{
				handle_sg(v);
			}

			void on(int64_t *v) final
			{
				handle_sg(v);
			}

			void on(float32_t *v) final
			{
				handle_sg(v);
			}

			void on(float64_t *v) final
			{
				handle_sg(v);
			}

			void on(floatmax_t *v) final
			{
				handle_sg(v);
			}

			void on(std::string *v) final
			{
				handle_sg(v->c_str());
			}

			void on(std::shared_ptr<SGObject> *v) final
			{
				handle_sg(v);
			}

			void on(char *v) final
			{
				handle_sg(v);
			}

			void on(uint8_t *v) final
			{
				handle_sg(v);
			}

			void on(uint16_t *v) final
			{
				handle_sg(v);
			}

			void on(uint32_t *v) final
			{
				handle_sg(v);
			}

			void on(uint64_t *v) final
			{
				handle_sg(v);
			}

			void on(complex128_t *v) final
			{
				handle_sg(v);
			}

			void enter_matrix(index_t *rows, index_t *cols) final
			{
				// initialise some variables needed to initialise a array	
				dims = {(std::ptrdiff_t) *rows, (std::ptrdiff_t) *cols};
				if (m_type == SG_TYPE_TO_INTERFACE::NONE)
					{
						current_i = 0;
						m_type = SG_TYPE_TO_INTERFACE::MATRIX;
					}
				else
				{
					m_nested_type = SG_TYPE_TO_INTERFACE::MATRIX;
					nested_current_i = 0;
					m_nested_interface_obj = nullptr;
				}
			}

			void enter_vector(index_t *size) final
			{
				dims = {(std::ptrdiff_t) *size};
				if (m_type == SG_TYPE_TO_INTERFACE::NONE)
				{
					current_i = 0;
					m_type = SG_TYPE_TO_INTERFACE::VECTOR;
				}
				else
				{
					m_nested_type = SG_TYPE_TO_INTERFACE::VECTOR;
					nested_current_i = 0;
					m_nested_interface_obj = nullptr;
				}
			}

			void enter_std_vector(size_t *size) final
			{
				list_size = *size;
				if (m_type == SG_TYPE_TO_INTERFACE::NONE)
				{
					current_i = 0;
					m_type = SG_TYPE_TO_INTERFACE::STD_VECTOR;
				}
				else
					m_nested_type = SG_TYPE_TO_INTERFACE::STD_VECTOR;
			}

			void enter_map(size_t *size) final
			{
				error("Casting of std::map to interface not implemented!");
			}

			void exit_matrix(index_t *rows, index_t *cols) final
			{
				m_nested_interface_obj = nullptr;
				if (m_type == SG_TYPE_TO_INTERFACE::MATRIX)
					m_type = SG_TYPE_TO_INTERFACE::NONE;
				m_nested_type = SG_TYPE_TO_INTERFACE::NONE;
			}

			void exit_vector(index_t *size) final
			{
				m_nested_interface_obj = nullptr;
				if (m_type == SG_TYPE_TO_INTERFACE::VECTOR)
					m_type = SG_TYPE_TO_INTERFACE::NONE;
				m_nested_type = SG_TYPE_TO_INTERFACE::NONE;
			}

			void exit_std_vector(size_t *size) final
			{
				if (m_type == SG_TYPE_TO_INTERFACE::STD_VECTOR)
					m_type = SG_TYPE_TO_INTERFACE::NONE;
				m_nested_type = SG_TYPE_TO_INTERFACE::NONE;
			}

			void exit_map(size_t *size) final
			{
				error("Casting of std::map to interface not implemented!");
			}

			void enter_matrix_row(index_t *rows, index_t *cols) final
			{
			}

			void exit_matrix_row(index_t *rows, index_t *cols) final
			{
			}

		private:

			GetterVisitorInterface(InterfaceBaseType& obj, std::vector<std::ptrdiff_t> dims_, 
				size_t current_i_, SG_TYPE_TO_INTERFACE type): 
				AnyVisitor(),
				m_interface_obj(&obj), 
				dims(dims_), 
				current_i(current_i_),
				m_type(type)
			{
			}

			template <typename T>
			void handle_sg(const T* v)
			{
				switch (m_type)
				{
					case SG_TYPE_TO_INTERFACE::VECTOR:
					case SG_TYPE_TO_INTERFACE::MATRIX:
						handle_array(v);
					break;
					case SG_TYPE_TO_INTERFACE::STD_VECTOR:
						handle_list(v);
					break;
					default:
						*m_interface_obj = static_cast<Derived*>(this)->sg_to_interface(v);
				}
			}

			template <typename T>
			void handle_array(const T* v)
			{
				if constexpr(std::is_same_v<T, char>)
				{
					// if it is char we will just get the whole buffer from the SGVector<char>
					// and make it a string. This is a special case of SGVector, where we don't 
					// convert to an array, but use a string instead.
					if (!(*m_interface_obj))
						*m_interface_obj = SWIG_FromCharPtrAndSize(v, dims[0]);
				}
				else
				{
					// this is an array but we haven't instatiated one yet,
					// so let's create one with the pointer to the first element
					if (!dims.empty() && !(*m_interface_obj))
						*m_interface_obj = static_cast<Derived*>(this)->template create_array<T>(v, dims);		
					else if (dims.empty() && !(*m_interface_obj))
						error("Could not determine the number of dimensions in array!");
					// we copy all the data once with create_array, so we don't need to keep 
					// updating the values of the array
					// so we do nothing after this
				}
			}

			template <typename T>
			void handle_list(const T* v)
			{
				bool new_obj = m_nested_interface_obj ? false : true;
				if (!(*m_interface_obj))
					*m_interface_obj = static_cast<Derived*>(this)->template create_new_list<T>(list_size);

				auto nested_visitor = GetterVisitorInterface(m_nested_interface_obj, dims, nested_current_i, m_nested_type);
				
				if (m_nested_type == SG_TYPE_TO_INTERFACE::VECTOR || m_nested_type == SG_TYPE_TO_INTERFACE::MATRIX)
					nested_visitor.handle_array(const_cast<T*>(v));
				else
					nested_visitor.on(const_cast<T*>(v));

				// we only do this once, from here on the nested_visitor will directly
				// modify the m_nested_interface_obj
				if (m_nested_interface_obj && new_obj)
				{
					static_cast<Derived*>(this)->template append_to_list<T>(*m_interface_obj, m_nested_interface_obj, current_i);
					current_i++;
				}
				else if (!m_nested_interface_obj && !new_obj)
					error("Could not cast shogun type {}!", demangled_type<T>().c_str());

				nested_current_i++;
			}

			InterfaceBaseType* m_interface_obj;
			std::vector<std::ptrdiff_t> dims;
			size_t current_i;
			InterfaceBaseType m_nested_interface_obj = nullptr;
			size_t nested_current_i;
			size_t list_size;
		
		protected:
			SG_TYPE_TO_INTERFACE m_type;
			SG_TYPE_TO_INTERFACE m_nested_type;
		};
	}
%}
