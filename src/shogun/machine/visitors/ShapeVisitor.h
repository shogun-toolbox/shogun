/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Gil Hoben
 */

#include <shogun/lib/any.h>
#include <shogun/util/converters.h>

namespace shogun
{
	class ShapeVisitor: public AnyVisitor
	{
		public:
			ShapeVisitor(): m_size(0) 
		{
		}

			virtual void on(bool*) final
			{
				++m_size;
			}
			virtual void on(std::vector<bool>::reference*) final
			{
				++m_size;
			}
			virtual void on(char*) final
			{
				++m_size;
			}
			virtual void on(int8_t*) final
			{
				++m_size;
			}
			virtual void on(uint8_t*) final
			{
				++m_size;
			}
			virtual void on(int16_t*) final
			{
				++m_size;			
			}
			virtual void on(uint16_t*) final
			{
				++m_size;
			}
			virtual void on(int32_t*) final
			{
				++m_size;
			}	

			virtual void on(uint32_t*) final
			{
				++m_size;
			}

			virtual void on(int64_t*) final
			{
				++m_size;
			}

			virtual void on(uint64_t*) final
			{
				++m_size;
			}

			virtual void on(float32_t*) final
			{
				++m_size;
			}

			virtual void on(float64_t*) final
			{
				++m_size;
			}

			virtual void on(floatmax_t*) final
			{
				++m_size;
			}

			virtual void on(complex128_t*) final
			{
				++m_size;
			}

			virtual void on(std::shared_ptr<SGObject>*) final
			{
				++m_size;
			}

			virtual void on(std::string*) final
			{
				++m_size;
			}

			virtual void enter_matrix(index_t* rows, index_t* cols)
			{
				m_dims = {*rows, *cols};
			}

			virtual void enter_vector(index_t* size)
			{
				m_dims = {*size};
			}
			virtual void enter_std_vector(size_t* size)
			{
				m_dims = {utils::safe_convert<index_t>(*size)};
			}
			virtual void enter_map(size_t* size)
			{
			}

			virtual void enter_matrix_row(index_t* rows, index_t* cols)
			{
			}

			virtual void exit_matrix_row(index_t* rows, index_t* cols)
			{
			}

			virtual void exit_matrix(index_t* rows, index_t* cols)
			{
			}

			virtual void exit_vector(index_t* size)
			{
			}

			virtual void exit_std_vector(size_t* size)
			{
			}

			virtual void exit_map(size_t* size)
			{
			}

			std::vector<index_t> get_dims() const noexcept
			{
				return m_dims;
			}

			size_t get_size() const noexcept
			{
				return m_size;
			}


		private:
			std::vector<index_t> m_dims;
			size_t m_size;
	};
}
