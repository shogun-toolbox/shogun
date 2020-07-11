#ifndef __SHAPE_VISITOR_H__
#define __SHAPE_VISITOR_H__
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

			void on(bool*) final
			{
				++m_size;
			}
			void on(std::vector<bool>::reference*) final
			{
				++m_size;
			}
			void on(char*) final
			{
				++m_size;
			}
			void on(int8_t*) final
			{
				++m_size;
			}
			void on(uint8_t*) final
			{
				++m_size;
			}
			void on(int16_t*) final
			{
				++m_size;			
			}
			void on(uint16_t*) final
			{
				++m_size;
			}
			void on(int32_t*) final
			{
				++m_size;
			}	

			void on(uint32_t*) final
			{
				++m_size;
			}

			void on(int64_t*) final
			{
				++m_size;
			}

			void on(uint64_t*) final
			{
				++m_size;
			}

			void on(float32_t*) final
			{
				++m_size;
			}

			void on(float64_t*) final
			{
				++m_size;
			}

			void on(floatmax_t*) final
			{
				++m_size;
			}

			void on(complex128_t*) final
			{
				++m_size;
			}

			void on(std::shared_ptr<SGObject>*) final
			{
				++m_size;
			}

			void on(std::string*) final
			{
				++m_size;
			}

			void on(AutoValueEmpty*) final
			{
				++m_size;
			}

			void enter_auto_value(bool* is_empty) final
			{
			}

			void enter_matrix(index_t* rows, index_t* cols) override
			{
				m_dims = {*rows, *cols};
			}

			void enter_vector(index_t* size) override
			{
				m_dims = {*size};
			}
			void enter_std_vector(size_t* size) override
			{
				m_dims = {utils::safe_convert<index_t>(*size)};
			}
			void enter_map(size_t* size) override
			{
			}

			void enter_matrix_row(index_t* rows, index_t* cols) override
			{
			}

			void exit_matrix_row(index_t* rows, index_t* cols) override
			{
			}

			void exit_matrix(index_t* rows, index_t* cols) override
			{
			}

			void exit_vector(index_t* size) override
			{
			}

			void exit_std_vector(size_t* size) override
			{
			}

			void exit_map(size_t* size) override
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
#endif /* __SHAPE_VISITOR_H__ */
