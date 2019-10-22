/** This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Viktor Gal
 */
#ifndef __BITSERY_VISITOR__
#define __BITSERY_VISITOR__

#include <shogun/lib/any.h>
#include <shogun/io/SGIO.h>

namespace shogun
{
	namespace io
	{
		namespace detail
		{
			static const size_t kNullObjectMagic = std::numeric_limits<size_t>::max();

			template <class S, class T>
			class BitseryVisitor : public AnyVisitor
			{
			public:
				BitseryVisitor(S& s):
					AnyVisitor(), m_s(s) {}

				void on(bool* v) override
				{
					m_s.boolValue(*v);
				}
				void on(std::vector<bool>::reference* v) override
				{
					bool tmp = *v;
					m_s.boolValue(tmp);
				}
				void on(char* v) override
				{
					m_s.value1b(*v);
				}
				void on(int8_t* v) override
				{
					m_s.value1b(*v);
				}
				void on(uint8_t* v) override
				{
					m_s.value1b(*v);
				}
				void on(int16_t* v) override
				{
					m_s.value2b(*v);
				}
				void on(uint16_t* v) override
				{
					m_s.value2b(*v);
				}
				void on(int32_t* v) override
				{
					m_s.value4b(*v);
				}
				void on(uint32_t* v) override
				{
					m_s.value4b(*v);
				}
				void on(int64_t* v) override
				{
					m_s.value8b(*v);
				}
				void on(uint64_t* v) override
				{
					m_s.value8b(*v);
				}
				void on(float* v) override
				{
					m_s.value4b(*v);
				}
				void on(float64_t* v) override
				{
					m_s.value8b(*v);
				}
				void on(floatmax_t* v) override
				{
					static_cast<T*>(this)->on_floatmax(m_s, v);
				}
				void on(complex128_t* v) override
				{
					static_cast<T*>(this)->on_complex(m_s, v);
				}
				void on(std::string* v) override
				{
					m_s.text1b(*v, 48);
				}
				void enter_matrix(index_t* rows, index_t* cols) override
				{
					m_s.value4b(*rows);
					m_s.value4b(*cols);
				}
				void enter_vector(index_t* size) override
				{
					m_s.value4b(*size);
				}
				void enter_std_vector(size_t* size) override
				{
					m_s.value8b(*size);
				}
				void enter_map(size_t* size) override
				{
				}

				void on(std::shared_ptr<SGObject>* v) override
				{
					static_cast<T*>(this)->on_object(m_s, v);
				}

				void enter_matrix_row(index_t *rows, index_t *cols) override {}
				void exit_matrix_row(index_t *rows, index_t *cols) override {}
				void exit_matrix(index_t* rows, index_t* cols) override {}
				void exit_vector(index_t* size) override {}
				void exit_std_vector(size_t* size) override {}
				void exit_map(size_t* size) override {}

			private:
				S& m_s;
				SG_DELETE_COPY_AND_ASSIGN(BitseryVisitor);
			};
		} // namespace detail
	} // namespace io
} // namespace shogun

#endif
