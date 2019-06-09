/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Viktor Gal, Giovanni De Toni
 */


#ifndef SHOGUN_TOSTRINGVISITOR_H
#define SHOGUN_TOSTRINGVISITOR_H

#include <shogun/base/range.h>
#include <shogun/lib/any.h>
#include <shogun/lib/SGMatrix.h>
#include <shogun/lib/SGVector.h>

#include <sstream>

namespace shogun {

	/**
	 * Visitor which converts an any value to its
	 * string representation.
	 */
	class ToStringVisitor : public AnyVisitor {

	public:
		ToStringVisitor(std::stringstream *ss) : AnyVisitor(), m_stream(ss) {
		}

		virtual void on(bool *v);

		virtual void on(int8_t *v);

		virtual void on(int16_t *v);

		virtual void on(int32_t *v);

		virtual void on(int64_t *v);

		virtual void on(float32_t *v);
		virtual void on(float64_t *v);
		virtual void on(floatmax_t *v);

		virtual void on(CSGObject **v);

		virtual void on(char *string);

		virtual void on(uint8_t *uint8);

		virtual void on(uint16_t *uint16);

		virtual void on(uint32_t *uint32);

		virtual void on(uint64_t *uint64);

		virtual void on(complex128_t *complex128);

		virtual void enter_matrix(index_t *rows, index_t *cols);

		virtual void enter_vector(index_t *size);

		virtual void enter_std_vector(size_t *size);

		virtual void enter_map(size_t *size);

		virtual void exit_matrix(index_t *rows, index_t *cols);

		virtual void exit_vector(index_t *size);

		virtual void exit_std_vector(size_t *size);

		virtual void exit_map(size_t *size);


	private:
		std::stringstream &stream() {
			return *m_stream;
		}

		template<class T>
		void to_string(SGMatrix <T> *m) {
			if (m) {
				stream() << "Matrix<" << demangled_type<T>() << ">(" << m->num_rows
						 << "," << m->num_cols << "): [";
				for (auto col : range(m->num_cols)) {
					stream() << "[";
					for (auto row : range(m->num_rows)) {
						stream() << (*m)(row, col);
						if (row < m->num_rows - 1)
							stream() << ",";
					}
					stream() << "]";
					if (col < m->num_cols)
						stream() << ",";
				}
				stream() << "]";
			}
		}

		template<class T>
		void to_string(SGVector <T> *v) {
			if (v) {
				stream() << "Vector<" << demangled_type<T>() << ">(" << v->vlen
						 << "): [";
				for (auto i : range(v->vlen)) {
					stream() << (*v)[i];
					if (i < v->vlen - 1)
						stream() << ",";
				}
				stream() << "]";
			}
		}

	private:
		std::stringstream *m_stream;
	};
}

#endif //SHOGUN_TOSTRINGVISITOR_H
