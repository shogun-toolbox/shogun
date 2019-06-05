/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Viktor Gal, Giovanni De Toni
 */


#ifndef SHOGUN_TOSTRINGVISITOR_H
#define SHOGUN_TOSTRINGVISITOR_H

#include <shogun/base/range.h>
#include <shogun/lib/any.h>

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

		virtual void on(int32_t *v);

		virtual void on(int64_t *v);

		virtual void on(float *v);

		virtual void on(double *v);

		virtual void on(long double *v);

		virtual void on(CSGObject **v);

		virtual void on(SGVector<int> *v);

		virtual void on(SGVector<float> *v);

		virtual void on(SGVector<double> *v);

		virtual void on(SGMatrix<int> *mat);

		virtual void on(SGMatrix<float> *mat);

		virtual void on(SGMatrix<double> *mat);

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
