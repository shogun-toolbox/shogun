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
		ToStringVisitor(std::stringstream *ss) : AnyVisitor(), m_stream(ss), m_buffer() {
		}

		void on(bool *v) override;

		void on(std::vector<bool>::reference *v) override;

		void on(int8_t *v) override;

		void on(int16_t *v) override;

		void on(int32_t *v) override;

		void on(int64_t *v) override;

		void on(float32_t *v) override;
		void on(float64_t *v) override;
		void on(floatmax_t *v) override;

		void on(std::string *v) override;

		void on(std::shared_ptr<SGObject>* v) override;

		void on(char *string) override;

		void on(uint8_t *uint8) override;

		void on(uint16_t *uint16) override;

		void on(uint32_t *uint32) override;

		void on(uint64_t *uint64) override;

		void on(complex128_t *complex128) override;

		void on(AutoValueEmpty*) override;

		void enter_matrix(index_t *rows, index_t *cols) override;

		void enter_vector(index_t *size) override;

		void enter_std_vector(size_t *size) override;

		void enter_map(size_t *size) override;

		void enter_auto_value(bool *is_empty) override;

		void exit_matrix(index_t *rows, index_t *cols) override;

		void exit_vector(index_t *size) override;

		void exit_std_vector(size_t *size) override;

		void exit_map(size_t *size) override;

		void enter_matrix_row(index_t *rows, index_t *cols) override;
		void exit_matrix_row(index_t *rows, index_t *cols) override;

	private:
		std::stringstream &stream() {
			return *m_stream;
		}

	private:
		std::stringstream *m_stream;
		std::string m_buffer;
	};
}

#endif //SHOGUN_TOSTRINGVISITOR_H
