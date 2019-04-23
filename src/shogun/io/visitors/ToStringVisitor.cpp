/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Viktor Gal, Giovanni De Toni
 */

#include <shogun/base/SGObject.h>
#include <shogun/io/visitors/ToStringVisitor.h>

using namespace shogun;

void ToStringVisitor::on(bool *v) {
	stream() << (*v ? "true" : "false") << m_buffer;
}

void ToStringVisitor::on(int32_t *v) {
	stream() << *v << m_buffer;
}

void ToStringVisitor::on(int64_t *v) {
	stream() << *v << m_buffer;
}

void ToStringVisitor::on(int8_t *v) {
	stream() << *v << m_buffer;
}

void ToStringVisitor::on(int16_t *v) {
	stream() << *v << m_buffer;
}

void ToStringVisitor::on(std::string *v) {
	stream() << *v << m_buffer;
}

void ToStringVisitor::on(std::shared_ptr<SGObject>* v) {
	if (*v) {
		stream() << (*v)->get_name() << "(...)" << m_buffer;
	} else {
		stream() << "null" << m_buffer;
	}
}

void ToStringVisitor::on(char *string) {
	stream() << *string << m_buffer;
}

void ToStringVisitor::on(uint8_t *uint8) {
	stream() << *uint8 << m_buffer;
}

void ToStringVisitor::on(uint16_t *uint16) {
	stream() << *uint16 << m_buffer;
}

void ToStringVisitor::on(uint32_t *uint32) {
	stream() << *uint32 << m_buffer;
}

void ToStringVisitor::on(uint64_t *uint64) {
	stream() << *uint64 << m_buffer;
}

void ToStringVisitor::on(complex128_t *complex128) {
	stream() << *complex128 << m_buffer;
}

void ToStringVisitor::enter_matrix(index_t *rows, index_t *cols) {
	stream() << "Matrix<"<< *rows << "x" << *cols << ">( ";
	m_buffer = ", ";
}

void ToStringVisitor::enter_vector(index_t *size) {
	stream() << "Vector<" << *size << ">( ";
	m_buffer = ", ";
}

void ToStringVisitor::enter_std_vector(size_t *size) {
	stream() << "Vector<" << *size << ">( ";
	m_buffer = ", ";
}

void ToStringVisitor::enter_map(size_t *size) {
	stream() << "Map<" << *size << ">( ";
	m_buffer = ", ";
}

void ToStringVisitor::on(float32_t *v) {
	stream() << *v << m_buffer;
}

void ToStringVisitor::on(float64_t *v) {
	stream() << *v << m_buffer;
}

void ToStringVisitor::on(floatmax_t *v) {
	stream() << *v << m_buffer;
}

void ToStringVisitor::exit_matrix(index_t *rows, index_t *cols) {
	stream() << ")";
	m_buffer.clear();
}

void ToStringVisitor::exit_vector(index_t *size) {
	stream().seekp(-2, std::ios_base::end);
	stream() << " )";
	m_buffer.clear();
}

void ToStringVisitor::exit_std_vector(size_t *size) {
	stream().seekp(-2, std::ios_base::end);
	stream() << " )";
	m_buffer.clear();
}

void ToStringVisitor::exit_map(size_t *size) {
	stream().seekp(-2, std::ios_base::end);
	stream() << " )";
	m_buffer.clear();
}

void ToStringVisitor::enter_matrix_row(index_t *rows, index_t *cols)
{
	stream() << "[ ";
}

void ToStringVisitor::exit_matrix_row(index_t *rows, index_t *cols)
{
	stream() << "]\n";
}
