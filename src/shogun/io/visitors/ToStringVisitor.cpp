/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Viktor Gal, Giovanni De Toni
 */

#include <shogun/base/SGObject.h>
#include <shogun/io/visitors/ToStringVisitor.h>

using namespace shogun;

void ToStringVisitor::on(bool *v) {
	stream() << (*v ? "true" : "false") << " ";
}

void ToStringVisitor::on(int32_t *v) {
	stream() << *v << " ";
}

void ToStringVisitor::on(int64_t *v) {
	stream() << *v << " ";
}

void ToStringVisitor::on(int8_t *v) {
	stream() << *v << " ";
}

void ToStringVisitor::on(int16_t *v) {
	stream() << *v << " ";
}

void ToStringVisitor::on(std::string *v) {
	stream() << *v << " ";
}

void ToStringVisitor::on(std::shared_ptr<SGObject>* v) {
	if (*v) {
		stream() << (*v)->get_name() << "(...) ";
	} else {
		stream() << "null ";
	}
}

void ToStringVisitor::on(char *string) {
	stream() << *string << " ";
}

void ToStringVisitor::on(uint8_t *uint8) {
	stream() << *uint8 << " ";
}

void ToStringVisitor::on(uint16_t *uint16) {
	stream() << *uint16 << " ";
}

void ToStringVisitor::on(uint32_t *uint32) {
	stream() << *uint32 << " ";
}

void ToStringVisitor::on(uint64_t *uint64) {
	stream() << *uint64 << " ";
}

void ToStringVisitor::on(complex128_t *complex128) {
	stream() << *complex128 << " ";
}

void ToStringVisitor::enter_matrix(index_t *rows, index_t *cols) {
	stream() << "Matrix<"<< *rows << "x" << *cols << ">( ";
}

void ToStringVisitor::enter_vector(index_t *size) {
	stream() << "Vector<" << *size << ">( ";
}

void ToStringVisitor::enter_std_vector(size_t *size) {
	stream() << "Vector<" << *size << ">( ";
}

void ToStringVisitor::enter_map(size_t *size) {
	stream() << "Map<" << *size << ">( ";
}

void ToStringVisitor::on(float32_t *v) {
	stream() << *v << " ";
}

void ToStringVisitor::on(float64_t *v) {
	stream() << *v << " ";
}

void ToStringVisitor::on(floatmax_t *v) {
	stream() << *v << " ";
}

void ToStringVisitor::exit_matrix(index_t *rows, index_t *cols) {
	stream() << ")";
}

void ToStringVisitor::exit_vector(index_t *size) {
	stream() << ")";
}

void ToStringVisitor::exit_std_vector(size_t *size) {
	stream() << ")";
}

void ToStringVisitor::exit_map(size_t *size) {
	stream() << ")";
}

void ToStringVisitor::enter_matrix_row(index_t *rows, index_t *cols)
{
	stream() << "[ ";
}

void ToStringVisitor::exit_matrix_row(index_t *rows, index_t *cols)
{
	stream() << "]\n";
}
