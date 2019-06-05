/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Viktor Gal, Giovanni De Toni
 */

#include <shogun/base/SGObject.h>
#include <shogun/io/visitors/ToStringVisitor.h>

using namespace shogun;

void ToStringVisitor::on(bool *v) {
	stream() << (*v ? "true" : "false");
}

void ToStringVisitor::on(int32_t *v) {
	stream() << *v;
}

void ToStringVisitor::on(int64_t *v) {
	stream() << *v;
}

void ToStringVisitor::on(float *v) {
	stream() << *v;
}

void ToStringVisitor::on(double *v) {
	stream() << *v;
}

void ToStringVisitor::on(long double *v) {
	stream() << *v;
}

void ToStringVisitor::on(CSGObject **v) {
	if (*v) {
		stream() << (*v)->get_name() << "(...)";
	} else {
		stream() << "null";
	}
}
void ToStringVisitor::on(SGVector<int> *v) {
	to_string(v);
}

void ToStringVisitor::on(SGVector<float> *v) {
	to_string(v);
}

void ToStringVisitor::on(SGVector<double> *v) {
	to_string(v);
}

void ToStringVisitor::on(SGMatrix<int> *mat) {
	to_string(mat);
}

void ToStringVisitor::on(SGMatrix<float> *mat) {
	to_string(mat);
}

void ToStringVisitor::on(SGMatrix<double> *mat) {
	to_string(mat);
}
