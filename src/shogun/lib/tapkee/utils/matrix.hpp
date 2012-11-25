#ifndef TAPKEE_MATRIX_H_
#define TAPKEE_MATRIX_H_

void centerMatrix()
{
	Matrix<Scalar,1,Dynamic> col_means = this->colwise().mean();
	Scalar grand_mean = this->mean();
	this->array() += grand_mean;
	this->rowwise() -= col_means;
	this->colwise() -= col_means.transpose();
}

#endif
