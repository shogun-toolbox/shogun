#ifndef TAPKEE_MATRIX_H_
#define TAPKEE_MATRIX_H_

namespace tapkee 
{

void centerMatrix(tapkee::DenseMatrix& matrix)
{
	Eigen::Matrix<tapkee::DenseMatrix::Scalar,1,Eigen::Dynamic> col_means = matrix.colwise().mean();
	tapkee::DenseMatrix::Scalar grand_mean = matrix.mean();
	matrix.array() += grand_mean;
	matrix.rowwise() -= col_means;
	matrix.colwise() -= col_means.transpose();
};

}
#endif
