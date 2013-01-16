#ifndef TAPKEE_MATRIX_H_
#define TAPKEE_MATRIX_H_

namespace tapkee 
{

void centerMatrix(DenseMatrix& matrix)
{
	Eigen::Matrix<DenseMatrix::Scalar,1,Eigen::Dynamic> col_means = matrix.colwise().mean();
	DenseMatrix::Scalar grand_mean = matrix.mean();
	matrix.array() += grand_mean;
	matrix.rowwise() -= col_means;
	matrix.colwise() -= col_means.transpose();
};

}
#endif
