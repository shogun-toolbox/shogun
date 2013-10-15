#ifndef TAPKEE_MATRIX_H_
#define TAPKEE_MATRIX_H_

namespace tapkee
{
namespace tapkee_internal
{

void centerMatrix(DenseMatrix& matrix)
{
	DenseVector col_means = matrix.colwise().mean().transpose();
	DenseMatrix::Scalar grand_mean = matrix.mean();
	matrix.array() += grand_mean;
	matrix.rowwise() -= col_means.transpose();
	matrix.colwise() -= col_means;
}

} // End of tapkee_internal
} // End of tapkee
#endif
