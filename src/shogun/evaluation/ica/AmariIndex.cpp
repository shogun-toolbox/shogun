#include "AmariIndex.h"

#ifdef HAVE_EIGEN3

#include <shogun/mathematics/Math.h>
#include <shogun/mathematics/eigen3.h>

using namespace shogun;
using namespace Eigen;

float64_t amari_index(SGMatrix<float64_t> SGW, SGMatrix<float64_t> SGA, bool standardize)
{
	Map<MatrixXd> W(SGW.matrix,SGW.num_rows,SGW.num_cols);
	Map<MatrixXd> A(SGA.matrix,SGA.num_rows,SGA.num_cols);

	REQUIRE(W.rows() == W.cols(), "amari_index - W must be square\n")
	REQUIRE(A.rows() == A.cols(), "amari_index - A must be square\n")
	REQUIRE(W.rows() == A.rows(), "amari_index - A and W must be the same size\n")
	REQUIRE(W.rows() >= 2, "amari_index - input must be at least 2x2\n")

	// normalizing both mixing matrices
	if (standardize)
	{
		for (int r = 0; r < W.rows(); r++)
		{
			W.row(r).normalize();
			if (W.row(r).maxCoeff() < -1*W.row(r).minCoeff())
				W.row(r) *= -1;
		}

		A = A.inverse();
		for (int r = 0; r < A.rows(); r++)
		{
			A.row(r).normalize();
			if (A.row(r).maxCoeff() < -1*A.row(r).minCoeff())
				A.row(r) *= -1;
		}
		A = A.inverse();

		bool swap = false;
		do
		{
			swap = false;
			for (int j = 1; j < A.cols(); j++)
			{
				if (A(0,j) < A(0,j-1))
				{
					A.col(j).swap(A.col(j-1));
					swap = true;
				}
			}

		} while(swap);
	}

	// calculating the permutation matrix
	MatrixXd P = (W * A).cwiseAbs();
	int k = P.rows();

	// summing the error in the permutation matrix
	MatrixXd E1(k,k);
	for (int r = 0; r < k; r++)
		E1.row(r) = P.row(r) / P.row(r).maxCoeff();

	float64_t row_error = (E1.rowwise().sum().array()-1).sum();

	MatrixXd E2(k,k);
	for (int c = 0; c < k; c++)
		E2.col(c) = P.col(c) / P.col(c).maxCoeff();

	float64_t col_error = (E2.colwise().sum().array()-1).sum();

	return 1.0 / (float)(2*k*(k-1)) * (row_error + col_error);

}
#endif //HAVE_EIGEN3
