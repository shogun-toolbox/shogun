/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Sergey Lisitsyn, Soumyajit De, Viktor Gal, Heiko Strathmann,
 *          Bjoern Esser, Soeren Sonnenburg
 */

#ifndef EIGEN3_H_
#define EIGEN3_H_

#ifdef __cplusplus
#include <complex>
#define lapack_complex_float std::complex<float>
#define lapack_complex_double std::complex<double>
#endif

#include <shogun/lib/config.h>

//#define EIGEN_RUNTIME_NO_MALLOC
#include <Eigen/Eigen>
#include <Eigen/Dense>
#include <Eigen/Sparse>

namespace shogun
{
template<class T> class SGSparseMatrix;

/** @brief This class contains some utilities for Eigen3 Sparse Matrix
 * integration with shogun. Currently it provides a method for
 * converting SGSparseMatrix to Eigen3 SparseMatrix.
 */
template<typename T> class EigenSparseUtil
{
	public:
	/** Converts a SGSparseMatrix to Eigen3 SparseMatrix by copying
	 * its non-zero co-efficients to a eigen3 SparseMatrix.
	 *
	 * @param sg_matrix the SGSparseMatrix
	 * @return Eigen3 SparseMatrix representation of sg_matrix
	 */
	static Eigen::SparseMatrix<T> toEigenSparse(SGSparseMatrix<T> sg_matrix);
};

}


#endif
