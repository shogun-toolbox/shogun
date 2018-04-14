/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Sergey Lisitsyn, Soumyajit De, Viktor Gal, Heiko Strathmann, 
 *          Bjoern Esser, Soeren Sonnenburg
 */

#ifndef EIGEN3_H_
#define EIGEN3_H_

#include <shogun/lib/config.h>

	//#define EIGEN_RUNTIME_NO_MALLOC
	#include <Eigen/Eigen>
	#include <Eigen/Dense>
	#include <Eigen/Sparse>

#if ((EIGEN_WORLD_VERSION == 3) && (EIGEN_MAJOR_VERSION == 2) && \
	((EIGEN_MINOR_VERSION == 91) || (EIGEN_MINOR_VERSION == 92)))
	// Regression has been introduced to eigen develop (3.3alpha1+):
	// http://eigen.tuxfamily.org/bz/show_bug.cgi?id=1229
	// until this is not fixed we need to copy the matrix and calculate the log
	#define EIGEN_WITH_LOG_BUG_1229 1
#endif

#if ((EIGEN_WORLD_VERSION == 3) && (EIGEN_MAJOR_VERSION == 2) && \
	((EIGEN_MINOR_VERSION >= 91)))
	// Eigen operator bug that was introduced somewhere in 3.3+
	// TODO put reference and version when it got fixed
	// c.f. github isse #3486
	#define EIGEN_WITH_OPERATOR_BUG 1
#endif

#if ((EIGEN_WORLD_VERSION == 3) && (EIGEN_MAJOR_VERSION == 2) && \
	((EIGEN_MINOR_VERSION >= 93)))
	#define EIGEN_WITH_TRANSPOSITION_BUG 1
#endif
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
