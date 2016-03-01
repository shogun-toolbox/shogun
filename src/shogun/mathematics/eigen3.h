/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (C) 2012 Sergey Lisitsyn
 */

#ifndef EIGEN3_H_
#define EIGEN3_H_

#include <shogun/lib/config.h>

	//#define EIGEN_RUNTIME_NO_MALLOC
	#include <Eigen/Eigen>
	#include <Eigen/Dense>
	#if EIGEN_VERSION_AT_LEAST(3,0,93)
		#include <Eigen/Sparse>
	#else
		#define EIGEN_YES_I_KNOW_SPARSE_MODULE_IS_NOT_STABLE_YET
		#include <unsupported/Eigen/SparseExtra>

		#ifndef DOXYGEN_SHOULD_SKIP_THIS
		// Triplet definition for Eigen3 backword compatibility
		namespace Eigen {
		template <typename T> struct Triplet
		{
			Triplet(index_t colIndex, index_t rowIndex, T valueT) :
			ecol(colIndex), erow(rowIndex), evalue(valueT)
			{
			}
			index_t col() const { return ecol; };
			index_t row() const { return erow; };
			T value() const { return evalue; };
			index_t ecol;
			index_t erow;
			T evalue;
		};

		// SimplicialLLT definition for Eigen3 backword compatibility
		template <typename T> class SimplicialLLT
		: public SimplicialCholesky<T,Lower>
		{
		public:
			SimplicialLLT()
			{
				SimplicialCholesky<T>::setMode(SimplicialCholeskyLLt);
			}
			inline const T matrixL()
			{
				return SimplicialCholesky<T>::m_matrix;
			}
			inline const T matrixU()
			{
				return SimplicialCholesky<T>::m_matrix.transpose();
			}
		};
		}
		#endif //DOXYGEN_SHOULD_SKIP_THIS

	#endif	//EIGEN_VERSION_AT_LEAST(3,0,93)

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
