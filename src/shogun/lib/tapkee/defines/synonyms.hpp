/* This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Copyright (c) 2012-2013 Sergey Lisitsyn
 */

#ifndef TAPKEE_DEFINES_SYNONYMS_H_
#define TAPKEE_DEFINES_SYNONYMS_H_

/* Tapkee includes */
#include <shogun/lib/tapkee/defines/types.hpp>
#include <shogun/lib/tapkee/defines/stdtypes.hpp>
/* End of Tapkee includes */

namespace tapkee
{
namespace tapkee_internal
{

#ifdef EIGEN_YES_I_KNOW_SPARSE_MODULE_IS_NOT_STABLE_YET
	template <typename T> struct Triplet
	{
		Triplet(IndexType colIndex, IndexType rowIndex, T valueT) :
			col_(colIndex), row_(rowIndex), value_(valueT)
		{
		}
		IndexType col() const { return col_; };
		IndexType row() const { return row_; };
		T value() const { return value_; };
		IndexType col_;
		IndexType row_;
		T value_;
	};
	typedef Triplet<tapkee::ScalarType> SparseTriplet;
#else // EIGEN_YES_I_KNOW_SPARSE_MODULE_IS_NOT_STABLE_YET
	typedef Eigen::Triplet<tapkee::ScalarType> SparseTriplet;
#endif // EIGEN_YES_I_KNOW_SPARSE_MODULE_IS_NOT_STABLE_YET

	typedef TAPKEE_INTERNAL_VECTOR<tapkee::tapkee_internal::SparseTriplet> SparseTriplets;
	typedef TAPKEE_INTERNAL_VECTOR<tapkee::IndexType> LocalNeighbors;
	typedef TAPKEE_INTERNAL_VECTOR<tapkee::tapkee_internal::LocalNeighbors> Neighbors;
	typedef TAPKEE_INTERNAL_PAIR<tapkee::DenseMatrix,tapkee::DenseVector> EigendecompositionResult;
	typedef TAPKEE_INTERNAL_VECTOR<tapkee::IndexType> Landmarks;
	typedef TAPKEE_INTERNAL_PAIR<tapkee::SparseWeightMatrix,tapkee::DenseDiagonalMatrix> Laplacian;
	typedef TAPKEE_INTERNAL_PAIR<tapkee::DenseSymmetricMatrix,tapkee::DenseSymmetricMatrix> DenseSymmetricMatrixPair;
	typedef TAPKEE_INTERNAL_PAIR<tapkee::SparseMatrix,tapkee::tapkee_internal::Neighbors> SparseMatrixNeighborsPair;

#if defined(TAPKEE_USE_PRIORITY_QUEUE) && defined(TAPKEE_USE_FIBONACCI_HEAP)
	#error "Can't use both priority queue and fibonacci heap at the same time"
#endif
#if !defined(TAPKEE_USE_PRIORITY_QUEUE) && !defined(TAPKEE_USE_FIBONACCI_HEAP)
	#define TAPKEE_USE_PRIORITY_QUEUE
#endif

} // End of namespace tapkee_internal

}

#endif
