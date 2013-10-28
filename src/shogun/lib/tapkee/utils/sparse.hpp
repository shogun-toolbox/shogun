/* This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Copyright (c) 2012-2013 Vladyslav Gorbatiuk
 */

#ifndef TAPKEE_SPARSE_H_
#define TAPKEE_SPARSE_H_

/* Tapkee includes */
#include <shogun/lib/tapkee/defines.hpp>
 /* End of Tapkee includes */

namespace tapkee
{
namespace tapkee_internal
{

SparseMatrix sparse_matrix_from_triplets(const SparseTriplets& sparse_triplets, IndexType m, IndexType n)
{
#ifdef EIGEN_YES_I_KNOW_SPARSE_MODULE_IS_NOT_STABLE_YET
	Eigen::DynamicSparseMatrix<ScalarType> dynamic_weight_matrix(m, n);
	dynamic_weight_matrix.reserve(sparse_triplets.size());
	for (SparseTriplets::const_iterator it=sparse_triplets.begin(); it!=sparse_triplets.end(); ++it)
		dynamic_weight_matrix.coeffRef(it->col(),it->row()) += it->value();
	SparseMatrix matrix(dynamic_weight_matrix);
#else
	SparseMatrix matrix(m, n);
	matrix.setFromTriplets(sparse_triplets.begin(),sparse_triplets.end());
#endif
	return matrix;
}

}
}

#endif
