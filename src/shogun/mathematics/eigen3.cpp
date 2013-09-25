/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Soumyajit De
 */

#include <shogun/lib/common.h>

#ifdef HAVE_EIGEN3
#include <shogun/lib/SGSparseMatrix.h>
#include <shogun/lib/SGSparseVector.h>
#include <shogun/mathematics/eigen3.h>

using namespace Eigen;

namespace shogun
{

template<typename T>
SparseMatrix<T> EigenSparseUtil<T>::toEigenSparse(SGSparseMatrix<T> sg_matrix)
	{
		REQUIRE(sg_matrix.num_vectors>0,
			"EigenSparseUtil::toEigenSparse(): \
			Number of rows must be positive!\n");
		REQUIRE(sg_matrix.num_features>0,
			"EigenSparseUtil::toEigenSparse(): \
			Number of cols must be positive!\n");
		REQUIRE(sg_matrix.sparse_matrix,
			"EigenSparseUtil::toEigenSparse(): \
			sg_matrix is not initialized!\n");

		index_t num_rows=sg_matrix.num_vectors;
		index_t num_cols=sg_matrix.num_features;

		typedef Eigen::Triplet<T> SparseTriplet;

		std::vector<SparseTriplet> tripletList;
		for (index_t i=0; i<num_rows; ++i)
		{
			for (index_t k=0; k<sg_matrix[i].num_feat_entries; ++k)
			{
				index_t &index_i=i;
				index_t &index_j=sg_matrix[i].features[k].feat_index;
				T &val=sg_matrix[i].features[k].entry;
				tripletList.push_back(SparseTriplet(index_i, index_j, val));
			}
		}

#ifdef EIGEN_YES_I_KNOW_SPARSE_MODULE_IS_NOT_STABLE_YET
		DynamicSparseMatrix<T> dM(num_rows, num_cols);
		dM.reserve(tripletList.size());

		for (typename std::vector<SparseTriplet>::iterator it=tripletList.begin(); 
			it!=tripletList.end(); ++it )
			dM.coeffRef(it->row(), it->col())+=it->value();

		SparseMatrix<T> M(dM);
#else // EIGEN_YES_I_KNOW_SPARSE_MODULE_IS_NOT_STABLE_YET
		SparseMatrix<T> M(num_rows, num_cols);
		M.setFromTriplets(tripletList.begin(), tripletList.end());
#endif // EIGEN_YES_I_KNOW_SPARSE_MODULE_IS_NOT_STABLE_YET

		return M;
	}

template class EigenSparseUtil<bool>;
template class EigenSparseUtil<float64_t>;
template class EigenSparseUtil<complex128_t>;
}
#endif //HAVE_EIGEN3
