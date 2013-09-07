/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Soumyajit De
 */
 
#include <shogun/lib/common.h>

#include <shogun/lib/SGVector.h>
#include <shogun/lib/SGSparseVector.h>
#include <shogun/lib/SGSparseMatrix.h>
#ifdef HAVE_EIGEN3
#include <shogun/mathematics/eigen3.h>
#include <gtest/gtest.h>

using namespace shogun;
using namespace Eigen;

TEST(SGSparseMatrix, multiply_float64_int32)
{
	const int32_t size=10;
	const int32_t num_feat=size/2;

	SGSparseMatrix<float64_t> m(size, size);
	for (index_t i=0; i<size; ++i)
	{
		m.sparse_matrix[i]=SGSparseVector<float64_t>(num_feat);
		for (index_t j=0; j<num_feat; ++j)
		{
			SGSparseVectorEntry<float64_t> entry;
			entry.feat_index=(j+1)*2;
			entry.entry=0.5;
			m.sparse_matrix[i].features[j]=entry;
		}
	}

	SGVector<int32_t> v(size);
	v.set_const(2);

	SGVector<float64_t> result=m*v;
	Map<VectorXd> r(result.vector, result.vlen);

	EXPECT_NEAR(r.norm(), 12.64911064067351809115, 1E-16);
}

TEST(SGSparseMatrix, multiply_complex64_int32)
{
	const int32_t size=10;
	const int32_t num_feat=size/2;

	SGSparseMatrix<complex64_t> m(size, size);
	for (index_t i=0; i<size; ++i)
	{
		m.sparse_matrix[i]=SGSparseVector<complex64_t>(num_feat);
		for (index_t j=0; j<num_feat; ++j)
		{
			SGSparseVectorEntry<complex64_t> entry;
			entry.feat_index=(j+1)*2;
			entry.entry=complex64_t(0.5, 0.75);
			m.sparse_matrix[i].features[j]=entry;
		}
	}

	SGVector<int32_t> v(size);
	v.set_const(2);

	SGVector<complex64_t> result=m*v;
	Map<VectorXcd> r(result.vector, result.vlen);

	EXPECT_NEAR(r.norm(), 22.80350850198275836078, 1E-16);
}

TEST(SGSparseMatrix, multiply_complex64_float64)
{
	const int32_t size=10;
	const int32_t num_feat=size/2;

	SGSparseMatrix<complex64_t> m(size, size);
	for (index_t i=0; i<size; ++i)
	{
		m.sparse_matrix[i]=SGSparseVector<complex64_t>(num_feat);
		for (index_t j=0; j<num_feat; ++j)
		{
			SGSparseVectorEntry<complex64_t> entry;
			entry.feat_index=(j+1)*2;
			entry.entry=complex64_t(0.5, 0.75);
			m.sparse_matrix[i].features[j]=entry;
		}
	}

	SGVector<float64_t> v(size);
	v.set_const(2);

	SGVector<complex64_t> result=m*v;
	Map<VectorXcd> r(result.vector, result.vlen);

	EXPECT_NEAR(r.norm(), 22.80350850198275836078, 1E-16);
}
#endif // HAVE_EIGEN3

TEST(SGSparseMatrix, access_by_index)
{
	const index_t size=2;

	SGSparseMatrix<int32_t> m(size, size);
	for (index_t i=0; i<size; ++i)
		m(i,i)=i+1;
	m.sort_features();

	for (index_t i=0; i<size; ++i)
		EXPECT_EQ(m(i,i), i+1);
}
