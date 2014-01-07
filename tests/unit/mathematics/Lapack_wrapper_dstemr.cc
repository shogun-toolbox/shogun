/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Soumyajit De
 */

#include <lib/common.h>

#ifdef HAVE_LAPACK
#ifdef HAVE_EIGEN3
#ifdef HAVE_ATLAS
#include <lib/SGVector.h>
#include <mathematics/lapack.h>
#include <mathematics/eigen3.h>
#include <gtest/gtest.h>

using namespace shogun;
using namespace Eigen;

TEST(Lapack_wrapper, dstemr)
{
	int32_t size=10;

	// main diagonal of a fixed tridiagonal matrix
	SGVector<float64_t> diag(size);
	for (index_t i=0; i<size; ++i)
		diag[i]=i+1;

	// subdiagonal, fixed valued
	SGVector<float64_t> subdiag(size);
	subdiag.set_const(0.5);

	int32_t M=0;
	SGVector<float64_t> w(size);
	int32_t tryrac=0.0;
	int32_t info=0;

	// computing all eigenvalues
	wrap_dstemr('N', 'I', size, diag.vector, subdiag.vector,
		0.0, 0.0, 1, size, &M, w.vector, NULL, 1, 1, NULL, tryrac, &info);

	ASSERT(info==0);

	// checking with eigen3 dense eigen solver
	MatrixXd m=MatrixXd::Zero(size, size);
	// filling the diagonal
	for (index_t i=0; i<size; ++i)
		m(i,i)=i+1;

	// filling the subdiagonals
	for (index_t i=0; i<size-1; ++i)
		m(i,i+1)=m(i+1,i)=0.5;

	VectorXcd eigenvals=m.eigenvalues();
	Map<VectorXd> map(w.vector, w.vlen);

	EXPECT_NEAR((map.cast<complex128_t>()-eigenvals).norm(), 0.0, 1E-10);
}
#endif // HAVE_ATLAS
#endif // HAVE_EIGEN3
#endif // HAVE_LAPACK
