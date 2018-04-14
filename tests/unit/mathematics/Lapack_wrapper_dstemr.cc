/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soumyajit De, Pan Deng, Bjoern Esser, Viktor Gal
 */

#include <gtest/gtest.h>

#include <shogun/lib/common.h>

#ifdef HAVE_LAPACK
#ifdef HAVE_ATLAS
#include <shogun/lib/SGVector.h>
#include <shogun/mathematics/lapack.h>
#include <shogun/mathematics/eigen3.h>

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
#endif // HAVE_LAPACK
