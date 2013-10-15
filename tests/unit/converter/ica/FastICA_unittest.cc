#include <shogun/base/init.h>
#include <shogun/lib/common.h>
#include <shogun/features/DenseFeatures.h>
#include <gtest/gtest.h>

#ifdef HAVE_EIGEN3

#include <shogun/mathematics/Math.h>
#include <shogun/mathematics/eigen3.h>

#include <shogun/converter/ica/FastICA.h>
#include <shogun/evaluation/ica/PermutationMatrix.h>

using namespace Eigen;

typedef Matrix< float64_t, Dynamic, Dynamic, ColMajor > EMatrix;
typedef Matrix< float64_t, Dynamic, 1, ColMajor > EVector;

using namespace shogun;

TEST(CFastICA, blind_source_separation)
{
	// Generate sample data
	int FS = 4000;
	EVector t(FS+1, true);
	t.setLinSpaced(FS+1,0,1);

	// Source Signals
	EMatrix S(2,FS+1);
	for(int i = 0; i < FS+1; i++)
	{
		S(0,i) = sin(2*M_PI*55*t[i]);
		S(1,i) = cos(2*M_PI*100*t[i]);
	}

	// Mixing Matrix
	EMatrix A(2,2);
	A(0,0) = 1;    A(0,1) = 0.85;
	A(1,0) = 0.55;  A(1,1) = 1;

	// Mix signals
	SGMatrix<float64_t> X(2,FS+1);
	Eigen::Map<EMatrix> EX(X.matrix,2,FS+1);
	EX = A * S;
	CDenseFeatures< float64_t >* mixed_signals = new CDenseFeatures< float64_t >(X);

	// Separate
	CFastICA* ica = new CFastICA();
	SG_REF(ica);
	CFeatures* signals = ica->apply(mixed_signals);
	SG_REF(signals);

	// Close to a permutation matrix (with random scales)
	Eigen::Map<EMatrix> EA(ica->get_mixing_matrix().matrix,2,2);
	SGMatrix<float64_t> P(2,2);
	Eigen::Map<EMatrix> EP(P.matrix,2,2);
	EP = EA.inverse() * A;

	// Test if output is correct
	bool isperm = is_permutation_matrix(P);
	EXPECT_EQ(isperm,true);

	SG_UNREF(ica);
	SG_UNREF(mixed_signals);
	SG_UNREF(signals);
}

#endif //HAVE_EIGEN3
