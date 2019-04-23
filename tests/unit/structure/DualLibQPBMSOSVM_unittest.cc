/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Thoralf Klein, Heiko Strathmann, Akash Shivram, Sergey Lisitsyn
 */

#include <shogun/lib/config.h>
#ifdef USE_GPL_SHOGUN
#include <gtest/gtest.h>

#include <shogun/lib/SGVector.h>
#include <shogun/lib/SGSparseVector.h>
#include <shogun/lib/SGSparseMatrix.h>

#include <shogun/features/SparseFeatures.h>
#include <shogun/structure/MulticlassSOLabels.h>
#include <shogun/labels/StructuredLabels.h>

#include <shogun/structure/MulticlassModel.h>
#include <shogun/structure/DualLibQPBMSOSVM.h>

using namespace shogun;

SGVector<float64_t> create_test_labels(int32_t N)
{
	SGVector<float64_t> labs(N);

	for (int32_t i=0; i<N; i++)
	{
		// labs[i] = float64_t(i/3);
		labs[i] = float64_t(i%5);
	}

	return labs;
}

SGSparseMatrix<float64_t> create_test_features(int32_t N, int32_t feat_dim, int32_t num_feat)
{
	SGSparseMatrix<float64_t> feats(feat_dim, N);

	for (int32_t i=0; i<N; i++)
	{
		feats.sparse_matrix[i] = SGSparseVector<float64_t>(num_feat);
		int32_t f = 0;

		ASSERT(f < num_feat);
		feats.sparse_matrix[i].features[f].feat_index = 0;
		feats.sparse_matrix[i].features[f].entry = i;
		f++;

		ASSERT(f < num_feat);
		feats.sparse_matrix[i].features[f].feat_index = 1;
		feats.sparse_matrix[i].features[f].entry = i%2 - 0.5;
		f++;

		ASSERT(f < num_feat);
		feats.sparse_matrix[i].features[f].feat_index = 2;
		feats.sparse_matrix[i].features[f].entry = i%2;
		f++;

		ASSERT(f < num_feat);
		feats.sparse_matrix[i].features[f].feat_index = 3;
		feats.sparse_matrix[i].features[f].entry = i%3;
		f++;
	}

	return feats;
}

class DualLibQPBMSOSVMTestLoopSolvers : public ::testing::TestWithParam<ESolver> {
  // You can implement all the usual fixture class members here.
  // To access the test parameter, call GetParam() from class
  // TestWithParam<T>.
};

TEST_P(DualLibQPBMSOSVMTestLoopSolvers,train_small_problem_and_predict)
{
	// toy data
	int32_t N        = 100;
	int32_t feat_dim = 5;
	int32_t num_feat = 4;

	// Create train labels
	SGVector<float64_t> labs = create_test_labels(N);
	auto labels = std::make_shared<MulticlassSOLabels>(labs);

	// Create train features
	SGSparseMatrix<float64_t> feats = create_test_features(N, feat_dim, num_feat);
	auto features = std::make_shared<SparseFeatures< float64_t >>(feats);

	// Create SO model, SO-SVM
	auto model = std::make_shared<MulticlassModel>(features, labels);
	auto sosvm = std::make_shared<DualLibQPBMSOSVM>(model, labels, 1e3);
	

	sosvm->set_cleanAfter(10);
	sosvm->set_cleanICP(1);
	sosvm->set_TolRel(0.01);
	sosvm->set_cp_models(1);
	sosvm->set_solver(GetParam());

	// sosvm->set_verbose(true);
	sosvm->set_BufSize(8);

	sosvm->train();

	BmrmStatistics res = sosvm->get_result();
	//SG_SPRINT("result = { Fp=%lf, Fd=%lf, nIter=%d, nCP=%d, nzA=%d, exitflag=%d }\n",
	//		res.Fp, res.Fd, res.nIter, res.nCP, res.nzA, res.exitflag);

	ASSERT_LE(res.nCP, 8u);
	ASSERT_LE(res.nzA, 8u);
	ASSERT_LE(res.exitflag, 0);

	auto out = sosvm->apply()->as<StructuredLabels>();

	// Compute error
	//-------------------------------------------------------------------------
	float64_t error=0.0;

	for (int32_t i=0; i<num_feat; ++i)
	{
		auto rn = out->get_label(i)->as<RealNumber>();
		error+=(rn->value==labs.get_element(i)) ? 0.0 : 1.0;
		
	}

	// SG_SPRINT("Error = %lf %% \n", error/num_feat*100);
	ASSERT_LE(error/num_feat*100, 75.0);

}

INSTANTIATE_TEST_CASE_P(IterateAllBMSOSolvers,
                        DualLibQPBMSOSVMTestLoopSolvers,
                        ::testing::Values(BMRM, PPBMRM, P3BMRM, NCBM));
#endif //USE_GPL_SHOGUN
