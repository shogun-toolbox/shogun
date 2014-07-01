#include <shogun/distance/EuclideanDistance.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/labels/MulticlassLabels.h>
#include <shogun/lib/SGMatrix.h>
#include <shogun/multiclass/KNN.h>
#include <shogun/lib/SGVector.h>
#include <shogun/base/init.h>
#include <gtest/gtest.h>

using namespace shogun;

void gen_rand_data(SGVector<float64_t> &lab, SGMatrix<float64_t> &feat)
{
	index_t dims = feat.num_rows;
	index_t num = lab.vlen;

	for (int32_t i = 0; i < num; i++) {
		if (i < num / 2) {
			lab[i] = 0;

			for (int32_t j = 0; j < dims; j++)
				feat(j, i) = CMath::random(-10.0, 10.0);
		} else {
			lab[i] = 1.0;

			for (int32_t j = 0; j < dims; j++)
				feat(j, i) = CMath::random(-10.0, 10.0);
		}
	}
}


void test_knn(CKNN::KNNMode mode)
{
	init_shogun_with_defaults();
	
	index_t num = 6;
	index_t dim = 2;

	SGMatrix<float64_t> features(dim, num);

	SGVector<float64_t> labels(num);

	gen_rand_data(labels, features);

	CMulticlassLabels* multilabels = new CMulticlassLabels(labels);

	CDenseFeatures<float64_t>* denseFeatures = new CDenseFeatures<float64_t>(
				features);

	CKNN* kdtree = new CKNN(3,
				new CEuclideanDistance(denseFeatures, denseFeatures), multilabels);
	kdtree->set_mode(kdtree->KDTree);
	kdtree->train();

	SGMatrix<float64_t> test(2, 1);
		test(0, 0) = 4.0;
		test(1, 0) = 3.0;

	CDenseFeatures<float64_t>* testFeatures = new CDenseFeatures<float64_t>(
				test);

	CMulticlassLabels* multiTestLab = kdtree->apply_multiclass(testFeatures);
	SGVector<int32_t> lab =
				((CMulticlassLabels*) multiTestLab)->get_int_labels();

	CKNN* knn = new CKNN(3,
					new CEuclideanDistance(denseFeatures, denseFeatures), multilabels);
	knn->train();

	CMulticlassLabels* multitest_at = knn->apply_multiclass(testFeatures);
	SGVector<int32_t> lab_at =
					((CMulticlassLabels*) multitest_at)->get_int_labels();

	ASSERT_NE(0, lab_at.size());
	ASSERT_NE(0, lab.size());
	ASSERT_EQ(lab.size(), lab_at.size());
	ASSERT_EQ(lab_at[0], lab[0]);

	SG_UNREF(kdtree);
	exit_shogun();
}

TEST(KNN, KDTREE)
{
	test_knn(CKNN::KDTree);
}
