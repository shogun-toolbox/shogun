#include <shogun/classifier/svm/LibLinear.h>
#include <shogun/features/DataGenerator.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/evaluation/ContingencyTableEvaluation.h>
#include <shogun/mathematics/Math.h>
#include <gtest/gtest.h>

using namespace shogun;

#ifdef HAVE_LAPACK
TEST(LibLinearTest,train_L2R_LR)
{
	LIBLINEAR_SOLVER_TYPE liblinear_solver_type = L2R_LR;
	index_t num_samples = 50;
	CMath::init_random(5);
	SGMatrix<float64_t> data =
		CDataGenerator::generate_gaussians(num_samples, 2, 2);
	CDenseFeatures<float64_t> features(data);

	SGVector<index_t> train_idx(num_samples), test_idx(num_samples);
	SGVector<float64_t> labels(num_samples);
	for (index_t i = 0, j = 0; i < data.num_cols; ++i)
	{
		if (i % 2 == 0)
			train_idx[j] = i;
		else
			test_idx[j++] = i;

		labels[i/2] = (i < data.num_cols/2) ? 1.0 : -1.0;
	}

	CDenseFeatures<float64_t>* train_feats = (CDenseFeatures<float64_t>*)features.copy_subset(train_idx);
	CDenseFeatures<float64_t>* test_feats =  (CDenseFeatures<float64_t>*)features.copy_subset(test_idx);
	CBinaryLabels* ground_truth = new CBinaryLabels(labels);

	CLibLinear* ll = new CLibLinear();

	CBinaryLabels* pred = NULL;
	float64_t liblin_accuracy;
	CContingencyTableEvaluation* eval = new CContingencyTableEvaluation();

	ll->set_bias_enabled(true);
	ll->set_features((CDenseFeatures<float64_t> *)(train_feats->clone()));
	ll->set_labels(ground_truth);

	ll->set_liblinear_solver_type(liblinear_solver_type);
	ll->train();
	pred = ll->apply_binary(test_feats);
	liblin_accuracy = eval->evaluate(pred, ground_truth);

	EXPECT_NEAR(liblin_accuracy, 1.0, 1e-6);

	SG_UNREF(train_feats);
	SG_UNREF(test_feats);
	SG_UNREF(pred);
}

TEST(LibLinearTest,train_L2R_L2LOSS_SVC_DUAL)
{
	LIBLINEAR_SOLVER_TYPE liblinear_solver_type = L2R_L2LOSS_SVC_DUAL;
	index_t num_samples = 50;
	CMath::init_random(5);
	SGMatrix<float64_t> data =
		CDataGenerator::generate_gaussians(num_samples, 2, 2);
	CDenseFeatures<float64_t> features(data);

	SGVector<index_t> train_idx(num_samples), test_idx(num_samples);
	SGVector<float64_t> labels(num_samples);
	for (index_t i = 0, j = 0; i < data.num_cols; ++i)
	{
		if (i % 2 == 0)
			train_idx[j] = i;
		else
			test_idx[j++] = i;

		labels[i/2] = (i < data.num_cols/2) ? 1.0 : -1.0;
	}

	CDenseFeatures<float64_t>* train_feats = (CDenseFeatures<float64_t>*)features.copy_subset(train_idx);
	CDenseFeatures<float64_t>* test_feats =  (CDenseFeatures<float64_t>*)features.copy_subset(test_idx);
	CBinaryLabels* ground_truth = new CBinaryLabels(labels);

	CLibLinear* ll = new CLibLinear();

	CBinaryLabels* pred = NULL;
	float64_t liblin_accuracy;
	CContingencyTableEvaluation* eval = new CContingencyTableEvaluation();

	ll->set_bias_enabled(true);
	ll->set_features((CDenseFeatures<float64_t> *)(train_feats->clone()));
	ll->set_labels(ground_truth);

	ll->set_liblinear_solver_type(liblinear_solver_type);
	ll->train();
	pred = ll->apply_binary(test_feats);
	liblin_accuracy = eval->evaluate(pred, ground_truth);

	EXPECT_NEAR(liblin_accuracy, 1.0, 1e-6);

	SG_UNREF(train_feats);
	SG_UNREF(test_feats);
	SG_UNREF(pred);
}

TEST(LibLinearTest,train_L2R_L2LOSS_SVC)
{
	LIBLINEAR_SOLVER_TYPE liblinear_solver_type = L2R_L2LOSS_SVC;
	index_t num_samples = 50;
	CMath::init_random(5);
	SGMatrix<float64_t> data =
		CDataGenerator::generate_gaussians(num_samples, 2, 2);
	CDenseFeatures<float64_t> features(data);

	SGVector<index_t> train_idx(num_samples), test_idx(num_samples);
	SGVector<float64_t> labels(num_samples);
	for (index_t i = 0, j = 0; i < data.num_cols; ++i)
	{
		if (i % 2 == 0)
			train_idx[j] = i;
		else
			test_idx[j++] = i;

		labels[i/2] = (i < data.num_cols/2) ? 1.0 : -1.0;
	}

	CDenseFeatures<float64_t>* train_feats = (CDenseFeatures<float64_t>*)features.copy_subset(train_idx);
	CDenseFeatures<float64_t>* test_feats =  (CDenseFeatures<float64_t>*)features.copy_subset(test_idx);
	CBinaryLabels* ground_truth = new CBinaryLabels(labels);

	CLibLinear* ll = new CLibLinear();

	CBinaryLabels* pred = NULL;
	float64_t liblin_accuracy;
	CContingencyTableEvaluation* eval = new CContingencyTableEvaluation();

	ll->set_bias_enabled(true);
	ll->set_features((CDenseFeatures<float64_t> *)(train_feats->clone()));
	ll->set_labels(ground_truth);

	ll->set_liblinear_solver_type(liblinear_solver_type);
	ll->train();
	pred = ll->apply_binary(test_feats);
	liblin_accuracy = eval->evaluate(pred, ground_truth);

	EXPECT_NEAR(liblin_accuracy, 1.0, 1e-6);

	SG_UNREF(train_feats);
	SG_UNREF(test_feats);
	SG_UNREF(pred);
}

TEST(LibLinearTest,train_L2R_L1LOSS_SVC_DUAL)
{
	LIBLINEAR_SOLVER_TYPE liblinear_solver_type = L2R_L1LOSS_SVC_DUAL;
	index_t num_samples = 50;
	CMath::init_random(5);
	SGMatrix<float64_t> data =
		CDataGenerator::generate_gaussians(num_samples, 2, 2);
	CDenseFeatures<float64_t> features(data);

	SGVector<index_t> train_idx(num_samples), test_idx(num_samples);
	SGVector<float64_t> labels(num_samples);
	for (index_t i = 0, j = 0; i < data.num_cols; ++i)
	{
		if (i % 2 == 0)
			train_idx[j] = i;
		else
			test_idx[j++] = i;

		labels[i/2] = (i < data.num_cols/2) ? 1.0 : -1.0;
	}

	CDenseFeatures<float64_t>* train_feats = (CDenseFeatures<float64_t>*)features.copy_subset(train_idx);
	CDenseFeatures<float64_t>* test_feats =  (CDenseFeatures<float64_t>*)features.copy_subset(test_idx);
	CBinaryLabels* ground_truth = new CBinaryLabels(labels);

	CLibLinear* ll = new CLibLinear();

	CBinaryLabels* pred = NULL;
	float64_t liblin_accuracy;
	CContingencyTableEvaluation* eval = new CContingencyTableEvaluation();

	ll->set_bias_enabled(true);
	ll->set_features((CDenseFeatures<float64_t> *)(train_feats->clone()));
	ll->set_labels(ground_truth);

	ll->set_liblinear_solver_type(liblinear_solver_type);
	ll->train();
	pred = ll->apply_binary(test_feats);
	liblin_accuracy = eval->evaluate(pred, ground_truth);

	EXPECT_NEAR(liblin_accuracy, 1.0, 1e-6);

	SG_UNREF(train_feats);
	SG_UNREF(test_feats);
	SG_UNREF(pred);
}

TEST(LibLinearTest,train_L1R_L2LOSS_SVC)
{
	LIBLINEAR_SOLVER_TYPE liblinear_solver_type = L1R_L2LOSS_SVC;
		index_t num_samples = 50;
		CMath::init_random(5);
		SGMatrix<float64_t> data =
			CDataGenerator::generate_gaussians(num_samples, 2, 2);
		CDenseFeatures<float64_t> features(data);

		SGVector<index_t> train_idx(num_samples), test_idx(num_samples);
		SGVector<float64_t> labels(num_samples);
		for (index_t i = 0, j = 0; i < data.num_cols; ++i)
		{
			if (i % 2 == 0)
				train_idx[j] = i;
			else
				test_idx[j++] = i;

			labels[i/2] = (i < data.num_cols/2) ? 1.0 : -1.0;
		}

		CDenseFeatures<float64_t>* train_feats = (CDenseFeatures<float64_t>*)features.copy_subset(train_idx);
		CDenseFeatures<float64_t>* test_feats =  (CDenseFeatures<float64_t>*)features.copy_subset(test_idx);

		SGMatrix<float64_t> train_matrix = train_feats->get_feature_matrix();
		SGMatrix<float64_t>::transpose_matrix(train_matrix.matrix, train_matrix.num_rows, train_matrix.num_cols);
		train_feats = new CDenseFeatures<float64_t>(train_matrix);

		CBinaryLabels* ground_truth = new CBinaryLabels(labels);

		CLibLinear* ll = new CLibLinear();

		CBinaryLabels* pred = NULL;
		float64_t liblin_accuracy;
		CContingencyTableEvaluation* eval = new CContingencyTableEvaluation();

		ll->set_bias_enabled(true);
		ll->set_features((CDenseFeatures<float64_t> *)(train_feats->clone()));
		ll->set_labels(ground_truth);

		ll->set_liblinear_solver_type(liblinear_solver_type);
		ll->train();
		pred = ll->apply_binary(test_feats);
		liblin_accuracy = eval->evaluate(pred, ground_truth);

		EXPECT_NEAR(liblin_accuracy, 1.0, 1e-6);

		SG_UNREF(train_feats);
		SG_UNREF(test_feats);
		SG_UNREF(pred);
}

TEST(LibLinearTest,train_L1R_LR)
{
	LIBLINEAR_SOLVER_TYPE liblinear_solver_type = L1R_LR;
		index_t num_samples = 50;
		CMath::init_random(5);
		SGMatrix<float64_t> data =
			CDataGenerator::generate_gaussians(num_samples, 2, 2);
		CDenseFeatures<float64_t> features(data);

		SGVector<index_t> train_idx(num_samples), test_idx(num_samples);
		SGVector<float64_t> labels(num_samples);
		for (index_t i = 0, j = 0; i < data.num_cols; ++i)
		{
			if (i % 2 == 0)
				train_idx[j] = i;
			else
				test_idx[j++] = i;

			labels[i/2] = (i < data.num_cols/2) ? 1.0 : -1.0;
		}

		CDenseFeatures<float64_t>* train_feats = (CDenseFeatures<float64_t>*)features.copy_subset(train_idx);
		CDenseFeatures<float64_t>* test_feats =  (CDenseFeatures<float64_t>*)features.copy_subset(test_idx);

		SGMatrix<float64_t> train_matrix = train_feats->get_feature_matrix();
		SGMatrix<float64_t>::transpose_matrix(train_matrix.matrix, train_matrix.num_rows, train_matrix.num_cols);
		train_feats = new CDenseFeatures<float64_t>(train_matrix);

		CBinaryLabels* ground_truth = new CBinaryLabels(labels);

		CLibLinear* ll = new CLibLinear();

		CBinaryLabels* pred = NULL;
		float64_t liblin_accuracy;
		CContingencyTableEvaluation* eval = new CContingencyTableEvaluation();

		ll->set_bias_enabled(true);
		ll->set_features((CDenseFeatures<float64_t> *)(train_feats->clone()));
		ll->set_labels(ground_truth);

		ll->set_liblinear_solver_type(liblinear_solver_type);
		ll->train();
		pred = ll->apply_binary(test_feats);
		liblin_accuracy = eval->evaluate(pred, ground_truth);

		EXPECT_NEAR(liblin_accuracy, 1.0, 1e-6);

		SG_UNREF(train_feats);
		SG_UNREF(test_feats);
		SG_UNREF(pred);
}

TEST(LibLinearTest,train_L2R_LR_DUAL)
{
	LIBLINEAR_SOLVER_TYPE liblinear_solver_type = L2R_LR_DUAL;
	index_t num_samples = 50;
	CMath::init_random(5);
	SGMatrix<float64_t> data =
		CDataGenerator::generate_gaussians(num_samples, 2, 2);
	CDenseFeatures<float64_t> features(data);

	SGVector<index_t> train_idx(num_samples), test_idx(num_samples);
	SGVector<float64_t> labels(num_samples);
	for (index_t i = 0, j = 0; i < data.num_cols; ++i)
	{
		if (i % 2 == 0)
			train_idx[j] = i;
		else
			test_idx[j++] = i;

		labels[i/2] = (i < data.num_cols/2) ? 1.0 : -1.0;
	}

	CDenseFeatures<float64_t>* train_feats = (CDenseFeatures<float64_t>*)features.copy_subset(train_idx);
	CDenseFeatures<float64_t>* test_feats =  (CDenseFeatures<float64_t>*)features.copy_subset(test_idx);
	CBinaryLabels* ground_truth = new CBinaryLabels(labels);

	CLibLinear* ll = new CLibLinear();

	CBinaryLabels* pred = NULL;
	float64_t liblin_accuracy;
	CContingencyTableEvaluation* eval = new CContingencyTableEvaluation();

	ll->set_bias_enabled(true);
	ll->set_features((CDenseFeatures<float64_t> *)(train_feats->clone()));
	ll->set_labels(ground_truth);

	ll->set_liblinear_solver_type(liblinear_solver_type);
	ll->train();
	pred = ll->apply_binary(test_feats);
	liblin_accuracy = eval->evaluate(pred, ground_truth);

	EXPECT_NEAR(liblin_accuracy, 1.0, 1e-6);

	SG_UNREF(train_feats);
	SG_UNREF(test_feats);
	SG_UNREF(pred);
}
#endif //HAVE_LAPACK
