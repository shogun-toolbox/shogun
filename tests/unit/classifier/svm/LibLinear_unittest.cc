#include <shogun/classifier/svm/LibLinear.h>
#include <shogun/features/DataGenerator.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/evaluation/ContingencyTableEvaluation.h>
#include <shogun/mathematics/Math.h>
#include <gtest/gtest.h>

using namespace shogun;

#ifdef HAVE_LAPACK
TEST(LibLinearTest,train)
{
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

	CDenseFeatures<float64_t>* train_feats_l1 = (CDenseFeatures<float64_t>*)features.copy_subset(train_idx);
	CDenseFeatures<float64_t>* test_feats_l1 =  (CDenseFeatures<float64_t>*)features.copy_subset(test_idx);

	SGMatrix<float64_t> train_matrix = train_feats_l1->get_feature_matrix();
	SGMatrix<float64_t> test_matrix = test_feats_l1->get_feature_matrix();

	SGMatrix<float64_t>::transpose_matrix(train_matrix.matrix, train_matrix.num_rows, train_matrix.num_cols);
	SGMatrix<float64_t>::transpose_matrix(test_matrix.matrix, test_matrix.num_rows, test_matrix.num_cols);

	train_feats_l1 = new CDenseFeatures<float64_t>(train_matrix);
	test_feats_l1 = new CDenseFeatures<float64_t>(test_matrix);

	CBinaryLabels* ground_truth = new CBinaryLabels(labels);

	LIBLINEAR_SOLVER_TYPE ll_list[7] = {
			/// L2 regularized linear logistic regression
			L2R_LR,
			/// L2 regularized SVM with L2-loss using dual coordinate descent
			L2R_L2LOSS_SVC_DUAL,
			/// L2 regularized SVM with L2-loss using newton in the primal
			L2R_L2LOSS_SVC,
			/// L2 regularized linear SVM with L1-loss using dual coordinate descent
			L2R_L1LOSS_SVC_DUAL,
			/// L1 regularized SVM with L2-loss using dual coordinate descent
			L1R_L2LOSS_SVC,
			/// L1 regularized logistic regression
			L1R_LR,
			/// L2 regularized linear logistic regression via dual
			L2R_LR_DUAL
	};

	CLibLinear* ll = new CLibLinear();
	CLibLinear* ll_l1 = new CLibLinear();
	CBinaryLabels* pred = NULL;
	float64_t liblin_accuracy;
	CContingencyTableEvaluation* eval = new CContingencyTableEvaluation();

	ll->set_bias_enabled(true);
	ll->set_features((CDenseFeatures<float64_t> *)(train_feats->clone()));
	ll->set_labels(ground_truth);

	ll_l1->set_bias_enabled(true);
	ll_l1->set_features((CDenseFeatures<float64_t> *)(train_feats_l1->clone()));
	ll_l1->set_labels(ground_truth);

	for(index_t i = 0; i < 7; i++)
	{
		LIBLINEAR_SOLVER_TYPE liblinear_solver_type = ll_list[i];
		SG_SPRINT("Begin LibLinear %d \n", liblinear_solver_type);
		if(liblinear_solver_type == L1R_L2LOSS_SVC ||
				(liblinear_solver_type == L1R_LR) )
		{
			SG_SPRINT("num_fea: %d\tnum_vec: %d\n",
					ll_l1->get_features()->get_dim_feature_space(),
					ll_l1->get_features()->get_num_vectors());
			ll_l1->set_liblinear_solver_type(ll_list[i]);
			SG_SPRINT("Begin of L1 Training %d.\n", liblinear_solver_type);
			ll_l1->train();
			SG_SPRINT("End of L1 Training %d.\n", liblinear_solver_type);
			pred = ll_l1->apply_binary(test_feats);
			liblin_accuracy = eval->evaluate(pred, ground_truth);
			SG_SPRINT("LibLinear(%d) accuracy: %f\n", liblinear_solver_type, liblin_accuracy);
			EXPECT_NEAR(liblin_accuracy, 1.0, 1e-6);
			SG_SPRINT("num_fea: %d\tnum_vec: %d\n",
								ll_l1->get_features()->get_dim_feature_space(),
								ll_l1->get_features()->get_num_vectors());
			ll_l1->set_features((CDenseFeatures<float64_t> *)(train_feats_l1->clone()));
		}
		else
		{
			ll->set_liblinear_solver_type(ll_list[i]);
			SG_SPRINT("Begin of Training %d.\n", liblinear_solver_type);
			ll->train();
			SG_SPRINT("End of Training %d.\n", liblinear_solver_type);
			pred = ll->apply_binary(test_feats);
			liblin_accuracy = eval->evaluate(pred, ground_truth);
			SG_SPRINT("LibLinear(%d) accuracy: %f\n", liblinear_solver_type, liblin_accuracy);
			EXPECT_NEAR(liblin_accuracy, 1.0, 1e-6);
			ll->set_features((CDenseFeatures<float64_t> *)(train_feats->clone()));
		}
		SG_SPRINT("\n");
	}

	SG_UNREF(train_feats);
	SG_UNREF(test_feats);
	SG_UNREF(pred);
}
#endif //HAVE_LAPACK
