/*
* Copyright (c) The Shogun Machine Learning Toolbox
* Written (w) 2014 Parijat Mazumdar
* All rights reserved.
*
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions are met:
*
* 1. Redistributions of source code must retain the above copyright notice, this
* list of conditions and the following disclaimer.
* 2. Redistributions in binary form must reproduce the above copyright notice,
* this list of conditions and the following disclaimer in the documentation
* and/or other materials provided with the distribution.
*
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
* ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
* WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
* DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
* ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
* (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
* LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
* ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
* (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
* SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*
* The views and conclusions contained in the software and documentation are those
* of the authors and should not be interpreted as representing official policies,
* either expressed or implied, of the Shogun Development Team.
*/

#include <shogun/classifier/svm/LibLinear.h>
#include <shogun/features/DataGenerator.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/evaluation/ContingencyTableEvaluation.h>
#include <shogun/mathematics/Math.h>
#include <gtest/gtest.h>

using namespace shogun;

#ifdef HAVE_LAPACK

//Generate Data for L1 regularized,
//it's data need to be transposed first
void generate_data_l1(CDenseFeatures<float64_t>* &train_feats,
					 CDenseFeatures<float64_t>* &test_feats,
					 CBinaryLabels* &ground_truth)
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

	train_feats = (CDenseFeatures<float64_t>*)features.copy_subset(train_idx);
	test_feats =  (CDenseFeatures<float64_t>*)features.copy_subset(test_idx);

	SGMatrix<float64_t> train_matrix = train_feats->get_feature_matrix();
	SGMatrix<float64_t>::transpose_matrix(train_matrix.matrix,
			train_matrix.num_rows, train_matrix.num_cols);
	train_feats = new CDenseFeatures<float64_t>(train_matrix);

	ground_truth = new CBinaryLabels(labels);
}

//Generate Data for L2 regularized
void generate_data_l2(CDenseFeatures<float64_t>* &train_feats,
					 CDenseFeatures<float64_t>* &test_feats,
					 CBinaryLabels* &ground_truth)
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

	train_feats = (CDenseFeatures<float64_t>*)features.copy_subset(train_idx);
	test_feats =  (CDenseFeatures<float64_t>*)features.copy_subset(test_idx);
	ground_truth = new CBinaryLabels(labels);
}

TEST(LibLinearTest,train_L2R_LR)
{
	LIBLINEAR_SOLVER_TYPE liblinear_solver_type = L2R_LR;

	CDenseFeatures<float64_t>* train_feats = NULL;
	CDenseFeatures<float64_t>* test_feats = NULL;
	CBinaryLabels* ground_truth = NULL;

	generate_data_l2(train_feats, test_feats, ground_truth);

	CLibLinear* ll = new CLibLinear();

	CBinaryLabels* pred = NULL;
	float64_t liblin_accuracy;
	CContingencyTableEvaluation* eval = new CContingencyTableEvaluation();

	ll->set_bias_enabled(true);
	ll->set_features(train_feats);
	ll->set_labels(ground_truth);

	ll->set_liblinear_solver_type(liblinear_solver_type);
	ll->train();
	pred = ll->apply_binary(test_feats);
	liblin_accuracy = eval->evaluate(pred, ground_truth);

	EXPECT_NEAR(liblin_accuracy, 1.0, 1e-6);

	SG_UNREF(ll);
	SG_UNREF(test_feats);
	SG_UNREF(eval);
	SG_UNREF(pred);
}

TEST(LibLinearTest,train_L2R_L2LOSS_SVC_DUAL)
{
	LIBLINEAR_SOLVER_TYPE liblinear_solver_type = L2R_L2LOSS_SVC_DUAL;

	CDenseFeatures<float64_t>* train_feats = NULL;
	CDenseFeatures<float64_t>* test_feats = NULL;
	CBinaryLabels* ground_truth = NULL;

	generate_data_l2(train_feats, test_feats, ground_truth);

	CLibLinear* ll = new CLibLinear();

	CBinaryLabels* pred = NULL;
	float64_t liblin_accuracy;
	CContingencyTableEvaluation* eval = new CContingencyTableEvaluation();

	ll->set_bias_enabled(true);
	ll->set_features(train_feats);
	ll->set_labels(ground_truth);

	ll->set_liblinear_solver_type(liblinear_solver_type);
	ll->train();
	pred = ll->apply_binary(test_feats);
	liblin_accuracy = eval->evaluate(pred, ground_truth);

	EXPECT_NEAR(liblin_accuracy, 1.0, 1e-6);

	SG_UNREF(ll);
	SG_UNREF(test_feats);
	SG_UNREF(eval);
	SG_UNREF(pred);
}

TEST(LibLinearTest,train_L2R_L2LOSS_SVC)
{
	LIBLINEAR_SOLVER_TYPE liblinear_solver_type = L2R_L2LOSS_SVC;

	CDenseFeatures<float64_t>* train_feats = NULL;
	CDenseFeatures<float64_t>* test_feats = NULL;
	CBinaryLabels* ground_truth = NULL;

	generate_data_l2(train_feats, test_feats, ground_truth);

	CLibLinear* ll = new CLibLinear();

	CBinaryLabels* pred = NULL;
	float64_t liblin_accuracy;
	CContingencyTableEvaluation* eval = new CContingencyTableEvaluation();

	ll->set_bias_enabled(true);
	ll->set_features(train_feats);
	ll->set_labels(ground_truth);

	ll->set_liblinear_solver_type(liblinear_solver_type);
	ll->train();
	pred = ll->apply_binary(test_feats);
	liblin_accuracy = eval->evaluate(pred, ground_truth);

	EXPECT_NEAR(liblin_accuracy, 1.0, 1e-6);

	SG_UNREF(ll);
	SG_UNREF(test_feats);
	SG_UNREF(eval);
	SG_UNREF(pred);
}

TEST(LibLinearTest,train_L2R_L1LOSS_SVC_DUAL)
{
	LIBLINEAR_SOLVER_TYPE liblinear_solver_type = L2R_L1LOSS_SVC_DUAL;

	CDenseFeatures<float64_t>* train_feats = NULL;
	CDenseFeatures<float64_t>* test_feats = NULL;
	CBinaryLabels* ground_truth = NULL;

	generate_data_l2(train_feats, test_feats, ground_truth);

	CLibLinear* ll = new CLibLinear();

	CBinaryLabels* pred = NULL;
	float64_t liblin_accuracy;
	CContingencyTableEvaluation* eval = new CContingencyTableEvaluation();

	ll->set_bias_enabled(true);
	ll->set_features(train_feats);
	ll->set_labels(ground_truth);

	ll->set_liblinear_solver_type(liblinear_solver_type);
	ll->train();
	pred = ll->apply_binary(test_feats);
	liblin_accuracy = eval->evaluate(pred, ground_truth);

	EXPECT_NEAR(liblin_accuracy, 1.0, 1e-6);

	SG_UNREF(ll);
	SG_UNREF(test_feats);
	SG_UNREF(eval);
	SG_UNREF(pred);
}

TEST(LibLinearTest,train_L1R_L2LOSS_SVC)
{
	LIBLINEAR_SOLVER_TYPE liblinear_solver_type = L1R_L2LOSS_SVC;

	CDenseFeatures<float64_t>* train_feats = NULL;
	CDenseFeatures<float64_t>* test_feats = NULL;
	CBinaryLabels* ground_truth = NULL;

	generate_data_l1(train_feats, test_feats, ground_truth);

	CLibLinear* ll = new CLibLinear();

	CBinaryLabels* pred = NULL;
	float64_t liblin_accuracy;
	CContingencyTableEvaluation* eval = new CContingencyTableEvaluation();

	ll->set_bias_enabled(true);
	ll->set_features(train_feats);
	ll->set_labels(ground_truth);

	ll->set_liblinear_solver_type(liblinear_solver_type);
	ll->train();
	pred = ll->apply_binary(test_feats);
	liblin_accuracy = eval->evaluate(pred, ground_truth);

	EXPECT_NEAR(liblin_accuracy, 1.0, 1e-6);

	SG_UNREF(ll);
	SG_UNREF(test_feats);
	SG_UNREF(eval);
	SG_UNREF(pred);
}

TEST(LibLinearTest,train_L1R_LR)
{
	LIBLINEAR_SOLVER_TYPE liblinear_solver_type = L1R_LR;

	CDenseFeatures<float64_t>* train_feats = NULL;
	CDenseFeatures<float64_t>* test_feats = NULL;
	CBinaryLabels* ground_truth = NULL;

	generate_data_l1(train_feats, test_feats, ground_truth);

	CLibLinear* ll = new CLibLinear();

	CBinaryLabels* pred = NULL;
	float64_t liblin_accuracy;
	CContingencyTableEvaluation* eval = new CContingencyTableEvaluation();

	ll->set_bias_enabled(true);
	ll->set_features(train_feats);
	ll->set_labels(ground_truth);

	ll->set_liblinear_solver_type(liblinear_solver_type);
	ll->train();
	pred = ll->apply_binary(test_feats);
	liblin_accuracy = eval->evaluate(pred, ground_truth);

	EXPECT_NEAR(liblin_accuracy, 1.0, 1e-6);

	SG_UNREF(ll);
	SG_UNREF(test_feats);
	SG_UNREF(eval);
	SG_UNREF(pred);
}

TEST(LibLinearTest,train_L2R_LR_DUAL)
{
	LIBLINEAR_SOLVER_TYPE liblinear_solver_type = L2R_LR_DUAL;

	CDenseFeatures<float64_t>* train_feats = NULL;
	CDenseFeatures<float64_t>* test_feats = NULL;
	CBinaryLabels* ground_truth = NULL;

	generate_data_l2(train_feats, test_feats, ground_truth);

	CLibLinear* ll = new CLibLinear();

	CBinaryLabels* pred = NULL;
	float64_t liblin_accuracy;
	CContingencyTableEvaluation* eval = new CContingencyTableEvaluation();

	ll->set_bias_enabled(true);
	ll->set_features(train_feats);
	ll->set_labels(ground_truth);

	ll->set_liblinear_solver_type(liblinear_solver_type);
	ll->train();
	pred = ll->apply_binary(test_feats);
	liblin_accuracy = eval->evaluate(pred, ground_truth);

	EXPECT_NEAR(liblin_accuracy, 1.0, 1e-6);

	SG_UNREF(ll);
	SG_UNREF(test_feats);
	SG_UNREF(eval);
	SG_UNREF(pred);
}
#endif //HAVE_LAPACK
