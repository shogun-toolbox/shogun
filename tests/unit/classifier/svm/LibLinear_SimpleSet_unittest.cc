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
void generate_data_l1_simple(CDenseFeatures<float64_t>* &train_feats,
					 CDenseFeatures<float64_t>* &test_feats,
					 CBinaryLabels* &ground_truth)
{
	index_t num_samples = 10;
	SGMatrix<float64_t> train_data(10, 2);
	SGMatrix<float64_t> test_data(2, 10);

	train_data(0,0)=-1;	train_data(0,1)=1;
	train_data(1,0)=-1;	train_data(1,1)=0;
	train_data(2,0)=-1;	train_data(2,1)=-1;
	train_data(3,0)=-2;	train_data(3,1)=0;
	train_data(4,0)=0;	train_data(4,1)=-1;
	train_data(5,0)=2;	train_data(5,1)=2;
	train_data(6,0)=3;	train_data(6,1)=1;
	train_data(7,0)=3;	train_data(7,1)=2;
	train_data(8,0)=3;	train_data(8,1)=3;
	train_data(9,0)=4;	train_data(9,1)=2;

	test_data(0,0)=-1;	test_data(1,0)=1;
	test_data(0,1)=-1;	test_data(1,1)=0;
	test_data(0,2)=-1;	test_data(1,2)=-1;
	test_data(0,3)=-2;	test_data(1,3)=0;
	test_data(0,4)=0;	test_data(1,4)=-1;
	test_data(0,5)=2;	test_data(1,5)=2;
	test_data(0,6)=3;	test_data(1,6)=1;
	test_data(0,7)=3;	test_data(1,7)=2;
	test_data(0,8)=3;	test_data(1,8)=3;
	test_data(0,9)=4;	test_data(1,9)=2;

	train_feats = new CDenseFeatures<float64_t>(train_data);
	test_feats = new CDenseFeatures<float64_t>(test_data);

	SGVector<float64_t> labels(num_samples);
	for (index_t i = 0; i < num_samples; ++i)
	{
		labels[i] = (i < num_samples/2) ? 1.0 : -1.0;
	}

	ground_truth = new CBinaryLabels(labels);

	SG_REF(ground_truth);
	SG_REF(train_feats);
	SG_REF(test_feats);
}

//Generate Data for L2 regularized
void generate_data_l2_simple(CDenseFeatures<float64_t>* &train_feats,
					 CDenseFeatures<float64_t>* &test_feats,
					 CBinaryLabels* &ground_truth)
{
	index_t num_samples = 10;
		SGMatrix<float64_t> train_data(2, 10);
		SGMatrix<float64_t> test_data(2, 10);

		train_data(0,0)=-1;	train_data(1,0)=1;
		train_data(0,1)=-1;	train_data(1,1)=0;
		train_data(0,2)=-1;	train_data(1,2)=-1;
		train_data(0,3)=-2;	train_data(1,3)=0;
		train_data(0,4)=0;	train_data(1,4)=-1;
		train_data(0,5)=2;	train_data(1,5)=2;
		train_data(0,6)=3;	train_data(1,6)=1;
		train_data(0,7)=3;	train_data(1,7)=2;
		train_data(0,8)=3;	train_data(1,8)=3;
		train_data(0,9)=4;	train_data(1,9)=2;

		test_data(0,0)=-1;	test_data(1,0)=1;
		test_data(0,1)=-1;	test_data(1,1)=0;
		test_data(0,2)=-1;	test_data(1,2)=-1;
		test_data(0,3)=-2;	test_data(1,3)=0;
		test_data(0,4)=0;	test_data(1,4)=-1;
		test_data(0,5)=2;	test_data(1,5)=2;
		test_data(0,6)=3;	test_data(1,6)=1;
		test_data(0,7)=3;	test_data(1,7)=2;
		test_data(0,8)=3;	test_data(1,8)=3;
		test_data(0,9)=4;	test_data(1,9)=2;

		train_feats = new CDenseFeatures<float64_t>(train_data);
		test_feats = new CDenseFeatures<float64_t>(test_data);

		SGVector<float64_t> labels(num_samples);
		for (index_t i = 0; i < num_samples; ++i)
		{
			labels[i] = (i < num_samples/2) ? 1.0 : -1.0;
		}

		ground_truth = new CBinaryLabels(labels);

		SG_REF(ground_truth);
		SG_REF(train_feats);
		SG_REF(test_feats);
}

TEST(LibLinearSimSetTest,train_L2R_LR)
{
	LIBLINEAR_SOLVER_TYPE liblinear_solver_type = L2R_LR;

	SGVector<float64_t> t_w(2);
	t_w[0] = -1.150365336474293;
	t_w[1] = -0.4144481723881207;

	CDenseFeatures<float64_t>* train_feats = NULL;
	CDenseFeatures<float64_t>* test_feats = NULL;
	CBinaryLabels* ground_truth = NULL;

	generate_data_l2_simple(train_feats, test_feats, ground_truth);

	CLibLinear* ll = new CLibLinear();

	CBinaryLabels* pred = NULL;
	float64_t liblin_accuracy;
	CContingencyTableEvaluation* eval = new CContingencyTableEvaluation();

	ll->set_bias_enabled(false);
	ll->set_features(train_feats);
	ll->set_labels(ground_truth);

	ll->set_liblinear_solver_type(liblinear_solver_type);
	ll->train();

	pred = ll->apply_binary(test_feats);

	liblin_accuracy = eval->evaluate(pred, ground_truth);

	for(int i=0;i<t_w.vlen;i++)
		EXPECT_NEAR(ll->get_w()[i], t_w[i], 1e-5);

	EXPECT_NEAR(liblin_accuracy, 1.0, 1e-5);

	SG_UNREF(ll);
	SG_UNREF(train_feats);
	SG_UNREF(test_feats);
	SG_UNREF(ground_truth);
	SG_UNREF(eval);
	SG_UNREF(pred);
}

TEST(LibLinearSimSetTest,train_L2R_L2LOSS_SVC_DUAL)
{
	LIBLINEAR_SOLVER_TYPE liblinear_solver_type = L2R_L2LOSS_SVC_DUAL;

	SGVector<float64_t> t_w(2);
	t_w[0] = -0.9523799021273924;
	t_w[1] = -0.3809534312059407;

	CDenseFeatures<float64_t>* train_feats = NULL;
	CDenseFeatures<float64_t>* test_feats = NULL;
	CBinaryLabels* ground_truth = NULL;

	generate_data_l2_simple(train_feats, test_feats, ground_truth);

	CLibLinear* ll = new CLibLinear();

	CBinaryLabels* pred = NULL;
	float64_t liblin_accuracy;
	CContingencyTableEvaluation* eval = new CContingencyTableEvaluation();

	ll->set_bias_enabled(false);
	ll->set_features(train_feats);
	ll->set_labels(ground_truth);

	ll->set_liblinear_solver_type(liblinear_solver_type);
	ll->train();

	pred = ll->apply_binary(test_feats);
	liblin_accuracy = eval->evaluate(pred, ground_truth);

	for(int i=0;i<t_w.vlen;i++)
		EXPECT_NEAR(ll->get_w()[i], t_w[i], 1e-5);

	EXPECT_NEAR(liblin_accuracy, 1.0, 1e-5);

	SG_UNREF(ll);
	SG_UNREF(train_feats);
	SG_UNREF(test_feats);
	SG_UNREF(ground_truth);
	SG_UNREF(eval);
	SG_UNREF(pred);
}

TEST(LibLinearSimSetTest,train_L2R_L2LOSS_SVC)
{
	LIBLINEAR_SOLVER_TYPE liblinear_solver_type = L2R_L2LOSS_SVC;

	SGVector<float64_t> t_w(2);
	t_w[0] = -0.9523809523809524;
	t_w[1] = -0.3809523809523809;

	CDenseFeatures<float64_t>* train_feats = NULL;
	CDenseFeatures<float64_t>* test_feats = NULL;
	CBinaryLabels* ground_truth = NULL;

	generate_data_l2_simple(train_feats, test_feats, ground_truth);

	CLibLinear* ll = new CLibLinear();

	CBinaryLabels* pred = NULL;
	float64_t liblin_accuracy;
	CContingencyTableEvaluation* eval = new CContingencyTableEvaluation();

	ll->set_bias_enabled(false);
	ll->set_features(train_feats);
	ll->set_labels(ground_truth);

	ll->set_liblinear_solver_type(liblinear_solver_type);
	ll->train();

	pred = ll->apply_binary(test_feats);
	liblin_accuracy = eval->evaluate(pred, ground_truth);

	for(int i=0;i<t_w.vlen;i++)
		EXPECT_NEAR(ll->get_w()[i], t_w[i], 1e-5);

	EXPECT_NEAR(liblin_accuracy, 1.0, 1e-5);

	SG_UNREF(ll);
	SG_UNREF(train_feats);
	SG_UNREF(test_feats);
	SG_UNREF(ground_truth);
	SG_UNREF(eval);
	SG_UNREF(pred);
}

TEST(LibLinearSimSetTest,train_L2R_L1LOSS_SVC_DUAL)
{
	LIBLINEAR_SOLVER_TYPE liblinear_solver_type = L2R_L1LOSS_SVC_DUAL;

	SGVector<float64_t> t_w(2);
	t_w[0] = -1;
	t_w[1] = 2.775557561562891e-17 ;

	CDenseFeatures<float64_t>* train_feats = NULL;
	CDenseFeatures<float64_t>* test_feats = NULL;
	CBinaryLabels* ground_truth = NULL;

	generate_data_l2_simple(train_feats, test_feats, ground_truth);

	CLibLinear* ll = new CLibLinear();

	CBinaryLabels* pred = NULL;
	float64_t liblin_accuracy;
	CContingencyTableEvaluation* eval = new CContingencyTableEvaluation();

	ll->set_bias_enabled(false);
	ll->set_features(train_feats);
	ll->set_labels(ground_truth);

	ll->set_liblinear_solver_type(liblinear_solver_type);
	ll->train();

	pred = ll->apply_binary(test_feats);
	liblin_accuracy = eval->evaluate(pred, ground_truth);

	for(int i=0;i<t_w.vlen;i++)
		EXPECT_NEAR(ll->get_w()[i], t_w[i], 1e-5);

	EXPECT_NEAR(liblin_accuracy, 0.9, 1e-5);

	SG_UNREF(ll);
	SG_UNREF(train_feats);
	SG_UNREF(test_feats);
	SG_UNREF(ground_truth);
	SG_UNREF(eval);
	SG_UNREF(pred);
}

TEST(LibLinearSimSetTest,train_L1R_L2LOSS_SVC)
{
	LIBLINEAR_SOLVER_TYPE liblinear_solver_type = L1R_L2LOSS_SVC;

	SGVector<float64_t> t_w(2);
	t_w[0] = -0.8333333333333333;
	t_w[1] = -0.1666666666666667;

	CDenseFeatures<float64_t>* train_feats = NULL;
	CDenseFeatures<float64_t>* test_feats = NULL;
	CBinaryLabels* ground_truth = NULL;

	generate_data_l1_simple(train_feats, test_feats, ground_truth);

	CLibLinear* ll = new CLibLinear();

	CBinaryLabels* pred = NULL;
	float64_t liblin_accuracy;
	CContingencyTableEvaluation* eval = new CContingencyTableEvaluation();

	ll->set_bias_enabled(false);
	ll->set_features(train_feats);
	ll->set_labels(ground_truth);

	ll->set_liblinear_solver_type(liblinear_solver_type);
	ll->train();

	pred = ll->apply_binary(test_feats);
	liblin_accuracy = eval->evaluate(pred, ground_truth);

	for(int i=0;i<t_w.vlen;i++)
		EXPECT_NEAR(ll->get_w()[i], t_w[i], 1e-5);

	EXPECT_NEAR(liblin_accuracy, 1.0, 1e-5);

	SG_UNREF(ll);
	SG_UNREF(train_feats);
	SG_UNREF(test_feats);
	SG_UNREF(ground_truth);
	SG_UNREF(eval);
	SG_UNREF(pred);
}

TEST(LibLinearSimSetTest,train_L1R_LR)
{
	LIBLINEAR_SOLVER_TYPE liblinear_solver_type = L1R_LR;

	SGVector<float64_t> t_w(2);
	t_w[0] = -1.378683364127616 ;
	t_w[1] = -0;

	CDenseFeatures<float64_t>* train_feats = NULL;
	CDenseFeatures<float64_t>* test_feats = NULL;
	CBinaryLabels* ground_truth = NULL;

	generate_data_l1_simple(train_feats, test_feats, ground_truth);

	CLibLinear* ll = new CLibLinear();

	CBinaryLabels* pred = NULL;
	float64_t liblin_accuracy;
	CContingencyTableEvaluation* eval = new CContingencyTableEvaluation();

	ll->set_bias_enabled(false);
	ll->set_features(train_feats);
	ll->set_labels(ground_truth);

	ll->set_liblinear_solver_type(liblinear_solver_type);
	ll->train();

	pred = ll->apply_binary(test_feats);
	liblin_accuracy = eval->evaluate(pred, ground_truth);

	for(int i=0;i<t_w.vlen;i++)
		EXPECT_NEAR(ll->get_w()[i], t_w[i], 1e-5);

	EXPECT_NEAR(liblin_accuracy, 1.0, 1e-5);

	SG_UNREF(ll);
	SG_UNREF(train_feats);
	SG_UNREF(test_feats);
	SG_UNREF(ground_truth);
	SG_UNREF(eval);
	SG_UNREF(pred);
}

TEST(LibLinearSimSetTest,train_L2R_LR_DUAL)
{
	LIBLINEAR_SOLVER_TYPE liblinear_solver_type = L2R_LR_DUAL;

	SGVector<float64_t> t_w(2);
	t_w[0] = -1.150367316729321 ;
	t_w[1] = -0.414449095403961;

	CDenseFeatures<float64_t>* train_feats = NULL;
	CDenseFeatures<float64_t>* test_feats = NULL;
	CBinaryLabels* ground_truth = NULL;

	generate_data_l2_simple(train_feats, test_feats, ground_truth);

	CLibLinear* ll = new CLibLinear();

	CBinaryLabels* pred = NULL;
	float64_t liblin_accuracy;
	CContingencyTableEvaluation* eval = new CContingencyTableEvaluation();

	ll->set_bias_enabled(false);
	ll->set_features(train_feats);
	ll->set_labels(ground_truth);

	ll->set_liblinear_solver_type(liblinear_solver_type);
	ll->train();

	pred = ll->apply_binary(test_feats);
	liblin_accuracy = eval->evaluate(pred, ground_truth);

	for(int i=0;i<t_w.vlen;i++)
		EXPECT_NEAR(ll->get_w()[i], t_w[i], 1e-5);

	EXPECT_NEAR(liblin_accuracy, 1.0, 1e-5);

	SG_UNREF(ll);
	SG_UNREF(train_feats);
	SG_UNREF(test_feats);
	SG_UNREF(ground_truth);
	SG_UNREF(eval);
	SG_UNREF(pred);
}

//-----------------------BIAS------------------------------
TEST(LibLinearSimSetTest,train_L2R_LR_BIAS)
{
	LIBLINEAR_SOLVER_TYPE liblinear_solver_type = L2R_LR;

	SGVector<float64_t> t_w(3);
	t_w[0] = -1.074173966275961 ;
	t_w[1] = -0.4636306285427702;
	t_w[2] = 0.6182115884788618;

	CDenseFeatures<float64_t>* train_feats = NULL;
	CDenseFeatures<float64_t>* test_feats = NULL;
	CBinaryLabels* ground_truth = NULL;

	generate_data_l2_simple(train_feats, test_feats, ground_truth);

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

	for(int i=0;i<t_w.vlen;i++)
		EXPECT_NEAR(ll->get_w()[i], t_w[i], 1e-5);
	EXPECT_NEAR(ll->get_bias(), t_w[2], 1e-5);

	EXPECT_NEAR(liblin_accuracy, 1.0, 1e-5);

	SG_UNREF(ll);
	SG_UNREF(train_feats);
	SG_UNREF(test_feats);
	SG_UNREF(ground_truth);
	SG_UNREF(eval);
	SG_UNREF(pred);
}

TEST(LibLinearSimSetTest,train_L2R_L2LOSS_SVC_DUAL_BIAS)
{
	LIBLINEAR_SOLVER_TYPE liblinear_solver_type = L2R_L2LOSS_SVC_DUAL;

	SGVector<float64_t> t_w(3);
	t_w[0] = -0.5153970026404913 ;
	t_w[1] = -0.2463534232497313;
	t_w[2] = 0.5737439568971296;

	CDenseFeatures<float64_t>* train_feats = NULL;
	CDenseFeatures<float64_t>* test_feats = NULL;
	CBinaryLabels* ground_truth = NULL;

	generate_data_l2_simple(train_feats, test_feats, ground_truth);

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

	for(int i=0;i<t_w.vlen;i++)
		EXPECT_NEAR(ll->get_w()[i], t_w[i], 1e-5);
	EXPECT_NEAR(ll->get_bias(), t_w[2], 1e-5);

	EXPECT_NEAR(liblin_accuracy, 1.0, 1e-5);

	SG_UNREF(ll);
	SG_UNREF(train_feats);
	SG_UNREF(test_feats);
	SG_UNREF(ground_truth);
	SG_UNREF(eval);
	SG_UNREF(pred);
}

TEST(LibLinearSimSetTest,train_L2R_L2LOSS_SVC_BIAS)
{
	LIBLINEAR_SOLVER_TYPE liblinear_solver_type = L2R_L2LOSS_SVC;

	SGVector<float64_t> t_w(3);
	t_w[0] = -0.5153970826580226 ;
	t_w[1] = -0.2463533225283631;
	t_w[2] = 0.5737439222042139;

	CDenseFeatures<float64_t>* train_feats = NULL;
	CDenseFeatures<float64_t>* test_feats = NULL;
	CBinaryLabels* ground_truth = NULL;

	generate_data_l2_simple(train_feats, test_feats, ground_truth);

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

	for(int i=0;i<t_w.vlen;i++)
		EXPECT_NEAR(ll->get_w()[i], t_w[i], 1e-5);
	EXPECT_NEAR(ll->get_bias(), t_w[2], 1e-5);

	EXPECT_NEAR(liblin_accuracy, 1.0, 1e-5);

	SG_UNREF(ll);
	SG_UNREF(train_feats);
	SG_UNREF(test_feats);
	SG_UNREF(ground_truth);
	SG_UNREF(eval);
	SG_UNREF(pred);
}

TEST(LibLinearSimSetTest,train_L2R_L1LOSS_SVC_DUAL_BIAS)
{
	LIBLINEAR_SOLVER_TYPE liblinear_solver_type = L2R_L1LOSS_SVC_DUAL;

	SGVector<float64_t> t_w(3);
	t_w[0] = -0.5714285446051303;
	t_w[1] = -0.2857143192435871;
	t_w[2] = 0.7142857276974349;

	CDenseFeatures<float64_t>* train_feats = NULL;
	CDenseFeatures<float64_t>* test_feats = NULL;
	CBinaryLabels* ground_truth = NULL;

	generate_data_l2_simple(train_feats, test_feats, ground_truth);

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

	for(int i=0;i<t_w.vlen;i++)
		EXPECT_NEAR(ll->get_w()[i], t_w[i], 1e-5);
	EXPECT_NEAR(ll->get_bias(), t_w[2], 1e-5);

	EXPECT_NEAR(liblin_accuracy, 1.0, 1e-5);

	SG_UNREF(ll);
	SG_UNREF(train_feats);
	SG_UNREF(test_feats);
	SG_UNREF(ground_truth);
	SG_UNREF(eval);
	SG_UNREF(pred);
}

TEST(LibLinearSimSetTest,train_L1R_L2LOSS_SVC_BIAS)
{
	LIBLINEAR_SOLVER_TYPE liblinear_solver_type = L1R_L2LOSS_SVC;

	SGVector<float64_t> t_w(3);
	t_w[0] = -0.5118958301270533;
	t_w[1] = -0.1428645860052333;
	t_w[2] = 0.44643750320628;

	CDenseFeatures<float64_t>* train_feats = NULL;
	CDenseFeatures<float64_t>* test_feats = NULL;
	CBinaryLabels* ground_truth = NULL;

	generate_data_l1_simple(train_feats, test_feats, ground_truth);

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

	for(int i=0;i<t_w.vlen;i++)
		EXPECT_NEAR(ll->get_w()[i], t_w[i], 1e-5);
	EXPECT_NEAR(ll->get_bias(), t_w[2], 1e-5);

	EXPECT_NEAR(liblin_accuracy, 1.0, 1e-5);

	SG_UNREF(ll);
	SG_UNREF(train_feats);
	SG_UNREF(test_feats);
	SG_UNREF(ground_truth);
	SG_UNREF(eval);
	SG_UNREF(pred);
}

TEST(LibLinearSimSetTest,train_L1R_LR_BIAS)
{
	LIBLINEAR_SOLVER_TYPE liblinear_solver_type = L1R_LR;

	SGVector<float64_t> t_w(3);
	t_w[0] = -1.365213552966976;
	t_w[1] = -0;
	t_w[2] = 0.06345876011584652;

	CDenseFeatures<float64_t>* train_feats = NULL;
	CDenseFeatures<float64_t>* test_feats = NULL;
	CBinaryLabels* ground_truth = NULL;

	generate_data_l1_simple(train_feats, test_feats, ground_truth);

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

	for(int i=0;i<t_w.vlen;i++)
		EXPECT_NEAR(ll->get_w()[i], t_w[i], 4e-5);
	EXPECT_NEAR(ll->get_bias(), t_w[2], 4e-5);

	EXPECT_NEAR(liblin_accuracy, 1.0, 1e-5);

	SG_UNREF(ll);
	SG_UNREF(train_feats);
	SG_UNREF(test_feats);
	SG_UNREF(ground_truth);
	SG_UNREF(eval);
	SG_UNREF(pred);
}

TEST(LibLinearSimSetTest,train_L2R_LR_DUAL_BIAS)
{
	LIBLINEAR_SOLVER_TYPE liblinear_solver_type = L2R_LR_DUAL;

	SGVector<float64_t> t_w(3);
	t_w[0] = -1.074172813563463;
	t_w[1] = -0.4636342439865924;
	t_w[2] = 0.6182083181247333;

	CDenseFeatures<float64_t>* train_feats = NULL;
	CDenseFeatures<float64_t>* test_feats = NULL;
	CBinaryLabels* ground_truth = NULL;

	generate_data_l2_simple(train_feats, test_feats, ground_truth);

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

	for(int i=0;i<t_w.vlen;i++)
		EXPECT_NEAR(ll->get_w()[i], t_w[i], 1e-5);
	EXPECT_NEAR(ll->get_bias(), t_w[2], 1e-5);

	EXPECT_NEAR(liblin_accuracy, 1.0, 1e-5);

	SG_UNREF(ll);
	SG_UNREF(train_feats);
	SG_UNREF(test_feats);
	SG_UNREF(ground_truth);
	SG_UNREF(eval);
	SG_UNREF(pred);
}
#endif //HAVE_LAPACK
