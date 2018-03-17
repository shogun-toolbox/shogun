/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2014 pl8787
 */
#include <gtest/gtest.h>
#include "utils/Utils.h"

#include <shogun/classifier/svm/LibLinear.h>
#include <shogun/features/DataGenerator.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/evaluation/ContingencyTableEvaluation.h>
#include <shogun/mathematics/Math.h>
#include <gtest/gtest.h>
#include <map>
#include <string>
#include <vector>

using namespace shogun;

#ifdef HAVE_LAPACK

class LibLinear : public ::testing::Test
{
public:
	CDenseFeatures<float64_t>* train_feats;
	CDenseFeatures<float64_t>* test_feats;
	CBinaryLabels* ground_truth;

	virtual void SetUp()
	{
		sg_rand->set_seed(1);
	}
	virtual void TearDown()
	{
		SG_UNREF(train_feats);
		SG_UNREF(test_feats);
		SG_UNREF(ground_truth);
	}
	//Helper that tests can call to drastically reduce code
	void train_with_solver
	(LIBLINEAR_SOLVER_TYPE llst, bool biasEnable, bool l1)
	{
		LIBLINEAR_SOLVER_TYPE liblinear_solver_type = llst;

		if (l1)
				generate_data_l1();
		else
				generate_data_l2();

		CLibLinear* ll = new CLibLinear();

		CBinaryLabels* pred = NULL;
		float64_t liblin_accuracy;
		CContingencyTableEvaluation* eval = new CContingencyTableEvaluation();

		ll->set_bias_enabled(biasEnable);
		ll->set_features(train_feats);
		ll->set_labels(ground_truth);

		ll->set_liblinear_solver_type(liblinear_solver_type);
		ll->train();
		pred = ll->apply_binary(test_feats);
		SG_REF(pred);

		liblin_accuracy = eval->evaluate(pred, ground_truth);

		EXPECT_NEAR(liblin_accuracy, 1.0, 1e-6);

		SG_UNREF(ll);
		SG_UNREF(eval);
		SG_UNREF(pred);
	}
	void train_with_solver_simple(
		LIBLINEAR_SOLVER_TYPE llst, bool biasEnable, bool l1, SGVector<float64_t> t_w, bool C_value=false) //C_value only for L2R_L1LOSS_SVC_DUAL
	{
		LIBLINEAR_SOLVER_TYPE liblinear_solver_type = llst;

		if (l1)
				generate_data_l1_simple();
		else
				generate_data_l2_simple();

		CLibLinear* ll = new CLibLinear();

		CBinaryLabels* pred = NULL;
		float64_t liblin_accuracy;
		CContingencyTableEvaluation* eval = new CContingencyTableEvaluation();

		ll->set_bias_enabled(biasEnable);
		ll->set_features(train_feats);
		if (C_value)
				ll->set_C(0.1,0.1); //Only in the case of L2R_L1LOSS_SVC_DUAL
		ll->set_labels(ground_truth);
		ll->set_liblinear_solver_type(liblinear_solver_type);
		ll->train();

		pred = ll->apply_binary(test_feats);
		SG_REF(pred);
		liblin_accuracy = eval->evaluate(pred, ground_truth);

		if (!biasEnable)
		{
			for(int i=0;i<t_w.vlen;i++)
				EXPECT_NEAR(ll->get_w()[i], t_w[i], 1e-5);
		}
		else
		{
			for(int i=0;i<t_w.vlen;i++)
				EXPECT_NEAR(ll->get_w()[i], t_w[i], 4e-5);
			EXPECT_NEAR(ll->get_bias(), t_w[2], 4e-5);
		}
		EXPECT_NEAR(liblin_accuracy, 1.0, 1e-5);

		SG_UNREF(ll);
		SG_UNREF(eval);
		SG_UNREF(pred);
	}

protected:
	void generate_data_l1()
	{
		generate_data("L1");
	}
	void generate_data_l2()
	{
		generate_data("L2");
	}
	void generate_data_l1_simple()
	{
		generate_data_simple("L1_SIMPLE");
	}
	void generate_data_l2_simple()
	{
			generate_data_simple("L2_SIMPLE");
	}
	void generate_data(std::string type) //Type either "L1" or "L2"
	{
		/*
			First part is shared by l1 and l2
		*/

		index_t num_samples = 50;
		CMath::init_random(5);
		SGMatrix<float64_t> data =
			CDataGenerator::generate_gaussians(num_samples, 2, 2);
		CDenseFeatures<float64_t> features(data);

		/*
			Generate the random data
		*/
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
		/*
			Now fill in train_feats, test_feats, and ground truth
		*/

		train_feats = (CDenseFeatures<float64_t>*)features.copy_subset(train_idx);
		test_feats =  (CDenseFeatures<float64_t>*)features.copy_subset(test_idx);

		if (type == "L1")
		{
			//transpose
			SGMatrix<float64_t> train_matrix = train_feats->get_feature_matrix();
			SGMatrix<float64_t>::transpose_matrix(train_matrix.matrix,
					train_matrix.num_rows, train_matrix.num_cols);

			SG_UNREF(train_feats);

			train_feats = new CDenseFeatures<float64_t>(train_matrix);

			ground_truth = new CBinaryLabels(labels);

			/*
				Now we ref ground_truth[l1_index], train_feats[l1_index]  ( test_feats[l1_index] was referenced in copy_subset)
			*/
			SG_REF(ground_truth);
			SG_REF(train_feats);
		}
		else
		{
				//type=="L2", no need to transpose
				ground_truth = new CBinaryLabels(labels);
				SG_REF(ground_truth);
		}
	}
	void generate_data_simple(std::string type) //Type either "L1_SIMPLE" or "L2_SIMPLE"
	{
			int32_t rows, cols;
			if (type=="L1_SIMPLE")
			{
				rows = 10;
				cols = 2;
			}
			else
			{
				//type == L2_SIMPLE
				rows = 2;
				cols = 10;
			}
			SGMatrix<float64_t> train_data(rows, cols);
			SGMatrix<float64_t> test_data(2, 10); //Always 2x10, doesn't matter l1 or l2

			/*
				We have to transpose the data if its l2. If it is l1, then leave it as it is (Since this is the data of l1 originally)
			*/
			index_t num_samples = 10;
			std::vector<int32_t> x{0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7,8,8,9,9};
			std::vector<int32_t> y{0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1};
			std::vector<int32_t> z{-1,1,-1,0,-1,-1,-2,0,0,-1,2,2,3,1,3,2,3,3,4,2};

			for(index_t i=0; i<num_samples*2; ++i){
				if (type=="L1_SIMPLE")
						train_data(x[i],y[i])=z[i];
				else
						train_data(y[i], x[i])=z[i]; //transpose
				test_data(y[i], x[i])=z[i];
			}
			/*
				The remainder is left regardless of weather this is for l1 or l2
			*/
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
};

TEST_F(LibLinear, train_L2R_LR)
{
	LIBLINEAR_SOLVER_TYPE liblinear_solver_type = L2R_LR;
	train_with_solver(liblinear_solver_type, false, false); //No bias, and not l1
}

TEST_F(LibLinear, train_L2R_L2LOSS_SVC_DUAL)
{
		LIBLINEAR_SOLVER_TYPE liblinear_solver_type = L2R_L2LOSS_SVC_DUAL;
		train_with_solver(liblinear_solver_type, false, false); //No bias, and not l1
}

TEST_F(LibLinear, train_L2R_L2LOSS_SVC)
{
		LIBLINEAR_SOLVER_TYPE liblinear_solver_type = L2R_L2LOSS_SVC;
		train_with_solver(liblinear_solver_type, false, false); //No bias, and not l1
}

TEST_F(LibLinear, train_L2R_L1LOSS_SVC_DUAL)
{
		LIBLINEAR_SOLVER_TYPE liblinear_solver_type = L2R_L1LOSS_SVC_DUAL;
		train_with_solver(liblinear_solver_type, false, false); //No bias, and not l1

}

TEST_F(LibLinear, train_L1R_L2LOSS_SVC)
{
		LIBLINEAR_SOLVER_TYPE liblinear_solver_type = L1R_L2LOSS_SVC;
		train_with_solver(liblinear_solver_type, false, true); //No bias, and l1
}

TEST_F(LibLinear, train_L1R_LR)
{
	LIBLINEAR_SOLVER_TYPE liblinear_solver_type = L1R_LR;
	train_with_solver(liblinear_solver_type, false, true); //No bias, and l1
}

TEST_F(LibLinear, train_L2R_LR_DUAL)
{
	LIBLINEAR_SOLVER_TYPE liblinear_solver_type = L2R_LR_DUAL;
	train_with_solver(liblinear_solver_type, false, false); //No bias, and not l1
}

TEST_F(LibLinear, train_L2R_LR_BIAS)
{
	LIBLINEAR_SOLVER_TYPE liblinear_solver_type = L2R_LR;
	train_with_solver(liblinear_solver_type, true, false); // bias, and not l1
}

TEST_F(LibLinear, train_L2R_L2LOSS_SVC_DUAL_BIAS)
{
		LIBLINEAR_SOLVER_TYPE liblinear_solver_type = L2R_L2LOSS_SVC_DUAL;
		train_with_solver(liblinear_solver_type, true, false); // bias, and not l1
}

TEST_F(LibLinear, train_L2R_L2LOSS_SVC_BIAS)
{
		LIBLINEAR_SOLVER_TYPE liblinear_solver_type = L2R_L2LOSS_SVC;
		train_with_solver(liblinear_solver_type, true, false); // bias, and not l1
}

TEST_F(LibLinear, train_L2R_L1LOSS_SVC_DUAL_BIAS)
{
		LIBLINEAR_SOLVER_TYPE liblinear_solver_type = L2R_L1LOSS_SVC_DUAL;
		train_with_solver(liblinear_solver_type, true, false); // bias, and not l1

}

TEST_F(LibLinear, train_L1R_L2LOSS_SVC_BIAS)
{
		LIBLINEAR_SOLVER_TYPE liblinear_solver_type = L1R_L2LOSS_SVC;
		train_with_solver(liblinear_solver_type, true, true); // bias, and l1
}

TEST_F(LibLinear, train_L1R_LR_BIAS)
{
	LIBLINEAR_SOLVER_TYPE liblinear_solver_type = L1R_LR;
	train_with_solver(liblinear_solver_type, true, true); // bias, and l1
}

TEST_F(LibLinear, train_L2R_LR_DUAL_BIAS)
{
	LIBLINEAR_SOLVER_TYPE liblinear_solver_type = L2R_LR_DUAL;
	train_with_solver(liblinear_solver_type, true, false); // bias, and not l1
}

/*
* --------------------------------
*	Simple set tests start from here
* --------------------------------
*/

TEST_F(LibLinear, simple_set_train_L2R_LR)
{
	LIBLINEAR_SOLVER_TYPE liblinear_solver_type = L2R_LR;
	SGVector<float64_t> t_w(2);
	t_w[0] = -1.150365336474293;
	t_w[1] = -0.4144481723881207;
	train_with_solver_simple(liblinear_solver_type, false, false, t_w); //no bias, not l1
}

TEST_F(LibLinear, simple_set_train_L2R_L2LOSS_SVC_DUAL)
{
	LIBLINEAR_SOLVER_TYPE liblinear_solver_type = L2R_L2LOSS_SVC_DUAL;
	SGVector<float64_t> t_w(2);
	t_w[0] = -0.9523799021273924;
	t_w[1] = -0.3809534312059407;

	train_with_solver_simple(liblinear_solver_type, false, false,t_w); //no bias, not l1
}

TEST_F(LibLinear, simple_set_train_L2R_L2LOSS_SVC)
{
	SGVector<float64_t> t_w(2);
	t_w[0] = -0.9523799021273924;
	t_w[1] = -0.3809534312059407;

	LIBLINEAR_SOLVER_TYPE liblinear_solver_type = L2R_L2LOSS_SVC;
	train_with_solver_simple(liblinear_solver_type, false, false, t_w); //no bias, not l1
}

TEST_F(LibLinear, simple_set_train_L2R_L1LOSS_SVC_DUAL)
{
	SGVector<float64_t> t_w(2);
	t_w[0] = -0.5;
	t_w[1] = -0.1 ;
	LIBLINEAR_SOLVER_TYPE liblinear_solver_type = L2R_L1LOSS_SVC_DUAL;
	train_with_solver_simple(liblinear_solver_type, false, false, t_w, true); //no bias, not l1, pass C_value
}

TEST_F(LibLinear, simple_set_train_L1R_L2LOSS_SVC)
{
	SGVector<float64_t> t_w(2);
	t_w[0] = -0.8333333333333333;
	t_w[1] = -0.1666666666666667;

	LIBLINEAR_SOLVER_TYPE liblinear_solver_type = L1R_L2LOSS_SVC;
	train_with_solver_simple(liblinear_solver_type, false, true, t_w); //no bias, and l1
}

TEST_F(LibLinear, simple_set_train_L1R_LR)
{
	LIBLINEAR_SOLVER_TYPE liblinear_solver_type = L1R_LR;
	SGVector<float64_t> t_w(2);
	t_w[0] = -1.378683364127616 ;
	t_w[1] = -0;
	train_with_solver_simple(liblinear_solver_type, false, true, t_w); //no bias, l1
}

TEST_F(LibLinear, simple_set_train_L2R_LR_DUAL)
{
	LIBLINEAR_SOLVER_TYPE liblinear_solver_type = L2R_LR_DUAL;
	SGVector<float64_t> t_w(2);
	t_w[0] = -1.150367316729321 ;
	t_w[1] = -0.414449095403961;
	train_with_solver_simple(liblinear_solver_type, false, false, t_w); //no bias, not l1
}

TEST_F(LibLinear, simple_set_train_L2R_LR_BIAS)
{
	LIBLINEAR_SOLVER_TYPE liblinear_solver_type = L2R_LR;
	SGVector<float64_t> t_w(3);
	t_w[0] = -1.074173966275961 ;
	t_w[1] = -0.4636306285427702;
	t_w[2] = 0.6182115884788618;
	train_with_solver_simple(liblinear_solver_type, true, false, t_w); //no bias, not l1
}

TEST_F(LibLinear, simple_set_train_L2R_L2LOSS_SVC_DUAL_BIAS)
{
	LIBLINEAR_SOLVER_TYPE liblinear_solver_type = L2R_L2LOSS_SVC_DUAL;
	SGVector<float64_t> t_w(3);
	t_w[0] = -0.5153970026404913 ;
	t_w[1] = -0.2463534232497313;
	t_w[2] = 0.5737439568971296;
	train_with_solver_simple(liblinear_solver_type, true, false, t_w); //bias, not l1
}

TEST_F(LibLinear, simple_set_train_L2R_L2LOSS_SVC_BIAS)
{
	LIBLINEAR_SOLVER_TYPE liblinear_solver_type = L2R_L2LOSS_SVC;
	SGVector<float64_t> t_w(3);
	t_w[0] = -0.5153970826580226 ;
	t_w[1] = -0.2463533225283631;
	t_w[2] = 0.5737439222042139;
	train_with_solver_simple(liblinear_solver_type, true, false, t_w); //bias, not l1
}

TEST_F(LibLinear, simple_set_train_L2R_L1LOSS_SVC_DUAL_BIAS)
{
	LIBLINEAR_SOLVER_TYPE liblinear_solver_type = L2R_L1LOSS_SVC_DUAL;
	SGVector<float64_t> t_w(3);
	t_w[0] = -0.5714285446051303;
	t_w[1] = -0.2857143192435871;
	t_w[2] = 0.7142857276974349;
	train_with_solver_simple(liblinear_solver_type, true, false, t_w); //bias, not l1
}

TEST_F(LibLinear, simple_set_train_L1R_L2LOSS_SVC_BIAS)
{
	LIBLINEAR_SOLVER_TYPE liblinear_solver_type = L1R_L2LOSS_SVC;
	SGVector<float64_t> t_w(3);
	t_w[0] = -0.5118958301270533;
	t_w[1] = -0.1428645860052333;
	t_w[2] = 0.44643750320628;
	train_with_solver_simple(liblinear_solver_type, true, true, t_w); // bias, l1
}
TEST_F(LibLinear, simple_set_train_L1R_LR_BIAS)
{
	LIBLINEAR_SOLVER_TYPE liblinear_solver_type = L1R_LR;
	SGVector<float64_t> t_w(3);
	t_w[0] = -1.365213552966976;
	t_w[1] = -0;
	t_w[2] = 0.06345876011584652;
	train_with_solver_simple(liblinear_solver_type, true, true, t_w); // bias, l1
}
TEST_F(LibLinear, simple_set_train_L2R_LR_DUAL_BIAS)
{
	LIBLINEAR_SOLVER_TYPE liblinear_solver_type = L2R_LR_DUAL;
	SGVector<float64_t> t_w(3);
	t_w[0] = -1.074172813563463;
	t_w[1] = -0.4636342439865924;
	t_w[2] = 0.6182083181247333;
	train_with_solver_simple(liblinear_solver_type, true, false, t_w); // bias, not l1
}
#endif //HAVE_LAPACK
