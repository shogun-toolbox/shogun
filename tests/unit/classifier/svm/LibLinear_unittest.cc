/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: pl8787, Elfarouk Yasser
 */

#include <shogun/classifier/svm/LibLinear.h>
#include <shogun/features/DataGenerator.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/evaluation/ContingencyTableEvaluation.h>
#include <shogun/mathematics/Math.h>
#include <gtest/gtest.h>

using namespace shogun;


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

	void train_with_solver
	(LIBLINEAR_SOLVER_TYPE llst, bool biasEnable, bool l1)
	{
		LIBLINEAR_SOLVER_TYPE liblinear_solver_type = llst;

		if (l1)
				generate_data_l1();
		else
				generate_data_l2();

		auto ll = new CLibLinear();
		SG_REF(ll);

		auto eval = new CContingencyTableEvaluation();
		SG_REF(eval);

		ll->set_bias_enabled(biasEnable);
		ll->set_features(train_feats);
		ll->set_labels(ground_truth);

		ll->set_liblinear_solver_type(liblinear_solver_type);
		ll->train();
		auto pred = ll->apply_binary(test_feats);
		SG_REF(pred);

		auto liblin_accuracy = eval->evaluate(pred, ground_truth);

		EXPECT_NEAR(liblin_accuracy, 1.0, 1e-6);

		SG_UNREF(eval);
		SG_UNREF(ll);
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

		auto ll = new CLibLinear();
		SG_REF(ll);

		auto eval = new  CContingencyTableEvaluation();
		SG_REF(eval);

		ll->set_bias_enabled(biasEnable);
		ll->set_features(train_feats);
		if (C_value)
			ll->set_C(0.1,0.1); //Only in the case of L2R_L1LOSS_SVC_DUAL
		ll->set_labels(ground_truth);
		ll->set_liblinear_solver_type(liblinear_solver_type);
		ll->train();

		auto pred = ll->apply_binary(test_feats);
		SG_REF(pred);

		auto liblin_accuracy = eval->evaluate(pred, ground_truth);

		if (!biasEnable)
		{
			for (auto i : range(t_w.vlen))
				EXPECT_NEAR(ll->get_w()[i], t_w[i], 1e-5);
		}
		else
		{
			for (auto i : range(t_w.vlen))
				EXPECT_NEAR(ll->get_w()[i], t_w[i], 4e-5);
			EXPECT_NEAR(ll->get_bias(), t_w[2], 4e-5);
		}
		EXPECT_NEAR(liblin_accuracy, 1.0, 1e-5);

		SG_UNREF(ll);
		SG_UNREF(eval);
		SG_UNREF(pred);
	}

protected:
	void generate_data_l2()
	{
		index_t num_samples = 50;
		sg_rand->set_seed(5);


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
			Now fill in train_feats and test_feats, and ground_truth
		*/
		train_feats = (CDenseFeatures<float64_t>*)features.copy_subset(train_idx);
		test_feats =  (CDenseFeatures<float64_t>*)features.copy_subset(test_idx);
		ground_truth = new CBinaryLabels(labels);

		SG_REF(ground_truth);
	}
	void generate_data_l1()
	{
		generate_data_l2();
		train_feats = train_feats->get_transposed();
		SG_REF(train_feats);
	}
	void generate_data_l2_simple()
	{
		SGMatrix<float64_t> train_data(2, 10);
		SGMatrix<float64_t> test_data(2, 10);

		index_t num_samples = 10;
		SGVector<int32_t> x{0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7,8,8,9,9};
		SGVector<int32_t> y{0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1};
		SGVector<int32_t> z{-1,1,-1,0,-1,-1,-2,0,0,-1,2,2,3,1,3,2,3,3,4,2};

		for(index_t i=0; i<num_samples*2; ++i){
			train_data(y[i], x[i])=z[i];
			test_data(y[i], x[i])=z[i];
		}

		train_feats = new CDenseFeatures<float64_t>(train_data);
		test_feats = new  CDenseFeatures<float64_t>(test_data);

		SGVector<float64_t> labels(num_samples);
		for (index_t i = 0; i < num_samples; ++i)
		{
			labels[i] = (i < num_samples/2) ? 1.0 : -1.0;
		}
		ground_truth = new CBinaryLabels(labels);

		SG_REF(train_feats);
		SG_REF(test_feats);
		SG_REF(ground_truth);

	}
	void generate_data_l1_simple()
	{
		generate_data_l2_simple();
		train_feats = train_feats->get_transposed();
		SG_REF(train_feats);
	}
};

TEST_F(LibLinear, train_L2R_LR)
{
	LIBLINEAR_SOLVER_TYPE liblinear_solver_type = L2R_LR;
	//No bias, and not l1
	train_with_solver(liblinear_solver_type, false, false);
}

TEST_F(LibLinear, train_L2R_L2LOSS_SVC_DUAL)
{
		LIBLINEAR_SOLVER_TYPE liblinear_solver_type = L2R_L2LOSS_SVC_DUAL;
		//No bias, and not l1
		train_with_solver(liblinear_solver_type, false, false);
}

TEST_F(LibLinear, train_L2R_L2LOSS_SVC)
{
		LIBLINEAR_SOLVER_TYPE liblinear_solver_type = L2R_L2LOSS_SVC;
		//No bias, and not l1
		train_with_solver(liblinear_solver_type, false, false);
}

TEST_F(LibLinear, train_L2R_L1LOSS_SVC_DUAL)
{
		LIBLINEAR_SOLVER_TYPE liblinear_solver_type = L2R_L1LOSS_SVC_DUAL;
		//No bias, and not l1
		train_with_solver(liblinear_solver_type, false, false);

}

TEST_F(LibLinear, train_L1R_L2LOSS_SVC)
{
		LIBLINEAR_SOLVER_TYPE liblinear_solver_type = L1R_L2LOSS_SVC;
		//No bias, and l1
		train_with_solver(liblinear_solver_type, false, true);
}

TEST_F(LibLinear, train_L1R_LR)
{
	LIBLINEAR_SOLVER_TYPE liblinear_solver_type = L1R_LR;
	//No bias, and l1
	train_with_solver(liblinear_solver_type, false, true);
}

TEST_F(LibLinear, train_L2R_LR_DUAL)
{
	LIBLINEAR_SOLVER_TYPE liblinear_solver_type = L2R_LR_DUAL;
	//No bias, and not l1
	train_with_solver(liblinear_solver_type, false, false);
}

TEST_F(LibLinear, train_L2R_LR_BIAS)
{
	LIBLINEAR_SOLVER_TYPE liblinear_solver_type = L2R_LR;
	// bias, and not l1
	train_with_solver(liblinear_solver_type, true, false);
}

TEST_F(LibLinear, train_L2R_L2LOSS_SVC_DUAL_BIAS)
{
		LIBLINEAR_SOLVER_TYPE liblinear_solver_type = L2R_L2LOSS_SVC_DUAL;
		// bias, and not l1
		train_with_solver(liblinear_solver_type, true, false);
}

TEST_F(LibLinear, train_L2R_L2LOSS_SVC_BIAS)
{
		LIBLINEAR_SOLVER_TYPE liblinear_solver_type = L2R_L2LOSS_SVC;
		// bias, and not l1
		train_with_solver(liblinear_solver_type, true, false);
}

TEST_F(LibLinear, train_L2R_L1LOSS_SVC_DUAL_BIAS)
{
		LIBLINEAR_SOLVER_TYPE liblinear_solver_type = L2R_L1LOSS_SVC_DUAL;
		// bias, and not l1
		train_with_solver(liblinear_solver_type, true, false);

}

TEST_F(LibLinear, train_L1R_L2LOSS_SVC_BIAS)
{
		LIBLINEAR_SOLVER_TYPE liblinear_solver_type = L1R_L2LOSS_SVC;
		// bias, and l1
		train_with_solver(liblinear_solver_type, true, true);
}

TEST_F(LibLinear, train_L1R_LR_BIAS)
{
	LIBLINEAR_SOLVER_TYPE liblinear_solver_type = L1R_LR;
	// bias, and l1
	train_with_solver(liblinear_solver_type, true, true);
}

TEST_F(LibLinear, train_L2R_LR_DUAL_BIAS)
{
	LIBLINEAR_SOLVER_TYPE liblinear_solver_type = L2R_LR_DUAL;
	// bias, and not l1
	train_with_solver(liblinear_solver_type, true, false);
}

TEST_F(LibLinear, simple_set_train_L2R_LR)
{
	LIBLINEAR_SOLVER_TYPE liblinear_solver_type = L2R_LR;
	SGVector<float64_t> t_w(2);
	t_w[0] = -1.150365336474293;
	t_w[1] = -0.4144481723881207;
	//no bias, not l1
	train_with_solver_simple(liblinear_solver_type, false, false, t_w);
}

TEST_F(LibLinear, simple_set_train_L2R_L2LOSS_SVC_DUAL)
{
	LIBLINEAR_SOLVER_TYPE liblinear_solver_type = L2R_L2LOSS_SVC_DUAL;
	SGVector<float64_t> t_w(2);
	t_w[0] = -0.9523799021273924;
	t_w[1] = -0.3809534312059407;
	//no bias, not l1
	train_with_solver_simple(liblinear_solver_type, false, false,t_w);
}

TEST_F(LibLinear, simple_set_train_L2R_L2LOSS_SVC)
{
	SGVector<float64_t> t_w(2);
	t_w[0] = -0.9523799021273924;
	t_w[1] = -0.3809534312059407;

	LIBLINEAR_SOLVER_TYPE liblinear_solver_type = L2R_L2LOSS_SVC;
	//no bias, not l1
	train_with_solver_simple(liblinear_solver_type, false, false, t_w);
}

TEST_F(LibLinear, simple_set_train_L2R_L1LOSS_SVC_DUAL)
{
	SGVector<float64_t> t_w(2);
	t_w[0] = -0.5;
	t_w[1] = -0.1 ;
	LIBLINEAR_SOLVER_TYPE liblinear_solver_type = L2R_L1LOSS_SVC_DUAL;
	//no bias, not l1, pass C_value
	train_with_solver_simple(liblinear_solver_type, false, false, t_w, true);
}

TEST_F(LibLinear, simple_set_train_L1R_L2LOSS_SVC)
{
	SGVector<float64_t> t_w(2);
	t_w[0] = -0.8333333333333333;
	t_w[1] = -0.1666666666666667;

	LIBLINEAR_SOLVER_TYPE liblinear_solver_type = L1R_L2LOSS_SVC;
	//no bias, and l1
	train_with_solver_simple(liblinear_solver_type, false, true, t_w);
}

TEST_F(LibLinear, simple_set_train_L1R_LR)
{
	LIBLINEAR_SOLVER_TYPE liblinear_solver_type = L1R_LR;
	SGVector<float64_t> t_w(2);
	t_w[0] = -1.378683364127616 ;
	t_w[1] = -0;
	//no bias, l1
	train_with_solver_simple(liblinear_solver_type, false, true, t_w);
}

TEST_F(LibLinear, simple_set_train_L2R_LR_DUAL)
{
	LIBLINEAR_SOLVER_TYPE liblinear_solver_type = L2R_LR_DUAL;
	SGVector<float64_t> t_w(2);
	t_w[0] = -1.150367316729321 ;
	t_w[1] = -0.414449095403961;
	//no bias, not l1
	train_with_solver_simple(liblinear_solver_type, false, false, t_w);
}

TEST_F(LibLinear, simple_set_train_L2R_LR_BIAS)
{
	LIBLINEAR_SOLVER_TYPE liblinear_solver_type = L2R_LR;
	SGVector<float64_t> t_w(3);
	t_w[0] = -1.074173966275961 ;
	t_w[1] = -0.4636306285427702;
	t_w[2] = 0.6182115884788618;
	//no bias, not l1
	train_with_solver_simple(liblinear_solver_type, true, false, t_w);
}

TEST_F(LibLinear, simple_set_train_L2R_L2LOSS_SVC_DUAL_BIAS)
{
	LIBLINEAR_SOLVER_TYPE liblinear_solver_type = L2R_L2LOSS_SVC_DUAL;
	SGVector<float64_t> t_w(3);
	t_w[0] = -0.5153970026404913 ;
	t_w[1] = -0.2463534232497313;
	t_w[2] = 0.5737439568971296;
	//bias, not l1
	train_with_solver_simple(liblinear_solver_type, true, false, t_w);
}

TEST_F(LibLinear, simple_set_train_L2R_L2LOSS_SVC_BIAS)
{
	LIBLINEAR_SOLVER_TYPE liblinear_solver_type = L2R_L2LOSS_SVC;
	SGVector<float64_t> t_w(3);
	t_w[0] = -0.5153970826580226 ;
	t_w[1] = -0.2463533225283631;
	t_w[2] = 0.5737439222042139;
	//bias, not l1
	train_with_solver_simple(liblinear_solver_type, true, false, t_w);
}

TEST_F(LibLinear, simple_set_train_L2R_L1LOSS_SVC_DUAL_BIAS)
{
	LIBLINEAR_SOLVER_TYPE liblinear_solver_type = L2R_L1LOSS_SVC_DUAL;
	SGVector<float64_t> t_w(3);
	t_w[0] = -0.5714285446051303;
	t_w[1] = -0.2857143192435871;
	t_w[2] = 0.7142857276974349;
	//bias, not l1
	train_with_solver_simple(liblinear_solver_type, true, false, t_w);
}

TEST_F(LibLinear, simple_set_train_L1R_L2LOSS_SVC_BIAS)
{
	LIBLINEAR_SOLVER_TYPE liblinear_solver_type = L1R_L2LOSS_SVC;
	SGVector<float64_t> t_w(3);
	t_w[0] = -0.5118958301270533;
	t_w[1] = -0.1428645860052333;
	t_w[2] = 0.44643750320628;
	// bias, l1
	train_with_solver_simple(liblinear_solver_type, true, true, t_w);
}
TEST_F(LibLinear, simple_set_train_L1R_LR_BIAS)
{
	LIBLINEAR_SOLVER_TYPE liblinear_solver_type = L1R_LR;
	SGVector<float64_t> t_w(3);
	t_w[0] = -1.365213552966976;
	t_w[1] = -0;
	t_w[2] = 0.06345876011584652;
	// bias, l1
	train_with_solver_simple(liblinear_solver_type, true, true, t_w);
}
TEST_F(LibLinear, simple_set_train_L2R_LR_DUAL_BIAS)
{
	LIBLINEAR_SOLVER_TYPE liblinear_solver_type = L2R_LR_DUAL;
	SGVector<float64_t> t_w(3);
	t_w[0] = -1.074172813563463;
	t_w[1] = -0.4636342439865924;
	t_w[2] = 0.6182083181247333;
	// bias, not l1
	train_with_solver_simple(liblinear_solver_type, true, false, t_w);
}
