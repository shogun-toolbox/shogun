/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: pl8787, Elfarouk Yasser
 */

#include <gtest/gtest.h>
#include <shogun/classifier/svm/LibLinear.h>
#include <shogun/features/DataGenerator.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/evaluation/ContingencyTableEvaluation.h>
#include <shogun/mathematics/Math.h>

#include <random>

using namespace shogun;


class LibLinearFixture : public ::testing::Test
{
public:
	std::shared_ptr<DenseFeatures<float64_t>> train_feats;
	std::shared_ptr<DenseFeatures<float64_t>> test_feats;
	std::shared_ptr<BinaryLabels> ground_truth;
	std::mt19937_64 prng;

	virtual void SetUp()
	{
		prng = std::mt19937_64(17);
	}

	virtual void TearDown()
	{



	}

	void train_with_solver
	(LIBLINEAR_SOLVER_TYPE llst, bool biasEnable, bool l1)
	{
		LIBLINEAR_SOLVER_TYPE liblinear_solver_type = llst;

		if (l1)
				generate_data_l1();
		else
				generate_data_l2();

		auto ll = std::make_shared<LibLinear>();


		auto eval = std::make_shared<ContingencyTableEvaluation>();


		ll->set_bias_enabled(biasEnable);
		ll->set_features(train_feats);
		ll->set_labels(ground_truth);

		ll->set_liblinear_solver_type(liblinear_solver_type);
		ll->train();
		auto pred = ll->apply_binary(test_feats);

		auto liblin_accuracy = eval->evaluate(pred, ground_truth);

		EXPECT_NEAR(liblin_accuracy, 1.0, 1e-6);
	}
	void train_with_solver_simple(
		LIBLINEAR_SOLVER_TYPE llst, bool biasEnable, bool l1, SGVector<float64_t> t_w, bool C_value=false) //C_value only for L2R_L1LOSS_SVC_DUAL
	{
		int32_t seed = 100;
		LIBLINEAR_SOLVER_TYPE liblinear_solver_type = llst;

		if (l1)
				generate_data_l1_simple();
		else
				generate_data_l2_simple();

		auto ll = std::make_shared<LibLinear>();


		auto eval = std::make_shared<ContingencyTableEvaluation>();


		ll->set_bias_enabled(biasEnable);
		ll->set_features(train_feats);
		if (C_value)
			ll->set_C(0.1,0.1); //Only in the case of L2R_L1LOSS_SVC_DUAL
		ll->set_labels(ground_truth);
		ll->set_liblinear_solver_type(liblinear_solver_type);
		ll->put("seed", seed);
		ll->train();

		auto pred = ll->apply_binary(test_feats);

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
	}

protected:
	void generate_data_l2()
	{
		index_t num_samples = 50;

		SGMatrix<float64_t> data =
			DataGenerator::generate_gaussians(num_samples, 2, 2, prng);

		DenseFeatures<float64_t> features(data);

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
		train_feats = features.copy_subset(train_idx)->as<DenseFeatures<float64_t>>();
		test_feats =  features.copy_subset(test_idx)->as<DenseFeatures<float64_t>>();
		ground_truth = std::make_shared<BinaryLabels>(labels);


	}
	void generate_data_l1()
	{
		generate_data_l2();
		auto old_train_feats = train_feats;
		train_feats = train_feats->get_transposed();


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

		train_feats = std::make_shared<DenseFeatures<float64_t>>(train_data);
		test_feats = std::make_shared<DenseFeatures<float64_t>>(test_data);

		SGVector<float64_t> labels(num_samples);
		for (index_t i = 0; i < num_samples; ++i)
		{
			labels[i] = (i < num_samples/2) ? 1.0 : -1.0;
		}
		ground_truth = std::make_shared<BinaryLabels>(labels);





	}
	void generate_data_l1_simple()
	{
		generate_data_l2_simple();
		auto old_train_feats = train_feats;
		train_feats = train_feats->get_transposed();


	}
};

TEST_F(LibLinearFixture, train_L2R_LR)
{
	LIBLINEAR_SOLVER_TYPE liblinear_solver_type = L2R_LR;
	//No bias, and not l1
	train_with_solver(liblinear_solver_type, false, false);
}

TEST_F(LibLinearFixture, train_L2R_L2LOSS_SVC_DUAL)
{
		LIBLINEAR_SOLVER_TYPE liblinear_solver_type = L2R_L2LOSS_SVC_DUAL;
		//No bias, and not l1
		train_with_solver(liblinear_solver_type, false, false);
}

TEST_F(LibLinearFixture, train_L2R_L2LOSS_SVC)
{
		LIBLINEAR_SOLVER_TYPE liblinear_solver_type = L2R_L2LOSS_SVC;
		//No bias, and not l1
		train_with_solver(liblinear_solver_type, false, false);
}

TEST_F(LibLinearFixture, train_L2R_L1LOSS_SVC_DUAL)
{
		LIBLINEAR_SOLVER_TYPE liblinear_solver_type = L2R_L1LOSS_SVC_DUAL;
		//No bias, and not l1
		train_with_solver(liblinear_solver_type, false, false);

}

TEST_F(LibLinearFixture, train_L1R_L2LOSS_SVC)
{
		LIBLINEAR_SOLVER_TYPE liblinear_solver_type = L1R_L2LOSS_SVC;
		//No bias, and l1
		train_with_solver(liblinear_solver_type, false, true);
}

TEST_F(LibLinearFixture, train_L1R_LR)
{
	LIBLINEAR_SOLVER_TYPE liblinear_solver_type = L1R_LR;
	//No bias, and l1
	train_with_solver(liblinear_solver_type, false, true);
}

TEST_F(LibLinearFixture, train_L2R_LR_DUAL)
{
	LIBLINEAR_SOLVER_TYPE liblinear_solver_type = L2R_LR_DUAL;
	//No bias, and not l1
	train_with_solver(liblinear_solver_type, false, false);
}

TEST_F(LibLinearFixture, train_L2R_LR_BIAS)
{
	LIBLINEAR_SOLVER_TYPE liblinear_solver_type = L2R_LR;
	// bias, and not l1
	train_with_solver(liblinear_solver_type, true, false);
}

TEST_F(LibLinearFixture, train_L2R_L2LOSS_SVC_DUAL_BIAS)
{
		LIBLINEAR_SOLVER_TYPE liblinear_solver_type = L2R_L2LOSS_SVC_DUAL;
		// bias, and not l1
		train_with_solver(liblinear_solver_type, true, false);
}

TEST_F(LibLinearFixture, train_L2R_L2LOSS_SVC_BIAS)
{
		LIBLINEAR_SOLVER_TYPE liblinear_solver_type = L2R_L2LOSS_SVC;
		// bias, and not l1
		train_with_solver(liblinear_solver_type, true, false);
}

TEST_F(LibLinearFixture, train_L2R_L1LOSS_SVC_DUAL_BIAS)
{
		LIBLINEAR_SOLVER_TYPE liblinear_solver_type = L2R_L1LOSS_SVC_DUAL;
		// bias, and not l1
		train_with_solver(liblinear_solver_type, true, false);

}

TEST_F(LibLinearFixture, train_L1R_L2LOSS_SVC_BIAS)
{
		LIBLINEAR_SOLVER_TYPE liblinear_solver_type = L1R_L2LOSS_SVC;
		// bias, and l1
		train_with_solver(liblinear_solver_type, true, true);
}

TEST_F(LibLinearFixture, train_L1R_LR_BIAS)
{
	LIBLINEAR_SOLVER_TYPE liblinear_solver_type = L1R_LR;
	// bias, and l1
	train_with_solver(liblinear_solver_type, true, true);
}

TEST_F(LibLinearFixture, train_L2R_LR_DUAL_BIAS)
{
	LIBLINEAR_SOLVER_TYPE liblinear_solver_type = L2R_LR_DUAL;
	// bias, and not l1
	train_with_solver(liblinear_solver_type, true, false);
}

TEST_F(LibLinearFixture, simple_set_train_L2R_LR)
{
	LIBLINEAR_SOLVER_TYPE liblinear_solver_type = L2R_LR;
	SGVector<float64_t> t_w(2);
	t_w[0] = -1.150365336474293;
	t_w[1] = -0.4144481723881207;
	//no bias, not l1
	train_with_solver_simple(liblinear_solver_type, false, false, t_w);
}

TEST_F(LibLinearFixture, simple_set_train_L2R_L2LOSS_SVC_DUAL)
{
	LIBLINEAR_SOLVER_TYPE liblinear_solver_type = L2R_L2LOSS_SVC_DUAL;
	SGVector<float64_t> t_w(2);
	t_w[0] = -0.9523799021273924;
	t_w[1] = -0.3809534312059407;
	//no bias, not l1
	train_with_solver_simple(liblinear_solver_type, false, false,t_w);
}

TEST_F(LibLinearFixture, simple_set_train_L2R_L2LOSS_SVC)
{
	SGVector<float64_t> t_w(2);
	t_w[0] = -0.9523799021273924;
	t_w[1] = -0.3809534312059407;

	LIBLINEAR_SOLVER_TYPE liblinear_solver_type = L2R_L2LOSS_SVC;
	//no bias, not l1
	train_with_solver_simple(liblinear_solver_type, false, false, t_w);
}

TEST_F(LibLinearFixture, simple_set_train_L2R_L1LOSS_SVC_DUAL)
{
	SGVector<float64_t> t_w(2);
	t_w[0] = -0.5;
	t_w[1] = -0.1 ;
	LIBLINEAR_SOLVER_TYPE liblinear_solver_type = L2R_L1LOSS_SVC_DUAL;
	//no bias, not l1, pass C_value
	train_with_solver_simple(liblinear_solver_type, false, false, t_w, true);
}

TEST_F(LibLinearFixture, simple_set_train_L1R_L2LOSS_SVC)
{
	SGVector<float64_t> t_w(2);
	t_w[0] = -0.8333333333333333;
	t_w[1] = -0.1666666666666667;

	LIBLINEAR_SOLVER_TYPE liblinear_solver_type = L1R_L2LOSS_SVC;
	//no bias, and l1
	train_with_solver_simple(liblinear_solver_type, false, true, t_w);
}

TEST_F(LibLinearFixture, simple_set_train_L1R_LR)
{
	LIBLINEAR_SOLVER_TYPE liblinear_solver_type = L1R_LR;
	SGVector<float64_t> t_w(2);
	t_w[0] = -1.378683364127616 ;
	t_w[1] = -0;
	//no bias, l1
	train_with_solver_simple(liblinear_solver_type, false, true, t_w);
}

TEST_F(LibLinearFixture, simple_set_train_L2R_LR_DUAL)
{
	LIBLINEAR_SOLVER_TYPE liblinear_solver_type = L2R_LR_DUAL;
	SGVector<float64_t> t_w(2);
	t_w[0] = -1.150367316729321 ;
	t_w[1] = -0.414449095403961;
	//no bias, not l1
	train_with_solver_simple(liblinear_solver_type, false, false, t_w);
}

TEST_F(LibLinearFixture, simple_set_train_L2R_LR_BIAS)
{
	LIBLINEAR_SOLVER_TYPE liblinear_solver_type = L2R_LR;
	SGVector<float64_t> t_w(3);
	t_w[0] = -1.074173966275961 ;
	t_w[1] = -0.4636306285427702;
	t_w[2] = 0.6182115884788618;
	//no bias, not l1
	train_with_solver_simple(liblinear_solver_type, true, false, t_w);
}

TEST_F(LibLinearFixture, simple_set_train_L2R_L2LOSS_SVC_DUAL_BIAS)
{
	LIBLINEAR_SOLVER_TYPE liblinear_solver_type = L2R_L2LOSS_SVC_DUAL;
	SGVector<float64_t> t_w(3);
	t_w[0] = -0.5153970026404913 ;
	t_w[1] = -0.2463534232497313;
	t_w[2] = 0.5737439568971296;
	//bias, not l1
	train_with_solver_simple(liblinear_solver_type, true, false, t_w);
}

TEST_F(LibLinearFixture, simple_set_train_L2R_L2LOSS_SVC_BIAS)
{
	LIBLINEAR_SOLVER_TYPE liblinear_solver_type = L2R_L2LOSS_SVC;
	SGVector<float64_t> t_w(3);
	t_w[0] = -0.5153970826580226 ;
	t_w[1] = -0.2463533225283631;
	t_w[2] = 0.5737439222042139;
	//bias, not l1
	train_with_solver_simple(liblinear_solver_type, true, false, t_w);
}

TEST_F(LibLinearFixture, simple_set_train_L2R_L1LOSS_SVC_DUAL_BIAS)
{
	LIBLINEAR_SOLVER_TYPE liblinear_solver_type = L2R_L1LOSS_SVC_DUAL;
	SGVector<float64_t> t_w(3);
	t_w[0] = -0.5714285446051303;
	t_w[1] = -0.2857143192435871;
	t_w[2] = 0.7142857276974349;
	//bias, not l1
	train_with_solver_simple(liblinear_solver_type, true, false, t_w);
}

TEST_F(LibLinearFixture, simple_set_train_L1R_L2LOSS_SVC_BIAS)
{
	LIBLINEAR_SOLVER_TYPE liblinear_solver_type = L1R_L2LOSS_SVC;
	SGVector<float64_t> t_w(3);
	t_w[0] = -0.5118958301270533;
	t_w[1] = -0.1428645860052333;
	t_w[2] = 0.44643750320628;
	// bias, l1
	train_with_solver_simple(liblinear_solver_type, true, true, t_w);
}
TEST_F(LibLinearFixture, simple_set_train_L1R_LR_BIAS)
{
	LIBLINEAR_SOLVER_TYPE liblinear_solver_type = L1R_LR;
	SGVector<float64_t> t_w(3);
	t_w[0] = -1.365213552966976;
	t_w[1] = -0;
	t_w[2] = 0.06345876011584652;
	// bias, l1
	train_with_solver_simple(liblinear_solver_type, true, true, t_w);
}
TEST_F(LibLinearFixture, simple_set_train_L2R_LR_DUAL_BIAS)
{
	LIBLINEAR_SOLVER_TYPE liblinear_solver_type = L2R_LR_DUAL;
	SGVector<float64_t> t_w(3);
	t_w[0] = -1.074172813563463;
	t_w[1] = -0.4636342439865924;
	t_w[2] = 0.6182083181247333;
	// bias, not l1
	train_with_solver_simple(liblinear_solver_type, true, false, t_w);
}
