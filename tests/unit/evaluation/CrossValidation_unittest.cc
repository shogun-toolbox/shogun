/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Heiko Strathmann
 */

#include <gtest/gtest.h>
#include <shogun/base/ShogunEnv.h>
#include <shogun/evaluation/CrossValidation.h>
#include <shogun/evaluation/CrossValidationSplitting.h>
#include <shogun/evaluation/ContingencyTableEvaluation.h>
#include <shogun/evaluation/MulticlassAccuracy.h>
#include <shogun/evaluation/MeanSquaredError.h>

#include <shogun/kernel/GaussianKernel.h>
#include <shogun/distance/EuclideanDistance.h>

#include <shogun/classifier/svm/LibSVM.h>
#include <shogun/classifier/Perceptron.h>
#include <shogun/classifier/svm/LibLinear.h>
#include <shogun/multiclass/MulticlassLibLinear.h>
#include <shogun/multiclass/KNN.h>
#include <shogun/regression/LinearRidgeRegression.h>
#include <shogun/regression/KernelRidgeRegression.h>
#include <shogun/regression/svr/LibLinearRegression.h>

#include <shogun/mathematics/NormalDistribution.h>
#include <shogun/mathematics/linalg/LinalgNamespace.h>



using namespace shogun;


#include <gtest/gtest.h>

using namespace shogun;
using namespace std;

template <class T>
class CrossValidationTests : public ::testing::Test
{
protected:
	void SetUp()
	{
	}

	void init()
	{
		machine = new T();
		SG_REF(machine)

		this->generate_data(this->machine->get_machine_problem_type());

		if (auto* casted = dynamic_cast<CKernelMachine*>(machine))
			casted->set_kernel(new CGaussianKernel());
		if (auto* casted = dynamic_cast<CDistanceMachine*>(machine))
			casted->set_distance(new CEuclideanDistance());

		auto ss = new CCrossValidationSplitting(labels, 5);
		CEvaluation* ec = nullptr;
		switch (machine->get_machine_problem_type())
		{
		case PT_BINARY:
			ec = new CAccuracyMeasure();
			break;
		case PT_MULTICLASS:
			ec = new CMulticlassAccuracy();
			break;
		case PT_REGRESSION:
			ec = new CMeanSquaredError();
			break;
		default:
			SG_SNOTIMPLEMENTED
			break;
		}
		cv = new CCrossValidation(machine, features, labels, ss, ec);
		cv->set_num_runs(3);
		SG_REF(cv);
	}

	void clean()
	{
		SG_UNREF(features)
		SG_UNREF(labels)
		SG_UNREF(machine)
		SG_UNREF(cv);
	}

	void TearDown()
	{
	}

	auto test_single_thread()
	{
		init();
		this->cv->put("seed", 1);
		env()->set_num_threads(1);
		auto result = cv->evaluate()->get<float64_t>("mean");
		clean();
		return result;
	}

	auto test_multi_thread()
	{
		init();
		this->cv->put("seed", 1);
		env()->set_num_threads(4);
		auto result = cv->evaluate()->get<float64_t>("mean");
		clean();
		return result;
	}

	void generate_data(EProblemType pt)
	{
		auto N = 50;
		auto D = 5;

		std::mt19937_64 prng(57);
		NormalDistribution<float64_t> randn;
		UniformRealDistribution<float64_t> rand(0,1);
		UniformIntDistribution<int32_t> randi(0,2);

		SGMatrix<float64_t> X(D,N);
		for (auto i : range(D*N))
			X.matrix[i] = randn(prng);
		features = new CDenseFeatures<float64_t>(X);

		SGVector<float64_t> y_reg(N);
		SGVector<float64_t> y_binary(N);
		SGVector<float64_t> y_mc(N);

		for (auto i : range(N))
		{
			auto redux = linalg::mean(X.get_column(i));

			y_reg[i] = redux + std::sin(redux) + 1;
			y_mc[i] = redux<0 ? 0 : 1;
			y_binary[i] = y_mc[i] * 2 -1;

			// noise
			y_reg[i] +=  randn(prng)*0.1;
			if (rand(prng)>0.1)
			{
				y_binary[i] *= (-1);
				y_mc[i] = (int32_t(y_mc[i]) + randi(prng)) % 3;
			}
		}

		switch (pt)
		{
		case PT_BINARY:
		case PT_CLASS:
		{
			labels = new CBinaryLabels(y_binary);
			break;
		}

		case PT_MULTICLASS:
		{
			labels = new CMulticlassLabels(y_mc);
			break;
		}

		case PT_REGRESSION:
			labels = new CRegressionLabels(y_reg);
			break;

		default:
			SG_SERROR("Unsupported problem type: %d\n", pt);
			FAIL();
		}

		SG_REF(features)
		SG_REF(labels)
	}

	CFeatures* features;
	CLabels* labels;
	CCrossValidation* cv;
	T* machine;
};

typedef ::testing::Types<CLibSVM, CPerceptron, CLibLinear,
		CMulticlassLibLinear, CLinearRidgeRegression, CKNN,
		CKernelRidgeRegression, CLibLinearRegression>
MachineTypes;

TYPED_TEST_CASE(CrossValidationTests, MachineTypes);

TYPED_TEST(CrossValidationTests, execute_single_thread)
{
	this->test_single_thread();
}

TYPED_TEST(CrossValidationTests, execute_multi_thread)
{
	this->test_multi_thread();
}

TYPED_TEST(CrossValidationTests, single_multi_same_result)
{
	auto single = this->test_single_thread();
	auto multi = this->test_multi_thread();

	EXPECT_NEAR(single, multi, 1e-7);
}
