/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Heiko Strathmann
 */
#include <gtest/gtest.h>
#include <shogun/labels/MappedLabels.h>
#include <shogun/lib/SGVector.h>
#include <shogun/lib/SGMatrix.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/machine/Machine.h>
#include <shogun/labels/MappedBinaryLabels.h>
#include <shogun/labels/MappedMulticlassLabels.h>


#include "environments/LinearTestEnvironment.h"

extern LinearTestEnvironment* linear_test_env;

using namespace shogun;

/** Mock class that asserts correct provided internal label type and returns
 * the same internal label type in train and apply respectively.
 * E.g. LibSVM expects CBinaryLabels and returns CBinaryLabels.
 * Ensures that all machines can rely on particular internal labels, depending
 * on problem type.
 */
class MockMachine : public CMachine
{
public:
	MockMachine(EProblemType pt) : CMachine()
	{
//		get_global_io()->set_loglevel(MSG_GCDEBUG);
		m_problem_type = pt;
	}

	virtual const char* get_name() const override { return "MockMachine"; }


protected:
	virtual bool train_machine(CFeatures*) override
	{
		switch (m_problem_type)
		{
		case PT_BINARY:
		{
			// binary are -1, +1
			auto casted = m_labels->as<CBinaryLabels>();
			ASSERT(casted);
			for (auto l : casted->get_labels())
				EXPECT_TRUE(l == -1 || l == 1);
			for (auto i : range(casted->get_num_labels()))
			{
				auto l = casted->get_label(i);
				EXPECT_TRUE(l == -1 || l == 1);
			}
			break;
		}

		case PT_MULTICLASS:
		{
			// multiclass are 0,1,2,3, ... (contiguous integers)
			auto casted = m_labels->as<CMulticlassLabels>();
			ASSERT(casted);
			for (auto l : casted->get_labels())
				EXPECT_TRUE(l >= 0 && l == int64_t(l));
			auto uniq = casted->get_labels().unique();
			for (auto i : range(uniq.size()))
				EXPECT_EQ(uniq[i], i);
			break;
		}

		default:
			EXPECT_TRUE(false);
		}
		return true;
	}

	virtual CBinaryLabels* apply_binary(CFeatures*) override
	{
		auto labs = SGVector<float64_t>(10);
		for (auto i : range(labs.size()))
			labs[i] = 2*(i%2)-1;
		auto result = new CBinaryLabels(labs);
		SG_REF(result);
		return result;
	}

	virtual CMulticlassLabels* apply_multiclass(CFeatures*) override
	{
		auto labs = SGVector<float64_t>(10);
		auto num_classes = m_labels->as<CMulticlassLabels>()->get_labels().unique().vlen;
		ASSERT(num_classes);
		for (auto i : range(labs.size()))
			labs[i] = i % num_classes;
		auto result = new CMulticlassLabels(labs);
		SG_REF(result);
		return result;
	}

	virtual EProblemType get_machine_problem_type() const
	{
		return m_problem_type;
	}

	EProblemType m_problem_type;
};

class MappedLabelsFixture : public ::testing::Test
{
protected:
	void SetUp()
	{
		feats = nullptr;
		m = nullptr;
		result = nullptr;
	}

	void TearDown()
	{
		SG_UNREF(feats);
		SG_UNREF(m);
		SG_UNREF(result);
	}

public:
	CLabels* train_and_apply(CLabels* labs, EProblemType pt)
	{
		ASSERT(labs);

		X = SGMatrix<float64_t>(1, labs->get_num_labels());
		feats = new CDenseFeatures<float64_t>(X);
		SG_REF(feats);

		m = new MockMachine(pt);
		SG_REF(m);

		m->set_labels(labs);
		m->train(feats);
		result = m->apply();
		return result;
	}

	template <class USER_PROVIDED, class MAPPING, class EXPECTED_INTERNAL>
	void test_mapped_labels(SGVector<float64_t> user_provided, SGVector<float64_t> expected_internal, EProblemType pt)
	{
		auto labs = some<USER_PROVIDED>(user_provided);
		auto applied = train_and_apply(labs, pt);

		// CMachine::apply returns the same class as user provided
		EXPECT_TRUE(dynamic_cast<USER_PROVIDED*>(applied));

		// internal labels are of appropriate class and mapping works
		auto internal = m->get_labels();
		auto casted = dynamic_cast<MAPPING*>(internal);
		ASSERT_TRUE(casted);
		EXPECT_TRUE(casted->get_labels().equals(expected_internal));

		// inversion mapping works: mapping of the internal labels gives original
		auto invert_labels = some<EXPECTED_INTERNAL>(expected_internal);
		auto inverted = casted->invert(invert_labels);
		EXPECT_TRUE(inverted->get_labels().equals(labs->get_labels()));
		SG_UNREF(inverted);
		SG_UNREF(internal);
	}

	template <class USER_PROVIDED>
	void test_mapped_labels(SGVector<float64_t> user_provided, EProblemType pt)
	{
		auto labs = some<USER_PROVIDED>(user_provided);
		auto applied = train_and_apply(labs, pt);

		// CMachine::apply returns the same class as user provided
		EXPECT_TRUE(dynamic_cast<USER_PROVIDED*>(applied));

		// internal labels are the same instance as user provided
		auto converted = m->get_labels();
		EXPECT_TRUE(converted == labs);
		SG_UNREF(converted);
	}

	SGMatrix<float64_t> X;
	CFeatures* feats;
	CMachine* m;
	CLabels* result;
};

TEST_F(MappedLabelsFixture, mc_as_bin)
{
	test_mapped_labels<CMulticlassLabels, CMappedBinaryLabels, CBinaryLabels>({0,2,0,2}, {-1,1,-1,1}, PT_BINARY);
}

TEST_F(MappedLabelsFixture, bin_as_same_bin)
{
	test_mapped_labels<CBinaryLabels>({-1,1,-1,1}, PT_BINARY);
}

TEST_F(MappedLabelsFixture, reg_as_bin)
{
	test_mapped_labels<CRegressionLabels, CMappedBinaryLabels, CBinaryLabels>({-2.1,3.2,3.2,-2.1}, {-1,1,1,-1}, PT_BINARY);
}

TEST_F(MappedLabelsFixture, mc_as_same_mc)
{
	test_mapped_labels<CMulticlassLabels>({0,1,2,3}, PT_MULTICLASS);
}

TEST_F(MappedLabelsFixture, non_contiguous_mc_as_mc)
{
	test_mapped_labels<CMulticlassLabels, CMappedMulticlassLabels, CMulticlassLabels>({0,2,4,6}, {0,1,2,3}, PT_MULTICLASS);
}
TEST_F(MappedLabelsFixture, bin_as_mc)
{
	test_mapped_labels<CBinaryLabels, CMappedMulticlassLabels, CMulticlassLabels>({-1,1,-1,1}, {0,1,0,1}, PT_MULTICLASS);
}

TEST_F(MappedLabelsFixture, reg_as_mc)
{
	test_mapped_labels<CRegressionLabels, CMappedMulticlassLabels, CMulticlassLabels>({-2.1,3.2,3.2,1.7}, {0,2,2,1}, PT_MULTICLASS);
}

