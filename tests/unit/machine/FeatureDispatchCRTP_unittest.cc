#include "environments/LinearTestEnvironment.h"
#include <gtest/gtest.h>
#include <shogun/base/some.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/features/StringFeatures.h>
#include <shogun/machine/FeatureDispatchCRTP.h>
#include <shogun/machine/LinearMachine.h>

using namespace shogun;
extern LinearTestEnvironment* linear_test_env;

class CMockLinear : public CDenseFeaturesDispatch<CMockLinear, CMachine>
{
public:
	CMockLinear() : CDenseFeaturesDispatch<CMockLinear, CMachine>()
	{
	}
	~CMockLinear()
	{
	}
	template <typename T>
	bool train_machine_templated(CDenseFeatures<T>* data)
	{
		if (data->get_feature_type() == F_DREAL)
			return true;
		return false;
	}
	virtual const char* get_name() const
	{
		return "CMockLinear";
	}
};

TEST(FeatureDispatchCRTP, train_with_dense)
{
	auto features = SGMatrix<float64_t>(2, 2);
	auto features_short = SGMatrix<float32_t>(2, 2);
	features.set_const(1);
	features_short.set_const(1);
	auto labels = SGVector<float64_t>({1, -1});

	auto mock_machine = some<CMockLinear>();
	mock_machine->set_labels(some<CBinaryLabels>(labels));

	// cannot train with null features
	EXPECT_THROW(mock_machine->train(), ShogunException);
	// features are F_DREAL and C_DENSE
	EXPECT_TRUE(mock_machine->train(some<CDenseFeatures<float64_t>>(features)));
	// features are F_SHORTREAL and C_DENSE
	EXPECT_FALSE(
	    mock_machine->train(some<CDenseFeatures<float32_t>>(features_short)));
}
