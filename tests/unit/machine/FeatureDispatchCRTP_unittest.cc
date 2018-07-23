#include "utils/Utils.h"
#include <gtest/gtest.h>
#include <shogun/base/some.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/features/StringFeatures.h>
#include <shogun/machine/FeatureDispatchCRTP.h>
#include <shogun/machine/LinearMachine.h>

using namespace shogun;

class CMockModel : public CDenseRealDispatch<CMockModel, CMachine>
{
public:
	CMockModel() : CDenseRealDispatch<CMockModel, CMachine>()
	{
	}
	~CMockModel()
	{
	}
	template <typename T>
	bool train_machine_templated(CDenseFeatures<T>* data)
	{
		if (data->get_feature_type() == val)
			return true;
		return false;
	}
	virtual const char* get_name() const
	{
		return "CMockModel";
	}

	EFeatureType val;
};

typedef ::testing::Types<float32_t, float64_t, floatmax_t> SGFloatTypes;

template <typename T>
class FeatureDispatchCRTP : public ::testing::Test
{
};

TYPED_TEST_CASE(FeatureDispatchCRTP, SGFloatTypes);

TYPED_TEST(FeatureDispatchCRTP, train_with_dense)
{
	auto feats = SGMatrix<TypeParam>(2, 2);
	feats.set_const(1);
	auto features = some<CDenseFeatures<TypeParam>>(feats);
	auto labels = SGVector<float64_t>({1, -1});

	auto mock_machine = some<CMockModel>();
	mock_machine->set_labels(some<CBinaryLabels>(labels));

	mock_machine->val = features->get_feature_type();
	EXPECT_TRUE(mock_machine->train(features));
}

TEST(TrainDense, train_dense_with_int)
{
	auto feats = SGMatrix<int16_t>(2, 2);
	feats.set_const(1);
	auto features = some<CDenseFeatures<int16_t>>(feats);
	auto labels = SGVector<float64_t>({1, -1});

	auto mock_machine = some<CMockModel>();
	mock_machine->set_labels(some<CBinaryLabels>(labels));

	EXPECT_THROW(mock_machine->train(features), ShogunException);
}

TEST(TrainDense, train_dense_with_string)
{
	auto strings = generateRandomStringData(2);
	auto features = some<CStringFeatures<char>>(strings, ALPHANUM);

	auto labels = SGVector<float64_t>({1, -1});

	auto mock_machine = some<CMockModel>();
	mock_machine->set_labels(some<CBinaryLabels>(labels));
	EXPECT_THROW(mock_machine->train(features), ShogunException);
}
