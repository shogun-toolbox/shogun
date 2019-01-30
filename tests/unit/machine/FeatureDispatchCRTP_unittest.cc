#include "utils/Utils.h"
#include <gtest/gtest.h>
#include <shogun/base/some.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/features/StringFeatures.h>
#include <shogun/machine/FeatureDispatchCRTP.h>
#include <shogun/machine/LinearMachine.h>

#include "sg_gtest_utilities.h"

using namespace shogun;

class CDenseRealMockMachine
    : public CDenseRealDispatch<CDenseRealMockMachine, CMachine>
{
public:
	CDenseRealMockMachine(EFeatureType f)
	    : CDenseRealDispatch<CDenseRealMockMachine, CMachine>()
	{
		m_expected_feature_type = f;
	}
	~CDenseRealMockMachine()
	{
	}
	template <typename T>
	bool train_machine_templated(CDenseFeatures<T>* data)
	{
		if (data->get_feature_type() == m_expected_feature_type)
			return true;
		return false;
	}
	virtual const char* get_name() const
	{
		return "CDenseRealMockMachine";
	}

	EFeatureType m_expected_feature_type;
};

class CStringMockMachine
    : public CStringFeaturesDispatch<CStringMockMachine, CMachine>
{
public:
	CStringMockMachine(EFeatureType f)
	    : CStringFeaturesDispatch<CStringMockMachine, CMachine>()
	{
		m_expected_feature_type = f;
	}
	~CStringMockMachine()
	{
	}
	template <typename T>
	bool train_machine_templated(CStringFeatures<T>* data)
	{
		if (data->get_feature_type() == m_expected_feature_type)
			return true;
		return false;
	}
	virtual const char* get_name() const
	{
		return "CStringMockMachine";
	}

	EFeatureType m_expected_feature_type;
};


template <typename T>
class DenseDispatchCRTP : public ::testing::Test
{
};

SG_TYPED_TEST_CASE(DenseDispatchCRTP, sg_real_types);

TYPED_TEST(DenseDispatchCRTP, train_with_dense)
{
	auto feats = SGMatrix<TypeParam>(2, 2);
	feats.set_const(1);
	auto features = some<CDenseFeatures<TypeParam>>(feats);
	auto labels = SGVector<float64_t>({1, -1});

	auto mock_machine =
	    some<CDenseRealMockMachine>(features->get_feature_type());
	mock_machine->set_labels(some<CBinaryLabels>(labels));

	EXPECT_TRUE(mock_machine->train(features));
}

typedef ::testing::Types<uint8_t, char, uint16_t> SGCharTypes;

template <typename T>
class StringDispatchCRTP : public ::testing::Test
{
};

TYPED_TEST_CASE(StringDispatchCRTP, SGCharTypes);

TYPED_TEST(StringDispatchCRTP, train_with_string)
{
	auto strings = generateRandomStringData<TypeParam>(2);
	auto features = some<CStringFeatures<TypeParam>>(strings, ALPHANUM);

	auto labels = SGVector<float64_t>({1, -1});

	auto mock_machine = some<CStringMockMachine>(features->get_feature_type());
	mock_machine->set_labels(some<CBinaryLabels>(labels));

	EXPECT_TRUE(mock_machine->train(features));
}

TEST(TrainDense, train_dense_with_wrong_feature_type)
{
	auto feats = SGMatrix<int16_t>(2, 2);
	feats.set_const(1);
	auto features = some<CDenseFeatures<int16_t>>(feats);
	auto labels = SGVector<float64_t>({1, -1});

	auto mock_machine =
	    some<CDenseRealMockMachine>(features->get_feature_type());
	mock_machine->set_labels(some<CBinaryLabels>(labels));

	EXPECT_THROW(mock_machine->train(features), ShogunException);
}

TEST(TrainDense, train_dense_with_wrong_feature_class)
{
	auto strings = generateRandomStringData(2);
	auto features = some<CStringFeatures<char>>(strings, ALPHANUM);

	auto labels = SGVector<float64_t>({1, -1});

	auto mock_machine =
	    some<CDenseRealMockMachine>(features->get_feature_type());
	mock_machine->set_labels(some<CBinaryLabels>(labels));
	EXPECT_THROW(mock_machine->train(features), ShogunException);
}
