#include "sg_gtest_utilities.h"

#include "utils/Utils.h"
#include <shogun/features/DenseFeatures.h>
#include <shogun/features/StringFeatures.h>
#include <shogun/machine/FeatureDispatchCRTP.h>
#include <shogun/machine/LinearMachine.h>


using namespace shogun;

class CDenseRealMockMachine
    : public DenseRealDispatch<CDenseRealMockMachine, Machine>
{
public:
	CDenseRealMockMachine(EFeatureType f)
	    : DenseRealDispatch<CDenseRealMockMachine, Machine>()
	{
		m_expected_feature_type = f;
	}
	~CDenseRealMockMachine()
	{
	}
	template <typename T>
	bool train_machine_templated(std::shared_ptr<DenseFeatures<T>> data)
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
    : public CStringFeaturesDispatch<CStringMockMachine, Machine>
{
public:
	CStringMockMachine(EFeatureType f)
	    : CStringFeaturesDispatch<CStringMockMachine, Machine>()
	{
		m_expected_feature_type = f;
	}
	~CStringMockMachine()
	{
	}
	template <typename T>
	bool train_machine_templated(std::shared_ptr<StringFeatures<T>> data)
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
	auto features = std::make_shared<DenseFeatures<TypeParam>>(feats);
	auto labels = SGVector<float64_t>({1, -1});

	auto mock_machine =
	    std::make_shared<CDenseRealMockMachine>(features->get_feature_type());
	mock_machine->set_labels(std::make_shared<BinaryLabels>(labels));

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
	auto features = std::make_shared<StringFeatures<TypeParam>>(strings, ALPHANUM);

	auto labels = SGVector<float64_t>({1, -1});

	auto mock_machine = std::make_shared<CStringMockMachine>(features->get_feature_type());
	mock_machine->set_labels(std::make_shared<BinaryLabels>(labels));

	EXPECT_TRUE(mock_machine->train(features));
}

TEST(TrainDense, train_dense_with_wrong_feature_type)
{
	auto feats = SGMatrix<int16_t>(2, 2);
	feats.set_const(1);
	auto features = std::make_shared<DenseFeatures<int16_t>>(feats);
	auto labels = SGVector<float64_t>({1, -1});

	auto mock_machine =
	    std::make_shared<CDenseRealMockMachine>(features->get_feature_type());
	mock_machine->set_labels(std::make_shared<BinaryLabels>(labels));

	EXPECT_THROW(mock_machine->train(features), ShogunException);
}

TEST(TrainDense, train_dense_with_wrong_feature_class)
{
	auto strings = generateRandomStringData(2);
	auto features = std::make_shared<StringFeatures<char>>(strings, ALPHANUM);

	auto labels = SGVector<float64_t>({1, -1});

	auto mock_machine =
	    std::make_shared<CDenseRealMockMachine>(features->get_feature_type());
	mock_machine->set_labels(std::make_shared<BinaryLabels>(labels));
	EXPECT_THROW(mock_machine->train(features), ShogunException);
}
