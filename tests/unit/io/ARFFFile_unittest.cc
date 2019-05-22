/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Gil Hoben
 */

#include <shogun/features/DenseFeatures.h>
#include <shogun/io/ARFFFile.h>

#include <sg_gtest_utilities.h>

using namespace shogun;

// Tolerance values for tests
template <typename T>
constexpr T get_epsilon()
{
	return std::numeric_limits<T>::epsilon();
}

// convert type to the supported enums
template <typename T>
constexpr EPrimitiveType convert_type_to_enum()
{
	return EPrimitiveType::PT_UNDEFINED;
}

template <>
constexpr EPrimitiveType convert_type_to_enum<int8_t>()
{
	return EPrimitiveType::PT_INT8;
}

template <>
constexpr EPrimitiveType convert_type_to_enum<int16_t>()
{
	return EPrimitiveType::PT_INT16;
}

template <>
constexpr EPrimitiveType convert_type_to_enum<int32_t>()
{
	return EPrimitiveType::PT_INT32;
}

template <>
constexpr EPrimitiveType convert_type_to_enum<int64_t>()
{
	return EPrimitiveType::PT_INT64;
}

template <>
constexpr EPrimitiveType convert_type_to_enum<float32_t>()
{
	return EPrimitiveType::PT_FLOAT32;
}

template <>
constexpr EPrimitiveType convert_type_to_enum<float64_t>()
{
	return EPrimitiveType::PT_FLOAT64;
}

template <>
constexpr EPrimitiveType convert_type_to_enum<floatmax_t>()
{
	return EPrimitiveType::PT_FLOATMAX;
}

template <typename T>
class ARFF_typed_tests : public ::testing::Test
{
};

SG_TYPED_TEST_CASE(
    ARFF_typed_tests,
    Types<int8_t, int16_t, int32_t, int64_t, float32_t, float64_t, floatmax_t>);

TYPED_TEST(ARFF_typed_tests, Parse_numeric)
{
	auto type = convert_type_to_enum<TypeParam>();

	std::string test = "@relation test_numeric \n"
	                   "%\n"
	                   "% \n"
	                   "@attribute VAR1 numeric \n"
	                   "@attribute VAR2 real \n"
	                   "% \n"
	                   "% \n"
	                   "@data \n"
	                   "50, 5.1 \n"
	                   "45, 4.13 ";
	auto ss = std::make_shared<std::istringstream>(test);
	auto s = std::shared_ptr<std::istream>(ss);

	auto parser = std::make_unique<ARFFDeserializer>(s, type);
	parser->read();
	auto result = parser->get_features();
	ASSERT_EQ(result->get_num_elements(), 2);

	auto col1 =
	    result->get_first_element()->template as<CDenseFeatures<TypeParam>>();
	auto col2 =
	    result->get_next_element()->template as<CDenseFeatures<TypeParam>>();

	SGVector<TypeParam> solution1{50, 45};
	SGVector<TypeParam> solution2{static_cast<TypeParam>(5.1),
	                              static_cast<TypeParam>(4.13)};

	ASSERT_EQ(col1->get_feature_matrix()[0], solution1[0]);
	ASSERT_EQ(col1->get_feature_matrix()[1], solution1[1]);

	EXPECT_NEAR(
	    col2->get_feature_matrix()[0], solution2[0], get_epsilon<TypeParam>());
	EXPECT_NEAR(
	    col2->get_feature_matrix()[1], solution2[1], get_epsilon<TypeParam>());
	ASSERT_EQ(parser->get_relation(), "test_numeric");
}

TEST(ARFFFileTest, Parse_datetime)
{
	std::string test = "@relation test_date \n"
	                   "% \n"
	                   "% \n"
	                   "@attribute PERIOD_DATE date \"yyyy-MM-dd Z\" \n"
	                   "@attribute VAR1 numeric \n"
	                   "% \n"
	                   "% \n"
	                   "@data \n"
	                   "\"2019-01-10 +0000\", 50 \n"
	                   "\"2019-02-10 -0100\", 26 \n"
	                   "\"2019-03-10 +0000\", 34 \n"
	                   "\"2019-04-10 +0000\", 41 \n"
	                   "\"2019-05-10 +0000\", 44 \n"
	                   "\"2019-06-10 +0000\", 45 ";

	auto ss = std::make_shared<std::istringstream>(test);
	auto s = std::shared_ptr<std::istream>(ss);

	SGVector<float64_t> solution1{1547078400, 1549760400, 1552176000,
	                              1554854400, 1557446400, 1560124800};

	SGVector<float64_t> solution2{50, 26, 34, 41, 44, 45};

	auto parser = std::make_unique<ARFFDeserializer>(s);
	parser->read();
	auto result = parser->get_features();
	ASSERT_EQ(result->get_num_elements(), 2);

	auto col1 = result->get_first_element()->as<CDenseFeatures<float64_t>>();
	auto mat1 = col1->get_feature_matrix();
	auto col2 = result->get_next_element()->as<CDenseFeatures<float64_t>>();
	auto mat2 = col2->get_feature_matrix();
	ASSERT_EQ(mat1.size(), 6);
	for (int i = 0; i < 6; ++i)
	{
		ASSERT_EQ(mat1[i], solution1[i]);
		ASSERT_EQ(mat2[i], solution2[i]);
	}
	ASSERT_EQ(parser->get_relation(), "test_date");
}

TEST(ARFFFileTest, Parse_string)
{
	std::string test = "@relation test_string \n"
	                   "@attribute VAR1 string \n"
	                   "@attribute VAR2 numeric \n"
	                   "@data \n"
	                   "\"test1\", 50 \n"
	                   "\"test2\", 26 \n"
	                   "\"test3\", 34 \n"
	                   "test1, 41 \n"
	                   "test2, 44 \n"
	                   "test3, 45 ";

	auto ss = std::make_shared<std::istringstream>(test);
	auto s = std::shared_ptr<std::istream>(ss);

	std::vector<const char*> solution1{"test1", "test2", "test3",
	                                   "test1", "test2", "test3"};
	SGVector<float64_t> solution2{50, 26, 34, 41, 44, 45};

	auto parser = std::make_unique<ARFFDeserializer>(s);
	parser->read();
	auto result = parser->get_features();
	ASSERT_EQ(result->get_num_elements(), 2);
	auto col1 = result->get_first_element()->as<CStringFeatures<char>>();
	auto col2 = result->get_next_element()->as<CDenseFeatures<float64_t>>();
	auto mat2 = col2->get_feature_matrix();
	ASSERT_EQ(col1->get_num_vectors(), 6);
	for (int i = 0; i < col1->get_num_vectors(); ++i)
	{
		auto row = col1->get_feature_vector(i);
		for (auto j = 0; j < col1->get_max_vector_length(); ++j)
			ASSERT_EQ(row[j], solution1[i][j]);
		ASSERT_EQ(mat2[i], solution2[i]);
	}
	ASSERT_EQ(parser->get_relation(), "test_string");
}

TEST(ARFFFileTest, Parse_nominal)
{
	std::string test =
	    "@relation test_nominal \n"
	    "% \n"
	    "% \n"
	    "@attribute \"Twist n\' Shout\" {\"a\", b, \"c 1\", \'¯\\_(ツ)_/¯\'} \n"
	    "@attribute VAR2 numeric \n"
	    "% \n"
	    "% \n"
	    "@data \n"
	    "	\'a\', 50 \n"
	    "b, 26 \n"
	    "\"b\"	, 34 \n"
	    "	\'c 1\'  , 41 \n"
	    "% the row below can be replaced if it causes issues...\n"
	    "\"¯\\_(ツ)_/¯\", 44 \n"
	    "a, 45 ";

	auto ss = std::make_shared<std::istringstream>(test);
	auto s = std::shared_ptr<std::istream>(ss);

	SGVector<float64_t> solution1{0, 1, 1, 2, 3, 0};
	SGVector<float64_t> solution2{50, 26, 34, 41, 44, 45};
	std::vector<std::string> nom_values_result{"a", "b", "c 1", "¯\\_(ツ)_/¯"};

	auto parser = std::make_unique<ARFFDeserializer>(s);
	parser->read();
	auto result = parser->get_features();
	ASSERT_EQ(result->get_num_elements(), 2);

	auto col1 = result->get_first_element()->as<CDenseFeatures<float64_t>>();
	auto mat1 = col1->get_feature_matrix();
	auto col2 = result->get_next_element()->as<CDenseFeatures<float64_t>>();
	auto mat2 = col2->get_feature_matrix();
	ASSERT_EQ(mat1.size(), 6);
	for (int i = 0; i < 6; ++i)
	{
		ASSERT_EQ(mat1[i], solution1[i]);
		ASSERT_EQ(mat2[i], solution2[i]);
	}
	auto nom_values = parser->get_nominal_values("Twist n\' Shout");
	ASSERT_EQ(nom_values.size(), nom_values_result.size());
	for (int i = 0; i < nom_values.size(); ++i)
	{
		ASSERT_EQ(nom_values[i], nom_values_result[i]);
	}

	ASSERT_EQ(parser->get_relation(), "test_nominal");
	ASSERT_EQ(parser->get_feature_names().size(), 2);
	ASSERT_EQ(parser->get_feature_names()[0], "Twist n\' Shout");
	ASSERT_EQ(parser->get_feature_names()[1], "VAR2");
}