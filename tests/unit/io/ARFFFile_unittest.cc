/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Gil Hoben
 */

#include <shogun/io/ARFFFile.h>

#include <gtest/gtest.h>
#include <shogun/features/DenseFeatures.h>

using namespace shogun;

TEST(ARFFFileTest, Parse_numeric)
{
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

	auto parser = std::make_unique<ARFFDeserializer>(s);
	parser->read();
	auto result = parser->get_features();
	ASSERT_EQ(result->get_num_feature_obj(), 2);

	auto col1 = result->get_feature_obj(0)->as<CDenseFeatures<float64_t>>();
	auto col2 = result->get_feature_obj(1)->as<CDenseFeatures<float64_t>>();

	ASSERT_EQ(col1->get_feature_matrix()[0], 50);
	ASSERT_EQ(col1->get_feature_matrix()[1], 45);

	ASSERT_EQ(col2->get_feature_matrix()[0], 5.1);
	ASSERT_EQ(col2->get_feature_matrix()[1], 4.13);
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
	ASSERT_EQ(result->get_num_feature_obj(), 2);

	auto col1 = result->get_feature_obj(0)->as<CDenseFeatures<float64_t>>();
	auto mat1 = col1->get_feature_matrix();
	auto col2 = result->get_feature_obj(1)->as<CDenseFeatures<float64_t>>();
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
	ASSERT_EQ(result->get_num_feature_obj(), 2);
	auto col1 = result->get_feature_obj(0)->as<CStringFeatures<char>>();
	auto col2 = result->get_feature_obj(1)->as<CDenseFeatures<float64_t>>();
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

	auto parser = std::make_unique<ARFFDeserializer>(s);
	parser->read();
	auto result = parser->get_features();
	ASSERT_EQ(result->get_num_feature_obj(), 2);

	auto col1 = result->get_feature_obj(0)->as<CDenseFeatures<float64_t>>();
	auto mat1 = col1->get_feature_matrix();
	auto col2 = result->get_feature_obj(1)->as<CDenseFeatures<float64_t>>();
	auto mat2 = col2->get_feature_matrix();
	ASSERT_EQ(mat1.size(), 6);
	for (int i = 0; i < 6; ++i)
	{
		ASSERT_EQ(mat1[i], solution1[i]);
		ASSERT_EQ(mat2[i], solution2[i]);
	}
	ASSERT_EQ(parser->get_relation(), "test_nominal");
	ASSERT_EQ(parser->get_feature_names().size(), 2);
	ASSERT_EQ(parser->get_feature_names()[0], "Twist n\' Shout");
	ASSERT_EQ(parser->get_feature_names()[1], "VAR2");
}