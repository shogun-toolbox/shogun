/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Gil Hoben
 */

#include <shogun/io/ARFFFile.h>

#include <gtest/gtest.h>

using namespace shogun;

TEST(ARFFFileTest, Parse_numeric)
{
	std::string test = "@relation test \n"
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
	auto s = std::dynamic_pointer_cast<std::istream>(ss);

	SGVector<float64_t> solution{50, 45, 5.1, 4.13};

	auto parser = std::make_unique<ARFFDeserializer>(s);
	parser->read();
	auto result = parser->get_data();
	ASSERT_EQ(result.size(), 4);
	for (int i = 0; i < 4; ++i)
		ASSERT_EQ(result[i], solution[i]);
}

TEST(ARFFFileTest, Parse_datetime)
{
	std::string test = "@relation test \n"
	                   "% \n"
	                   "% \n"
	                   "@attribute PERIOD_DATE date \"yyyy-MM-dd Z\" \n"
	                   "@attribute VAR1 numeric \n"
	                   "@attribute VAR2 numeric \n"
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
	auto s = std::dynamic_pointer_cast<std::istream>(ss);

	SGVector<float64_t> solution{1547078400, 1549760400, 1552176000,
	                             1554854400, 1557446400, 1560124800};

	auto parser = std::make_unique<ARFFDeserializer>(s);
	parser->read();
	auto result = parser->get_data();
	ASSERT_EQ(result.size(), 12);
	for (int i = 0; i < 6; ++i)
		ASSERT_EQ(result[i], solution[i]);
}
