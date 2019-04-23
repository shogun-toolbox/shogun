/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Evgeniy Andreev, Thoralf Klein, Bjoern Esser, Viktor Gal
 */

#include <shogun/io/LineReader.h>
#include <shogun/lib/DelimiterTokenizer.h>
#include <shogun/lib/SGVector.h>

#include <cstring>
#include <gtest/gtest.h>

using namespace shogun;

const int max_line_length = 1024*1024;
const int max_num_lines = 100;

TEST(LineReaderTest, constructor)
{
	FILE* fin=fopen(__FILE__, "r");

	auto tokenizer=std::make_shared<DelimiterTokenizer>();
	tokenizer->delimiters['\n']=1;


	auto reader=std::make_shared<LineReader>(fin, tokenizer);
	EXPECT_TRUE(reader->has_next());

	fclose(fin);
}

TEST(LineReaderTest, read_yourself)
{
	SGVector<char> strings[max_num_lines];
	SGVector<char> temp_string(max_line_length);
	int lines_count=0;


	FILE* fin=fopen(__FILE__, "r");

	auto tokenizer=std::make_shared<DelimiterTokenizer>();
	tokenizer->delimiters['\n']=1;


	auto reader=std::make_shared<LineReader>(max_line_length, fin, tokenizer);
	EXPECT_TRUE(reader->has_next());

	// read all strings from source code using LineReader
	while (reader->has_next())
	{
		strings[lines_count]=reader->read_line();
		lines_count++;
	}

	// now read lines using getline
	// and check it on equality
	rewind(fin);
	lines_count=0;
	while (fgets(temp_string.vector, temp_string.vlen, fin)!=NULL)
	{
		for (int i=0; i<strings[lines_count].vlen; i++)
		{
			EXPECT_EQ(temp_string.vector[i], strings[lines_count].vector[i]);
		}
		lines_count++;
		temp_string=SGVector<char>(max_line_length);
	}

	fclose(fin);
}

