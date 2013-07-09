/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Evgeniy Andreev (gsomix)
 */

#include <shogun/base/init.h>
#include <shogun/io/LineReader.h>
#include <shogun/lib/SGVector.h>
#include <gtest/gtest.h>

#include <cstring>

using namespace shogun;

const int max_line_length = 256;
const int max_num_lines = 100;

TEST(LineReaderTest, constructor)
{
	CLineReader* reader;

	FILE* fin=fopen("io/LineReader_unittest.cc", "r");
	reader=new CLineReader(fin);
	EXPECT_TRUE(reader->has_next_line());
	SG_UNREF(reader);

	fclose(fin);
}

TEST(LineReaderTest, read_yourself)
{
	
	SGVector<char> strings[max_num_lines];
	SGVector<char> temp_string(max_line_length);
	int lines_count;
	
	CLineReader* reader;

	FILE* fin=fopen("io/LineReader_unittest.cc", "r");
	reader=new CLineReader(max_line_length, fin);
	EXPECT_TRUE(reader->has_next_line());

	// read all strings from source code using LineReader
	lines_count=0;
	while (reader->has_next_line())
	{
		strings[lines_count]=reader->get_next_line();
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
	}
	SG_UNREF(reader);

	fclose(fin);	
}
