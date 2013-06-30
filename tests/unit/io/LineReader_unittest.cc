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
	int lines_count;

	char* temp=NULL;
	size_t temp_size=0;
	
	CLineReader* reader;

	FILE* fin=fopen("io/LineReader_unittest.cc", "r");
	reader=new CLineReader(fin, max_line_length);
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
	while (getline(&temp, &temp_size, fin)!=-1)
	{
		for (int i=0; i<strings[lines_count].vlen; i++)
		{
			EXPECT_EQ(temp[i], strings[lines_count].vector[i]);
		}
		lines_count++;
	}
	SG_UNREF(reader);

	fclose(fin);	
}
