/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Evgeniy Andreev (gsomix)
 */

#include <shogun/base/init.h>
#include <shogun/lib/CircularBuffer.h>
#include <shogun/lib/DelimiterTokenizer.h>
#include <gtest/gtest.h>

#include <cstring>

using namespace shogun;

TEST(CircularBufferTest, constructor)
{
	int buffer_size=1024;

	CCircularBuffer* buffer;

	// default constructor
	// buffer should be 0-sized
	// w/o available and contained elements
	buffer=new CCircularBuffer();
	EXPECT_EQ(0, buffer->available());
	EXPECT_EQ(0, buffer->num_bytes_contained());
	SG_UNREF(buffer);

	// constructor with parameters
	// now some elements are available
	// but still none are contained
	buffer=new CCircularBuffer(buffer_size);
	EXPECT_EQ(buffer_size, buffer->available());
	EXPECT_EQ(0, buffer->num_bytes_contained());
	SG_UNREF(buffer);
}

TEST(CircularBufferTest, push_pop)
{
	int buffer_size=64;
	int result;

	CCircularBuffer* buffer;
	SGVector<char> test_string((char*)"CircularBuffer", 14, false);
	SGVector<char> tmp_string;

	// default constructor
	// we cannot push to 0-sized buffer
	buffer=new CCircularBuffer();
	result=buffer->push(test_string);
	EXPECT_EQ(0, result);
	SG_UNREF(buffer);

	// push
	// try write to buffer and check state
	buffer=new CCircularBuffer(buffer_size);
	result=buffer->push(test_string);
	EXPECT_EQ(test_string.vlen, result);
	EXPECT_EQ(buffer_size-test_string.vlen, buffer->available());
	EXPECT_EQ(test_string.vlen, buffer->num_bytes_contained());

	// pop
	// same with pop
	tmp_string=buffer->pop(test_string.vlen);
	EXPECT_EQ(test_string.vlen, tmp_string.vlen);
	EXPECT_EQ(buffer_size, buffer->available());
	EXPECT_EQ(0, buffer->num_bytes_contained());
	for (int i=0; i<tmp_string.vlen; i++)
	{
		EXPECT_EQ(test_string.vector[i], tmp_string.vector[i]);
	}

	SG_UNREF(buffer);
}

TEST(CircularBufferTest, stress_test)
{
	// we push string that slightly longer than half of buffer
	// at each iteration data is recorded with litte shift
	// so with many iterations we can test all memory of buffer
	// for write, read and find opertions
	int repeat=1024;
	int buffer_size=64;

	CCircularBuffer* buffer;
	CDelimiterTokenizer* tokenizer;

	SGVector<char> tmp_string;
	SGVector<char> test_string((char*)"all your bayes are belong to us! ", 33, false);

	buffer=new CCircularBuffer(buffer_size);

	tokenizer=new CDelimiterTokenizer();
	tokenizer->delimiters[' ']=1;
	SG_REF(tokenizer);

	buffer->set_tokenizer(tokenizer);

	EXPECT_EQ(buffer_size, buffer->available());
	EXPECT_EQ(0, buffer->num_bytes_contained());

	for (int i=0; i<repeat; i++)
	{
		buffer->push(test_string);

		int num_read;
		index_t start;
		while ((num_read=buffer->next_token_idx(start))>0)
		{
			buffer->skip_characters(start);
			tmp_string=buffer->pop(num_read);
			buffer->skip_characters(1);
		}
	}

	EXPECT_EQ(buffer_size, buffer->available());
	EXPECT_EQ(0, buffer->num_bytes_contained());

	SG_UNREF(buffer);
	SG_UNREF(tokenizer);
}
